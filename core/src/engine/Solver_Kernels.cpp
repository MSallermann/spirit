#include <engine/Solver_Kernels.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <utility/Exception.hpp>

#include <Eigen/Dense>

#include <array>
#include <algorithm>

using namespace Utility;
using Utility::Constants::Pi;

namespace Engine
{
namespace Solver_Kernels
{
    #ifndef SPIRIT_USE_CUDA

    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out)
    {
        #pragma omp parallel for
        for (unsigned int i = 0; i < spins.size(); ++i)
        {
            Vector3 A = 0.5 * force[i];

            // 1/determinant(A)
            scalar detAi = 1.0 / (1 + A.squaredNorm());

            // calculate equation without the predictor?
            Vector3 a2 = spins[i] - spins[i].cross(A);

            out[i][0] = (a2[0] * (A[0] * A[0] + 1   ) + a2[1] * (A[0] * A[1] - A[2]) + a2[2] * (A[0] * A[2] + A[1])) * detAi;
            out[i][1] = (a2[0] * (A[1] * A[0] + A[2]) + a2[1] * (A[1] * A[1] + 1   ) + a2[2] * (A[1] * A[2] - A[0])) * detAi;
            out[i][2] = (a2[0] * (A[2] * A[0] - A[1]) + a2[1] * (A[2] * A[1] + A[0]) + a2[2] * (A[2] * A[2] + 1   )) * detAi;
        }
    }


    void full_inexact_line_search(const Data::Spin_System & system,
        const vectorfield & image, vectorfield & image_displaced,
        const vectorfield & force, const vectorfield & force_displaced,
        const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step)
    {
        // Calculate geodesic distance between image and image_displaced, if not pre-determined
        scalar r = Manifoldmath::dist_geodesic(image, image_displaced);
        if( r < 1e-6 )
        {
            step_size = 0;
            return;
        }

        scalar E0 = system.hamiltonian->Energy(image);
        // E0 = this->systems[img]->E;
        scalar Er = system.hamiltonian->Energy(image_displaced);

        scalar g0 = 0;
        scalar gr = 0;
        #pragma omp parallel for reduction(+:g0) reduction(+:gr)
        for( int i=0; i<image.size(); ++i )
        {
            g0 += force[i].dot(axis[i]);
            // TODO: displace dir by rotating into other spin
            // ACTUALLY: the direction is orthogonal to the rotation plane, so it does not change
            gr += ( image_displaced[i].cross(force_displaced[i]) ).dot(axis[i]);
        }

        // Approximate ine search
        ++n_step;
        step_size *= inexact_line_search(r, E0, Er, g0, gr);// * Constants::gamma / Constants::mu_B;
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image_displaced[i]);
        }
        Er = system.hamiltonian->Energy(image_displaced);
        // this->Calculate_Force( this->configurations_displaced, this->forces_displaced );
        if( n_step < 20 && Er > E0+std::abs(E0)*1e-4 )
        {
            full_inexact_line_search(system, image, image_displaced, force, force_displaced, angle, axis, step_size, n_step);
        }
    }

    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual)
    {
        scalar dt = 1e-3;
        // scalar dt = this->systems[0]->llg_parameters->dt;

        scalar top=0, bot=0;

        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set residuals
            residual_last[i] = residual[i];
            residual[i] = image[i].cross(force[i]);
            // TODO: this is for comparison with VP etc. and needs to be fixed!
            //       in fact, all solvers should use the force, not dt*force=displacement
            force_virtual[i] = dt * residual[i];

            bot += residual_last[i].dot(residual_last[i]);
            // Polak-Ribiere formula
            // TODO: this finite difference *should* be done covariantly (i.e. displaced)
            // Vectormath::rotate(residual_last[i], axis[i], step_size * angle[i], residual_last[i]);
            top += residual[i].dot( residual[i] - residual_last[i] );
            // Fletcher-Reeves formula
            // top += residual[i].dot( residual[i] );
        }
        if( std::abs(bot) > 0 )
            return std::max(top/bot, scalar(0));
        else
            return 0;
    }

    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis)
    {
        scalar dir_max = 0;
        #pragma omp parallel for reduction(max : dir_max)
        for( int i=0; i<direction.size(); ++i )
        {
            // direction = residual + beta*direction
            direction[i] = residual[i] + beta*direction[i];
            scalar dir_norm_i = direction[i].norm();
            // direction[i] = residual[i] + beta[img]*residual_last[i];
            axis[i] = direction[i].normalized();
            if( dir_norm_i > dir_max )
                dir_max = dir_norm_i;
            // dir_avg += dir_norm_i;
            // angle[i] = direction[i].norm();
        }
        return dir_max;
    }

    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle, scalar normalization, const vectorfield & image, vectorfield & image_displaced)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            // Set rotation
            angle[i] = direction[i].norm() / normalization;
            // Rotate
            Vectormath::rotate(image[i], axis[i], angle[i], image_displaced[i]);
        }
    }

    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis, scalarfield & angle, scalar step_size)
    {
        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            Vectormath::rotate(image[i], axis[i], step_size * angle[i], image[i]);
            Vectormath::rotate(residual[i], axis[i], step_size * angle[i], residual[i]);
        }
    }

    // Calculates the residuals for a certain spin configuration
    void ncg_stereo_eval(   std::vector<Vector2> & a_residuals, std::vector<Vector2> & a_residuals_last, const vectorfield & spins,
                            const vectorfield & forces, std::vector<Eigen::Matrix<scalar,3,2>> & jacobians, const scalarfield & a3_coords )
    {
        // Get Jacobians
        #pragma omp parallel for
        for(int i=0; i < spins.size(); i++)
        {
            auto & J        = jacobians[i];
            const auto & s  = spins[i];
            const auto & a3 = a3_coords[i];

            J(0,0) =  s[1]*s[1]  + s[2]*(s[2] + a3);
            J(0,1) = -s[0]*s[1];
            J(1,0) = -s[0]*s[1];
            J(1,1) =  s[0]*s[0]  + s[2]*(s[2] + a3);
            J(2,0) = -s[0]*(s[2] + a3);
            J(2,1) = -s[1]*(s[2] + a3);

            // If the two adresses are different we save the old residuals
            if( &a_residuals != &a_residuals_last )
                a_residuals_last[i] = a_residuals[i];

            a_residuals[i]      = forces[i].transpose() * J;
        }
    }

    void ncg_stereo_a_to_spins(const vector2field & a_coords, const scalarfield & a3_coords, vectorfield & spins)
    {
        #pragma omp_parallel_for
        for(int i=0; i<a_coords.size(); i++)
        {
            auto &        s = spins[i];
            const auto &  a = a_coords[i];
            const auto & a3 = a3_coords[i];

            s[0] = 2*a[0] / (1 + a[0]*a[0] + a[1]*a[1]);
            s[1] = 2*a[1] / (1 + a[0]*a[0] + a[1]*a[1]);
            s[2] = a3 * (1 - a[0]*a[0] - a[1]*a[1]) / (1 + a[0]*a[0] + a[1]*a[1]);
        }
    }

    void ncg_stereo_spins_to_a(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords)
    {
        #pragma omp_parallel_for
        for(int i=0; i<spins.size(); i++)
        {
            const auto & s = spins[i];
            auto &       a = a_coords[i];
            auto &      a3 = a3_coords[i];

            a3 = (s[2] > 0) ? 1 : -1;
            a[0] = s[0] / (1 + s[2]*a3);
            a[1] = s[1] / (1 + s[2]*a3);
        }
    }

    void ncg_stereo_check_coordinates(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions)
    {
        // Check if we need to reset the maps
        bool reset = false;

        #pragma omp parallel for
        for( int i=0; i<spins.size(); i++ )
        {
            // If for one spin the z component deviates too much from the pole we perform a reset for *all* spins
            // Note I am not sure why we reset for all spins ... but this in agreement with the method of F. Rybakov
            if(spins[i][2]*a3_coords[i] < -0.5)
            {
                reset = true;
                break;
            }
        }

        if(reset)
        {
            std::cout << fmt::format("resetting coordinates\n");
            for( int i=0; i<spins.size(); ++i )
            {
                const auto & s = spins[i];
                auto &       a = a_coords[i];
                auto &      a3 = a3_coords[i];

                if( spins[i][2]*a3_coords[i] < 0 )
                {
                    // Transform coordinates to optimal map
                    a3 = (s[2] > 0) ? 1 : -1;
                    a[0] = s[0] / (1 + s[2]*a3);
                    a[1] = s[1] / (1 + s[2]*a3);

                    // Also transform search direction to new map
                    a_directions[i] *= (1 - a3 * s[2]) / (1 + a3 * s[2]);
                }
            }
        }
    }


    bool ncg_stereo_line_search(const Data::Spin_System & system, vectorfield & image, vectorfield & image_displaced,
                                vector2field & a_residuals, const vector2field & a_residuals_displaced, vector2field & a_directions,
                                vector2field & a_coords, vector2field & a_coords_displaced, scalarfield a3_coords, scalar & step_size, int & n_step,
                                scalar E0, scalar g0, scalar gr, scalar a_direction_norm)
    {
        std::cout << fmt::format(" -- Performing Line Search -- \n");
        fmt::print("n_step = {}\n", n_step);
        fmt::print("step_size = {}\n", step_size);

        scalar Er = system.hamiltonian->Energy(image_displaced);

        // std::cout << fmt::format("energy           {:.15f}\n", E0);
        // std::cout << fmt::format("energy_displaced {:.15f}\n", Er);

        // // Get directional derivatives with respect to search direction in atlas space
        // scalar gr = 0;
        // #pragma omp parallel for reduction(+:g0) reduction(+:gr)
        // for( int i=0; i<image.size(); ++i )
        // {
        //     // g0 -= a_residuals[i].dot(a_directions[i]) / a_direction_norm;
        //     gr -= a_residuals_displaced[i].dot(a_directions[i]) / a_direction_norm;
        // }

        ++n_step;
        scalar factor = inexact_line_search(step_size, E0, Er, g0, gr);

        fmt::print("factor = {}\n", factor);
        if(!isnan(factor))
        {
            step_size *= factor;
        } else {
            // fmt::print("Going to E0\n");
            step_size = 0;
            return true;
        }

        #pragma omp parallel for
        for( int i=0; i<image.size(); ++i )
        {
            a_coords_displaced[i] = a_coords[i] + step_size * a_directions[i];
        }
        Solver_Kernels::ncg_stereo_a_to_spins(a_coords_displaced, a3_coords, image_displaced);

        Er = system.hamiltonian->Energy(image_displaced);
        // std::cout << fmt::format("new energy_displaced {:.15f}\n", Er);


        if( n_step < 20 && Er > E0 )
        {
            ncg_stereo_line_search(system, image, image_displaced,
                                a_residuals, a_residuals_displaced, a_directions,
                                a_coords, a_coords_displaced, a3_coords, step_size, n_step, E0, g0, gr, a_direction_norm);
            return false;
        }
        return true;
        // scalar  c1 = -2 * (energy_displaced - energy) / (r*r*r) + (gr + g0) / (r*r);
        // scalar  c2 =  3 * (energy_displaced - energy) / (r*r) - (gr + 2*g0) / r;
        // scalar& c3 = g0;
        // scalar& c4 = energy;
        // scalar temp = c2*c2 - 3*c1*c3;

        // fmt::print("a_direction_norm = {}\n", a_direction_norm);
        // std::cout << fmt::format("Polynomial values\n");
        // std::cout << fmt::format("    r = {}\n", r);
        // std::cout << fmt::format("   g0 = {}\n", g0);
        // std::cout << fmt::format("   gr = {}\n", gr);
        // std::cout << fmt::format("   c1 = {}\n", c1);
        // std::cout << fmt::format("   c2 = {}\n", c2);
        // std::cout << fmt::format("   c3 = {}\n", c3);
        // std::cout << fmt::format("   c4 = {}\n", c4);
        // std::cout << fmt::format("   c2*c2 - 3*c1*c3 = {}\n", temp);
    }
    #endif
}
}