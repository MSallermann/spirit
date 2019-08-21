 #pragma once

#include <utility/Constants.hpp>

#include <algorithm>

using namespace Utility;

template <> inline
void Method_Solver<Solver::NCG_Stereo>::Initialize ()
{
    this->jmax    = 500;    // max iterations
    this->n       = 50;     // restart every n iterations XXX: what's the appropriate val?

    // Polak-Ribiere criterion
    this->beta  = scalarfield( this->noi, 0 );

    this->forces  = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->a_coords              = std::vector< vector2field >( this->noi, vector2field(this->nos, {0,0}) );
    this->a3_coords             = std::vector< scalarfield >( this->noi, scalarfield(this->nos, 1) );
    this->a_coords_displaced    = std::vector< vector2field >( this->noi, vector2field(this->nos, {0,0}) );
    this->a_directions          = std::vector< vector2field >( this->noi,  vector2field( this->nos, { 0, 0 } ) );
    this->a_residuals           = std::vector< vector2field > ( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->a_residuals_last      = std::vector< vector2field > ( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->a_residuals_displaced = std::vector< vector2field > ( this->noi, vector2field( this->nos, { 0, 0 } ) );
    this->jacobians             = std::vector< std::vector<Eigen::Matrix<scalar,3,2> > >( this->noi, std::vector<Eigen::Matrix<scalar,3,2 > > (this->nos));
    this->jacobians_displaced   = std::vector< std::vector<Eigen::Matrix<scalar,3,2> > >( this->noi, std::vector<Eigen::Matrix<scalar,3,2 > > (this->nos));
    this->reset_ncg             = std::vector< bool >( this->noi, false );

    this->configurations_displaced = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for (int i=0; i<this->noi; i++)
    {
        configurations_displaced[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos, {0, 0, 0} ) );
        Solver_Kernels::ncg_stereo_spins_to_a( *this->configurations[i], this->a_coords[i], this->a3_coords[i] );
    }

    // // Debug
    // for(int img=0; img<this->noi; ++img)
    // {
    //     for(int i=0; i<this->nos; ++i)
    //     {
    //         fmt::print("<NCG_Atlas> : Image [{}], spin [{}], {} = {}, {} = {}, {} = {}\n", img, i, "a1", a_coords[img][i][0], "a2", a_coords[img][i][1], "a3", a3_coords[img][i]);
    //     }
    // }
};


/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121)

    Template instantiation of the Simulation class for use with the NCG Solver
    The method of nonlinear conjugate gradients is a proven and effective solver.
*/

template <> inline
void Method_Solver<Solver::NCG_Stereo>::Iteration()
{
    // Current force
    this->Calculate_Force( this->configurations, this->forces );

    // Calculate Residuals for current parameters and save the old residuals
    #pragma omp parallel for collapse
    for( int img=0; img<this->noi; img++ )
    {
        Solver_Kernels::ncg_stereo_eval(this->a_residuals[img], this->a_residuals_last[img], *this->configurations[img], this->forces[img], this->jacobians[img], this->a3_coords[img]);
    }

    #pragma omp parallel for
    for (int img=0; img<this->noi; img++)
    {
        auto& image                 = *this->configurations[img];
        auto& image_displaced       = *this->configurations_displaced[img];
        auto& beta                  = this->beta[img];
        auto& a_coords              = this->a_coords[img];
        auto& a3_coords             = this->a3_coords[img];
        auto& a_coords_displaced    = this->a_coords_displaced[img];
        auto& a_directions          = this->a_directions[img];
        auto& a_residuals           = this->a_residuals[img];
        auto& a_residuals_last      = this->a_residuals_last[img];
        auto& a_residuals_displaced = this->a_residuals_displaced[img];
        auto& jacobians             = this->jacobians[img];
        auto& jacobians_displaced   = this->jacobians_displaced[img];

        // Calculate beta
        scalar top = 0, bot = 0;
        if(this->reset_ncg[img])
        {
            this->beta[img] = 0;
        } else {
            for(int i=0; i<this->nos; i++)
            {
                top += a_residuals[i].dot(a_residuals[i] - a_residuals_last[i]);
                bot += a_residuals_last[i].dot(a_residuals_last[i]);
            }
            if( std::abs(bot) > 0 ) {
                this->beta[img] = std::max(top/bot, scalar(0));
            } else {
                this->beta[img] = 0;
            }
        }

        // fmt::print("beta = {}\n", this->beta[img]);

        // Calculate new search direction
        #pragma omp parallel for
        for(int i=0; i<this->nos; i++)
        {
            a_directions[i] *= beta;
            a_directions[i] += a_residuals[i];
        }

        // Calculate the energy at current position as well as the directional derivative
        scalar E0 = this->systems[img]->hamiltonian->Energy(image);
        scalar g0 = 0;
        scalar gr = 0;
        scalar a_direction_norm = Solver_Kernels::ncg_stereo_atlas_norm(a_directions);
        #pragma omp parallel for reduction(+:g0) reduction(+:gr)
        for( int i=0; i<image.size(); ++i )
        {
            g0 += a_residuals[i].dot(a_directions[i]) / a_direction_norm;
            gr += a_residuals_displaced[i].dot(a_directions[i]) / a_direction_norm;
        }
        fmt::print("g0 = {}\n", g0);
        fmt::print("gr = {}\n", gr);

        // Calculate displaced spin directions and residual
        scalar step_size = 1;
        for(int i=0; i<this->nos; i++)
        {
            a_coords_displaced[i] = a_coords[i] + step_size * a_directions[i];
            fmt::print("a_directions    = {} {}\n", a_directions[i][0], a_directions[i][1]);
            fmt::print("Current coord   = {} {}\n", a_coords[i][0], a_coords[i][0]);
            fmt::print("displaced coord = {} {}\n", a_coords_displaced[i][0], a_coords_displaced[i][1]);
        }

        Solver_Kernels::ncg_stereo_a_to_spins(a_coords_displaced, a3_coords, image_displaced);
        Solver_Kernels::ncg_stereo_eval(a_residuals_displaced, a_residuals_displaced, image_displaced, this->forces[img], jacobians_displaced, a3_coords);

        int n_steps = 1;
        // const Data::Spin_System & system, const vector2field & a_residuals_displaced, vector2field & a_directions, vector2field & a_coords, scalarfield a3_coords, scalar E0, scalar g0, scalar & step_size, int & n_step
        this->reset_ncg[img] = Solver_Kernels::ncg_stereo_line_search(*this->systems[img], image, image_displaced, a_residuals, a_residuals_displaced, a_directions, a_coords, a_coords_displaced, a3_coords, step_size, n_steps, E0, g0, gr, a_direction_norm);
        this->reset_ncg[img] = false;

        fmt::print("------\n");
        for(int i=0; i<image.size(); i++)
        {
            fmt::print("old a coord = {} {}\n", a_coords[i][0], a_coords[i][1]);
            fmt::print("new a coord = {} {}\n", a_coords_displaced[i][0], a_coords_displaced[i][1]);
            fmt::print("old image = {} {} {}\n", image[i][0], image[i][1], image[i][2]);
            fmt::print("new image = {} {} {}\n", image_displaced[i][0], image_displaced[i][1], image_displaced[i][2]);

            a_coords[i] = a_coords_displaced[i];
            image[i]    = image_displaced[i];
        }
        fmt::print("------\n");
        Solver_Kernels::ncg_stereo_check_coordinates(image, a_coords, a3_coords, a_directions);
    }
}

template <> inline
std::string Method_Solver<Solver::NCG_Stereo>::SolverName()
{
    return "NCG_Atlas";
}

template <> inline
std::string Method_Solver<Solver::NCG_Stereo>::SolverFullName()
{
    return "Nonlinear conjugate gradients with Atlas method";
}