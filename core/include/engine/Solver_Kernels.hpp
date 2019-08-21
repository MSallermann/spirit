#pragma once
#ifndef SOLVER_KERNELS_H
#define SOLVER_KERNELS_H

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <engine/Vectormath_Defines.hpp>
#include <data/Spin_System.hpp>

namespace Engine
{
namespace Solver_Kernels
{
    // SIB
    void sib_transform(const vectorfield & spins, const vectorfield & force, vectorfield & out);

    // NCG
    inline scalar inexact_line_search(scalar r, scalar E0, scalar Er, scalar g0, scalar gr)
    {
        scalar c1 = - 2*(Er - E0) / std::pow(r, 3) + (gr + g0) / std::pow(r, 2);
        scalar c2 = 3*(Er - E0) / std::pow(r, 2) - (gr + 2*g0) / r;
        scalar c3 = g0;
        // scalar c4 = E0;
        return std::abs( (-c2 + std::sqrt(c2*c2 - 3*c1*c3)) / (3*c1) ) / r;
    }
    scalar ncg_beta_polak_ribiere(vectorfield & image, vectorfield & force, vectorfield & residual,
        vectorfield & residual_last, vectorfield & force_virtual);
    scalar ncg_dir_max(vectorfield & direction, vectorfield & residual, scalar beta, vectorfield & axis);
    void full_inexact_line_search(const Data::Spin_System & system,
        const vectorfield & image, vectorfield & image_displaced,
        const vectorfield & force, const vectorfield & force_displaced,
        const scalarfield & angle, const vectorfield & axis, scalar & step_size, int & n_step);
    void ncg_rotate(vectorfield & direction, vectorfield & axis, scalarfield & angle,
        scalar normalization, const vectorfield & image, vectorfield & image_displaced);
    void ncg_rotate_2(vectorfield & image, vectorfield & residual, vectorfield & axis,
        scalarfield & angle, scalar step_size);

    void ncg_stereo_eval( std::vector<Vector2> & residuals, std::vector<Vector2> & residuals_last, const vectorfield & spins,
                          const vectorfield & forces, std::vector<Eigen::Matrix<scalar,3,2>> & jacobians, const scalarfield & a3_coords );
    void ncg_stereo_a_to_spins(const vector2field & a_coords, const scalarfield & a3_coords, vectorfield & spins);
    void ncg_stereo_spins_to_a(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords);
    void ncg_stereo_check_coordinates(const vectorfield & spins, vector2field & a_coords, scalarfield & a3_coords, vector2field & a_directions);

    // bool ncg_stereo_line_search( const Data::Spin_System & system, const vector2field & a_residuals_displaced, vector2field & a_directions, vector2field & a_coords, scalarfield a3_coords, scalar E0, scalar g0, scalar & step_size, int & n_step);

    bool ncg_stereo_line_search( const Data::Spin_System & system, vectorfield & image, vectorfield & image_displaced,
                                 vector2field & a_residuals, const vector2field & a_residuals_displaced, vector2field & a_directions,
                                 vector2field & a_coords, vector2field & a_coords_displaced, scalarfield a3_coords, scalar & step_size,
                                 int & n_step, scalar E0, scalar g0, scalar gr, scalar a_direction_norm);

    inline scalar ncg_stereo_atlas_norm(const vector2field & a_coords)
    {
        scalar dist = 0;
        #pragma omp parallel for reduction(+:dist)
        for (unsigned int i = 0; i < a_coords.size(); ++i)
            dist += (a_coords[i]).squaredNorm();
        return sqrt(dist);
    }

    inline scalar ncg_stereo_atlas_distance(const vector2field & a_coords1, const vector2field & a_coords2)
    {
        scalar dist = 0;
        #pragma omp parallel for reduction(+:dist)
        for (unsigned int i = 0; i < a_coords2.size(); ++i)
            dist += (a_coords1[i] - a_coords2[i]).squaredNorm() ;
        return sqrt(dist);
    }
}
}

#endif