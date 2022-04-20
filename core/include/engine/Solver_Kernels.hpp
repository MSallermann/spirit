#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_KERNELS_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_KERNELS_HPP

#include <memory>
#include <vector>
#include <limits>

#include <Eigen/Core>
#include <complex>
#include <data/Spin_System.hpp>
#include <engine/Backend_par.hpp>
#include <engine/Backend_seq.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

namespace Engine
{
namespace Solver_Kernels
{

template<typename Propagate>
scalar backtracking_linesearch(
    Engine::Hamiltonian & ham, scalar linear_coeff_delta_e,
    scalar quadratic_coeff_delta_e, scalar energy_current, scalar ratio, scalar tau, const field<Vector3> & spins,
    field<Vector3> & spins_buffer, Propagate prop )
{

    scalar delta_e          = 0.0;
    scalar delta_e_expected = 0.0;

    scalar alpha = 1.0 / tau; // set alpha to 1/tau so that 1 is the first step size that is tried
    int nos      = spins.size();
    static vectorfield gradient_throwaway = vectorfield(nos);

    int MAX_ITER        = 20;
    int iter            = 0;
    bool stop_criterion = false;

    // fmt::print( "======\n" );

    // fmt::print( "spins[0] = {}\n", spins[0].transpose() );

    while( !stop_criterion )
    {
        iter++;
        alpha *= tau;
        delta_e_expected = linear_coeff_delta_e * alpha + quadratic_coeff_delta_e * alpha * alpha;
        scalar error_delta_e_expected = std::numeric_limits<scalar>::epsilon() * std::sqrt( linear_coeff_delta_e * alpha * linear_coeff_delta_e * alpha + quadratic_coeff_delta_e * alpha * alpha * quadratic_coeff_delta_e * alpha * alpha);

        // Propagate spins by alpha
        prop(spins, spins_buffer, alpha);
        // fmt::print( "spins_buffer[0] = {}", spins_buffer[0].transpose() );

        // Compute energy diff
        scalar energy_step = 0;
        ham.Gradient_and_Energy( spins_buffer, gradient_throwaway, energy_step ); // We are just interested in the energy, hence we dont need the oso gradient etc.

        scalar delta_e       = energy_step - energy_current;
        scalar error_delta_e = std::numeric_limits<scalar>::epsilon() * std::sqrt( (1.0+energy_step) * (1.0+energy_step) + (1.0+energy_current) * (1.0+energy_current) );

        scalar error_ratio   = std::abs(delta_e / delta_e_expected + std::numeric_limits<scalar>::epsilon()) * std::sqrt( std::pow(error_delta_e / delta_e, 2) + std::pow(error_delta_e_expected / delta_e_expected, 2) );

        bool linesearch_applicable = error_ratio < ratio * 1e-2;

        fmt::print( "======\n" );
        fmt::print( "iter                        {}\n", iter );
        fmt::print( "ratio                       {}\n", ratio );
        fmt::print( "alpha ls                    {:.15f}\n", alpha );
        fmt::print( "energy_step                 {:.15f}\n", energy_step );
        fmt::print( "energy_current              {:.15f}\n", energy_current );
        fmt::print( "delta_e ls                  {:.15f} +- {:.15f}\n", delta_e, error_delta_e );
        fmt::print( "delta_e_expected ls         {:.15f} +- {:.15f}\n", delta_e_expected, error_delta_e_expected );
        fmt::print( "delta_e_expected/delta_e ls {:.15f}\n", delta_e_expected / delta_e );
        fmt::print( "criterion {}\n", std::abs( std::abs( delta_e_expected / delta_e ) - 1 ) );

        if(!linesearch_applicable)
        {
            return alpha;
        }

        stop_criterion = std::abs( std::abs( delta_e_expected / delta_e ) - 1 ) < ratio || iter >= MAX_ITER;
    };
    // fmt::print( "Finished line search\n" );
    return alpha;
}


// SIB
void sib_transform( const vectorfield & spins, const vectorfield & force, vectorfield & out );

// OSO coordinates
void oso_rotate( vectorfield & configuration, const vectorfield & searchdir, scalar alpha=1.0 );
void oso_calc_gradients( vectorfield & residuals, const vectorfield & spins, const vectorfield & forces );
scalar maximum_rotation( const vectorfield & searchdir, scalar maxmove );

// Atlas coordinates
void atlas_calc_gradients(
    vector2field & residuals, const vectorfield & spins, const vectorfield & forces, const scalarfield & a3_coords );
void atlas_rotate(
    std::vector<std::shared_ptr<vectorfield>> & configurations, const std::vector<scalarfield> & a3_coords,
    const std::vector<vector2field> & searchdir );
bool ncg_atlas_check_coordinates(
    const std::vector<std::shared_ptr<vectorfield>> & spins, std::vector<scalarfield> & a3_coords, scalar tol = -0.6 );
void lbfgs_atlas_transform_direction(
    std::vector<std::shared_ptr<vectorfield>> & configurations, std::vector<scalarfield> & a3_coords,
    std::vector<field<vector2field>> & atlas_updates, std::vector<field<vector2field>> & grad_updates,
    std::vector<vector2field> & searchdir, std::vector<vector2field> & grad_pr, scalarfield & rho );

// LBFGS
template<typename Vec>
void lbfgs_get_searchdir(
    int & local_iter, scalarfield & rho, scalarfield & alpha, std::vector<field<Vec>> & q_vec,
    std::vector<field<Vec>> & searchdir, std::vector<field<field<Vec>>> & delta_a,
    std::vector<field<field<Vec>>> & delta_grad, const std::vector<field<Vec>> & grad,
    std::vector<field<Vec>> & grad_pr, const int num_mem, const scalar maxmove )
{
    // std::cerr << "lbfgs searchdir \n";
    static auto dot = [] SPIRIT_LAMBDA( const Vec & v1, const Vec & v2 ) { return v1.dot( v2 ); };
    static auto set = [] SPIRIT_LAMBDA( const Vec & x ) { return x; };

    scalar epsilon = sizeof( scalar ) == sizeof( float ) ? 1e-30 : 1e-300;

    int noi     = grad.size();
    int nos     = grad[0].size();
    int m_index = local_iter % num_mem; // memory index
    int c_ind   = 0;

    if( local_iter == 0 ) // gradient descent
    {
        for( int img = 0; img < noi; img++ )
        {
            Backend::par::set( grad_pr[img], grad[img], set );
            auto & dir   = searchdir[img];
            auto & g_cur = grad[img];
            Backend::par::set( dir, g_cur, [] SPIRIT_LAMBDA( const Vec & x ) { return -x; } );
            auto & da = delta_a[img];
            auto & dg = delta_grad[img];
            for( int i = 0; i < num_mem; i++ )
            {
                rho[i]   = 0.0;
                auto dai = da[i].data();
                auto dgi = dg[i].data();
                Backend::par::apply(
                    nos,
                    [dai, dgi] SPIRIT_LAMBDA( int idx )
                    {
                        dai[idx] = Vec::Zero();
                        dgi[idx] = Vec::Zero();
                    } );
            }
        }
    }
    else
    {
        for( int img = 0; img < noi; img++ )
        {
            auto da   = delta_a[img][m_index].data();
            auto dg   = delta_grad[img][m_index].data();
            auto g    = grad[img].data();
            auto g_pr = grad_pr[img].data();
            auto sd   = searchdir[img].data();
            Backend::par::apply(
                nos,
                [da, dg, g, g_pr, sd] SPIRIT_LAMBDA( int idx )
                {
                    da[idx] = sd[idx];
                    dg[idx] = g[idx] - g_pr[idx];
                } );
        }

        scalar rinv_temp = 0;
        for( int img = 0; img < noi; img++ )
            rinv_temp += Backend::par::reduce( delta_grad[img][m_index], delta_a[img][m_index], dot );

        if( rinv_temp > epsilon )
            rho[m_index] = 1.0 / rinv_temp;
        else
        {
            local_iter = 0;
            return lbfgs_get_searchdir(
                local_iter, rho, alpha, q_vec, searchdir, delta_a, delta_grad, grad, grad_pr, num_mem, maxmove );
        }

        for( int img = 0; img < noi; img++ )
            Backend::par::set( q_vec[img], grad[img], set );

        for( int k = num_mem - 1; k > -1; k-- )
        {
            c_ind       = ( k + m_index + 1 ) % num_mem;
            scalar temp = 0;
            for( int img = 0; img < noi; img++ )
                temp += Backend::par::reduce( delta_a[img][c_ind], q_vec[img], dot );

            alpha[c_ind] = rho[c_ind] * temp;
            for( int img = 0; img < noi; img++ )
            {
                auto q = q_vec[img].data();
                auto a = alpha.data();
                auto d = delta_grad[img].data();
                Backend::par::apply(
                    nos, [c_ind, q, a, d] SPIRIT_LAMBDA( int idx ) { q[idx] += -a[c_ind] * d[c_ind][idx]; } );
            }
        }

        scalar dy2 = 0;
        for( int img = 0; img < noi; img++ )
            dy2 += Backend::par::reduce( delta_grad[img][m_index], delta_grad[img][m_index], dot );

        for( int img = 0; img < noi; img++ )
        {
            scalar rhody2     = dy2 * rho[m_index];
            scalar inv_rhody2 = 0.0;
            if( rhody2 > epsilon )
                inv_rhody2 = 1.0 / rhody2;
            else
                inv_rhody2 = 1.0 / ( epsilon );
            Backend::par::set(
                searchdir[img], q_vec[img], [inv_rhody2] SPIRIT_LAMBDA( const Vec & q ) { return inv_rhody2 * q; } );
        }

        for( int k = 0; k < num_mem; k++ )
        {
            if( local_iter < num_mem )
                c_ind = k;
            else
                c_ind = ( k + m_index + 1 ) % num_mem;

            scalar rhopdg = 0;
            for( int img = 0; img < noi; img++ )
                rhopdg += Backend::par::reduce( delta_grad[img][c_ind], searchdir[img], dot );

            rhopdg *= rho[c_ind];

            for( int img = 0; img < noi; img++ )
            {
                auto sd   = searchdir[img].data();
                auto alph = alpha[c_ind];
                auto da   = delta_a[img][c_ind].data();
                Backend::par::apply(
                    nos, [sd, alph, da, rhopdg] SPIRIT_LAMBDA( int idx ) { sd[idx] += ( alph - rhopdg ) * da[idx]; } );
            }
        }

        for( int img = 0; img < noi; img++ )
        {
            auto g    = grad[img].data();
            auto g_pr = grad_pr[img].data();
            auto sd   = searchdir[img].data();
            Backend::par::apply(
                nos,
                [g, g_pr, sd] SPIRIT_LAMBDA( int idx )
                {
                    g_pr[idx] = g[idx];
                    sd[idx]   = -sd[idx];
                } );
        }
    }
    local_iter++;
}

} // namespace Solver_Kernels
} // namespace Engine

#endif