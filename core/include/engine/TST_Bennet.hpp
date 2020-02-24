#pragma once
#ifndef TST_BENNET_HPP
#define TST_BENNET_HPP

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
    namespace TST_Bennet
    {
        void Calculate(Data::TST_Bennet_Info & tst_bennet_info, int n_iterations_bennet);

        bool Get_Unstable_Mode(const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors);

        void Bennet_Minimum(int n_iteration, int n_initial, int n_decor, field<scalar> & bennet_results, const MatrixX & hessian_minimum, const MatrixX & hessian_sp, scalar energy_barrier);

        void Bennet_SP(int n_iteration, int n_initial, int n_decor, field<scalar> & vel_perp_results, field<scalar> & bennet_results, const MatrixX & hessian_sp, const MatrixX & hessian_minimum, const VectorX & perpendicular_velocity, scalar energy_barrier);

        struct MC_Tracker
        {
            // Total stats
            int n_trials      = 0;
            int n_rejected    = 0;
            scalar dist_width = 1;

            // Tracker for adaptive width
            scalar target_rejection_ratio = 0.5;
            int n_memory       = 100;
            int n_rejected_cur = 0;
            scalar rejection_ratio_cur = 0;

            void adapt(bool rejected)
            {
                if( n_trials % n_memory == 0 )
                {
                    if(n_rejected_cur > 0)
                    {
                        rejection_ratio_cur = scalar(n_rejected_cur) / scalar(n_memory);
                        dist_width = dist_width * target_rejection_ratio / rejection_ratio_cur;
                    } else {
                        dist_width *= 2.0;
                    }
                    n_rejected_cur = 0;
                }
                if (rejected) {
                    n_rejected_cur++;
                    n_rejected++;
                }
                n_trials++;
            }
        };

        // Performs one iteration of a Metropolis Monte Carlo algorithm, restricted to a cartesian hyperlane. The hyperplane is assumed to contain the origin (zero vector).
        template<typename Action_Function>
        void Hyperplane_Metropolis(Action_Function action_func, VectorX & state_old, VectorX & state_new, const VectorX & plane_normal, std::mt19937 & prng, MC_Tracker & mc)
        {
            int n = plane_normal.size();
            auto distribution = std::uniform_real_distribution<scalar>(0, 1);

            for (int idx=0; idx<n; ++idx)
            {
                scalar dS = mc.dist_width * (2*distribution(prng)-1); // random perturbation
                scalar proj_on_normal = dS * plane_normal[idx];
                state_new[idx] += dS;

                // Correct the move such that it lies in the hyperplane
                for(int j=0; j<n; j++)
                {
                    state_new[j] -= proj_on_normal * plane_normal[j];
                }

                // Energy difference of configurations with and without displacement
                scalar E_old  = action_func( state_old );
                scalar E_new  = action_func( state_new );
                scalar E_diff = E_new - E_old;

                // Metropolis criterion: reject the step if energy rose
                if( E_diff > 1e-14 )
                {
                    // Exponential factor
                    scalar exp_ediff  = std::exp( -E_diff );

                    // Metropolis random number
                    scalar x_metropolis = distribution(prng);

                    // Only reject if random number is larger than exponential
                    if( exp_ediff < x_metropolis)
                    {
                        // Restore the state
                        for(int j=0; j<n; j++)
                            state_new[j] = state_old[j];
                        mc.adapt(true);
                        continue;
                    }
                }
                state_old = state_new;
                mc.adapt(false);
            }
        }

        // Performs one iteration of a Metropolis Monte Carlo algorithm with a frozen X coordinate.
        template<typename Action_Function>
        void Freeze_X_Metropolis(Action_Function action_func, VectorX & state, std::mt19937 & prng, MC_Tracker & mc)
        {
            int n = state.size();
            auto distribution = std::uniform_real_distribution<scalar>(0, 1);

            for (int idx=1; idx<n; ++idx)
            {
                scalar dS = mc.dist_width * (2*distribution(prng)-1); // random perturbation

                // Energy difference of configurations with and without displacement
                scalar E_diff = action_func(state, idx, dS);

                // Metropolis criterion: reject the step if energy rose
                if( E_diff > 1e-14 )
                {
                    // Exponential factor
                    scalar exp_ediff = std::exp( -E_diff );

                    // Metropolis random number
                    scalar x_metropolis = distribution(prng);

                    // Only reject if random number is larger than exponential
                    if( exp_ediff < x_metropolis)
                    {
                        mc.adapt(true); // Rejected
                        continue;
                    }
                }
                state[idx] += dS;
                mc.adapt(false); // Accepted
            }
        }
    }
}

#endif