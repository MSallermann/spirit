#pragma once
#ifndef TST_BENNET_HPP
#define TST_BENNET_HPP

#include "Spirit_Defines.h"
#include <engine/Method.hpp>
#include <data/Spin_System.hpp>
#include <data/Spin_System_Chain.hpp>
#include <engine/Vectormath_Defines.hpp>
#include <engine/Vectormath.hpp>

#include <fstream>
#include <string>

namespace Engine
{
    namespace TST_Bennet
    {
        void Calculate(Data::TST_Bennet_Info & tst_bennet_info, int n_chain, int n_iterations_bennet);

        bool Get_Unstable_Mode(const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
            MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors);

        void Bennet_Minimum(int n_iteration, int n_initial, int n_decor, field<scalar> & bennet_results, const MatrixX & hessian_minimum, const MatrixX & hessian_sp, scalar shift_constant);

        void Bennet_SP(int n_iteration, int n_initial, int n_decor, field<scalar> & vel_perp_results, field<scalar> & bennet_results, const MatrixX & hessian_sp, const MatrixX & hessian_minimum, const VectorX & perpendicular_velocity, scalar shift_constant);

        // A small function for debugging purposes
        inline void saveMatrix(std::string fname, const MatrixX & matrix)
        {
            std::cout << "Saving matrix to file: " << fname << "\n";
            std::ofstream file(fname);
            if(file && file.is_open())
            {
                file << matrix;
            } else {
                std::cerr << "Could not save matrix!";
            }
        }

        template<typename Field_Like>
        inline void saveField(std::string fname, const Field_Like & field)
        {
            int n = field.size();
            std::cout << "Saving field to file: " << fname << "\n";
            std::ofstream file(fname);
            if(file && file.is_open())
            {
                for(int i=0; i<n; i++)
                    file << field[i] << "\n";
            } else {
                std::cerr << "Could not save field!";
            }
        }

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

        template<typename MC_Callback>
        inline scalar Bennet_Chain_Sampling(int n_images, int n_bennet, int n_initial, int n_decor, const MatrixX & hessian_min, const MatrixX & hessian_sp, const scalar shift_constant, MC_Callback & call_back, scalar & Z_ratio, scalar & Z_ratio_err)
        {
            field<scalar> bennet_expectation_values_p(n_images, 0);
            field<scalar> bennet_expectation_values_m(n_images, 0);
            field<scalar> bennet_err_p(n_images, 0);
            field<scalar> bennet_err_m(n_images, 0);

            field<scalar> bennet_results_p(n_bennet, 0);
            field<scalar> bennet_results_m(n_bennet, 0);

            std::mt19937 prng = std::mt19937(803);

            Z_ratio = 1;
            Z_ratio_err = 0;

            #pragma omp prallel for
            for(int img=0; img<n_images; img++)
            {
                Vectormath::fill(bennet_results_p, 0);
                Vectormath::fill(bennet_results_m, 0);

                MC_Tracker mc;
                mc.target_rejection_ratio = 0.5;
                mc.dist_width = 1;

                scalar slider = 1.0 - scalar(img)/(n_images-1);

                auto energy_diff = [&](const VectorX & state_old, const int idx, const scalar dS) 
                {
                    return 0.5 * (slider * hessian_min(idx, idx)  + (1-slider) * hessian_sp(idx,idx)) * dS*dS + state_old.dot( slider * hessian_min.row(idx) + (1-slider) * hessian_sp.row(idx)) * dS;
                };

                auto bennet_exp_p = [&] (const VectorX & state)
                {
                    return 1.0 / (1.0 + std::exp(0.5 * state.transpose() * (hessian_min - hessian_sp)/scalar(n_images-1) * state - shift_constant/scalar(n_images-1)));
                };

                auto bennet_exp_m = [&] (const VectorX & state)
                {
                    return 1.0 / (1.0 + std::exp(0.5 * state.transpose() * (hessian_sp - hessian_min)/scalar(n_images-1) * state + shift_constant/scalar(n_images-1)));
                };

                VectorX state = VectorX::Zero(hessian_sp.row(0).size());

                // Thermalize
                for(int i=0; i<n_initial; i++)
                {
                    Freeze_X_Metropolis(energy_diff, state, prng, mc);
                }

                // Sample
                for(int i=0; i<n_bennet; i++)
                {
                    for(int j=0; j < n_decor; j++)
                        Freeze_X_Metropolis(energy_diff, state, prng, mc);

                    Freeze_X_Metropolis(energy_diff, state, prng, mc);

                    call_back(img, i, state);

                    if(img>0)
                        bennet_results_p[i] = bennet_exp_p(state);

                    if(img<n_images-1)
                        bennet_results_m[i] = bennet_exp_m(state);
                }

                for(int k=0; k<n_bennet; k++)
                {
                    bennet_expectation_values_p[img] += bennet_results_p[k]/n_bennet;
                    bennet_expectation_values_m[img] += bennet_results_m[k]/n_bennet;
                }

                for(int k=0; k<n_bennet; k++)
                {
                    bennet_err_p[img] += std::pow(bennet_results_p[k] - bennet_expectation_values_p[img], 2);
                    bennet_err_m[img] += std::pow(bennet_results_m[k] - bennet_expectation_values_m[img], 2);
                }

                bennet_err_p[img] = std::sqrt(bennet_err_p[img]) / n_bennet;
                bennet_err_m[img] = std::sqrt(bennet_err_m[img]) / n_bennet;
            }

            // Combine forward and backward sampling
            for(int img=0; img<n_images-1; img++)
            {
                Z_ratio *= bennet_expectation_values_m[img] / bennet_expectation_values_p[img+1];
            }

            for(int img=0; img<n_images-1; img++)
            {
                Z_ratio_err += std::pow(bennet_err_m[img]/bennet_expectation_values_m[img], 2) + std::pow(bennet_err_p[img+1]/bennet_expectation_values_p[img+1], 2);
            }

            Z_ratio_err = Z_ratio * std::sqrt(Z_ratio_err);
        }

    }
}

#endif