#include <engine/TST_Bennet.hpp>
#include <engine/HTST.hpp>
#include <engine/Backend_par.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <GenEigsSolver.h>
#include <GenEigsRealShiftSolver.h>
#include <SymEigsSolver.h>

#include <iostream>

#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <math.h>
#include <fmt/format.h>

namespace C = Utility::Constants;

namespace Engine
{
    namespace TST_Bennet
    {
        void Calculate(Data::TST_Bennet_Info & tst_bennet_info, int n_iterations_bennet)
        {
            // This algorithm implements a statistical sampling method to calculate life times
            Log(Utility::Log_Level::All, Utility::Log_Sender::TST_Bennet, "--- Prefactor calculation");
            int N_INITIAL = 5000;
            int N_DECOR = 2;

            auto& image_minimum = *tst_bennet_info.minimum->spins;
            auto& image_sp = *tst_bennet_info.saddle_point->spins;
            int nos = image_minimum.size();

            scalar temperature = tst_bennet_info.minimum->llg_parameters->temperature;

            scalar e_sp;
            scalar e_minimum;
            scalar e_barrier;
            MatrixX orth_hessian_min, orth_hessian_sp;
            VectorX orth_perpendicular_velocity;
            {
                VectorX unstable_mode;
                VectorX eigenvalues;
                VectorX perpendicular_velocity = VectorX::Zero(2*nos);
                MatrixX tangent_basis, eigenvectors;

                e_sp = tst_bennet_info.saddle_point->hamiltonian->Energy(image_sp); // Energy

                vectorfield gradient_sp(nos, {0,0,0}); // Unconstrained gradient
                tst_bennet_info.saddle_point->hamiltonian->Gradient(image_sp, gradient_sp);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of saddle point embedding Hessian...");
                MatrixX hessian_sp = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                tst_bennet_info.saddle_point->hamiltonian->Hessian(image_sp, hessian_sp);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of saddle point constrained Hessian...");
                MatrixX hessian_sp_constrained = MatrixX::Zero(2*nos, 2*nos); // Constrained hessian

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Calculation of unstable mode...");
                Get_Unstable_Mode(image_sp, gradient_sp, hessian_sp, tangent_basis, hessian_sp_constrained, eigenvalues, eigenvectors);
                unstable_mode = eigenvectors.col(0);

                // Free memory
                gradient_sp.resize(0);
                eigenvalues.resize(0,0);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Calculation of dynamical matrix...");
                MatrixX dynamical_matrix(3*nos, 3*nos);
                HTST::Calculate_Dynamical_Matrix(image_sp, tst_bennet_info.saddle_point->geometry->mu_s, hessian_sp, dynamical_matrix);
                perpendicular_velocity =  eigenvectors.col(0).transpose() * tangent_basis.transpose() * dynamical_matrix * tangent_basis;

                // Free memory
                hessian_sp.resize(0,0);
                dynamical_matrix.resize(0,0);

                MatrixX tangent_basis_min;
                e_minimum = tst_bennet_info.minimum->hamiltonian->Energy(image_minimum); // Energy
                vectorfield gradient_minimum(nos, {0,0,0}); // Unconstrained gradient
                tst_bennet_info.minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of minimum embedding Hessian...");
                MatrixX hessian_minimum = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                tst_bennet_info.minimum->hamiltonian->Hessian(image_minimum, hessian_minimum);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of minimum constrained Hessian...");
                MatrixX hessian_minimum_constrained = MatrixX::Zero(2*nos, 2*nos);
                tangent_basis_min = MatrixX::Zero(3*nos, 2*nos);
                Manifoldmath::tangent_basis_spherical(image_minimum, tangent_basis_min);
                Manifoldmath::hessian_bordered(image_minimum, gradient_minimum, hessian_minimum, tangent_basis_min, hessian_minimum_constrained);

                // Free memory
                gradient_minimum.resize(0);
                hessian_minimum.resize(0,0);
                tangent_basis_min.resize(0,0);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Orthonormalizing Hessians...");
                MatrixX orth_basis = MatrixX::Zero(unstable_mode.size(), unstable_mode.size());
                orth_basis.diagonal() = VectorX::Ones(unstable_mode.size());
                orth_basis.col(0) = unstable_mode.normalized();

                MatrixX T = orth_basis.householderQr().householderQ(); // An orthonormal basis such that, T.transpose() * unstable_mode = (1,0,0,...,0)

                orth_hessian_min = T.transpose() * hessian_minimum_constrained * T;
                orth_hessian_sp  = T.transpose() * hessian_sp_constrained * T;
                orth_perpendicular_velocity = T.transpose() * perpendicular_velocity;

                e_barrier = (e_sp - e_minimum) / (C::k_B * temperature);

            // std::cout << "unstable_mode\n" << unstable_mode << "\n";
            // std::cout << "hessian_minimum_constrained\n" << hessian_minimum_constrained << "\n";
            // std::cout << "orth_hessian_min\n" << orth_hessian_min << "\n";
            // std::cout << "hessian_sp_constrained\n" << hessian_sp_constrained << "\n";
            // std::cout << "orth_hessian_sp\n" << orth_hessian_sp << "\n";
            // std::cout << "T\n" << T << "\n";
            // std::cout << "Vel_perp " << perpendicular_velocity.transpose() << "\n";
            // std::cout << "orht Vel_perp " << orth_perpendicular_velocity.transpose() << "\n";
            }
            scalar shift_constant = 0;

            // Sample minimium
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Sampling at minimum...");
            field<scalar> bennet_results_min(n_iterations_bennet, 0);
            Bennet_Minimum(n_iterations_bennet, N_INITIAL, N_DECOR, bennet_results_min, orth_hessian_min / (C::k_B * temperature), orth_hessian_sp / (C::k_B * temperature), shift_constant);

            // Expectation value
            scalar benn_min = 0; 
            for(int i=0; i<n_iterations_bennet; i++)
                benn_min += bennet_results_min[i];
            benn_min /= n_iterations_bennet;

            // Standard deviation
            scalar benn_min_var = 0;
            for(int i=0; i<n_iterations_bennet; i++)
                benn_min_var += (bennet_results_min[i]-benn_min) * (bennet_results_min[i]-benn_min);
            benn_min_var = std::sqrt( benn_min_var / n_iterations_bennet) / std::sqrt(n_iterations_bennet);
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Bennet expectation value at minimum: {} +- {}", benn_min, benn_min_var) );

            // Sample saddle point
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Sampling at saddle point...");
            field<scalar> bennet_results_sp(n_iterations_bennet, 0);
            field<scalar> vel_perp_results(n_iterations_bennet, 0);

            Bennet_SP(n_iterations_bennet, N_INITIAL, N_DECOR, vel_perp_results, bennet_results_sp, orth_hessian_sp / (C::k_B * temperature), orth_hessian_min / (C::k_B * temperature), orth_perpendicular_velocity, shift_constant);

            scalar benn_sp = 0;
            scalar vel_perp = 0;
            for(int i=0; i<n_iterations_bennet; i++)
            {
                benn_sp += bennet_results_sp[i];
                vel_perp += vel_perp_results[i];
            }
            benn_sp /= n_iterations_bennet;
            vel_perp /= n_iterations_bennet;

            scalar benn_sp_var = 0;
            scalar vel_perp_var = 0;
            for(int i=0; i<n_iterations_bennet; i++)
            {
                benn_sp_var += (bennet_results_sp[i]-benn_sp) * (bennet_results_sp[i]-benn_sp);
                vel_perp_var += (vel_perp_results[i]-vel_perp) * (vel_perp_results[i]-vel_perp);
            }
            benn_sp_var = std::sqrt( benn_sp_var / n_iterations_bennet) / std::sqrt(n_iterations_bennet);
            vel_perp_var = std::sqrt( vel_perp_var / n_iterations_bennet) / std::sqrt(n_iterations_bennet);

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Bennet expectation value at saddle point: {} +- {}", benn_sp, benn_sp_var) );

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Using the Bennet Acceptance ratio to compute the entropy contribution to the transition rate" );
            scalar q_min = 1.0/(2.0 * C::k_B * temperature) * orth_hessian_min(0,0);
            scalar unstable_mode_contribution_minimum = std::sqrt(C::Pi / q_min);

            scalar rate = C::g_e / (C::hbar) * vel_perp * benn_min / (benn_sp * unstable_mode_contribution_minimum) * std::exp(-e_barrier) * std::exp(shift_constant);
            scalar err_rate = rate * std::sqrt( std::pow(benn_sp_var/benn_sp, 2) + std::pow(benn_min_var/benn_min, 2) + std::pow(vel_perp_var/vel_perp, 2) );

            scalar Z_ratio = benn_min / benn_sp;
            scalar Z_ratio_err = Z_ratio * std::sqrt( std::pow(benn_min_var/benn_min, 2) + std::pow(benn_sp_var/benn_sp, 2) );

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Unstable mode contribution = {}", unstable_mode_contribution_minimum ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Zs/Zmin = {} +- {} ", Z_ratio, Z_ratio_err));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    vel_perp = {} +- {}", vel_perp, vel_perp_var ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    kbT = {}", temperature * C::k_B));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Delta E / kbT = {}", e_barrier));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Rate = {} +- {} [1/ps]", rate, err_rate));

            tst_bennet_info.benn_min     = benn_min;
            tst_bennet_info.err_benn_min = benn_min_var;
            tst_bennet_info.benn_sp      = benn_sp;
            tst_bennet_info.err_benn_sp  = benn_sp_var;
            tst_bennet_info.unstable_mode_contribution = unstable_mode_contribution_minimum;
            tst_bennet_info.n_iterations   = n_iterations_bennet;
            tst_bennet_info.rate           = rate;
            tst_bennet_info.rate_err       = err_rate;
            tst_bennet_info.vel_perp       = vel_perp;
            tst_bennet_info.vel_perp_err   = vel_perp_var;
            tst_bennet_info.energy_barrier = e_barrier;
            tst_bennet_info.Z_ratio = Z_ratio;
            tst_bennet_info.err_Z_ratio = Z_ratio_err;

            // Debug
            // {
            //     std::cerr << "====== Debug test =====\n";
            //     VectorX state_new = VectorX::Zero(3);
            //     VectorX state_old = VectorX::Zero(3);
            //     VectorX normal = VectorX::Zero(3);
            //     normal[0] = 0; normal[1] = 0; normal[2] = 0;
            //     MatrixX hessian = MatrixX::Zero(3,3);

            //     hessian <<  1,0,0,
            //                 0,1,0,
            //                 0,0,1;
            //     std::cerr << hessian << "\n";
            //     MC_Tracker mc;
            //     mc.target_rejection_ratio = 0.25;
            //     std::mt19937 prng = std::mt19937(21380);

            //     auto func = [&](const VectorX & state) {return 0.5 * state.transpose() * hessian * state;};
            //     VectorX v = VectorX::Zero(3);
            //     v << 1,1,1;
            //     std::cerr << "func({1,1,1}) = " << func(v) << "\n";
            //     v << 1,0,1;
            //     std::cerr << "func({1,0,1}) = " << func(v) << "\n";
            //     v << 0,0,1;
            //     std::cerr << "func({0,0,1}) = " << func(v) << "\n";
            //     v << 1,0,0;
            //     std::cerr << "func({1,0,0}) = " << func(v) << "\n";


            //     for(int i=0; i < 10000; i++)
            //     {
            //         Hyperplane_Metropolis(func, state_old, state_new, normal.normalized(), prng, mc);
            //         Backend::par::apply(state_old.size(), [&](int idx) { state_old[idx] = state_new[idx]; });
            //     }
            //     for(int i=0; i < 100000; i++)
            //     {
            //         Hyperplane_Metropolis(func, state_old, state_new, normal.normalized(), prng, mc);
            //         Backend::par::apply(state_old.size(), [&](int idx) { state_old[idx] = state_new[idx]; });
            //         std::cerr << state_old.transpose() << "\n";
            //     }
            //     // std::cerr << "n_trials     = " << mc.n_trials   << "\n";
            //     // std::cerr << "n_rejected   = " << mc.n_rejected << "\n";
            //     // std::cerr << "dist_width   = " << mc.dist_width << "\n";
            // }
        }

        bool Get_Unstable_Mode(const vectorfield & spins, const vectorfield & gradient, const MatrixX & hessian,
                MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues, MatrixX & eigenvectors)
        {
            int nos = spins.size();
            int n_modes = 1;

            hessian_constrained = MatrixX::Zero(2*nos, 2*nos);
            tangent_basis       = MatrixX::Zero(3*nos, 2*nos);
            Manifoldmath::hessian_bordered(spins, gradient, hessian, tangent_basis, hessian_constrained);

            // Create the Spectra Matrix product operation
            Spectra::DenseSymMatProd<scalar> op(hessian_constrained);

            // Create and initialize a Spectra solver
            Spectra::SymEigsSolver< scalar, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<scalar> > hessian_spectrum(&op, n_modes, 2*nos);
            hessian_spectrum.init();

            // Compute the specified spectrum, sorted by smallest real eigenvalue
            int nconv = hessian_spectrum.compute(100, 1e-10, int(Spectra::SMALLEST_ALGE));

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();

            // Retrieve the real eigenvectors
            eigenvectors = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return (hessian_spectrum.info() == Spectra::SUCCESSFUL) && (nconv > 0);
        }

        void Bennet_Minimum(int n_iteration, int n_initial, int n_decor, field<scalar> & bennet_results, const MatrixX & hessian_minimum, const MatrixX & hessian_sp, scalar shift_constant)
        {
            // Sample at Minimum
            VectorX state_min = VectorX::Zero(hessian_minimum.row(0).size());

            std::mt19937 prng = std::mt19937(803);

            MC_Tracker mc_min;
            mc_min.target_rejection_ratio = 0.5;
            mc_min.dist_width = 1;

            auto energy_diff = [&](const VectorX & state_old, const int idx, const scalar dS) 
            {
                return 0.5 * hessian_minimum(idx, idx) * dS*dS + state_old.dot(hessian_minimum.row(idx)) * dS;
            };

            auto bennet_exp = [&] (const VectorX & state)
            {
                return 1.0 / ( 1.0 + std::exp(0.5 * state.transpose() * (hessian_sp - hessian_minimum) * state + shift_constant));
            };

            for(int i=0; i < n_initial; i++)
            {
                Freeze_X_Metropolis(energy_diff, state_min, prng, mc_min);
            }

            for(int i=0; i < n_iteration; i++)
            {
                for(int j=0; j < n_decor; j++)
                {
                    Freeze_X_Metropolis(energy_diff, state_min, prng, mc_min);
                }
                Freeze_X_Metropolis(energy_diff, state_min, prng, mc_min);
                bennet_results[i] = bennet_exp(state_min);
            }
            // End new implementation

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Finished sampling at minimum.");
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    n_trials   = {}", mc_min.n_trials  ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    n_rejected = {}", mc_min.n_rejected));
        }

        void Bennet_SP(int n_iteration, int n_initial, int n_decor, field<scalar> & vel_perp_values, field<scalar> & bennet_results, const MatrixX & hessian_sp, const MatrixX & hessian_minimum, const VectorX & perpendicular_velocity, scalar shift_constant)
        {
            std::mt19937 prng = std::mt19937(2113);
            // Sample at SP
            VectorX state_sp = VectorX::Zero(hessian_sp.row(0).size());

            auto energy_diff = [&](const VectorX & state_old, const int idx, const scalar dS) 
            {
                return 0.5 * hessian_sp(idx, idx) * dS*dS + state_old.dot(hessian_sp.row(idx)) * dS;
            };

            auto bennet_exp = [&] (const VectorX & state)
            {
                return 1.0 / (1.0 + std::exp(0.5 * state.transpose() * (hessian_minimum - hessian_sp) * state - shift_constant));
            };

            MC_Tracker mc_sp;
            mc_sp.target_rejection_ratio = 0.5;
            mc_sp.dist_width = 1;

            for(int i=0; i < n_initial; i++)
            {
                Freeze_X_Metropolis(energy_diff, state_sp, prng, mc_sp);
            }

            for(int i=0; i < n_iteration; i++)
            {
                for(int j=0; j < n_decor; j++)
                {
                    Freeze_X_Metropolis(energy_diff, state_sp, prng, mc_sp);
                }
                Freeze_X_Metropolis(energy_diff, state_sp, prng, mc_sp);
                bennet_results[i] = bennet_exp(state_sp);
                vel_perp_values[i] = 0.5 * std::abs(perpendicular_velocity.dot(state_sp));
            }

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Finished sampling at saddle point.");
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    n_trials   = {}", mc_sp.n_trials  ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    n_rejected = {}", mc_sp.n_rejected));
        }
    }
}