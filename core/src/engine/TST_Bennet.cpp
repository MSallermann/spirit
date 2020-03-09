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
        void Calculate(Data::TST_Bennet_Info & tst_bennet_info, int n_chain, int n_iterations_bennet)
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
            MatrixX orth_hessian_min, orth_hessian_sp, hessian_min, hessian_sp;
            VectorX orth_perpendicular_velocity;
            VectorX perpendicular_velocity = VectorX::Zero(2*nos);
            VectorX eigenvalues;
            VectorX unstable_mode;
            MatrixX tangent_basis, eigenvectors;
            {
                MatrixX hessian_min_embed, hessian_sp_embed;
                e_sp = tst_bennet_info.saddle_point->hamiltonian->Energy(image_sp); // Energy

                vectorfield gradient_sp(nos, {0,0,0}); // Unconstrained gradient
                tst_bennet_info.saddle_point->hamiltonian->Gradient(image_sp, gradient_sp);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of saddle point embedding Hessian...");
                hessian_sp_embed = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                tst_bennet_info.saddle_point->hamiltonian->Hessian(image_sp, hessian_sp_embed);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of saddle point constrained Hessian...");
                MatrixX hessian_sp = MatrixX::Zero(2*nos, 2*nos); // Constrained hessian

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Calculation of unstable mode...");
                Get_Unstable_Mode(image_sp, gradient_sp, hessian_sp_embed, tangent_basis, hessian_sp, eigenvalues, eigenvectors);
                unstable_mode = eigenvectors.col(0);

                // Free memory
                gradient_sp.resize(0);
                eigenvalues.resize(0,0);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Calculation of dynamical matrix...");
                MatrixX dynamical_matrix(3*nos, 3*nos);
                HTST::Calculate_Dynamical_Matrix(image_sp, tst_bennet_info.saddle_point->geometry->mu_s, hessian_sp_embed, dynamical_matrix);
                perpendicular_velocity = eigenvectors.col(0).transpose() * tangent_basis.transpose() * dynamical_matrix * tangent_basis;

                // Free memory
                hessian_sp_embed.resize(0,0);
                dynamical_matrix.resize(0,0);

                MatrixX tangent_basis_min;
                e_minimum = tst_bennet_info.minimum->hamiltonian->Energy(image_minimum); // Energy
                vectorfield gradient_minimum(nos, {0,0,0}); // Unconstrained gradient
                tst_bennet_info.minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of minimum embedding Hessian...");
                hessian_min_embed = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                tst_bennet_info.minimum->hamiltonian->Hessian(image_minimum, hessian_min_embed);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Evaluation of minimum constrained Hessian...");
                MatrixX hessian_min = MatrixX::Zero(2*nos, 2*nos);
                tangent_basis_min = MatrixX::Zero(3*nos, 2*nos);
                Manifoldmath::tangent_basis_spherical(image_minimum, tangent_basis_min);
                Manifoldmath::hessian_bordered(image_minimum, gradient_minimum, hessian_min_embed, tangent_basis_min, hessian_min);

                // Free memory
                gradient_minimum.resize(0);
                hessian_min_embed.resize(0,0);
                tangent_basis_min.resize(0,0);

                Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Orthonormalizing Hessians...");
                MatrixX orth_basis = MatrixX::Zero(unstable_mode.size(), unstable_mode.size());
                orth_basis.diagonal() = VectorX::Ones(unstable_mode.size());
                orth_basis.col(0) = unstable_mode.normalized();

                MatrixX T = orth_basis.householderQr().householderQ(); // An orthonormal basis such that, T.transpose() * unstable_mode = (1,0,0,...,0)

                orth_hessian_min = T.transpose() * hessian_min * T;
                orth_hessian_sp  = T.transpose() * hessian_sp * T;
                orth_perpendicular_velocity = T.transpose() * perpendicular_velocity;

                e_barrier = (e_sp - e_minimum) / (C::k_B * temperature);
            }

            scalar shift_constant = 0;
            field<scalar> vel_perp_cb(n_iterations_bennet,0);
            auto cb = [&](int img, int i, const VectorX& state)
            {
                if(img == n_chain-1)
                {
                    vel_perp_cb[i] = 0.5 * std::abs(orth_perpendicular_velocity.dot(state));
                }
            };

            scalar Z_ratio, Z_ratio_err;
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Sampling of {} Hessians with {} iterations each...", n_chain, n_iterations_bennet));
            Bennet_Chain_Sampling(n_chain, n_iterations_bennet, N_INITIAL, N_DECOR, orth_hessian_min/(C::k_B * temperature), orth_hessian_sp/(C::k_B * temperature), shift_constant, cb, Z_ratio, Z_ratio_err);
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "    Finished sampling of Hessians...");

            scalar vel_perp = Vectormath::mean(vel_perp_cb);
            scalar vel_perp_err = 0;
            for(int i=0; i<n_iterations_bennet; i++)
                vel_perp_err += std::pow(vel_perp_cb[i] - vel_perp, 2);
            vel_perp_err = std::sqrt(vel_perp_err) / n_iterations_bennet;

            scalar q_min = 1.0/(2.0 * C::k_B * temperature) * orth_hessian_min(0,0);
            scalar unstable_mode_contribution = std::sqrt(C::Pi / q_min);

            scalar rate     = C::g_e / (C::hbar) * vel_perp * Z_ratio / unstable_mode_contribution * std::exp(-e_barrier) * std::exp(shift_constant);
            scalar err_rate = rate * std::sqrt( std::pow(Z_ratio_err/Z_ratio, 2) + std::pow(vel_perp_err/vel_perp, 2) );

            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, "--- Results:" );
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Unstable mode contribution = {}", unstable_mode_contribution ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Zs/Zmin = {} +- {} ", Z_ratio, Z_ratio_err));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    vel_perp = {} +- {}", vel_perp, vel_perp_err ));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    kbT = {}", temperature * C::k_B));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Delta E / kbT = {}", e_barrier));
            Log(Utility::Log_Level::Info, Utility::Log_Sender::TST_Bennet, fmt::format("    Rate = {} +- {} [1/ps]", rate, err_rate));

            tst_bennet_info.unstable_mode_contribution = unstable_mode_contribution;
            tst_bennet_info.n_iterations   = n_iterations_bennet;
            tst_bennet_info.rate           = rate;
            tst_bennet_info.rate_err       = err_rate;
            tst_bennet_info.vel_perp       = vel_perp;
            tst_bennet_info.vel_perp_err   = vel_perp_err;
            tst_bennet_info.energy_barrier = e_barrier;
            tst_bennet_info.Z_ratio = Z_ratio;
            tst_bennet_info.err_Z_ratio = Z_ratio_err;
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

        void Bennet_Minimum(int n_iteration, int n_initial, int n_decor, field<scalar> & bennet_results, const MatrixX & hessian_min, const MatrixX & hessian_sp, scalar shift_constant)
        {
            // Sample at Minimum
            VectorX state_min = VectorX::Zero(hessian_min.row(0).size());

            std::mt19937 prng = std::mt19937(803);

            MC_Tracker mc_min;
            mc_min.target_rejection_ratio = 0.5;
            mc_min.dist_width = 1;

            auto energy_diff = [&](const VectorX & state_old, const int idx, const scalar dS) 
            {
                return 0.5 * hessian_min(idx, idx) * dS*dS + state_old.dot(hessian_min.row(idx)) * dS;
            };

            auto bennet_exp = [&] (const VectorX & state)
            {
                return 1.0 / ( 1.0 + std::exp(0.5 * state.transpose() * (hessian_sp - hessian_min) * state + shift_constant));
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

        void Bennet_SP(int n_iteration, int n_initial, int n_decor, field<scalar> & vel_perp_values, field<scalar> & bennet_results, const MatrixX & hessian_sp, const MatrixX & hessian_min, const VectorX & perpendicular_velocity, scalar shift_constant)
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
                return 1.0 / (1.0 + std::exp(0.5 * state.transpose() * (hessian_min - hessian_sp) * state - shift_constant));
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