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

namespace C = Utility::Constants;

namespace Engine
{
    namespace TST_Bennet
    {
        void Calculate(Data::HTST_Info & htst_info, int n_iterations_bennet)
        {
            // This algorithm implements a statistical sampling method to calculate life times
            //  - Step 1: Calculate geodesic Hessian at minimum (completely analogous to HTST)
            //  - Step 2: Calculate geodesic Hessian and unstable mode at saddle point
            //  - Step 3: Orthonormalize the Hessians with respect to the unstable mode
            //  - Step 4: Perform statistical Monte Carlo sampling of the minimum, within the Hyperplane orthogonal to the unstable mode
            //  - Step 5: Perform statistical Monte Carlo sampling of the saddle point, within the Hyperplane orthogonal to the unstable mode
            //  - Step 6: Use the Bennet Acceptance ratio to compute the entropy contribution to the transition rate

            std::cerr << "TST_BENNET\n";

            auto& image_minimum = *htst_info.minimum->spins;
            auto& image_sp = *htst_info.saddle_point->spins;
            int nos = image_minimum.size();

            scalar temperature = htst_info.minimum->llg_parameters->temperature;

            // #### Begin Step 1: Calculate geodesic Hessian at minimum (completely analogous to HTST) ####
            scalar e_minimum;
            MatrixX tangent_basis_min, hessian_minimum_constrained;
            {
                e_minimum = htst_info.minimum->hamiltonian->Energy(image_minimum); // Energy
                vectorfield gradient_minimum(nos, {0,0,0}); // Unconstrained gradient
                htst_info.minimum->hamiltonian->Gradient(image_minimum, gradient_minimum);
                MatrixX hessian_minimum = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                htst_info.minimum->hamiltonian->Hessian(image_minimum, hessian_minimum);

                hessian_minimum_constrained = MatrixX::Zero(2*nos, 2*nos); // Constrained hessian 3Nx3N
                tangent_basis_min = MatrixX::Zero(3*nos, 2*nos);
                Manifoldmath::tangent_basis_spherical(image_minimum, tangent_basis_min);
                Manifoldmath::hessian_bordered(image_minimum, gradient_minimum, hessian_minimum, tangent_basis_min, hessian_minimum_constrained);
            }
            // #### End Step 1 ####


            // #### Begin Step 2: Calculate geodesic Hessian and unstable mode at saddle point ####
            scalar e_sp;
            MatrixX tangent_basis, hessian_sp_constrained, eigenvectors;
            VectorX unstable_mode;
            VectorX eigenvalues;
            VectorX perpendicular_velocity = VectorX::Zero(2*nos);
            scalar e_barrier;

            // ###### Evaluate SP quantitites #######
            {
                e_sp = htst_info.minimum->hamiltonian->Energy(image_sp); // Energy

                vectorfield gradient_sp(nos, {0,0,0}); // Unconstrained gradient
                htst_info.saddle_point->hamiltonian->Gradient(image_sp, gradient_sp);

                MatrixX hessian_sp = MatrixX::Zero(3*nos,3*nos); // Unconstrained hessian
                htst_info.saddle_point->hamiltonian->Hessian(image_sp, hessian_sp);

                hessian_sp_constrained = MatrixX::Zero(2*nos, 2*nos); // Constrained hessian

                Get_Unstable_Mode(image_sp, gradient_sp, hessian_sp, tangent_basis, hessian_sp_constrained, eigenvalues, eigenvectors);
                unstable_mode = eigenvectors.col(0);

                MatrixX dynamical_matrix(3*nos, 3*nos);
                HTST::Calculate_Dynamical_Matrix(image_sp, htst_info.saddle_point->geometry->mu_s, hessian_sp, dynamical_matrix);

                e_barrier = (e_sp - e_minimum) / (C::k_B * temperature);
                // e_barrier = 0;
 
                perpendicular_velocity =  eigenvectors.col(0).transpose() * tangent_basis.transpose() * dynamical_matrix * tangent_basis;

                std::cerr << "per vel projected\n" << tangent_basis.transpose() * dynamical_matrix * tangent_basis << "\n";
                        {
                            std::cerr << "-----------------\n";
                            std::cerr << "Calculated unstable mode\n";
                            std::cerr << "Eval  = " << eigenvalues[0] << "\n";
                            std::cerr << "mode  = " << eigenvectors.col(0).transpose() << "\n";
                            std::cerr << "vel  = " << perpendicular_velocity.col(0).transpose() << "\n";
                            std::cerr << "vel.size() = " << perpendicular_velocity.size() << "\n";
                            std::cerr << "tangent_basis  = \n" << tangent_basis << "\n";
                            std::cerr << "hessian_sp  = \n" << hessian_sp << "\n";
                            std::cerr << "Temperature = " << temperature << "\n";
                            std::cerr << "energy_barrier = " << e_barrier << "\n";
                            std::cerr << "kb * T = " << C::k_B * temperature << "\n";
                            std::cerr << "hessian_sp_constrained = \n" << hessian_sp_constrained << "\n";
                            std::cerr << "dynamical_matrix  = \n" << dynamical_matrix << "\n";
                            std::cerr << "perpendicular_velocity  = \n" << perpendicular_velocity.transpose() << "\n";
                            // std::cerr << tangent_basis.transpose() * dynamical_matrix * tangent_basis << "\n";
                            std::cerr << "-----------------\n";
                        }
            } 
            // #### End step 2 ####


            // #### Begin Step 3: Orthonormalize the Hessians with respect to the unstable mode ####
            MatrixX orth_basis = MatrixX::Zero(unstable_mode.size(), unstable_mode.size());
            orth_basis.diagonal() = VectorX::Ones(unstable_mode.size());
            orth_basis.col(0) = unstable_mode.normalized();
            MatrixX T = orth_basis.householderQr().householderQ(); // An orthonormal basis such that, T.transpose() * unstable_mode = (1,0,0,...,0)

            MatrixX orth_hessian_min = T.transpose() * hessian_minimum_constrained * T;
            MatrixX orth_hessian_sp  = T.transpose() * hessian_sp_constrained * T;
            VectorX orth_perpendicular_velocity = T.transpose() * perpendicular_velocity;
            // #### End Step 3 ####


            // #### Begin Step 4: Perform statistical Monte Carlo sampling of the minimum, within the Hyperplane orthogonal to the unstable mode ####
            field<scalar> bennet_results_min(n_iterations_bennet, 0);
            Bennet_Minimum(n_iterations_bennet, bennet_results_min, orth_hessian_min / (C::k_B * temperature), orth_hessian_sp / (C::k_B * temperature), e_barrier);

            scalar benn_min = 0; 
            for(int i=0; i<n_iterations_bennet; i++)
                benn_min += bennet_results_min[i];
            benn_min /= n_iterations_bennet;

            scalar benn_min_var = 0;
            for(int i=0; i<n_iterations_bennet; i++)
                benn_min_var += (bennet_results_min[i]-benn_min) * (bennet_results_min[i]-benn_min);
            benn_min_var = std::sqrt( benn_min_var / n_iterations_bennet) / std::sqrt(n_iterations_bennet);
            std::cerr << "benn_min = " << benn_min << " +- " << benn_min_var << "\n";
            // #### End Step 4 ####


            // #### Begin Step 5: Perform statistical Monte Carlo sampling of the saddle point, within the Hyperplane orthogonal to the unstable mode #### 
            scalar vel_perp=0;
            field<scalar> bennet_results_sp(n_iterations_bennet, 0);
            Bennet_SP(n_iterations_bennet, vel_perp, bennet_results_sp, orth_hessian_sp / (C::k_B * temperature), orth_hessian_min / (C::k_B * temperature), orth_perpendicular_velocity, e_barrier);

            scalar benn_sp = 0;
            for(int i=0; i<n_iterations_bennet; i++)
                benn_sp += bennet_results_sp[i];
            benn_sp /= n_iterations_bennet;

            scalar benn_sp_var = 0;
            for(int i=0; i<n_iterations_bennet; i++)
                benn_sp_var += (bennet_results_sp[i]-benn_sp) * (bennet_results_sp[i]-benn_sp);
            benn_sp_var = std::sqrt( benn_sp_var / n_iterations_bennet) / std::sqrt(n_iterations_bennet);
            std::cerr << "benn_sp = " << benn_sp << " +- " << benn_sp_var << "\n";
            // #### End Step 5 ####


            // #### Step 6: Use the Bennet Acceptance ratio to compute the entropy contribution to the transition rate ####
            scalar q_min = 1.0/(2.0 * C::k_B * temperature) * eigenvectors.col(0).normalized().transpose() * hessian_minimum_constrained * eigenvectors.col(0).normalized();
            scalar unstable_mode_contribution_minimum = std::sqrt(C::Pi / q_min);

            scalar rate = C::g_e / (C::hbar) * vel_perp * benn_min / (benn_sp * unstable_mode_contribution_minimum) ;
            std::cerr << "Unstable mode contribution " << unstable_mode_contribution_minimum << "\n";
            std::cerr << "Zs/Zm = " << benn_min / benn_sp << "\n";        
            std::cerr << "Rate  = " << rate << " [1/ps]\n";
            // #### End Step 6 ####

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
            int nconv = hessian_spectrum.compute(1000, 1e-10, int(Spectra::SMALLEST_ALGE));

            // Extract real eigenvalues
            eigenvalues = hessian_spectrum.eigenvalues().real();

            // Retrieve the real eigenvectors
            eigenvectors = hessian_spectrum.eigenvectors().real();

            // Return whether the calculation was successful
            return (hessian_spectrum.info() == Spectra::SUCCESSFUL) && (nconv > 0);
        }

        void Bennet_Minimum(int n_iteration, field<scalar> & bennet_results, const MatrixX & hessian_minimum, const MatrixX & hessian_sp, scalar energy_barrier)
        {
            // Sample at Minimum
            VectorX state_min_old = VectorX::Zero(hessian_minimum.row(0).size());
            VectorX state_min_new = VectorX::Zero(hessian_minimum.row(0).size());

            std::cerr << ">>>> Minimum <<<<\n";
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
                return 1.0 / ( 1.0 + std::exp(0.5 * state.transpose() * (hessian_sp - hessian_minimum) * state + energy_barrier));
            };

            for(int i=0; i < n_iteration; i++)
            {
                Freeze_X_Metropolis(energy_diff, state_min_old, state_min_new, prng, mc_min);
                state_min_old     = state_min_new;
                bennet_results[i] = bennet_exp(state_min_old);
            }
            // End new implementation

            std::cerr << "n_trials     = " << mc_min.n_trials   << "\n";
            std::cerr << "n_rejected   = " << mc_min.n_rejected << "\n";
            std::cerr << "dist_width   = " << mc_min.dist_width << "\n";
        }

        void Bennet_SP(int n_iteration, scalar & vel_perp_estimator, field<scalar> & bennet_results, const MatrixX & hessian_sp, const MatrixX & hessian_minimum, const VectorX & perpendicular_velocity, scalar energy_barrier)
        {
            std::cerr << "\n>>>>> Saddle_point <<<<<\n";

            std::mt19937 prng = std::mt19937(2113);

            // Sample at SP
            VectorX state_sp_old = VectorX::Zero(hessian_sp.row(0).size());
            VectorX state_sp_new = VectorX::Zero(hessian_sp.row(0).size());

            auto energy_diff = [&](const VectorX & state_old, const int idx, const scalar dS) 
            {
                return 0.5 * hessian_sp(idx, idx) * dS*dS + state_old.dot(hessian_sp.row(idx)) * dS;
            };

            auto bennet_exp = [&] (const VectorX & state)
            {
                return 1.0 / (1.0 + std::exp(0.5 * state.transpose() * (hessian_minimum - hessian_sp) * state - energy_barrier));
            };

            MC_Tracker mc_sp;
            mc_sp.target_rejection_ratio = 0.5;
            mc_sp.dist_width = 1;

            scalar vel_perp = 0;
            for(int i=0; i<n_iteration; i++)
            {
                Freeze_X_Metropolis(energy_diff, state_sp_old, state_sp_new, prng, mc_sp);
                state_sp_old = state_sp_new;
                bennet_results[i] = bennet_exp(state_sp_old);
                vel_perp += 0.5 * std::abs(perpendicular_velocity.dot(state_sp_old));
            }

            vel_perp  /= n_iteration;

            std::cerr << "n_trials   = " << mc_sp.n_trials   << "\n";
            std::cerr << "n_rejected = " << mc_sp.n_rejected << "\n";
            std::cerr << "dist_width = " << mc_sp.dist_width << "\n";
            std::cerr << "vel_perp = " << vel_perp << "\n";

            vel_perp_estimator = vel_perp;

        }
    }
}