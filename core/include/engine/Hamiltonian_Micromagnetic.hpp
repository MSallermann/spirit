#pragma once
#ifndef HAMILTONIAN_MICROMAGNETIC_H
#define HAMILTONIAN_MICROMAGNETIC_H

#include <vector>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>
#include "FFT.hpp"

namespace Engine
{
    // The Micromagnetic Hamiltonian
    class Hamiltonian_Micromagnetic : public Hamiltonian
    {
    public:

        Hamiltonian_Micromagnetic(
            scalar external_field_magnitude, Vector3 external_field_normal,
            scalar anisotropy_magnitude, Vector3 anisotropy_normal,
            scalar  exchange_stiffness,
            scalar  dmi,
            std::shared_ptr<Data::Geometry> geometry,
            int spatial_gradient_order,
            DDI_Method ddi_method, intfield ddi_n_periodic_images, bool ddi_pb_zero_padding, scalar ddi_radius,
            intfield boundary_conditions,
            Vector3 cell_sizes,
            scalar Ms
        );

        void Update_Interactions();

        void Update_Energy_Contributions() override;

        void Hessian(const vectorfield & spins, MatrixX & hessian) override;
        void Gradient(const vectorfield & spins, vectorfield & gradient) override;

        void Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions) override;
        void Energy_Update(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions, vectorfield & gradient);
        // Calculate the total energy for a single spin to be used in Monte Carlo.
        //      Note: therefore the energy of pairs is weighted x2 and of quadruplets x4.

        scalar Energy_Single_Spin(int ispin, const vectorfield & spins) override;

        // Hamiltonian name as string
        const std::string& Name() override;

        std::shared_ptr<Data::Geometry> geometry;
        scalar Ms;
        // ------------ ... ------------
        int spatial_gradient_order;

        // ------------ Single Spin Interactions ------------
        // External magnetic field across the sample
        scalar external_field_magnitude;
        Vector3 external_field_normal;

        scalar anisotropy_magnitude;
        Vector3 anisotropy_normal;

        // Exchange interaction
        scalar  exchange_stiffness;
        Matrix3 exchange_tensor;

        // DMI
        scalar  dmi_constant;
        int     dmi_chirality;
        Matrix3 dmi_tensor;

        DDI_Method  ddi_method;
        intfield    ddi_n_periodic_images;
        bool        ddi_pb_zero_padding;

        Vector3 cell_sizes;
        scalar cell_volume;

        //DDI cutoff variables
        scalar      ddi_cutoff_radius;
        pairfield   ddi_pairs;
        scalarfield ddi_magnitudes;
        vectorfield ddi_normals;

        vectorfield mult_spins;

        #ifdef SPIRIT_LOW_MEMORY
            scalarfield temp_energies;
        #endif

        pairfield neigh;
        // field<Matrix3> spatial_gradient;

    private:

        inline int fast_idx_from_translations(const std::array<int,3> & translations, const field<int> & n_cells, const field<int> & boundary_conditions)
        {
            std::array<int, 3> abc;
            for(int i=0; i<3; i++)
            {
                if( translations[i] >= n_cells[i] || translations[i] < 0)
                {
                    if(boundary_conditions[i] >= 1)
                    {
                        abc[i] = abc[i] % n_cells[i];
                    } else {
                        return -1;
                    }
                }
                abc[i] = translations[i];
            }
            return abc[0] + n_cells[0] * ( abc[1] + n_cells[1] * abc[2]);
        }

        // ------------ Effective Field Functions ------------
        // Calculate the Zeeman effective field of a single Spin
        void Gradient_Zeeman(vectorfield & gradient);
        // Calculate the Anisotropy effective field of a single Spin
        void Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient);
        // Calculate the exchange interaction effective field of a Spin Pair
        void Gradient_Exchange(const vectorfield & spins, vectorfield & gradient);
        // Calculate the DMI effective field of a Spin Pair
        void Gradient_DMI(const vectorfield & spins, vectorfield & gradient);
        void Spatial_Gradient(const vectorfield & spins);
        // Calculates the Dipole-Dipole contribution to the effective field of spin ispin within system s
        void Gradient_DDI(const vectorfield& spins, vectorfield & gradient);
        void Gradient_DDI_Cutoff(const vectorfield& spins, vectorfield & gradient);
        void Gradient_DDI_Direct(const vectorfield& spins, vectorfield & gradient);
        void Gradient_DDI_FFT(const vectorfield& spins, vectorfield & gradient);

        // ------------ Energy Functions ------------
        // Indices for Energy vector
        //int idx_zeeman, idx_anisotropy, idx_exchange, idx_dmi, idx_ddi;

        void Energy_From_Gradient(const vectorfield & spins, const vectorfield & gradient, const scalar factor, scalarfield & energy);

        void E_Zeeman(const vectorfield & spins, scalarfield & Energy);

        // Calculate the Anisotropy energy of a Spin System
        void E_Anisotropy(const vectorfield & spins, scalarfield & Energy);

        // Calculate the exchange interaction energy of a Spin System
        void E_Exchange(const vectorfield & spins, scalarfield & Energy);

        // Calculate the DMI energy of a Spin System
        void E_DMI(const vectorfield & spins, scalarfield & Energy);

        // calculates the Dipole-Dipole Energy
        void E_DDI(const vectorfield & spins, scalarfield & Energy);
        
        // Dipolar interactions
        void E_DDI_Direct(const vectorfield & spins, scalarfield & Energy);

        // Quadruplet
        //void E_Quadruplet(const vectorfield & spins, scalarfield & Energy);

        // Preparations for DDI-Convolution Algorithm
        void Prepare_DDI();
        void Clean_DDI();

        // Plans for FT / rFT
        FFT::FFT_Plan fft_plan_spins;
        FFT::FFT_Plan fft_plan_reverse;

        field<FFT::FFT_cpx_type> transformed_dipole_matrices;

        bool save_dipole_matrices = true;
        field<Matrix3> dipole_matrices;

        // Number of inter-sublattice contributions
        int n_inter_sublattice;
        // At which index to look up the inter-sublattice D-matrices
        field<int> inter_sublattice_lookup;

        // Lengths of padded system
        field<int> n_cells_padded;
        // Total number of padded spins per sublattice
        int sublattice_size;

        FFT::StrideContainer spin_stride;
        FFT::StrideContainer dipole_stride;

        //Calculate the FT of the padded D matriess
        void FFT_Dipole_Matrices(FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c);
        //Calculate the FT of the padded spins
        void FFT_Spins(const vectorfield & spins);

        //Bounds for nested for loops. Only important for the CUDA version
        field<int> it_bounds_pointwise_mult;
        field<int> it_bounds_write_gradients;
        field<int> it_bounds_write_spins;
        field<int> it_bounds_write_dipole;

        int prev_position=0;
        int iteration_num=0;
    };
}
#endif
