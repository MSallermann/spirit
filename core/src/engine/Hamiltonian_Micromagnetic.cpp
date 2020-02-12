#ifndef SPIRIT_USE_CUDA

#include <engine/Hamiltonian_Micromagnetic.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Neighbours.hpp>
#include <data/Spin_System.hpp>
#include <utility/Constants.hpp>
#include <algorithm>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <engine/Backend_par.hpp>
#include "FFT.hpp"
using namespace Data;
using namespace Utility;
namespace C = Utility::Constants_Micromagnetic;
using Engine::Vectormath::idx_from_pair;

namespace Engine
{
    Hamiltonian_Micromagnetic::Hamiltonian_Micromagnetic(
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
    ) : Hamiltonian(boundary_conditions), spatial_gradient_order(spatial_gradient_order), geometry(geometry),
        external_field_magnitude(external_field_magnitude), external_field_normal(external_field_normal),
        anisotropy_magnitude(anisotropy_magnitude), anisotropy_normal(anisotropy_normal),
        ddi_method(ddi_method), ddi_n_periodic_images(ddi_n_periodic_images), ddi_pb_zero_padding(ddi_pb_zero_padding), ddi_cutoff_radius(ddi_radius),
        exchange_stiffness(exchange_stiffness), dmi_constant(dmi),
        cell_sizes(cell_sizes), Ms(Ms)
    {

        cell_volume = cell_sizes[0] * cell_sizes[1] * cell_sizes[2];

        exchange_tensor <<  exchange_stiffness, 0, 0,
                            0, exchange_stiffness, 0, 
                            0, 0, exchange_stiffness;
        std::cout << "\n\nexchange_tensor\n" << exchange_tensor << "\n\n";

        dmi_tensor <<   dmi_constant, 0, 0,
                        0, dmi_constant, 0,
                        0, 0, dmi_constant;

        std::cout << "\n\ndmi_tensor\n" << dmi_tensor << "\n\n";
        std::cout << "\n\nMs " << Ms << "\n\n";

        std::cout << "\n\nexternal field magnitude " << external_field_magnitude << "\n\n";
        std::cout << "\n\nexternal field normal " << external_field_normal.transpose() << "\n\n";

        std::cout << "\n\nanisotropy magnitude " << anisotropy_magnitude << "\n\n";
        std::cout << "\n\nanisotropy normal " << anisotropy_normal.transpose() << "\n\n";

        std::cout << "\n\n ddi_method " << (int) ddi_method << "\n";

        this->Update_Interactions();
    }

    void Hamiltonian_Micromagnetic::Update_Interactions()
    {
        #if defined(SPIRIT_USE_OPENMP)
        // When parallelising (cuda or openmp), we need all neighbours per spin
        const bool use_redundant_neighbours = true;
        #else
        // When running on a single thread, we can ignore redundant neighbours
        const bool use_redundant_neighbours = false;
        #endif

        Pair t{0,0,{0,0,0}};
        neigh = pairfield(18, t);
        neigh[0].translations = {1, 0, 0};
        neigh[1].translations = {-1, 0, 0};
        neigh[2].translations = {0, 1, 0};
        neigh[3].translations = {0, -1, 0};
        neigh[4].translations = {0, 0, 1};
        neigh[5].translations = {0, 0, -1};
        neigh[6].translations = {1, 1, 0};
        neigh[7].translations = {-1, -1, 0};
        neigh[8].translations = {1, -1, 0};
        neigh[9].translations = {-1, 1, 0};
        neigh[10].translations = {1, 0, 1};
        neigh[11].translations = {-1, 0, -1};
        neigh[12].translations = {1, 0, -1};
        neigh[13].translations = {-1, 0, 1};
        neigh[14].translations = {0, 1, 1};
        neigh[15].translations = {0, -1, -1};
        neigh[16].translations = {0, 1, -1};
        neigh[17].translations = {0, -1, 1};

        // Dipole-dipole (FFT)
        this->Prepare_DDI();

        // Update, which terms still contribute
        this->Update_Energy_Contributions();
    }

    void Hamiltonian_Micromagnetic::Update_Energy_Contributions()
    {
       this->energy_contributions_per_spin = std::vector<std::pair<std::string, scalarfield>>(0);

        // External field
        if( std::abs(this->external_field_magnitude) > 1e-60 )
        {
            this->energy_contributions_per_spin.push_back({"Zeeman", scalarfield(0)});
            this->idx_zeeman = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_zeeman = -1;
        // Anisotropy
        if( std::abs(this->anisotropy_magnitude) > 1e-60 )
        {
            this->energy_contributions_per_spin.push_back({"Anisotropy", scalarfield(0) });
            this->idx_anisotropy = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_anisotropy = -1;
        // Exchange
        if( std::abs(this->exchange_stiffness) > 1e-60 )
        {
            this->energy_contributions_per_spin.push_back({"Exchange", scalarfield(0) });
            this->idx_exchange = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_exchange = -1;
        // DMI
        if( std::abs(this->dmi_constant) > 1e-60 )
        {
            this->energy_contributions_per_spin.push_back({"DMI", scalarfield(0) });
            this->idx_dmi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_dmi = -1;
        // Dipole-Dipole
        if( this->ddi_method != DDI_Method::None )
        {
            this->energy_contributions_per_spin.push_back({"DDI", scalarfield(0) });
            this->idx_ddi = this->energy_contributions_per_spin.size()-1;
        }
        else this->idx_ddi = -1;
    }

    void Hamiltonian_Micromagnetic::Energy_Contributions_per_Spin(const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions)
    {

        if( contributions.size() != this->energy_contributions_per_spin.size() )
        {
            contributions = this->energy_contributions_per_spin;
        }

        int nos = spins.size();
        for( auto& contrib : contributions )
        {
            // Allocate if not already allocated
            if (contrib.second.size() != nos) contrib.second = scalarfield(nos, 0);
            // Otherwise set to zero
            else Vectormath::fill(contrib.second, 0);
        }

        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Vectormath::fill(temp_gradient, {0,0,0});
        // Spatial_Gradient(spins);

        // External field
        if( this->idx_zeeman >=0 )
        {
            Gradient_Zeeman(temp_gradient);
            Energy_From_Gradient(spins, temp_gradient, 1.0/C::meV, contributions[idx_zeeman].second);
        }

        // Anisotropy
        if( this->idx_anisotropy >=0 )
        {
            Vectormath::fill(temp_gradient, {0,0,0});
            Gradient_Anisotropy(spins, temp_gradient);
            Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, contributions[idx_anisotropy].second);
        }

        // Exchange
        if( this->idx_exchange >=0 )
        {
            Vectormath::fill(temp_gradient, {0,0,0});
            Gradient_Exchange(spins, temp_gradient);
            Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, contributions[idx_exchange].second);
        }

        // DMI
        if( this->idx_dmi >=0 )
        { 
            Vectormath::fill(temp_gradient, {0,0,0});
            Gradient_DMI(spins, temp_gradient);
            Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, contributions[idx_dmi].second);
        }

        // DDI
        if( this->idx_ddi >=0 )
        {
            Vectormath::fill(temp_gradient, {0,0,0});
            Gradient_DDI(spins, temp_gradient);
            Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, contributions[idx_exchange].second);
        }
    }

    void Hamiltonian_Micromagnetic::Gradient(const vectorfield & spins, vectorfield & gradient)
    {
         // Set to zero
        Vectormath::fill(gradient, {0,0,0});
        // External field
        if( this->idx_zeeman >=0 ) this->Gradient_Zeeman(gradient);
        // Anisotropy
        if( this->idx_anisotropy >=0 ) this->Gradient_Anisotropy(spins, gradient);
        // Exchange
        if( this->idx_exchange >=0 ) this->Gradient_Exchange(spins, gradient);
        // DMI
        if( this->idx_dmi >=0 ) this->Gradient_DMI(spins, gradient);
        // DDI
        if( this->idx_ddi >=0 ) this->Gradient_DDI(spins, gradient);

        // Scale gradient so that it is in meV
        auto g = gradient.data();
        Backend::par::apply(gradient.size(), [g] SPIRIT_LAMBDA (int idx) {g[idx] /= C::meV;});
    }

    void Hamiltonian_Micromagnetic::Energy_From_Gradient(const vectorfield & spins, const vectorfield & gradient, const scalar factor, scalarfield & energy)
    {
        auto n = spins.size();
        auto s = spins.data();
        auto g = gradient.data();
        auto e = energy.data();
        auto set = [s,e,g,factor] SPIRIT_LAMBDA (int idx) { e[idx] = factor * s[idx].dot(g[idx]); };
        Backend::par::apply( n, set );
    }

    void Hamiltonian_Micromagnetic::E_Zeeman(const vectorfield & spins, scalarfield & energy)
    {
        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Gradient_Zeeman(temp_gradient);
        Energy_From_Gradient(spins, temp_gradient, 1.0/C::meV, energy);
    }

    void Hamiltonian_Micromagnetic::Gradient_Zeeman(vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            gradient[icell] -= Ms * cell_volume * external_field_magnitude * external_field_normal;
        }
    }

    void Hamiltonian_Micromagnetic::E_Anisotropy(const vectorfield & spins, scalarfield & energy)
    {
        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Gradient_Anisotropy(spins, temp_gradient);
        Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, energy);
    }

    // void Hamiltonian_Micromagnetic::E_Anisotropy(const vectorfield & spins, const vectorfield & gradient, scalarfield & Energy);
    void Hamiltonian_Micromagnetic::Gradient_Anisotropy(const vectorfield & spins, vectorfield & gradient)
    {
        #pragma omp parallel for
        for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        {
            gradient[icell] -= 2.0 * anisotropy_magnitude * anisotropy_normal * anisotropy_normal.dot(spins[icell]) * cell_volume;
        }
    }

    void Hamiltonian_Micromagnetic::E_Exchange(const vectorfield & spins, scalarfield & energy)
    {
        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Gradient_Exchange(spins, temp_gradient);
        Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, energy);
    }

    void Hamiltonian_Micromagnetic::Gradient_Exchange(const vectorfield & spins, vectorfield & gradient)
    {
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        std::array<int,3> plus_translations;
        std::array<int,3> minus_translations;
        #pragma omp parallel for collapse(3)
        for(int c=0; c < Nc; c++)
        {
            for(int b=0; b < Nb; b++)
            {
                for(int a=0; a < Na; a++)
                {
                    int ispin = a + Na * (b + Nb * c);
                    for (unsigned int i = 0; i < 3; ++i) //Iterate over directions for spatial gradient
                    {
                        plus_translations[0] = (a + neigh[2*i].translations[0]);
                        plus_translations[1] = (b + neigh[2*i].translations[1]);
                        plus_translations[2] = (c + neigh[2*i].translations[2]);
                        int ispin_plus = fast_idx_from_translations(plus_translations, geometry->n_cells, boundary_conditions);

                        minus_translations[0] = (a + neigh[2*i + 1].translations[0]);
                        minus_translations[1] = (b + neigh[2*i + 1].translations[1]);
                        minus_translations[2] = (c + neigh[2*i + 1].translations[2]);
                        int ispin_minus = fast_idx_from_translations(minus_translations, geometry->n_cells, boundary_conditions);

                        if (ispin_plus == -1) {
                            ispin_plus = ispin;
                        }
                        if (ispin_minus == -1) {
                            ispin_minus = ispin;
                        }

                        gradient[ispin] -= 2 * exchange_stiffness * cell_volume * (spins[ispin_plus] - 2 * spins[ispin] + spins[ispin_minus]) / (cell_sizes[i] * cell_sizes[i]);
                    }
                }
            }
        }
    }

    void Hamiltonian_Micromagnetic::Spatial_Gradient(const vectorfield & spins)
    {
        // #pragma omp parallel for
        // for( int icell = 0; icell < geometry->n_cells_total; ++icell )
        // {
        //     int ispin = icell;//basically id of a cell
        //     for (unsigned int i = 0; i < 3; ++i)
        //     {
        //         int ispin_plus  = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,geometry->atom_types, neigh[2 * i]);
        //         int ispin_minus = idx_from_pair(ispin, boundary_conditions, geometry->n_cells, geometry->n_cell_atoms,geometry->atom_types, neigh[2 * i + 1]);
        //         if (ispin_plus == -1) {
        //             ispin_plus = ispin;
        //         }
        //         if (ispin_minus == -1) {
        //             ispin_minus = ispin;
        //         }
        //         spatial_gradient[ispin](0, i) = (spins[ispin_plus][0] - spins[ispin_minus][0]) / (2 * cell_sizes[i]);
        //         spatial_gradient[ispin](1, i) = (spins[ispin_plus][1] - spins[ispin_minus][1]) / (2 * cell_sizes[i]);
        //         spatial_gradient[ispin](2, i) = (spins[ispin_plus][2] - spins[ispin_minus][2]) / (2 * cell_sizes[i]);
        //     }
        // }
    }

    void Hamiltonian_Micromagnetic::E_DMI(const vectorfield & spins, scalarfield & energy)
    {
        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Spatial_Gradient(spins);
        Gradient_DMI(spins, temp_gradient);
        Energy_From_Gradient(spins, temp_gradient, 0.5/C::meV, energy);
    }

    void Hamiltonian_Micromagnetic::Gradient_DMI(const vectorfield & spins, vectorfield & gradient)
    {
        scalar s0, s1, s2;
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        std::array<int,3> plus_translations;
        std::array<int,3> minus_translations;

        #pragma omp parallel for collapse(3)
        for(int c=0; c < Nc; c++)
        {
            for(int b=0; b < Nb; b++)
            {
                for(int a=0; a < Na; a++)
                {
                    int ispin = a + Na * (b + Nb * c);
                    for (unsigned int i = 0; i < 3; ++i) // Iterate over directions for spatial gradient
                    {
                        plus_translations[0] = (a + neigh[2*i].translations[0]);
                        plus_translations[1] = (b + neigh[2*i].translations[1]);
                        plus_translations[2] = (c + neigh[2*i].translations[2]);
                        int ispin_plus = fast_idx_from_translations(plus_translations, geometry->n_cells, boundary_conditions);

                        minus_translations[0] = (a + neigh[2*i + 1].translations[0]);
                        minus_translations[1] = (b + neigh[2*i + 1].translations[1]);
                        minus_translations[2] = (c + neigh[2*i + 1].translations[2]);
                        int ispin_minus = fast_idx_from_translations(minus_translations, geometry->n_cells, boundary_conditions);

                        if (ispin_plus == -1) {
                            ispin_plus = ispin;
                        }
                        if (ispin_minus == -1) {
                            ispin_minus = ispin;
                        }

                        s0 = (spins[ispin_plus][0] - spins[ispin_minus][0]) / (2 * cell_sizes[i]);
                        s1 = (spins[ispin_plus][1] - spins[ispin_minus][1]) / (2 * cell_sizes[i]);
                        s2 = (spins[ispin_plus][2] - spins[ispin_minus][2]) / (2 * cell_sizes[i]);

                        gradient[ispin][0] += cell_volume * (2 * dmi_tensor(1, i) * s2 - 2 * dmi_tensor(2, i) * s1);
                        gradient[ispin][1] += cell_volume * (2 * dmi_tensor(2, i) * s0 - 2 * dmi_tensor(0, i) * s2);
                        gradient[ispin][2] += cell_volume * (2 * dmi_tensor(0, i) * s1 - 2 * dmi_tensor(1, i) * s0);
                    }
                }
            }
        }
    }

    void Hamiltonian_Micromagnetic::Gradient_DDI(const vectorfield & spins, vectorfield & gradient)
    {
        this->Gradient_DDI_FFT(spins, gradient);
    }
    void Hamiltonian_Micromagnetic::Gradient_DDI_Cutoff(const vectorfield & spins, vectorfield & gradient)
    {
        // TODO
    }

    void Hamiltonian_Micromagnetic::Gradient_DDI_FFT(const vectorfield & spins, vectorfield & gradient)
    {

        FFT_Spins(spins);
        auto& ft_D_matrices = transformed_dipole_matrices;

        auto& ft_spins = fft_plan_spins.cpx_ptr;

        auto& res_iFFT = fft_plan_reverse.real_ptr;
        auto& res_mult = fft_plan_reverse.cpx_ptr;

        int number_of_mults = it_bounds_pointwise_mult[0] * it_bounds_pointwise_mult[1] * it_bounds_pointwise_mult[2];

        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        // TODO: also parallelize over i_b1
        // Loop over basis atoms (i.e sublattices) and add contribution of each sublattice
        int idx, idx_d;

        // Workaround for compability with intel compiler
        const int c_n_cell_atoms = geometry->n_cell_atoms;
        const int * c_it_bounds_pointwise_mult = it_bounds_pointwise_mult.data();

        // Loop over basis atoms (i.e sublattices)
        #pragma omp parallel for collapse(3)
        for( int c = 0; c < c_it_bounds_pointwise_mult[2]; ++c )
        {
            for( int b = 0; b < c_it_bounds_pointwise_mult[1]; ++b )
            {
                for( int a = 0; a < c_it_bounds_pointwise_mult[0]; ++a )
                {
                    idx = a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                    idx_d  = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;

                    auto& fs_x = ft_spins[idx                       ];
                    auto& fs_y = ft_spins[idx + 1 * spin_stride.comp];
                    auto& fs_z = ft_spins[idx + 2 * spin_stride.comp];

                    auto& fD_xx = ft_D_matrices[idx_d                    ];
                    auto& fD_xy = ft_D_matrices[idx_d + 1 * dipole_stride.comp];
                    auto& fD_xz = ft_D_matrices[idx_d + 2 * dipole_stride.comp];
                    auto& fD_yy = ft_D_matrices[idx_d + 3 * dipole_stride.comp];
                    auto& fD_yz = ft_D_matrices[idx_d + 4 * dipole_stride.comp];
                    auto& fD_zz = ft_D_matrices[idx_d + 5 * dipole_stride.comp];

                    FFT::addTo(res_mult[idx + 0 * spin_stride.comp], FFT::mult3D(fD_xx, fD_xy, fD_xz, fs_x, fs_y, fs_z), true);
                    FFT::addTo(res_mult[idx + 1 * spin_stride.comp], FFT::mult3D(fD_xy, fD_yy, fD_yz, fs_x, fs_y, fs_z), true);
                    FFT::addTo(res_mult[idx + 2 * spin_stride.comp], FFT::mult3D(fD_xz, fD_yz, fD_zz, fs_x, fs_y, fs_z), true);
                }
            }// end iteration over padded lattice cells
        }// end iteration over second sublattice

        // Inverse Fourier Transform
        FFT::batch_iFour_3D(fft_plan_reverse);

       // Workaround for compability with intel compiler
        const int * c_n_cells = geometry->n_cells.data();
        // Place the gradients at the correct positions and mult with correct mu
        for( int c = 0; c < c_n_cells[2]; ++c )
        {
            for( int b = 0; b < c_n_cells[1]; ++b )
            {
                for( int a = 0; a < c_n_cells[0]; ++a )
                {
                    int idx_orig = geometry->n_cell_atoms * (a + Na * (b + Nb * c));
                    int idx = a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;
                    gradient[idx_orig][0] -= Ms * cell_volume * res_iFFT[idx                       ] / sublattice_size;
                    gradient[idx_orig][1] -= Ms * cell_volume * res_iFFT[idx + 1 * spin_stride.comp] / sublattice_size;
                    gradient[idx_orig][2] -= Ms * cell_volume * res_iFFT[idx + 2 * spin_stride.comp] / sublattice_size;
                }
            }
        }//end iteration sublattice 1
    }//end Field_DipoleDipole

    void Hamiltonian_Micromagnetic::Gradient_DDI_Direct(const vectorfield & spins, vectorfield & gradient)
    {
        //TODO
    }

    void Hamiltonian_Micromagnetic::FFT_Spins(const vectorfield & spins)
    {
        //size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];
        int n_cell_atoms = geometry->n_cell_atoms;

        auto& fft_spin_inputs = fft_plan_spins.real_ptr;

        //iterate over the **original** system
        #pragma omp parallel for collapse(4)
        for( int c = 0; c < Nc; ++c )
        {
            for( int b = 0; b < Nb; ++b )
            {
                for( int a = 0; a < Na; ++a )
                {
                    for( int bi = 0; bi < n_cell_atoms; ++bi )
                    {
                        int idx_orig = bi + n_cell_atoms * (a + Na * (b + Nb * c));
                        int idx      = bi * spin_stride.basis + a * spin_stride.a + b * spin_stride.b + c * spin_stride.c;

                        fft_spin_inputs[idx                        ] = spins[idx_orig][0] * (cell_volume * Ms);
                        fft_spin_inputs[idx + 1 * spin_stride.comp ] = spins[idx_orig][1] * (cell_volume * Ms);
                        fft_spin_inputs[idx + 2 * spin_stride.comp ] = spins[idx_orig][2] * (cell_volume * Ms);
                    }
                }
            }
        }//end iteration over basis
        FFT::batch_Four_3D(fft_plan_spins);
    }

    void Hamiltonian_Micromagnetic::FFT_Dipole_Matrices(FFT::FFT_Plan & fft_plan_dipole, int img_a, int img_b, int img_c)
    {
         // Size of original geometry
        int Na = geometry->n_cells[0];
        int Nb = geometry->n_cells[1];
        int Nc = geometry->n_cells[2];

        auto& fft_dipole_inputs = fft_plan_dipole.real_ptr;
        scalar mult = C::mu_0 / (4*C::Pi);

        // Iterate over the padded system
        const int * c_n_cells_padded = n_cells_padded.data(); // Workaround for intel compilers..
        #pragma omp parallel for collapse(3)
        for( int c = 0; c < c_n_cells_padded[2]; ++c )
        {
            for( int b = 0; b < c_n_cells_padded[1]; ++b )
            {
                for( int a = 0; a < c_n_cells_padded[0]; ++a )
                {
                    int a_idx = a < Na ? a : a - n_cells_padded[0];
                    int b_idx = b < Nb ? b : b - n_cells_padded[1];
                    int c_idx = c < Nc ? c : c - n_cells_padded[2];
                    scalar Dxx = 0, Dxy = 0, Dxz = 0, Dyy = 0, Dyz = 0, Dzz = 0;
                    Vector3 diff;
                    // Iterate over periodic images
                    for( int a_pb = - img_a; a_pb <= img_a; a_pb++ )
                    {
                        for( int b_pb = - img_b; b_pb <= img_b; b_pb++ )
                        {
                            for( int c_pb = -img_c; c_pb <= img_c; c_pb++ )
                            {
                                diff =  {  (a_idx + a_pb * Na) * cell_sizes[0],
                                         + (b_idx + b_pb * Nb) * cell_sizes[1],
                                         + (c_idx + c_pb * Nc) * cell_sizes[2] };

                                if( diff.norm() > 1e-10 )
                                {
                                    // Implementation of cuboid-cuboid DDI kernel
                                    for (int i = 0; i < 2; i++) {
                                        for (int j = 0; j < 2; j++) {
                                            for (int k = 0; k < 2; k++) {
                                                scalar r = sqrt((a_idx + i - 0.5f)*(a_idx + i - 0.5f)*cell_sizes[0]* cell_sizes[0] + (b_idx + j - 0.5f)*(b_idx + j-0.5f)*cell_sizes[1] * cell_sizes[1] + (c_idx + k - 0.5f)*(c_idx + k - 0.5f)*cell_sizes[2] * cell_sizes[2]);
                                                Dxx += mult * pow(-1.0f, i + j + k) * atan(((c_idx + k-0.5f) * (b_idx + j - 0.5f) * cell_sizes[1]*cell_sizes[2]/cell_sizes[0] / r / (a_idx + i - 0.5f)));
                                                Dxy -= mult * pow(-1.0f, i + j + k) * log((((c_idx + k - 0.5f) * cell_sizes[2] + r)));
                                                Dxz -= mult * pow(-1.0f, i + j + k) * log((((b_idx + j - 0.5f) * cell_sizes[1] + r)));
                                                Dyy += mult * pow(-1.0f, i + j + k) * atan(((a_idx + i-0.5f) * (c_idx + k - 0.5f) * cell_sizes[2]*cell_sizes[0]/cell_sizes[1] / r / (b_idx + j - 0.5f)));
                                                Dyz -= mult * pow(-1.0f, i + j + k) * log((((a_idx + i - 0.5f)* cell_sizes[0] + r)));
                                                Dzz += mult * pow(-1.0f, i + j + k) * atan(((b_idx + j-0.5f) * (a_idx + i - 0.5f) * cell_sizes[0]*cell_sizes[1]/cell_sizes[2] / r / (c_idx + k - 0.5f)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    int idx = a * dipole_stride.a + b * dipole_stride.b + c * dipole_stride.c;
                    fft_dipole_inputs[idx                         ] = Dxx;
                    fft_dipole_inputs[idx + 1 * dipole_stride.comp] = Dxy;
                    fft_dipole_inputs[idx + 2 * dipole_stride.comp] = Dxz;
                    fft_dipole_inputs[idx + 3 * dipole_stride.comp] = Dyy;
                    fft_dipole_inputs[idx + 4 * dipole_stride.comp] = Dyz;
                    fft_dipole_inputs[idx + 5 * dipole_stride.comp] = Dzz;
                }
            }
        }
        FFT::batch_Four_3D(fft_plan_dipole);
    }

    scalar Hamiltonian_Micromagnetic::Energy_Single_Spin(int ispin, const vectorfield & spins)
    {
        // DO NOT USE FOR MONTE CARLO ...
        return 0;
    }

    void Hamiltonian_Micromagnetic::E_DDI(const vectorfield& spins, scalarfield & energy)
    {
        auto temp_gradient = vectorfield(spins.size(), {0,0,0});
        Gradient_DDI_FFT(spins, temp_gradient);
        Energy_From_Gradient(spins, temp_gradient, 0.5, energy);
    }

    void Hamiltonian_Micromagnetic::Prepare_DDI()
    {
        Clean_DDI();
        if( ddi_method != DDI_Method::FFT )
            return;

        // We perform zero-padding in a lattice direction if the dimension of the system is greater than 1 *and*
        //  - the boundary conditions are open, or
        //  - the boundary conditions are periodic and zero-padding is explicitly requested
        n_cells_padded.resize(3);
        for(int i=0; i<3; i++)
        {
            n_cells_padded[i] = geometry->n_cells[i];
            bool perform_zero_padding = geometry->n_cells[i] > 1 && (boundary_conditions[i] == 0 || ddi_pb_zero_padding);
            if(perform_zero_padding)
                n_cells_padded[i] *= 2;
        }
        sublattice_size = n_cells_padded[0] * n_cells_padded[1] * n_cells_padded[2];

        FFT::FFT_Init();

        // Workaround for bug in kissfft
        // kissfft_ndr does not perform one-dimensional FFTs properly
        #ifndef SPIRIT_USE_FFTW
        int number_of_one_dims = 0;
        for( int i=0; i<3; i++ )
            if(n_cells_padded[i] == 1 && ++number_of_one_dims > 1)
                n_cells_padded[i] = 2;
        #endif

        // We dont need to transform over length 1 dims
        std::vector<int> fft_dims;
        for (int i = 2; i >= 0; i--) // Notice that reverse order is important!
        {
            if (n_cells_padded[i] > 1)
                fft_dims.push_back(n_cells_padded[i]);
        }

        // Create FFT plans
        FFT::FFT_Plan fft_plan_dipole  = FFT::FFT_Plan(fft_dims, false, 6, sublattice_size);
        fft_plan_spins   = FFT::FFT_Plan(fft_dims, false, 3 * geometry->n_cell_atoms, sublattice_size);
        fft_plan_reverse = FFT::FFT_Plan(fft_dims, true, 3 * geometry->n_cell_atoms, sublattice_size);

        #ifdef SPIRIT_USE_FFTW
            field<int*> temp_s = {&spin_stride.comp, &spin_stride.basis, &spin_stride.a, &spin_stride.b, &spin_stride.c};
            field<int*> temp_d = {&dipole_stride.comp, &dipole_stride.basis, &dipole_stride.a, &dipole_stride.b, &dipole_stride.c};;
            FFT::get_strides(temp_s, {3, 1, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2]});
            FFT::get_strides(temp_d, {6, 1, n_cells_padded[0], n_cells_padded[1], n_cells_padded[2]});
            it_bounds_pointwise_mult  = {   (n_cells_padded[0]/2 + 1), // due to redundancy in real fft
                                            n_cells_padded[1],
                                            n_cells_padded[2]
                                        };
        #else
            field<int*> temp_s = {&spin_stride.a, &spin_stride.b, &spin_stride.c, &spin_stride.comp, &spin_stride.basis};
            field<int*> temp_d = {&dipole_stride.a, &dipole_stride.b, &dipole_stride.c, &dipole_stride.comp, &dipole_stride.basis};;
            FFT::get_strides(temp_s, {n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 3, 1});
            FFT::get_strides(temp_d, {n_cells_padded[0], n_cells_padded[1], n_cells_padded[2], 6, 1});
            it_bounds_pointwise_mult  = {   n_cells_padded[0],
                                            n_cells_padded[1],
                                            n_cells_padded[2]
                                        };
            (it_bounds_pointwise_mult[fft_dims.size() - 1] /= 2 )++;
        #endif

        //perform FFT of dipole matrices
        int img_a = boundary_conditions[0] == 0 ? 0 : ddi_n_periodic_images[0];
        int img_b = boundary_conditions[1] == 0 ? 0 : ddi_n_periodic_images[1];
        int img_c = boundary_conditions[2] == 0 ? 0 : ddi_n_periodic_images[2];

        FFT_Dipole_Matrices(fft_plan_dipole, img_a, img_b, img_c);

        transformed_dipole_matrices = std::move(fft_plan_dipole.cpx_ptr);
    }//end prepare

    void Hamiltonian_Micromagnetic::Clean_DDI()
    {
        fft_plan_spins = FFT::FFT_Plan();
        fft_plan_reverse = FFT::FFT_Plan();
    }

    void Hamiltonian_Micromagnetic::Hessian(const vectorfield & spins, MatrixX & hessian)
    {
    }


    // Hamiltonian name as string
    static const std::string name = "Micromagnetic";
    const std::string& Hamiltonian_Micromagnetic::Name() { return name; }
}

#endif

