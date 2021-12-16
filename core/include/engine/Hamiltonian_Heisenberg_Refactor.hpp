#pragma once
#ifndef SPIRIT_HAMILTONIAN_HEISENBERG_REFACTOR
#define SPIRIT_HAMILTONIAN_HEISENBERG_REFACTOR

#include <Spirit/Hamiltonian.h>
#include <data/Geometry.hpp>
#include <engine/Neighbours.hpp>
#include <engine/Hamiltonian.hpp>
#include <engine/Heisenberg_Interactions.hpp>
#include <memory>

namespace Engine
{

// using namespace Heisenberg_Interactions;

enum class DDI_Method_R //TODO@Moritz: resolve name conflic with DDI_METHOD
{
    FFT    = SPIRIT_DDI_METHOD_FFT,
    FMM    = SPIRIT_DDI_METHOD_FMM,
    Cutoff = SPIRIT_DDI_METHOD_CUTOFF,
    None   = SPIRIT_DDI_METHOD_NONE
};

struct Zeeman_Configuration
{
    scalar magnitude;
    Vector3 normal;
};

struct Anisotropy_Configuration
{
    intfield indices;
    scalarfield magnitudes;
    vectorfield normals;
};

template<typename Pair_t>
struct Pair_Configuration
{
    field<Pair_t> pairs;

    void from_shells(const Data::Geometry & geometry, const scalarfield & shell_magnitudes, bool use_redundant)
    {
        pairs = field<Pair_t>(0);

        const int n_shells = shell_magnitudes.size();
        intfield shells_idx( 0 );
        field<Pair> base_pairs( 0 );

        // Generate neighbours
        Neighbours::Get_Neighbours_in_Shells(geometry, n_shells, base_pairs, shells_idx, use_redundant );

        for( std::size_t ipair = 0; ipair < base_pairs.size(); ++ipair )
        {
            Pair_t new_pair;
            new_pair.from_base_pair( base_pairs[ipair] );
            new_pair.magnitude = shell_magnitudes[shells_idx[ipair]];
            pairs.push_back( new_pair );
        }
    };

    void from_unique_pairs(const field<Pair_t> & pairs_in, bool use_redundant)
    {
        // Use direct list of pairs
        pairs = pairs_in;
        if( use_redundant )
        {
            for( std::size_t i = 0; i < pairs.size(); ++i )
            {
                auto & p = pairs_in[i];
                auto & t = p.translations;
                Pair_t new_pair;
                new_pair.from_conjugate(p);
                this->pairs.push_back( new_pair );
            }
        }
    }
};

struct Interaction_Pair : Pair
{
    void from_conjugate(const Interaction_Pair & other)
    {
        i = other.j;
        j = other.i;
        translations[0] = -other.translations[0];
        translations[1] = -other.translations[1];
        translations[2] = -other.translations[2];
    }

    void from_base_pair(const Pair & other)
    {
        i = other.i;
        j = other.j;
        translations[0] = other.translations[0];
        translations[1] = other.translations[1];
        translations[2] = other.translations[2];
    }
};

struct Exchange_Pair : Interaction_Pair
{
    int magnitude;
    void from_conjugate(const Exchange_Pair & other)
    {
        Interaction_Pair::from_conjugate(other);
        magnitude = other.magnitude;
    }
};

struct DMI_Pair : Interaction_Pair
{
    int magnitude;
    Vector3 normal;

    void from_conjugate(const DMI_Pair & other)
    {
        Interaction_Pair::from_conjugate(other);
        magnitude = other.magnitude;
        normal = -other.normal;
    }
};

using DMI_Configuration      = Pair_Configuration<DMI_Pair>;
using Exchange_Configuration = Pair_Configuration<Exchange_Pair>;

struct DDI_Configuration
{
    DDI_Method_R method;
    intfield n_periodic_images;
    bool pb_zero_padding;
    scalar radius;
};

struct Quadruplet_Configuration
{
    quadrupletfield quadruplets;
    scalarfield magnitudes;
};

class Hamiltonian_Heisenberg_R : public Engine::Hamiltonian
{
public:
    Hamiltonian_Heisenberg_R(
        Zeeman_Configuration external_field, Anisotropy_Configuration anisotropy,
        Exchange_Configuration exchange, DMI_Configuration dmi, DDI_Configuration ddi,
        Quadruplet_Configuration quadruplet, std::shared_ptr<Data::Geometry> geometry, intfield boundary_conditions );

    Zeeman_Configuration external_field;
    Anisotropy_Configuration anisotropy;
    Exchange_Configuration exchange;
    DMI_Configuration dmi;
    DDI_Configuration ddi;
    Quadruplet_Configuration quadruplet;
    std::shared_ptr<Data::Geometry> geometry;
    intfield boundary_conditions;

    void Update_Energy_Contributions() override;

    void Hessian( const vectorfield & spins, MatrixX & hessian ) override;

    void Sparse_Hessian( const vectorfield & spins, SpMatrixX & hessian ) override;

    void Gradient( const vectorfield & spins, vectorfield & gradient ) override;

    // TODO@Moritz base class hamiltonian still has Gradient_and_Energy() with scalar energy instead of scalarfield:
    void Gradient_and_Energy( const vectorfield & spins, vectorfield & gradient, scalarfield & energy );

    void Energy_Contributions_per_Spin(
        const vectorfield & spins, std::vector<std::pair<std::string, scalarfield>> & contributions ) override;

    scalar Energy_Single_Spin( int ispin, const vectorfield & spins ) override;

    // Hamiltonian name as string
    const std::string & Name() override;

protected:
    int idx_zeeman = -1, idx_anisotropy = -1, idx_dmi = -1, idx_exchange = -1, idx_ddi = -1, idx_quadruplet = -1;
};

} // namespace Engine

#endif