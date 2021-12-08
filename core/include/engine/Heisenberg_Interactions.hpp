#pragma once
#ifndef SPIRIT_HEISENBERG_INTERACTIONS
#define SPIRIT_HEISENBERG_INTERACTIONS

#ifdef SPIRIT_USE_CUDA
#define SPIRIT_INTERACTION_FUNC __forceinline__ __device__
#else
#define SPIRIT_INTERACTION_FUNC inline
#endif

#include <engine/Backend_par.hpp>
#include <engine/Vectormath_Defines.hpp>

namespace Engine
{
namespace Heisenberg_Interactions
{

template<typename descriptor_t>
SPIRIT_INTERACTION_FUNC int idx_from_pair( int ispin, int ipair )
{
    return 0;
}

template<typename descriptor_t>
SPIRIT_INTERACTION_FUNC void single_site_interaction(
    int ispin, const Vector3 * spins, Vector3 * gradients, scalar * energies, descriptor_t descriptor )
{
    if( descriptor.check( ispin ) )
    {
        gradients[ispin] += descriptor.gradient( ispin, spins );
        energies[ispin] += descriptor.energy( ispin, spins );
    }
}

template<typename descriptor_t>
SPIRIT_INTERACTION_FUNC void pair_interaction(
    int ispin, const Vector3 * spins, Vector3 * gradients, scalar * energies, descriptor_t descriptor )
{
    if( descriptor.check( ispin ) )
    {
        Vector3 gradient_ispin = { 0, 0, 0 };
        scalar energy_ispin    = 0.0f;
        for( auto ipair = 0; ipair < descriptor.n_pairs( ispin ); ++ipair )
        {
            const typename descriptor_t::pair_type & pair = descriptor.get_pair( ispin, ipair );
            int jspin                                     = idx_from_pair( ispin, pair );
            if( jspin > 0 )
            {
                gradient_ispin += pair.gradient( ispin, jspin, spins );
                energy_ispin += pair.energy( ispin, jspin, spins );
            }
        }
        gradients[ispin] += gradient_ispin;
        energies[ispin] += energy_ispin;
    }
}

struct Zeeman
{
    scalar magnitude;
    Vector3 normal;
    scalar * mu_s;

    SPIRIT_INTERACTION_FUNC bool check( int ispin )
    {
        return true;
    }

    SPIRIT_INTERACTION_FUNC Vector3 gradient( int ispin, const Vector3 * spins )
    {
        return mu_s[ispin] * magnitude * normal;
    }

    SPIRIT_INTERACTION_FUNC scalar energy( int ispin, const Vector3 * spins )
    {
        return mu_s[ispin] * magnitude * normal.dot( spins[ispin] );
    }
};

template<typename pair_t>
struct Homogeneous_Pair_Descriptor
{
    using pair_type = pair_t;
    int * offsets;
    int * m_n_pairs;
    pair_t * pairs;
    int n_cell_atoms;

    SPIRIT_INTERACTION_FUNC bool check( int ispin )
    {
        return true;
    }

    SPIRIT_INTERACTION_FUNC int n_pairs( int ispin )
    {
        return m_n_pairs[ispin % n_cell_atoms];
    }

    SPIRIT_INTERACTION_FUNC const Pair & get_pair( int ispin, int ipair )
    {
        return pairs[offsets[ispin % ipair]];
    }
};

struct Exchange_Pair : Pair
{
    scalar magnitude;
    SPIRIT_INTERACTION_FUNC Vector3 gradient( int ispin, int jspin, const Vector3 * spins )
    {
        return magnitude * spins[jspin];
    }

    SPIRIT_INTERACTION_FUNC scalar energy( int ispin, int jspin, const Vector3 * spins )
    {
        return magnitude * spins[ispin].dot( spins[jspin] );
    }
};

struct DMI_Pair : Pair
{
    scalar magnitude;
    Vector3 normal;
    SPIRIT_INTERACTION_FUNC Vector3 gradient( int ispin, int jspin, const Vector3 * spins )
    {
        return magnitude * spins[jspin].cross( normal );
    }

    SPIRIT_INTERACTION_FUNC scalar energy( int ispin, int jspin, const Vector3 * spins )
    {
        return 0.5 * magnitude * normal.dot( spins[ispin].cross( spins[jspin] ) );
    }
};

} // namespace Heisenberg_Interactions
} // namespace Engine
#endif