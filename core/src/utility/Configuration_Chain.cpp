#include <data/Spin_System.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Manifoldmath.hpp>
#include <utility/Configuration_Chain.hpp>
#include <utility/Configurations.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <engine/Backend_par.hpp>


namespace Utility
{
namespace Configuration_Chain
{

void Add_Noise_Temperature( std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2, scalar temperature )
{
    for( int img = idx_1 + 1; img <= idx_2 - 1; ++img )
    {
        Configurations::Add_Noise_Temperature( *c->images[img], temperature, img );
    }
}

void Homogeneous_Rotation( std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2 )
{
    // bool translations[3] = {true, true, true};
    // bool rotations[3]    = {true, true, true};
    // // bool translations[3] = {false, false, false};
    // // bool rotations[3]    = {false, false, false};
    // Transition_without_Modes( c, idx_1, idx_2, translations, rotations );
    // return;

    auto & spins_1 = *c->images[idx_1]->spins;
    auto & spins_2 = *c->images[idx_2]->spins;

    scalar angle, rot_angle;
    Vector3 rot_axis;
    Vector3 ex = { 1, 0, 0 };
    Vector3 ey = { 0, 1, 0 };

    bool antiparallel = false;
    for( int i = 0; i < c->images[0]->nos; ++i )
    {
        rot_angle = Engine::Vectormath::angle( spins_1[i], spins_2[i] );
        rot_axis  = spins_1[i].cross( spins_2[i] ).normalized();

        // If spins are antiparallel, we choose an arbitrary rotation axis
        if( std::abs( rot_angle - Constants::Pi ) < 1e-4 )
        {
            antiparallel = true;
            if( std::abs( spins_1[i].dot( ex ) ) - 1 > 1e-4 )
                rot_axis = ex;
            else
                rot_axis = ey;
        }

        // If they are not strictly parallel we can rotate
        if( rot_angle > 1e-8 )
        {
            for( int img = idx_1 + 1; img < idx_2; ++img )
            {
                angle = rot_angle * scalar( img - idx_1 ) / scalar( idx_2 - idx_1 );
                Engine::Vectormath::rotate( spins_1[i], rot_axis, angle, ( *c->images[img]->spins )[i] );
            }
        }
        // Otherwise we simply leave the spin untouched
        else
        {
            for( int img = idx_1 + 1; img < idx_2; ++img )
            {
                ( *c->images[img]->spins )[i] = spins_1[i];
            }
        }
    }
    if( antiparallel )
        Log( Log_Level::Warning, Log_Sender::All,
             "For the interpolation of antiparallel spins an arbitrary rotation axis has been chosen." );
}


void Transition_Without_Zero_Modes( std::shared_ptr<Data::Spin_System_Chain> c, int idx_1, int idx_2)
{
    auto & spins_1 = *c->images[idx_1]->spins;
    auto & spins_2 = *c->images[idx_2]->spins;

    int n_iterations_follow = 1;

    scalar delta_Rx = 10;

    int nos = c->images[0]->nos;

    field<Matrix3> jacobians(nos);

    std::vector<vectorfield> undesired_modes = std::vector<vectorfield>( 6, vectorfield( nos, Vector3::Zero() ) );

    vectorfield connecting_mode(nos, Vector3::Zero( ) );

    auto & spins_final = *c->images[idx_2]->spins;

    auto & geometry = *c->images[0]->geometry;
    auto & hamiltonian = c->images[0]->hamiltonian;

    // Axis and angle for spin propagation
    scalarfield angle = scalarfield(nos, 0);
    vectorfield axis  = vectorfield(nos, Vector3::Zero());

    bool translations[3] = {true, true, true};
    bool rotations[3]    = {true, true, true};
    std::vector<Vector3> translation_vectors  { {1,0,0}, {0,1,0}, {0,0,1} };
    std::vector<Vector3> rotation_axes        { {1,0,0}, {0,1,0}, {0,0,1} };
    std::vector<Vector3> rotation_centers     { geometry.center, geometry.center, geometry.center };

    for(int img = idx_1+1; img < idx_2; img++)
    {
        auto & spins_prev = *c->images[img-1]->spins;
        auto & spins_cur  = *c->images[img]->spins;

        spins_cur = spins_prev;

        // Engine::Vectormath::set_c_a(1, spins_prev, spins_cur);

        for(int iteration=0; iteration<n_iterations_follow; iteration++)
        {

            // Compute connecting mode
            Engine::Vectormath::set_c_a(1, spins_final, connecting_mode);
            Engine::Vectormath::add_c_a(-1, spins_cur, connecting_mode);
            Engine::Manifoldmath::project_tangential(connecting_mode, spins_cur);
            Engine::Manifoldmath::normalize( connecting_mode );

            // Compute the jacobians
            Engine::Vectormath::jacobian( spins_cur, geometry, hamiltonian->boundary_conditions, jacobians);

            // Project out translation
            for(int i=0; i<3; i++)
            {
                if(translations[i])
                {
                    Engine::Vectormath::translational_mode( undesired_modes[i], spins_cur, geometry, jacobians, translation_vectors[i] );
                    Engine::Manifoldmath::project_tangential( undesired_modes[i], spins_cur );
                    Engine::Manifoldmath::project_orthogonal( connecting_mode, undesired_modes[i] );
                }
            }

            // Project out tanslation rotations
            for(int i=3; i<6; i++)
            {
                if(rotations[i-3])
                {
                    // Engine::Vectormath::spin_spatial_rotational_mode( undesired_modes[i], spins_cur, geometry, jacobians, rotation_axes[i-3], rotation_centers[i-3] );
                    Engine::Vectormath::spin_rotational_mode( undesired_modes[i], spins_cur, rotation_axes[i-3] );
                    Engine::Manifoldmath::project_tangential( undesired_modes[i], spins_cur );
                    Engine::Manifoldmath::project_orthogonal( connecting_mode, undesired_modes[i] );
                }
            }

            Engine::Manifoldmath::normalize( connecting_mode );

            scalar total_rotated_angle_sq = 0;
            for( int idx = 0; idx < nos; idx++ )
            {
                angle[idx]              = connecting_mode[idx].norm() / n_iterations_follow;
                total_rotated_angle_sq += angle[idx] * angle[idx];
                axis[idx]               = spins_cur[idx].cross( connecting_mode[idx] ).normalized();
            }

            scalar dist_current = Engine::Manifoldmath::dist_geodesic( spins_cur, spins_final );
            scalar angle_scale  =  dist_current / ( total_rotated_angle_sq * 2 * (idx_2 - img) );

            Engine::Vectormath::scale( angle, angle_scale );
            Engine::Vectormath::rotate( spins_cur, axis, angle, spins_cur );

        }
    }
}

} // namespace Configuration_Chain
} // namespace Utility
