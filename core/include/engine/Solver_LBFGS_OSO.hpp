#pragma once
#ifndef SPIRIT_CORE_ENGINE_SOLVER_LBFGS_OSO_HPP
#define SPIRIT_CORE_ENGINE_SOLVER_LBFGS_OSO_HPP

#include <utility/Constants.hpp>
// #include <utility/Exception.hpp>

#include <algorithm>

using namespace Utility;

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Initialize()
{
    this->n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
    this->delta_a        = std::vector<field<vectorfield>>(
        this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->delta_grad = std::vector<field<vectorfield>>(
        this->noi, field<vectorfield>( this->n_lbfgs_memory, vectorfield( this->nos, { 0, 0, 0 } ) ) );
    this->rho            = scalarfield( this->n_lbfgs_memory, 0 );
    this->alpha          = scalarfield( this->n_lbfgs_memory, 0 );
    this->forces         = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_virtual = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->forces_previous = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->searchdir      = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad           = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->grad_pr        = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );
    this->q_vec          = std::vector<vectorfield>( this->noi, vectorfield( this->nos, { 0, 0, 0 } ) );

    this->configurations_temp = std::vector<std::shared_ptr<vectorfield>>( this->noi );
    for( int i = 0; i < this->noi; i++ )
        configurations_temp[i] = std::shared_ptr<vectorfield>( new vectorfield( this->nos ) );

    this->local_iter     = 0;
    this->maxmove        = Constants::Pi / 200.0;
};

/*
    Implemented according to Aleksei Ivanov's paper: https://arxiv.org/abs/1904.02669
    TODO: reference painless conjugate gradients
    See also Jorge Nocedal and Stephen J. Wright 'Numerical Optimization' Second Edition, 2006 (p. 121).
*/

template<>
inline void Method_Solver<Solver::LBFGS_OSO>::Iteration()
{
    // update forces which are -dE/ds
    this->Calculate_Force( this->configurations, this->forces );
    // calculate gradients for OSO
    for( int img = 0; img < this->noi; img++ )
    {
        auto & image    = *this->configurations[img];
        auto & grad_ref = this->grad[img];

        auto fv = this->forces_virtual[img].data();
        auto f  = this->forces[img].data();
        auto s  = image.data();

        Backend::par::apply( this->nos, [f, fv, s] SPIRIT_LAMBDA( int idx ) { fv[idx] = s[idx].cross( f[idx] ); } );

        Solver_Kernels::oso_calc_gradients( grad_ref, image, this->forces[img] );
    }

    // calculate search direction
    Solver_Kernels::lbfgs_get_searchdir(
        this->local_iter, this->rho, this->alpha, this->q_vec, this->searchdir, this->delta_a, this->delta_grad,
        this->grad, this->grad_pr, this->n_lbfgs_memory, maxmove );

    // Scale direction
    scalar scaling = 1;
    for( int img = 0; img < noi; img++ )
        scaling = std::min( Solver_Kernels::maximum_rotation( searchdir[img], maxmove ), scaling );


    for( int img = 0; img < noi; img++ )
    {
        Vectormath::scale( searchdir[img], scaling );
    }

    // Compute *energy* gradients and energy of current spins
    // Note that these gradients can be different from the forces which are returned by Calculate_Force
    // E.g. for a GNEB calculation the forces would also include the springs while the energy gradient does not
    scalarfield energy    = scalarfield(noi);
    scalarfield step_size = scalarfield(noi);
    scalarfield a = scalarfield(noi);
    scalarfield b = scalarfield(noi);
    scalarfield c = scalarfield(noi);

    for( int img = 0; img < noi; img++ )
    {
        this->systems[img]->hamiltonian->Gradient_and_Energy( *this->configurations[img], this->forces_previous[img], energy[img] );
        Solver_Kernels::oso_calc_gradients( forces_previous[img], *this->configurations[img], this->forces_previous[img] );
        c[img] = energy[img];
        b[img] = -Vectormath::dot( searchdir[img], forces_previous[img] );
    }

    // TODO: reduce gradient and energy evaluations as much as possible

    // rotate temporary spins
    for( int img = 0; img < noi; img++ )
    {
        Vectormath::set_c_a(1, *this->configurations[img], *this->configurations_temp[img]);
        Solver_Kernels::oso_rotate( *this->configurations_temp[img], this->searchdir[img] );
        this->systems[img]->hamiltonian->Gradient_and_Energy( *this->configurations_temp[img], this->forces_previous[img], energy[img] );
        Solver_Kernels::oso_calc_gradients( forces_previous[img], *this->configurations_temp[img], this->forces_previous[img] );
        a[img] = 0.5 * ( -Vectormath::dot( searchdir[img], forces_previous[img] ) - b[img] );
    }

    scalar alpha = 1;
    for( int img = 0; img < noi; img++ )
    {
        scalar epsilon = 5e-2;
        auto prop =
            [
                &searchdir = searchdir[img],
                img
            ]( const vectorfield & spins, vectorfield & spins_buffer, scalar alpha )
            {
                Vectormath::set_c_a(1, spins, spins_buffer);
                Solver_Kernels::oso_rotate( spins_buffer, searchdir, alpha );
            };

        scalar alpha_img = Solver_Kernels::backtracking_linesearch( *this->systems[img]->hamiltonian, b[img], a[img], c[img], epsilon, 0.5, *this->configurations[img], *this->configurations_temp[img], prop);
        alpha = std::min(alpha, alpha_img);
    }

    // fmt::print("alpha = {}\n", alpha);
    for( int img = 0; img < noi; img++ )
    {
        Solver_Kernels::oso_rotate( *this->configurations[img], searchdir[img], alpha );
    }

}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverName()
{
    return "LBFGS_OSO";
}

template<>
inline std::string Method_Solver<Solver::LBFGS_OSO>::SolverFullName()
{
    return "Limited memory Broyden-Fletcher-Goldfarb-Shanno using exponential transforms";
}

#endif