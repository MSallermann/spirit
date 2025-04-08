#pragma once

#include <engine/common/Interaction_Wrapper.hpp>
#include <engine/spin/interaction/Functor_Prototypes.hpp>
#include <utility/Exception.hpp>

namespace Engine
{

namespace Spin
{

namespace Interaction
{

using Common::Interaction::is_local;

template<typename state_type>
struct IWrapper : public Common::Interaction::IWrapper<state_type>
{
    using state_t = state_type;

    virtual std::size_t Sparse_Hessian_Size_per_Cell() const               = 0;
    virtual void Gradient( const state_t & state, vectorfield & gradient ) = 0;
    virtual void Hessian( const state_t & state, MatrixX & hessian )       = 0;

protected:
    constexpr IWrapper() = default;
};

template<typename InteractionType, typename Interface>
struct Wrapper : public Common::Interaction::Wrapper<InteractionType, Interface>
{
    using base_t = Common::Interaction::Wrapper<InteractionType, Interface>;

    using Interaction = InteractionType;
    using Data        = typename Interaction::Data;
    using Cache       = typename Interaction::Cache;

    using state_t = typename Interaction::state_t;

    constexpr Wrapper() noexcept = default;
    Wrapper( const Data & data, Cache & cache ) noexcept : base_t( data, cache ) {};

    std::size_t Sparse_Hessian_Size_per_Cell() const final
    {
        return Interaction::Sparse_Hessian_Size_per_Cell( this->data, this->cache );
    }

    void Gradient( const state_t & state, vectorfield & gradient ) final
    {
        if constexpr( is_local<InteractionType>::value )
        {
            if( this->indices.offsets.size() <= 1 )
                return;

            const int n_spans = this->indices.offsets.size() - 1;

            if( gradient.size() != n_spans )
                spirit_throw(
                    Utility::Exception_Classifier::Standard_Exception, Utility::Log_Level::Error,
                    fmt::format(
                        "Mismatched size for indices in Gradient caclulation (Interaction: '{}')", this->Name() ) );

            auto state_ptr          = static_cast<typename state_traits<state_t>::const_pointer>( state.data() );
            auto functor            = typename Interaction::Gradient( this->data, this->cache );
            const auto * idx_offset = this->indices.offsets.data();
            const auto * idx_data   = this->indices.data.data();
            auto * gradient_ptr     = gradient.data();

            Backend::for_each_n(
                SPIRIT_PAR Backend::make_counting_iterator<int>( 0 ), n_spans,
                [state_ptr, functor, idx_offset, idx_data, gradient_ptr] SPIRIT_LAMBDA( const int idx )
                {
                    gradient_ptr[idx] += functor(
                        Span( idx_data + idx_offset[idx], idx_offset[idx + 1] - idx_offset[idx] ), state_ptr );
                } );
        }
        else
        {
            std::invoke( typename Interaction::Gradient( this->data, this->cache ), state, gradient );
        }
    }

    template<typename Callable>
    void Hessian_Impl( const state_t & state, Callable && hessian )
    {
        if constexpr( is_local<InteractionType>::value )
        {
            if( this->indices.offsets.size() <= 1 )
                return;

            const int n_spans = this->indices.offsets.size() - 1;

            auto functor            = typename Interaction::Hessian( this->data, this->cache );
            const auto * idx_offset = this->indices.offsets.data();
            const auto * idx_data   = this->indices.data.data();

            Backend::cpu::for_each_n(
                Backend::make_counting_iterator( 0 ), n_spans,
                [&state, &hessian, functor, idx_offset, idx_data]( const int idx ) {
                    functor(
                        Span( idx_data + idx_offset[idx], idx_offset[idx + 1] - idx_offset[idx] ), state, hessian );
                } );
        }
        else
        {
            std::invoke( typename Interaction::Hessian( this->data, this->cache ), state, hessian );
        }
    }

    void Hessian( const state_t & state, MatrixX & hessian )
    {
        Hessian_Impl( state, Functor::dense_hessian_wrapper( hessian ) );
    }
};

} // namespace Interaction

} // namespace Spin

} // namespace Engine
