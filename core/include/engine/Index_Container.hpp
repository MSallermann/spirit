#pragma once

#include <data/Geometry.hpp>
#include <engine/Backend.hpp>
#include <engine/Span.hpp>
#include <engine/Vectormath_Defines.hpp>

#include <type_traits>

namespace Engine
{

// non-local IndexContainer is an empty class
template<typename Interaction, typename = void>
class IndexContainer
{
};

// local IndexContainer as a contiguous vector of indices grouped by offsets
template<typename InteractionType>
struct IndexContainer<InteractionType, std::void_t<typename InteractionType::Index>>
{
    using Interaction = InteractionType;
    using Index       = typename Interaction::Index;

    field<int> offsets{};
    field<Index> data{};

    IndexContainer() = default;
};

template<typename InteractionType, typename ValueType, typename = void>
auto make_index_container( ValueType && values ) -> IndexContainer<InteractionType>
{
    IndexContainer<InteractionType> container{};

    if( std::all_of( values.begin(), values.end(), []( const auto & item ) { return item.empty(); } ) )
        return container;

    container.offsets.reserve( values.size() + 1 );
    container.offsets.push_back( 0 );

    container.data.reserve( std::transform_reduce(
        values.begin(), values.end(), std::size_t( 0 ), std::plus<std::size_t>{},
        []( const auto & item ) { return item.size(); } ) );

    for( auto & item : values )
    {
        container.offsets.push_back( container.offsets.back() + item.size() );
        if constexpr( std::is_rvalue_reference_v<ValueType> )
            std::move( item.begin(), item.end(), std::back_inserter( container.data ) );
        else
            std::copy( item.begin(), item.end(), std::back_inserter( container.data ) );
    }
    return container;
}

template<typename InteractionType>
bool verify_index_container( const IndexContainer<InteractionType> & container )
{
    if( container.offsets.empty() )
        return true;

    if( container.offsets.front() != 0 )
        return false;

    for( int i = 1; i < container.offsets.size(); ++i )
    {
        if( container.offsets[i - 1] > container.offsets[i] )
            return false;
    }

    return container.offsets.back() <= container.data.size();
}

} // namespace Engine
