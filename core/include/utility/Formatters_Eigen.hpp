#pragma once
#ifndef SPIRIT_CORE_FORMATTERS_EIGEN_HPP
#define SPIRIT_CORE_FORMATTERS_EIGEN_HPP

#include <Eigen/Core>

#include <fmt/ostream.h>
#include <fmt/ranges.h>

template<typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>> : ostream_formatter
{
};

// opt-out of ranges formatting
template<typename T>
struct fmt::is_range<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
{
    static constexpr const bool value = false;
};

#endif
