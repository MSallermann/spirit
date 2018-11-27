#pragma once
#ifndef SIMPLE_FMM_DEFINES_HPP
#include <array>
#include <complex>
#include "Eigen/Core"
#include "Spirit_Defines.h"

namespace SimpleFMM
{
    // # ifndef scalar
    // #   define scalar double
    // # endif

    using Vector3c   = Eigen::Matrix<std::complex<scalar>, 3, 1>;
    using Vector3 = Eigen::Matrix<scalar, 3, 1>;
    using Matrix3 = Eigen::Matrix<scalar, 3, 3>;
    using Matrix3c = Eigen::Matrix<std::complex<scalar>, 3, 3>;
    using vectorfield = std::vector<Vector3>;
    using intfield = std::vector<int>;
    using scalarfield = std::vector<scalar>;
    using complexfield = std::vector<std::complex<scalar> >;
}

#endif