#pragma once
#ifndef SIMPLE_FMM_SPHERICAL_HARMONICS
#define SIMPLE_FMM_SPHERICAL_HARMONICS
#include <cmath>
#include <complex>
#include "fmm/SimpleFMM_Defines.hpp"
#include <iostream>
#include <stdexcept>
#include "fmm/Utility.hpp"
#include "sh/spherical_harmonics.h"

// #define PI 3.14159265359
namespace SimpleFMM
{
    namespace Spherical_Harmonics
    {
        using std::cos;
        using std::sin;
        using std::sqrt;
        using std::exp;
        using cplx = std::complex<scalar>;
        using Utility::minus_one_power;
        using Utility::factorial;
        
        static constexpr scalar PI = 3.1415926535897932384626433;
        // This should produce the ortho-normal spherical harmonics with condon-shortley phase
        inline std::complex<scalar> Spherical_Harm(int l, int m, scalar phi, scalar theta)
        {
            if(m > 0) {
                return 1/std::sqrt(2) * cplx(sh::EvalSH(l, m, phi, theta), sh::EvalSH(l, -m, phi, theta));
            } else if (m < 0) {
                return 1/std::sqrt(2) * cplx(minus_one_power(m) * sh::EvalSH(l, -m, phi, theta), minus_one_power(m+1) * sh::EvalSH(l, m, phi, theta));
            } else {
                return sh::EvalSH(l, m, phi, theta);
            }
        }

        inline scalar R_prefactor(int l, int m)
        {
            return sqrt(4*PI / ( (2*l + 1) * factorial(l+m) * factorial(l-m) ) );
        }

        inline std::complex<scalar> R(int l, int m, scalar r, scalar phi, scalar theta)
        {
            return  std::pow(r, l) * R_prefactor(l, m) * Spherical_Harm(l, m, phi, theta);
        }

        inline scalar S_prefactor(int l, int m)
        {
            return sqrt(4*PI / (2*l + 1) * factorial(l+m) * factorial(l-m));
        }

        inline std::complex<scalar> S(int l, int m, scalar r, scalar phi, scalar theta)
        {
            return std::pow(r, -(l+1)) * S_prefactor(l, m) * Spherical_Harm(l, m, phi, theta);
        }
    }
}
#endif

