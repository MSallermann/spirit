#pragma once
#ifndef SIMPLE_FMM_SPHERICAL_HARMONICS
#define SIMPLE_FMM_SPHERICAL_HARMONICS
#include <cmath>
#include <complex>
#include "SimpleFMM_Defines.hpp"
#include <iostream>
#include <stdexcept>
#include "Utility.hpp"
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
        
        scalar PI = 3.14159265359;

        std::complex<scalar> Spherical_Harm(int l, int m, scalar phi, scalar theta)
        {
            if(m > 0)
            {
                return 1/std::sqrt(2) * cplx(sh::EvalSH(l, m, phi, theta), sh::EvalSH(l, -m, phi, theta));
            } else if (m < 0)
            {
                return 1/std::sqrt(2) * cplx(std::pow(-1, m) * sh::EvalSH(l, -m, phi, theta), std::pow(-1, m+1) * sh::EvalSH(l, m, phi, theta));
            } else {
                return sh::EvalSH(l, m, phi, theta);
            }
        }

        //Computes normalized laplacian spherical harmonics w/ condon-shortley phase
        //TODO: Do this properly!
        // std::complex<scalar> Spherical_Harm(int l, int m, scalar phi, scalar theta)
        // {
        //     if(std::abs(m) <= l)
        //     {
        //         if(l==0) 
        //         {
        //             return 0.5 * std::sqrt(1/PI);
        //         } 
        //         else if (l==1) 
        //         {
        //             if(m==-1)       return  0.5 * sqrt(1.5 / PI) * exp(cplx(0, -phi)) * sin(theta);
        //             else if (m==0)  return  0.5 * sqrt(3.0 / PI) * cos(theta);
        //             else if (m==1)  return -0.5 * sqrt(1.5 / PI) * exp(cplx(0, phi))  * sin(theta);
        //         } 
        //         else if (l==2) 
        //         {
        //             if(m==-2)       return  0.25 * sqrt(7.5 / PI) * exp(cplx(0, -2*phi)) * sin(theta) * sin(theta);
        //             else if(m==-1)  return  0.5  * sqrt(7.5 / PI) * exp(cplx(0, -phi))   * sin(theta) * cos(theta);
        //             else if(m==0)   return  0.25 * sqrt(5.0 / PI) * (3 * cos(theta) * cos(theta) - 1);
        //             else if(m==1)   return -0.5  * sqrt(7.5 / PI) * exp(cplx(0, phi))   * sin(theta) * cos(theta);
        //             else if(m==2)   return  0.25 * sqrt(7.5 / PI) * exp(cplx(0, 2*phi)) * sin(theta) * sin(theta);
        //         }
        //         else if (l==3) 
        //         {
        //             if(m==-3)       return  0.25 * sqrt(7.5 / PI) * exp(cplx(0, -2*phi)) * sin(theta) * sin(theta);
        //             else if(m==-2)  return  0.5  * sqrt(7.5 / PI) * exp(cplx(0, -phi))   * sin(theta) * cos(theta);
        //             else if(m==0)   return  0.25 * sqrt(5.0 / PI) * (3 * cos(theta) * cos(theta) - 1);
        //             else if(m==2)   return -0.5  * sqrt(7.5 / PI) * exp(cplx(0, phi))   * sin(theta) * cos(theta);
        //             else if(m==3)   return  0.25 * sqrt(7.5 / PI) * exp(cplx(0, 2*phi)) * sin(theta) * sin(theta);
        //         }
                 
        //         else {
        //             throw std::invalid_argument("Band index out of range!");
        //         }
        //     } else {
        //         throw std::invalid_argument("m out of range (|m| <= l not fulfilled)");
        //     }
        // }

        scalar R_prefactor(int l, int m)
        {
            return sqrt(4 * PI / ((2*l + 1) * Utility::factorial(l+m) * Utility::factorial(l-m)));
        }

        std::complex<scalar> R(int l, int m, scalar r, scalar phi, scalar theta)
        {
            return  std::pow(r, l) * R_prefactor(l, m) * Spherical_Harm(l, m, phi, theta);
        }

        scalar S_prefactor(int l, int m)
        {
            return sqrt(4 * PI / (2*l + 1) * Utility::factorial(l+m) * Utility::factorial(l-m));
        }

        std::complex<scalar> S(int l, int m, scalar r, scalar phi, scalar theta)
        {
            return std::pow(r, -(l+1)) * S_prefactor(l, m) * Spherical_Harm(l, m, phi, theta);
        }
    }
}
#endif

