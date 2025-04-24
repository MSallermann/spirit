#pragma once
#ifndef SPIRIT_CORE_UTILITY_CONSTANTS_HPP
#define SPIRIT_CORE_UTILITY_CONSTANTS_HPP
#include <Spirit/Spirit_Defines.h>

namespace Utility
{

/*
 * Constants by convention:
 * - Energy scale: Millielctronvolts
 * - Time scale: Picoseconds
 * - Magnetic fields scale: Tesla
 */
namespace Constants
{

// The Bohr Magneton [meV/T]
scalar constexpr mu_B = 0.057883817555;

// The vacuum permeability [T^2 m^3 / meV]
scalar constexpr mu_0 = 2.0133545 * 1e-28;

// The Boltzmann constant [meV/K]
scalar constexpr k_B = 0.08617330350;

// Planck constant [meV*ps/rad]
scalar constexpr hbar = 0.6582119514;

// Gyromagnetic ratio of electron [rad/(ps*T)]
// Also gives the Larmor precession frequency for electron
scalar constexpr gamma = 0.1760859644;

// Electron (Landé) g-factor = gamma * hbar / mu_B [unitless]
scalar constexpr g_e = 2.00231930436182;

// Millirydberg [mRy/meV]
scalar constexpr mRy = 1.0 / 13.605693009;

// erg [erg/meV]
scalar constexpr erg = 6.2415091 * 1e14;

// Pi [rad]
scalar constexpr Pi = 3.141592653589793238462643383279502884197169399375105820974;

// Pi/2 [rad]
scalar constexpr Pi_2 = 1.570796326794896619231321691639751442098584699687552910487;

} // namespace Constants

/*
 * Constants_mRy by convention:
 * - Energy scale: Millirydberg
 * - Time scale: Picoseconds
 * - Magnetic fields scale: Tesla
 */
namespace Constants_mRy
{

// The Bohr Magneton [mRy/T]
scalar constexpr mu_B = Constants::mu_B / Constants::mRy;

// The Boltzmann constant [mRy/K]
scalar constexpr k_B = Constants::k_B / Constants::mRy;

// Planck constant [mRy*ps/rad]
scalar constexpr hbar = Constants::hbar / Constants::mRy;

// Millielectronvolt [meV/mRy]
scalar constexpr meV = 1.0 / Constants::mRy;

// Gyromagnetic ratio of electron [rad/(ps*T)]
scalar constexpr gamma = 0.1760859644;

// Electron g-factor [unitless]
scalar constexpr g_e = 2.00231930436182;

// Pi [rad]
scalar constexpr Pi = 3.141592653589793238462643383279502884197169399375105820974;

// Pi/2 [rad]
scalar constexpr Pi_2 = 1.570796326794896619231321691639751442098584699687552910487;

} // namespace Constants_mRy

} // namespace Utility

#endif
