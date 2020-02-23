#pragma once
#ifndef TST_BENNET_INTERFACE_H
#define TST_BENNET_INTERFACE_H

#include "DLL_Define_Export.h"
struct State;

PREFIX float TST_Bennet_Calculate(State * state, int idx_image_minimum, int idx_image_sp, int n_iterations_bennet=5000, int idx_chain=-1) SUFFIX;
PREFIX void TST_Bennet_Get_Info(State * state, float * benn_min, float * err_benn_min, float * benn_sp, float * err_benn_sp, float * unstable_mode_contribution, float * rate, float * rate_err, int idx_chain=-1) SUFFIX;

#endif