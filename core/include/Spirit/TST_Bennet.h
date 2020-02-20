#pragma once
#ifndef TST_BENNET_INTERFACE_H
#define TST_BENNET_INTERFACE_H

#include "DLL_Define_Export.h"
struct State;

PREFIX float TST_Bennet_Calculate(State * state, int idx_image_minimum, int idx_image_sp, int n_iterations_bennet=5000, int idx_chain=-1) SUFFIX;

#endif