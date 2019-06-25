#pragma once
#ifndef MAINSCREENHPP
#define MAINSCREENHPP

#include "MainScreen.hpp"
#include "SpinWidget.hpp"
#include "SimulationWidget.hpp"
#include "Spirit/State.h"
#include <nanogui/layout.h>
#include <nanogui/button.h>
#include <nanogui/window.h>

using nanogui::Screen;
using namespace nanogui;

MainScreen::MainScreen(std::shared_ptr<State> state)
: Screen( Eigen::Vector2i(1024, 768), "NanoGUI Test", 
    /*resizable*/true, /*fullscreen*/false, /*colorBits*/8,
    /*alphaBits*/8, /*depthBits*/24, /*stencilBits*/8,
    /*nSamples*/0, /*glMajor*/4, /*glMinor*/1 ), state(state)
{
    SpinWidget * s        = new SpinWidget(this, state.get());
    SimulationWidget * s2 = new SimulationWidget(this, state.get(), *s, {this->size()[0]*0.25, this->size()[1]*0.25});
}

#endif