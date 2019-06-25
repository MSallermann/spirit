#pragma once
#ifndef SIMULATIONWIDGET_HPP
#define SIMULATIONWIDGET_HPP

#include <nanogui/window.h>
#include "Spirit/State.h"
#include "SpinWidget.hpp"

class SimulationWidget : public nanogui::Window
{
    public:
    State * state;
    SimulationWidget(nanogui::Widget * parent, State * state, SpinWidget & spin_widget, const Eigen::Vector2i & size = {-1,-1});
};

#endif