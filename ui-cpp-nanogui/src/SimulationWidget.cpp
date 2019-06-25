#include "SimulationWidget.hpp"
#include "Configurations.h"
#include <nanogui/button.h>
#include <nanogui/layout.h>
#include "Spirit/Simulation.h"
#include <iostream>
#include <thread>

using namespace nanogui;
SimulationWidget::SimulationWidget(nanogui::Widget* parent, State * state, SpinWidget & spin_widget, const Eigen::Vector2i & size) : nanogui::Window(parent), state(state)
{
    if(size[0] < 0 || size[1] < 0)
    {
        this->setSize(parent->size());
    }
    else
    {
        this->setSize(size);
    }
    this->setTitle("Simulation");
    this->setLayout(new nanogui::BoxLayout(nanogui::Orientation::Vertical) );

    nanogui::Button * b = new nanogui::Button(this, "Random" );
    b->setTooltip("Set a random spin configuration");
    b->setCallback([&] 
    { 
        Configuration_Random( this->state ); 
        spin_widget.updateData();
    });

    b = new nanogui::Button(this, "+z" );
    b->setTooltip("Set a +z spin configuration");
    b->setCallback([&] 
        { 
            Configuration_PlusZ( this->state ); 
            spin_widget.updateData();
        }
    );

    b = new nanogui::Button(this, "Start" );
    b->setTooltip("Start/Stop the simulation");
    b->setCallback([&, b] 
    {
        if(!Simulation_Running_On_Image(this->state))
        {
            auto th = new std::thread(&Simulation_LLG_Start, this->state, 0, -1, -1, false, -1, -1);
            spin_widget.updateData();
            b->setCaption("Stop");
        } else {
            Simulation_Stop_All(this->state);
            b->setCaption("Start");
        }
    });

    // this->performLayout();
    // this->performLayout(parent->mNVGContext);
    // this->draw(parent->mNVGContext);
}