#include <nanogui/nanogui.h>
#include <nanogui/button.h>
#include <nanogui/window.h>
#include <iostream>
#include "MainScreen.hpp"

struct State;

int main(int argc , char ** argv ) {
    std::string cfgfile = "input/input.cfg";
    nanogui::init();
    {
        std::unique_ptr<MainScreen> app(new MainScreen( std::shared_ptr<State>( State_Setup(cfgfile.c_str()), State_Delete ) ));
        app->performLayout();
        app->drawAll();
        app->setVisible(true);
        app->drawWidgets();
        nanogui::mainloop();
    }
    nanogui::shutdown();
    return 0;
}
