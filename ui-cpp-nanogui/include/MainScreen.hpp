#include "Spirit/State.h"
#include <nanogui/screen.h>

class MainScreen : public nanogui::Screen 
{
    public:
    std::shared_ptr<State> state;
    MainScreen(std::shared_ptr<State>);
};