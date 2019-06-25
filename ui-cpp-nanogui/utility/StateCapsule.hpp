#pragma once
#ifndef StateCapsule_HPP
#define StateCapsule_HPP

#include "Spirit/State.h"
#include <string>
#include <vector>

class State_Capsule 
{
    public:

    std::shared_ptr<State> _state;

    State_Capsule(std::string cfgfile, bool quiet) : 
    _state(State_Setup(cfgfile.c_str(), quiet))
    {

    } 

    State * get()
    {
        return _state.get();
    }

    ~State_Capsule()
    {
        State_Delete(this->_state.get());
    }

    std::vector<> Configuration_Change_Listeners;
    void Event_Configuration_Change()
    {
        for(auto f : Configuration_Change_Listeners)
        {

        }
    }
};

#endif