#pragma once
#ifndef SPINWIDGET_HPP
#define SPINWIDGET_HPP
#include "Spirit/State.h"
#include "Spirit/Geometry.h"
#include "Spirit/System.h"

#include <glm/glm.hpp>
#include <nanogui/window.h>
#include <nanogui/widget.h>
#include <nanogui/glcanvas.h>
#include <VFRendering/View.hxx>
#include <VFRendering/Geometry.hxx>
#include <VFRendering/CombinedRenderer.hxx>
#include <VFRendering/ArrowRenderer.hxx>

struct State_Info
{
    State_Info(State * state)
    {
        Build_From_State(state);
    }

    std::vector<glm::vec3> positions;

    int nos;
    int n_cell_atoms;
    int * n_cells  = new int[3];
    float * min    = new float[3];
    float * max    = new float[3];
    float * center = new float[3];

    void Build_From_State(State * state)
    {
        this->nos          = Geometry_Get_NOS(state);
        this->pos          = Geometry_Get_Positions(state);
        this->n_cell_atoms = Geometry_Get_N_Cell_Atoms(state);
        this->spins        = System_Get_Spin_Directions(state);
        Geometry_Get_Bounds(state, this->min, this->max);
        Geometry_Get_N_Cells(state, this->n_cells);
        Geometry_Get_Center(state, this->center);
        
        positions = std::vector<glm::vec3>(this->nos);
        
        for(int i = 0; i < this->nos; i++)
        {
            positions[i][0] = this->pos[3*i    ];
            positions[i][1] = this->pos[3*i + 1];
            positions[i][2] = this->pos[3*i + 2];
        }
    }

    ~State_Info()
    {
        delete[] n_cells;
        delete[] center;
        delete[] min;
        delete[] max;
    }

    private:
    double * pos;
    double * spins;
};

class SpinWidget : public nanogui::GLCanvas
{
    public:
    VFRendering::View view;
    VFRendering::VectorField * vf;
    VFRendering::Geometry geometry;
    State * state;
    std::vector<glm::vec3> directions;

    State_Info state_info;

    SpinWidget(Widget* parent, State * state, const Eigen::Vector2i & size = {-1,-1});

    std::vector<std::shared_ptr<VFRendering::RendererBase>> renderers;
    std::shared_ptr<VFRendering::ArrowRenderer> arrow_renderer;

    void updateData();
    void updateRenderers();
    void setSize(const Eigen::Vector2i & size);

    virtual bool mouseDragEvent(const Eigen::Vector2i &current_position,
        const Eigen::Vector2i &relative_position, int button, int modifiers) override
    {
        auto prev = current_position-relative_position;
        glm::vec2 previous = {prev.x(), prev.y()};
        glm::vec2 current = {current_position.x(), current_position.y()};
        view.mouseMove(previous, current, VFRendering::CameraMovementModes::ROTATE_BOUNDED);
        return true;
    }

    virtual bool scrollEvent(const Eigen::Vector2i &p, const Eigen::Vector2f &relative_position) override
    {
        view.mouseScroll(relative_position.y());
        return true;
    }

    virtual void drawGL() override;

    ~SpinWidget() {
        delete this->vf;
    }
};

#endif