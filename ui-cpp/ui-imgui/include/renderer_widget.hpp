#pragma once
#ifndef SPIRIT_IMGUI_RENDERER_WIDGET_HPP
#define SPIRIT_IMGUI_RENDERER_WIDGET_HPP

#include <VFRendering/RendererBase.hxx>
#include <VFRendering/VectorField.hxx>
#include <VFRendering/View.hxx>

#include <memory>

struct State;

namespace ui
{

enum class Colormap
{
    HSV,
    HSV_NO_Z,
    BLUE_WHITE_RED,
    BLUE_GREEN_RED,
    BLUE_RED,
    MONOCHROME
};

struct ColormapWidget
{
    Colormap colormap = Colormap::HSV;
    std::string colormap_implementation_str;

    float colormap_rotation = 0;
    bool colormap_invert_z  = false;
    bool colormap_invert_xy = false;
    glm::vec3 colormap_cardinal_a{ 1, 0, 0 };
    glm::vec3 colormap_cardinal_b{ 0, 1, 0 };
    glm::vec3 colormap_cardinal_c{ 0, 0, 1 };
    glm::vec3 colormap_monochrome_color{ 0.5f, 0.5f, 0.5f };

protected:
    ColormapWidget();
    void reset_colormap();
    void set_colormap_implementation( Colormap colormap );
    bool colormap_input();
};

struct RendererWidget
{
    std::shared_ptr<State> state;
    bool show_   = true;
    bool remove_ = false;
    std::shared_ptr<VFRendering::RendererBase> renderer;
    int id = 0;

    float filter_direction_min[3]{ -1, -1, -1 };
    float filter_direction_max[3]{ 1, 1, 1 };
    float filter_position_min[3]{ 0, 0, 0 };
    float filter_position_max[3]{ 1, 1, 1 };

    virtual void show();
    virtual void apply_settings();

protected:
    RendererWidget( std::shared_ptr<State> state ) : state( state ) {}
    virtual std::string name() const = 0;
    virtual void show_settings()     = 0;
    virtual void reset()             = 0;
    void show_filters();
    void reset_filters();
};

struct BoundingBoxRendererWidget : RendererWidget
{
    BoundingBoxRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void show() override;
    void apply_settings() override;

    float line_width    = 0;
    int level_of_detail = 10;
    bool draw_shadows   = false;

protected:
    std::string name() const override
    {
        return "Boundingbox";
    }
    void show_settings() override;
    void reset() override;
};

struct CoordinateSystemRendererWidget : RendererWidget
{
    CoordinateSystemRendererWidget( std::shared_ptr<State> state );
    void show() override;

protected:
    std::string name() const override
    {
        return "Coordinate system";
    }
    void show_settings() override;
    void reset() override;
};

struct DotRendererWidget : RendererWidget, ColormapWidget
{
    DotRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

    float size = 1;

protected:
    std::string name() const override
    {
        return "Dots";
    }
    void show_settings() override;
    void reset() override;
};

struct ArrowRendererWidget : RendererWidget, ColormapWidget
{
    ArrowRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

    float size = 1;
    int lod    = 10;

protected:
    std::string name() const override
    {
        return "Arrows";
    }
    void show_settings() override;
    void reset() override;
};

struct ParallelepipedRendererWidget : RendererWidget, ColormapWidget
{
    ParallelepipedRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

    float size = 1;

protected:
    std::string name() const override
    {
        return "Boxes";
    }
    void show_settings() override;
    void reset() override;
};

struct SphereRendererWidget : RendererWidget, ColormapWidget
{
    SphereRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

    float size = 0.1f;
    int lod    = 10;

protected:
    std::string name() const override
    {
        return "Spheres";
    }
    void show_settings() override;
    void reset() override;
};

struct SurfaceRendererWidget : RendererWidget, ColormapWidget
{
    SurfaceRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

protected:
    std::string name() const override
    {
        return "Surface";
    }
    void show_settings() override;
    void reset() override;
};

struct IsosurfaceRendererWidget : RendererWidget, ColormapWidget
{
    IsosurfaceRendererWidget(
        std::shared_ptr<State> state, const VFRendering::View & view, const VFRendering::VectorField & vectorfield );
    void apply_settings() override;

    float isovalue    = 0;
    int isocomponent  = 2;
    bool draw_shadows = true;
    bool flip_normals = false;

protected:
    std::string name() const override
    {
        return "Isosurface";
    }
    void show_settings() override;
    void reset() override;
    void set_isocomponent( int isocomponent );
    void set_lighting_implementation( bool draw_shadows );
};

} // namespace ui

#endif