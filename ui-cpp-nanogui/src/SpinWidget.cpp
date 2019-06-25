#include "SpinWidget.hpp"
#include <vector>

#include "glm/glm.hpp"
#include <glm/gtx/transform.hpp>
#include <VFRendering/Geometry.hxx>
#include <VFRendering/ArrowRenderer.hxx>
#include <VFRendering/VectorField.hxx>
#include "Spirit/Geometry.h"
#include "Spirit/System.h"
#include "MainScreen.hpp"

SpinWidget::SpinWidget(nanogui::Widget * parent, State * state, const Eigen::Vector2i & size) : nanogui::GLCanvas(parent), state(state), state_info(state)
{
    // Call this *before* setSize
    this->view = VFRendering::View();

    if(size[0] < 0 || size[1] < 0)
    {
        this->setSize(parent->size());
    }
    else
    {
        this->setSize(size);
    }

    glm::vec3 color = { 0.5, 0.5, 0.5 };
    this->view.setOption<VFRendering::View::Option::BACKGROUND_COLOR>(color);

    this->geometry   = VFRendering::Geometry(  state_info.positions );
    this->directions = std::vector<glm::vec3>( state_info.nos, {0,0,1} );
    this->vf         = new VFRendering::VectorField(this->geometry, this->directions);

    this->arrow_renderer = std::shared_ptr<VFRendering::ArrowRenderer>(new VFRendering::ArrowRenderer(this->view, *this->vf));
    this->updateRenderers();

    // Set View options
    VFRendering::Options options;
    options.set<VFRendering::View::Option::SYSTEM_CENTER>({state_info.center[0], state_info.center[1], state_info.center[2]});
    options.set<VFRendering::View::Option::COLORMAP_IMPLEMENTATION>(VFRendering::Utilities::getColormapImplementation(VFRendering::Utilities::Colormap::HSV));
    options.set<VFRendering::View::Option::CAMERA_POSITION>({state_info.center[0], state_info.center[1], 30});
    options.set<VFRendering::View::Option::CENTER_POSITION>({state_info.center[0], state_info.center[1], state_info.center[2]});
    options.set<VFRendering::View::Option::UP_VECTOR>({0, 1, 0});
    view.updateOptions(options);
    this->updateData();
}

void SpinWidget::updateRenderers()
{
    this->renderers = {};
    this->renderers.push_back(this->arrow_renderer);
    auto renderers_system = std::shared_ptr<VFRendering::CombinedRenderer>(new VFRendering::CombinedRenderer( this->view, this->renderers ));
    this->view.renderers({{renderers_system, {0.0, 0.0, 1.0, 1.0}}}, false);
}

void SpinWidget::setSize(const Eigen::Vector2i & size)
{
    nanogui::GLCanvas::setSize(size);
    auto size_x = this->size()[0]*this->screen()->pixelRatio();
    auto size_y = this->size()[1]*this->screen()->pixelRatio();
    view.setFramebufferSize(size_x, size_y);
}

void SpinWidget::updateData()
{
    double* dirs = System_Get_Spin_Directions(state);
    this->directions.resize(state_info.nos);
    for(int i=0; i<state_info.nos; i++)
    {
        this->directions[i][0] = dirs[3*i    ];
        this->directions[i][1] = dirs[3*i + 1];
        this->directions[i][2] = dirs[3*i + 2];

    }
    this->vf->updateVectors(this->directions);
}

void SpinWidget::drawGL()
{
    using namespace nanogui;
    this->updateData();
    glEnable(GL_DEPTH_TEST);
    this->view.draw();
    glDisable(GL_DEPTH_TEST);
}