#include "AdvancedGraph.hpp"
#include <nanogui/theme.h>
#include <nanogui/opengl.h>
#include <nanovg.h>
#include <nanogui/serializer/core.h>
#include <sstream>

AdvancedGraph::AdvancedGraph(Widget * parent, const std::string & caption) : Graph(parent, caption)
{};

void AdvancedGraph::setValues(const VectorXf & x_values, const VectorXf & y_values, const std::vector<Marker> & markers, const std::vector<Color> marker_colors, const std::vector<double> & marker_scales)
{
    if(x_values.size() != y_values.size())
    {
        throw "Value vectors x and y need to have the same length!";
    }
    if(x_values.size() != markers.size())
    {
        throw "Value vectors and marker vector need to have the same length!";
    }
    if(x_values.size() != marker_colors.size())
    {
        throw "Value vectors and marker_colors vector need to have the same length!";
    }
     if(x_values.size() != marker_scales.size())
    {
        throw "Value vectors and marker_scales vector need to have the same length!";
    }

    this->mValuesX = x_values;
    this->mValues = y_values;
    this->markers = markers;
    this->marker_colors = marker_colors;
    this->marker_scale = marker_scales;
}

void AdvancedGraph::setValues(const VectorXf & x_values, const VectorXf & y_values, const Marker & marker, const Color & color, double marker_scales )
{
    this->setValues(x_values, y_values, std::vector<Marker>(x_values.size(), marker), std::vector<Color>(x_values.size(), color), std::vector<double>(x_values.size(), marker_scales) );
}

void AdvancedGraph::setMarker(const std::vector<Marker> & markers)
{
    if( this->mValuesX.size() == markers.size() )
        this->markers = markers;
}

void AdvancedGraph::setMarkerColor(const std::vector<Color> & colors)
{
    if( this->mValuesX.size() == colors.size() )
        this->marker_colors = colors;
}

void AdvancedGraph::data_to_pixel(const nanogui::Vector2f & data, nanogui::Vector2i & pixel)
{
    pixel[0] = mPos.x() + (data.x() - mx_min) * (mSize.x() - 2*margin_x) / (mx_max - mx_min) + margin_x;
    pixel[1] = mPos.y() + (my_max - data.y()) * (mSize.y() - 2*margin_y) / (my_max - my_min) + margin_y;
}

void AdvancedGraph::pixel_to_data(const nanogui::Vector2i & pixel, nanogui::Vector2f & data)
{
    data[0] =  (pixel.x()-mPos.x()-margin_x) * (mx_max - mx_min) / (mSize.x() - 2*margin_x) + mx_min;
    data[1] = -(pixel.y()-mPos.y()-margin_y) * (my_max - my_min) / (mSize.y() - 2*margin_y) + my_max;
}

void AdvancedGraph::setXLabel(const std::string & label)
{
    this->x_label = label;
}
void AdvancedGraph::setYLabel(const std::string & label)
{
    this->y_label = label;
}


// Helper Function for equilateral triangle pointing down
void nvgTriangDown(NVGcontext* ctx, float x, float y, float l)
{
    float h = 0.8660254038; // sqrt(3)/2
    nvgMoveTo(ctx, x, y + h*l*2.f/3.f);     // lower corner
    nvgLineTo(ctx, x - l/2.f, y - h*l/3.f); // upper left corner
    nvgLineTo(ctx, x + l/2.f, y - h*l/3.f); // upper right corner
    nvgLineTo(ctx, x, y + h*l*2.f/3.f);     // back to lower corner
}

// Helper Function for equilateral triangle pointing up
void nvgTriangUp(NVGcontext* ctx, float x, float y, float l)
{
    float h = 0.8660254038; // sqrt(3)/2
    nvgMoveTo(ctx, x, y - h*l*2.f/3.f);     // upper corner
    nvgLineTo(ctx, x - l/2.f, y + h*l/3.f); // lower left corner
    nvgLineTo(ctx, x + l/2.f, y + h*l/3.f); // lower right corner
    nvgLineTo(ctx, x, y - h*l*2.f/3.f);     // back to upper corner
}

void AdvancedGraph::draw_markers(NVGcontext * ctx)
{
    nanogui::Vector2i cur_pos = {0,0};

    for (size_t i = 0; i < (size_t) mValues.size(); i++) {

        data_to_pixel({mValuesX[i], mValues[i]}, cur_pos);

        if(this->markers[i] != Marker::NONE)
        {
            nvgBeginPath(ctx);
            nvgMoveTo(ctx, cur_pos.x(), cur_pos.y());
            nvgFillColor(ctx, marker_colors[i]);
        }
        if(this->markers[i] == Marker::CIRCLE)
        {
            nvgCircle(ctx, cur_pos.x(), cur_pos.y(), 5 * marker_scale[i]);
        } else if (this->markers[i] == Marker::SQUARE)
        {
            nvgRect(ctx, cur_pos.x()-3.75*marker_scale[i], cur_pos.y()-3.75*marker_scale[i], 7.5 * marker_scale[i], 7.5 * marker_scale[i]);
        } else if (this->markers[i] == Marker::TRIANG_UP)
        {
            nvgTriangUp(ctx, cur_pos.x(), cur_pos.y(), 12*marker_scale[i]);
        } else if (this->markers[i] == Marker::TRIANG_DOWN)
        {
            nvgTriangDown(ctx, cur_pos.x(), cur_pos.y(), 12*marker_scale[i]);
        }
        if(this->markers[i] != Marker::NONE)
        {
            nvgFill(ctx);
            nvgFillColor(ctx, mForegroundColor);
        }
    }
}

void AdvancedGraph::draw_ticks(NVGcontext * ctx)
{
    nanogui::Vector2i line_begin = {0,0};
    nanogui::Vector2f data_pt = {0,0};
    std::ostringstream streamObj;
    nvgBeginPath(ctx);

    // x ticks
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    for( int i = 0; i < n_ticks_x; i++ )
    {
        data_pt = {mx_min + (mx_max - mx_min) * i/n_ticks_x, my_min};
        data_to_pixel(data_pt, line_begin);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_begin.x(), line_begin.y() - tick_length);

        std::ostringstream().swap(streamObj);
        streamObj << data_pt.x();
        nvgText(ctx, line_begin.x(), line_begin.y(), streamObj.str().c_str(), NULL);
    }

    // y ticks
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_CENTER);
    for( int i = 0; i < n_ticks_y; i++ )
    {
        data_pt = {mx_min, my_min + (my_max - my_min) * i/n_ticks_y};
        data_to_pixel(data_pt, line_begin);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_begin.x() + tick_length, line_begin.y());
        std::ostringstream().swap(streamObj);
        streamObj << data_pt.y();

        // TODO: kind of an ugly hack so that the numbers dont overlap the y-axis
        nvgText(ctx, (mPos.x() + line_begin.x())/2, line_begin.y(), streamObj.str().c_str(), NULL);
    }

    nvgFontSize(ctx, 14.0f);
    nvgText(ctx, mPos.x() + 3, mPos.y() + 1, mCaption.c_str(), NULL);

    nvgStrokeColor(ctx, Color(0,0,0,255));
    nvgStrokeWidth(ctx, 2.0);
    nvgStroke(ctx);
}

void AdvancedGraph::draw_labels(NVGcontext * ctx)
{
    nvgBeginPath(ctx);
    nvgStrokeColor(ctx, Color(0,0,0,255));
    nvgTextAlign(ctx, NVG_ALIGN_CENTER | NVG_ALIGN_TOP);
    nvgText(ctx, mPos.x() + mSize.x()/2, mPos.y() + mSize.y() - margin_y/2, x_label.c_str(), NULL);
    nvgStrokeWidth(ctx, 1.0);
    nvgStroke(ctx);
}

void AdvancedGraph::draw_box(NVGcontext * ctx)
{
    nvgBeginPath(ctx);
    nvgRect(ctx, mPos.x() + margin_x, mPos.y() + margin_y, mSize.x() - 2*margin_x, mSize.y() - 2*margin_y);
    nvgStrokeColor(ctx, Color(100, 255));
    nvgStrokeWidth(ctx, 2.0);
    nvgStroke(ctx);
}

void AdvancedGraph::draw_grid(NVGcontext * ctx)
{
    nanogui::Vector2i line_begin = {0,0};
    nanogui::Vector2i line_end   = {0,0};
    nvgBeginPath(ctx);

    // Vertical lines
    for( int i = 0; i < n_ticks_x; i++ )
    {   
        data_to_pixel({mx_min + (mx_max - mx_min) * i/n_ticks_x, my_min}, line_begin);
        data_to_pixel({mx_min + (mx_max - mx_min) * i/n_ticks_x, my_max}, line_end);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_end.x(), line_end.y());
    }

    // Horizontal lines
    for( int i = 0; i < n_ticks_y; i++ )
    {
        data_to_pixel({mx_min, my_min + (my_max - my_min) * i/n_ticks_y}, line_begin);
        data_to_pixel({mx_max, my_min + (my_max - my_min) * i/n_ticks_y}, line_end);
        nvgMoveTo(ctx, line_begin.x(), line_begin.y());
        nvgLineTo(ctx, line_end.x(), line_end.y());
    }
    nvgStrokeColor(ctx, Color(60,60,60,100));
    nvgStrokeWidth(ctx, 1.0);
    nvgStroke(ctx);
}

void AdvancedGraph::draw_line_segments(NVGcontext * ctx)
{
    nanogui::Vector2i cur_pos = {0,0};
    nvgBeginPath(ctx);
    data_to_pixel({mValuesX[0], mValues[0]}, cur_pos);
    nvgMoveTo(ctx, cur_pos.x(), cur_pos.y()); // Begin path at first data point

    for (size_t i = 0; i < (size_t) mValues.size(); i++) {
        data_to_pixel({mValuesX[i], mValues[i]}, cur_pos);    
        nvgLineTo(ctx, cur_pos.x(), cur_pos.y());
    }

    nvgStrokeColor(ctx, Color(100, 255));
    nvgStrokeWidth(ctx, 2.0);
    nvgStroke(ctx);
    nvgFillColor(ctx, mForegroundColor);
}

void AdvancedGraph::draw(NVGcontext *ctx){
    // TODO: should probably throw some kind of error here
    if( mValues.size() != mValuesX.size() )
        return;
    if( mx_min >= mx_max || my_min >= my_max )
        return;
    Widget::draw(ctx);

    nvgBeginPath(ctx);
    nvgRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y());
    nvgFillColor(ctx, mBackgroundColor);
    nvgFill(ctx);

    if (mValues.size() < 2)
        return;

    draw_box(ctx);
    draw_grid(ctx);
    draw_line_segments(ctx);
    draw_markers(ctx);
    draw_ticks(ctx);
    draw_labels(ctx);

    nvgFontFace(ctx, "sans");

    if (!mCaption.empty()) {
        nvgFontSize(ctx, 14.0f);
        nvgTextAlign(ctx, NVG_ALIGN_LEFT | NVG_ALIGN_TOP);
        nvgFillColor(ctx, mTextColor);
        nvgText(ctx, mPos.x() + 3, mPos.y() + 1, mCaption.c_str(), NULL);
    }

    if (!mHeader.empty()) {
        nvgFontSize(ctx, 18.0f);
        nvgTextAlign(ctx, NVG_ALIGN_RIGHT | NVG_ALIGN_TOP);
        nvgFillColor(ctx, mTextColor);
        nvgText(ctx, mPos.x() + mSize.x() - 3, mPos.y() + 1, mHeader.c_str(), NULL);
    }

    if (!mFooter.empty()) {
        nvgFontSize(ctx, 15.0f);
        nvgTextAlign(ctx, NVG_ALIGN_RIGHT | NVG_ALIGN_BOTTOM);
        nvgFillColor(ctx, mTextColor);
        nvgText(ctx, mPos.x() + mSize.x() - 3, mPos.y() + mSize.y() - 1, mFooter.c_str(), NULL);
    }
    nvgBeginPath(ctx);
    nvgRect(ctx, mPos.x(), mPos.y(), mSize.x(), mSize.y());
    nvgStrokeColor(ctx, Color(100, 255));
    nvgStroke(ctx);
}