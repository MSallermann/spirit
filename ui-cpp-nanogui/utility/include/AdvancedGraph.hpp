#include <nanogui/common.h>
#include <nanogui/graph.h>

enum Marker
{
    NONE        = 0, 
    CIRCLE      = 1,
    SQUARE      = 2,
    TRIANG_UP   = 3,
    TRIANG_DOWN = 4
};

class AdvancedGraph : public nanogui::Graph
{
    using VectorXf = nanogui::VectorXf;
    using Color    = nanogui::Color;

    protected:
    nanogui::VectorXf   mValuesX;
    std::vector<Marker> markers;
    std::vector<Color>  marker_colors;
    std::vector<double> marker_scale;

    std::string x_label;
    std::string y_label;

    int n_ticks_x = 10;
    int n_ticks_y = 10;

    int tick_length = 5;

    bool grid = false;
    double mx_min;
    double mx_max;
    double my_min;
    double my_max;

    int margin_x = 30;
    int margin_y = 30;

    virtual void draw_markers(NVGcontext * ctx);
    virtual void draw_ticks(NVGcontext * ctx);
    virtual void draw_labels(NVGcontext * ctx);
    virtual void draw_line_segments(NVGcontext * ctx);
    virtual void draw_grid(NVGcontext * ctx);
    virtual void draw_box(NVGcontext * ctx);


    // Convert data point to pixels on the canvas
    virtual void data_to_pixel(const nanogui::Vector2f & data, nanogui::Vector2i & pixel);

    // Convert pixels on canvas to data point
    virtual void pixel_to_data(const nanogui::Vector2i & pixel, nanogui::Vector2f & data);

    public:
    AdvancedGraph(Widget * parent, const std::string & caption = "Untitled");

    // Sets x and y values
    // virtual void setValues(const VectorXf & values) = delete;
    virtual void setValues(const VectorXf & x_values, const VectorXf & y_values, const Marker & marker = Marker::NONE, const Color & color = Color(0,0,0,0), double marker_scales = 1.0);
    virtual void setValues(const VectorXf & x_values, const VectorXf & y_values, const std::vector<Marker> & markers, const std::vector<Color> marker_colors, const std::vector<double> & marker_scales);

    const VectorXf &valuesX() const { return mValuesX; }
    VectorXf &valuesX() { return mValuesX; }

    // Set x value
    void setXMin(double x_min) {mx_min = x_min;};
    // Set maximal x value
    void setXMax(double x_max) {mx_max = x_max;};

    // Set minimal y value
    void setYMin(double y_min) {my_min = y_min;};
    // Set maximal y value
    void setYMax(double y_max) {my_max = y_max;};

    // Set the color of the marker at idx
    void setMarkerColor(int idx, const Color & color);
    // Set the color of all markers
    void setMarkerColor(const Color & color);
    // Set the colors of the markers to colors
    void setMarkerColor(const std::vector<Color> & colors);

    // Set the marker at idx
    void setMarker(int idx, Marker marker);
    // Set all markers to marker
    void setMarker(Marker marker);
    // Set markers according to the vector markers
    void setMarker(const std::vector<Marker> & markers);

    // Set the marker scale at idx
    void setMarkerScale(int idx, double scale);
    // Set the marker scale for all markers
    void setMarkerScale(double scale);
    // Set the marker scales according to vector markers
    void setMarkerScale(const std::vector<Marker> & scales);

    void setXLabel(const std::string & label);
    void setYLabel(const std::string & label);


    virtual void draw(NVGcontext *ctx) override;


};