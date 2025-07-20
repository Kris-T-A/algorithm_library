#include "matplotlibcpp.h"
#include "pyplot_cpp/pyplot_cpp.h"
#include <iostream>

// Due to a bug in matplotlibcpp, we need to call this function to ensure the interpreter is killed
// before the program exits: 
// https://stackoverflow.com/questions/67533541/py-finalize-resulting-in-segmentation-fault-for-python-3-9-but-not-for-python/67577360#67577360
void killit()
{
    matplotlibcpp::detail::_interpreter::kill();
}

namespace Pyplotcpp
{
void plot(I::Real2D x)
{
    const auto nCols = x.cols();
    const auto nRows = x.rows();
    for (auto col = 0; col < nCols; col++)
    {
        const float *xPtr = x.col(col).data();
        std::vector<float> xVec(xPtr, xPtr + nRows);
        matplotlibcpp::plot(xVec);
        
    }
    std::atexit(killit);
}

void plot(I::Real2D x, I::Real2D y)
{
    const auto nCols = std::min(x.cols(), y.cols());
    const auto nRows = std::min(x.rows(), y.rows());
    for (auto col = 0; col < nCols; col++)
    {
        const float *xPtr = x.col(col).data();
        const float *yPtr = y.col(col).data();
        std::vector<float> xVec(xPtr, xPtr + nRows);
        std::vector<float> yVec(yPtr, yPtr + nRows);
        matplotlibcpp::plot(xVec, yVec);
    }
    std::atexit(killit);
}

// local function
inline Figure *imshow(I::Real2D x, const std::map<std::string, std::string> &options = {})
{
    const int nCols = static_cast<int>(x.cols());
    const int nRows = static_cast<int>(x.rows());
    const int colors = 1;
    Eigen::ArrayXXf y = x.transpose(); // matplotlib assumes data is transposed
    PyObject *figure;
    matplotlibcpp::imshow(y.data(), nRows, nCols, colors, options, &figure);
    std::atexit(killit);
    return reinterpret_cast<Figure *>(figure);
}

Figure *imagesc(I::Real2D x) 
{ 
    Figure *fig = imshow(x);
    std::atexit(killit);
    return fig; 
}

Figure *imagesc(I::Real2D x, std::array<float, 2> scaling)
{
    const std::map<std::string, std::string> sc{{"vmin", std::to_string(scaling[0])}, {"vmax", std::to_string(scaling[1])}};
    Figure *fig = imshow(x, sc);
    std::atexit(killit);
    return fig;
}

void title(const std::string &titlename) { matplotlibcpp::title(titlename); }

void xlim(float left, float right) { matplotlibcpp::xlim(left, right); }

void ylim(float down, float up) { matplotlibcpp::ylim(down, up); }

void xlabel(const std::string &label) { matplotlibcpp::xlabel(label); }

void ylabel(const std::string &label) { matplotlibcpp::ylabel(label); }

void colorbar(Figure *mat) { matplotlibcpp::colorbar(reinterpret_cast<PyObject *>(mat)); }

void save(const std::string &filename, const int dpi) { matplotlibcpp::save(filename, dpi); }

void show(const bool block) { matplotlibcpp::show(block); std::atexit(killit); }

void clear() { matplotlibcpp::clf(); }

void figure(long number) { matplotlibcpp::figure(number); }

void close() { matplotlibcpp::close(); std::cout << "Closed figure." << std::endl; }
} // namespace Pyplotcpp