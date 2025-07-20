#pragma once
#include <pyplot_cpp/pyplot_cpp.h>

void saveSpectrogram(const Eigen::ArrayXXf& spec, const std::string& outputName)
{
    float maxValue = spec.maxCoeff();
    Pyplotcpp::clear();
    auto figure = Pyplotcpp::imagesc(spec, {maxValue - 80, maxValue});
    Pyplotcpp::xlabel("Frame Number");
    Pyplotcpp::ylabel("Frequency bin");
    Pyplotcpp::colorbar(figure);
    Pyplotcpp::save(outputName, 1000);
}