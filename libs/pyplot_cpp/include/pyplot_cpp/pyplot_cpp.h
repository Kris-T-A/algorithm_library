#pragma once
#include "algorithm_library/interface/input_output.h"
#include <array>
#include <string>

namespace Pyplotcpp
{
struct Figure; // no definition of Figure, since we only use it to hold a pointer to PyObjects without leaking implementation details.

void plot(I::Real2D x);
void plot(I::Real2D x, I::Real2D y);

Figure *imagesc(I::Real2D x);
Figure *imagesc(I::Real2D x, std::array<float, 2> scaling);

void title(const std::string &titlename);
void xlim(float left, float right);
void ylim(float down, float up);
void xlabel(const std::string &label);
void ylabel(const std::string &label);
void colorbar(Figure *figure);

void save(const std::string &filename, const int dpi = 0);
void show(const bool block = true);
void clear();
void figure(long number = -1);
void close();
} // namespace Pyplotcpp