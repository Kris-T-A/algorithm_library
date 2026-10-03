#pragma once
#include "interface/interface.h"

// Calculate the 3-dimensional normal vector of a plane for each point in the plane.
// The input is a 2-dimensional matrix containing the z-value of each point in a x/y grid
// The output is a 2-dimensional matrix, where the 1st output dimension is 3 times the size of the 1st input dimension to hold the interleaved 3-dimensional normal vector.
//
// author: Kristian Timm Andersen

// --------------------------------------------- Filterbank Analysis ----------------------------------------------------------------

struct Normal3dConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nValuesX = 257;
        DEFINE_TUNABLE_COEFFICIENTS(nValuesX)
    };

    struct Parameters
    {
        float distance1 = 31.25f; // distance between values along 1st dimension
        float distance2 = 8.f;    // distance between values along 2nd dimension
        DEFINE_TUNABLE_PARAMETERS(distance1, distance2)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nValuesX, 10); } // number of y values is arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(3 * c.nValuesX, input.cols()); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nValuesX) && (input.cols() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == 3 * c.nValuesX) && (output.cols() > 0) && output.allFinite(); }
};

// Estimate normalized surface normals from a streamed height grid and store
// the three vector components interleaved along the output rows.
class Normal3d : public Algorithm<Normal3dConfiguration>
{
  public:
    Normal3d() = default;
    Normal3d(const Coefficients &c);
};
