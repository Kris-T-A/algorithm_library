#pragma once
#include "interface/interface.h"

// Spline interpolates the set of data points xGiven/yGiven evaluted at xDesired.
// The inputs must be ordered in increasing x order or the algorithm will give nonsense output.
//
// author: Kristian Timm Andersen

struct SplineConfiguration
{
    struct Input
    {
        I::Real2D xGiven;
        I::Real2D yGiven;
        I::Real2D xDesired;
    };

    using Output = O::Real2D;

    struct Coefficients
    {
        int nGiven = 10;
        DEFINE_TUNABLE_COEFFICIENTS(nGiven)
    };

    struct Parameters
    {
        float sFactor = 0.f;
        DEFINE_TUNABLE_PARAMETERS(sFactor)
    };

    static std::tuple<Eigen::ArrayXXf, Eigen::ArrayXXf, Eigen::ArrayXXf> initInput(const Coefficients &c)
    {
        Eigen::ArrayXXf xGiven = Eigen::ArrayXf::LinSpaced(c.nGiven, 0, static_cast<float>(c.nGiven - 1)).replicate(1, 2); // number of channels is arbitrary
        Eigen::ArrayXXf yGiven = Eigen::ArrayXXf::Random(c.nGiven, xGiven.cols());
        Eigen::ArrayXXf xDesired = Eigen::ArrayXf::LinSpaced(100, 0, static_cast<float>(c.nGiven - 1)).replicate(1, xGiven.cols()); // number of desired outputs is arbitrary
        return std::make_tuple(xGiven, yGiven, xDesired);
    }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXf::Zero(input.xDesired.rows(), input.xDesired.cols());
    } // size of desired y values is determined by xDesired

    static bool validInput(Input input, const Coefficients &c)
    {
        // check that values in xGiven and xDesired are ordered in increasing order
        for (auto channel = 0; channel < input.xGiven.cols(); channel++)
        {
            for (auto i = 1; i < input.xGiven.rows(); i++)
            {
                if (input.xGiven(i, channel) < input.xGiven(i - 1, channel)) { return false; } // if not ordered in increasing order
            }
            for (auto i = 1; i < input.xDesired.rows(); i++)
            {
                if (input.xDesired(i, channel) < input.xDesired(i - 1, channel)) { return false; } // if not ordered in increasing order
            }
        }
        bool flag = (input.xGiven.rows() == c.nGiven) && input.xGiven.allFinite();
        flag &= (input.yGiven.rows() == c.nGiven) && (input.yGiven.cols() == input.xGiven.cols()) && (input.yGiven.allFinite());
        flag &= (input.xDesired.cols() == input.xGiven.cols()) && input.xDesired.allFinite();
        return flag;
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() > 0) && (output.cols() > 0) && output.allFinite(); }
};

// Interpolate ordered x/y control points with cubic splines evaluated
// at the requested x positions.
class Spline : public Algorithm<SplineConfiguration>
{
  public:
    Spline() = default;
    Spline(const Coefficients &c);
};