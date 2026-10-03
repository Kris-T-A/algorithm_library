#pragma once
#include "interface/interface.h"

// Calculate power spectrum of a cascade of 2nd order filters.
//
// author: Kristian Timm Andersen

struct FilterPowerSpectrumConfiguration
{
    // input sos filter is structured as:
    // [b0 bb0 ... ]
    // [b1 bb1 ... ]
    // [b2 bb2 ... ]
    // [1  1   ... ]
    // [a1 aa1 ... ]
    // [a2 aa2 ... ]
    using Input = I::Real6X;
    using Output = O::Real;

    struct Coefficients
    {
        int nBands = 257; // number of frequency points from 0 to pi in output
        DEFINE_TUNABLE_COEFFICIENTS(nBands)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    // initInput might not give a stable filter, i.e. poles might not be inside unit circle
    static Eigen::Array<float, 6, Eigen::Dynamic> initInput(const Coefficients &c)
    {
        // filter sos array
        Eigen::Array<float, 6, Eigen::Dynamic> input = .1f * Eigen::Array<float, 6, Eigen::Dynamic>::Random(6, 3); // Number of 2nd order filters is arbitrary
        input.row(3).setOnes();
        return input;
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(c.nBands); } // frequency bins

    static bool validInput(Input input, const Coefficients &c)
    {
        bool flag = (input.rows() == 6) && input.allFinite();
        flag &= (input.row(3) == 1.f).all();
        return flag;
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && (output >= 0).all(); }
};

// Evaluate the power response of a cascade of second-order IIR sections
// on the configured frequency grid.
class FilterPowerSpectrum : public Algorithm<FilterPowerSpectrumConfiguration>
{
  public:
    FilterPowerSpectrum() = default;
    FilterPowerSpectrum(const Coefficients &c);
};
