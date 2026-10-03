#pragma once
#include "interface/interface.h"

// IIR filter
//
// author: Kristian Timm Andersen

struct IIRFilterConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        int nSos = 3;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, nSos)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(100, c.nChannels); } // time samples. Number of samples can be arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(input.rows(), c.nChannels); } // time samples

    static bool validInput(Input input, const Coefficients &c) { return (input.cols() == c.nChannels) && (input.rows() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.cols() == c.nChannels) && (output.rows() > 0) && output.allFinite(); }
};

// Filter each input channel through a configured cascade of second-order IIR sections,
// retaining filter state between blocks.
class IIRFilter : public Algorithm<IIRFilterConfiguration>
{
  public:
    IIRFilter() = default;
    IIRFilter(const Coefficients &c);

    // sos is structured as:
    // [b0 bb0 ... ]
    // [b1 bb1 ... ]
    // [b2 bb2 ... ]
    // [1  1   ... ]
    // [a1 aa1 ... ]
    // [a2 aa2 ... ]
    void setFilter(I::Real2D sos, float gain);

    Eigen::ArrayXXf getSosFilter() const;
    float getGain() const;
};
