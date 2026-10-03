#pragma once
#include "interface/interface.h"

// Gain Calculation given the aposteriori SNR as input: P_{s+n}/Phat_{n}
//
// author: Kristian Timm Andersen

struct GainCalculationConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        float filterbankRate = 125.f;
        int nBands = 257;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, filterbankRate, nBands)
    };

    struct Parameters
    {
        float minimumGaindB = -12.f;
        float upTConstant = 0.01f;
        float downTConstant = 0.01f;
        DEFINE_TUNABLE_PARAMETERS(minimumGaindB, upTConstant, downTConstant)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // a posteriori SNR

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Ones(c.nBands, c.nChannels); } // linear gain

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input.cols() == c.nChannels) && (input >= 0).all(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && (output.cols() == c.nChannels) && (output >= 0).all(); }
};

// Estimate the a priori signal-to-noise ratio from the supplied a posteriori ratios
// and convert it into smoothed noise-reduction gains.
class GainCalculation : public Algorithm<GainCalculationConfiguration>
{
  public:
    GainCalculation() = default;
    GainCalculation(const Coefficients &c);
};
