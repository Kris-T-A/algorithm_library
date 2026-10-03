#pragma once
#include "interface/interface.h"

struct NoiseReductionConfiguration
{
    using Input = I::Complex2D;
    using Output = O::Complex2D;

    struct Coefficients
    {
        int nBands = 257;
        int nChannels = 2;
        float filterbankRate = 125.f;
        enum NoiseReductionType { APRIORI, ML };
        NoiseReductionType algorithmType = APRIORI; // choose algorithm to use for calculating Noise reduction
        DEFINE_TUNABLE_ENUM(NoiseReductionType, {{APRIORI, "Apriori SNR"}, {ML, "Machine Learning"}})
        DEFINE_TUNABLE_COEFFICIENTS(nBands, nChannels, filterbankRate, algorithmType);
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXcf initInput(const Coefficients &c) { return Eigen::ArrayXXcf::Random(c.nBands, c.nChannels); } // complex input spectrum

    static Eigen::ArrayXXcf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXcf::Zero(c.nBands, c.nChannels); // complex output spectrum with noise reduction
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && (output.cols() == c.nChannels) && output.allFinite(); }
};

// Attenuate noise in complex input spectra using either activity-based noise
// estimation and a priori gains or the configured neural-network model.
class NoiseReduction : public Algorithm<NoiseReductionConfiguration>
{
  public:
    NoiseReduction() = default;
    NoiseReduction(const Coefficients &c);
};
