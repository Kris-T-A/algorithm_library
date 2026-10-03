#pragma once
#include "interface/interface.h"

struct MultiChannelNoiseReductionConfiguration
{
    struct Input
    {
        I::Complex2D xFreq;
        I::Boolean signalOfInterestFlag;
    };

    using Output = O::Complex;

    struct Coefficients
    {
        int nChannels = 4;
        float filterbankRate = 125.f;
        int nBands = 257;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, filterbankRate, nBands)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXXcf, bool> initInput(const Coefficients &c)
    {
        return std::make_tuple(Eigen::ArrayXXcf::Random(c.nBands, c.nChannels), true);
    } // complex frequency spectrum and activity flag

    static Eigen::ArrayXcf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXcf::Zero(c.nBands); } // complex frequency spectrum

    static bool validInput(Input input, const Coefficients &c) { return (input.xFreq.rows() == c.nBands) && (input.xFreq.cols() == c.nChannels) && input.xFreq.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && output.allFinite(); }
};

// Beamform multichannel spectra and apply noise-reduction gains derived from
// the beamformed signal and noise-reference powers.
class MultiChannelNoiseReduction : public Algorithm<MultiChannelNoiseReductionConfiguration>
{
  public:
    MultiChannelNoiseReduction() = default;
    MultiChannelNoiseReduction(const Coefficients &c);
};