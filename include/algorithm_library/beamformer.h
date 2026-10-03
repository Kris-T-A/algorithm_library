#pragma once
#include "interface/interface.h"

struct BeamformerConfiguration
{
    struct Input
    {
        I::Complex2D xFreq;
        I::Boolean signalOfInterestFlag;
    };

    struct Output
    {
        O::Complex yFreq;
        O::Complex noiseFreq;
    };

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

    static std::tuple<Eigen::ArrayXcf, Eigen::ArrayXcf> initOutput(Input input, const Coefficients &c)
    {
        return {Eigen::ArrayXcf::Zero(c.nBands), Eigen::ArrayXcf::Zero(c.nBands)};
    } // complex frequency spectrum

    static bool validInput(Input input, const Coefficients &c) { return (input.xFreq.rows() == c.nBands) && (input.xFreq.cols() == c.nChannels) && input.xFreq.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.yFreq.rows() == c.nBands) && output.yFreq.allFinite() && (output.noiseFreq.rows() == c.nBands) && output.noiseFreq.allFinite();
    }
};

// Combine multichannel complex spectra using MVDR beamforming, producing
// a signal-of-interest spectrum and a noise-reference spectrum.
class Beamformer : public Algorithm<BeamformerConfiguration>
{
  public:
    Beamformer() = default;
    Beamformer(const Coefficients &c);
};