#pragma once
#include "interface/interface.h"

// Spectral compressor
//
// author: Kristian Timm Andersen

struct SpectralCompressorConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        float sampleRate = 48000.f;
        int bufferSize = 1024;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, sampleRate, bufferSize)
    };

    struct Parameters
    {
        float thresholdDB = -12.f;
        int ratio = 10;
        float upTConstant = 0.01f;
        float downTConstant = 0.01f;
        enum ThresholdMode { ABSOLUTE_VALUE, ENERGY_VALUE };
        ThresholdMode thresholdMode = ENERGY_VALUE;
        DEFINE_TUNABLE_ENUM(ThresholdMode, {{ABSOLUTE_VALUE, "Absolute mode"}, {ENERGY_VALUE, "Energy mode"}})
        DEFINE_TUNABLE_PARAMETERS(thresholdDB, ratio, upTConstant, downTConstant, thresholdMode)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.bufferSize, c.nChannels); }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.bufferSize, c.nChannels); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && (output.cols() == c.nChannels) && output.allFinite(); }
};

// Compress audio dynamics in separate frequency bands using spectral gains
// with configurable threshold, ratio, and gain smoothing.
class SpectralCompressor : public Algorithm<SpectralCompressorConfiguration>
{
  public:
    SpectralCompressor() = default;
    SpectralCompressor(const Coefficients &c);

    float getDelaySamples() const;
};
