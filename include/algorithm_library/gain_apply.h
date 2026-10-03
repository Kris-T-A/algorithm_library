#pragma once
#include "interface/interface.h"

// Apply gain to signal.
//
// author: Kristian Timm Andersen

struct GainApplyConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        float sampleRate = 16000.f;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, sampleRate)
    };

    struct Parameters
    {
        float gain = 1.f;               // gain applied to every channel (can also be negative)
        float smoothingTConstant = 0.f; // gain smoothing time constant in seconds
        DEFINE_TUNABLE_PARAMETERS(gain, smoothingTConstant)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(100, c.nChannels); } // time samples. Number of samples can be arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXf::Zero(input.rows(), c.nChannels);
    } // time samples. Number of samples can be arbitrary

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() > 0) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() > 0) && (output.cols() == c.nChannels) && output.allFinite(); }
};

// Apply a global gain and per-channel gains to audio, smoothing gain changes
// with the configured time constant.
class GainApply : public Algorithm<GainApplyConfiguration>
{
  public:
    GainApply() = default;
    GainApply(const Coefficients &c);

    void setChannelGain(int channel, float gain); // apply gain to individual channel
    float getChannelGain(int channel) const;      // get gain applied to individual channel (without smoothing or overall gain)
};
