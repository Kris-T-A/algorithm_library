#pragma once
#include "interface/interface.h"

struct SingleChannelPathConfiguration
{
    using Input = I::Real;
    using Output = O::Real;

    struct Coefficients
    {
        int bufferSize = 128;
        float sampleRate = 16000.f;
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, sampleRate);
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time samples

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(c.bufferSize); } // time samples

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && output.allFinite(); }
};

// Remove DC, analyze a single audio channel into spectral frames, apply
// noise-reduction gains, and synthesize the filtered time-domain signal.
class SingleChannelPath : public Algorithm<SingleChannelPathConfiguration>
{
  public:
    SingleChannelPath() = default;
    SingleChannelPath(const Coefficients &c);
};
