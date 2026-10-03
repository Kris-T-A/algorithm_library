#pragma once
#include "interface/interface.h"

struct PreprocessingPathCoefficientsParameters
{
    struct Coefficients
    {
        int bufferSize = 128;
        float sampleRate = 16000.f;
        int nChannels = 4;
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, sampleRate, nChannels);
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getNChannelsOut(const Coefficients &c) { return 1; }
};

using PreprocessingPathConfiguration = ConfigurationBuffer<PreprocessingPathCoefficientsParameters>;

// Remove DC from multichannel audio, detect signal activity, and beamform
// in the frequency domain before synthesizing a single output channel.
class PreprocessingPath : public AlgorithmBuffer<PreprocessingPathConfiguration>
{
  public:
    PreprocessingPath() = default;
    PreprocessingPath(const Coefficients &c);
};
