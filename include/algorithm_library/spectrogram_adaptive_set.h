#pragma once
#include "algorithm_library/fft.h"
#include "interface/interface.h"

// A set of adaptive spectrograms supporting 1 channel input
//
// The spectrograms are processed in parallel with FFT size halving between each spectrogram and output power spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen

struct SpectrogramAdaptiveSetConfiguration
{
    using Input = I::Real;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int bufferSize = 1024;        // buffer size in the input (first filterbank)
        int nBands = 100;             // number of perceptual frequency bands in output
        int nSpectrograms = 3;        // each spectrogram doubles the number of outputs (at half the bufferSize)
        float sampleRate = 48000.0f;  // sample rate
        bool spectralTilt = true;     // apply 3dB / octave spectral weight
        float frequencyMin = 20.f;    // minimum frequency (Hz)
        float frequencyMax = 20000.f; // maximum frequency (Hz)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nSpectrograms, sampleRate, spectralTilt, frequencyMin, frequencyMax)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time domain signal

    static std::vector<Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXf> output(c.nSpectrograms);
        for (auto i = 0; i < c.nSpectrograms; i++)
        {
            int nFrames = 1 << i;
            output[i] = Eigen::ArrayXXf::Constant(c.nBands, nFrames, -200);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != c.nSpectrograms) { return false; }
        for (auto i = 0; i < c.nSpectrograms; i++)
        {
            int nFrames = 1 << i;
            if ((output[i].rows() != c.nBands) || (output[i].cols() != nFrames) || (!output[i].allFinite())) { return false; }
        }
        return true;
    }
};

class SpectrogramAdaptiveSet : public Algorithm<SpectrogramAdaptiveSetConfiguration>
{
  public:
    SpectrogramAdaptiveSet() = default;
    SpectrogramAdaptiveSet(const Coefficients &c);
};
