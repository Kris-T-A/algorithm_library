#pragma once
#include "algorithm_library/fft.h"
#include "interface/interface.h"

// An adaptive spectrogram supporting 1 channel input
//
// author: Kristian Timm Andersen

struct SpectrogramAdaptiveConfiguration
{
    using Input = I::Real;
    using Output = O::Real2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size
        int nBands = 2049;     // number of frequency bands in the first filterbank
        int nSpectrograms = 4; // each spectrogram halves the buffer size, so output contains 2^(nSpectrograms-1) frames
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        int nonlinearity = 0;  // nonlinearity factor where left/right side of window is reduced by a factor of 2^nonlinearity, 0 = no nonlinearity
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nSpectrograms, nFolds, nonlinearity)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time domain signal

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nBands, 1 << (c.nSpectrograms - 1)); } // output power spectrogram

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return ((output.rows() == c.nBands) && (output.cols() == (1 << (c.nSpectrograms - 1))) && output.allFinite() && (output >= 0).all());
    }
};

// Analysis filterbank
class SpectrogramAdaptive : public Algorithm<SpectrogramAdaptiveConfiguration>
{
  public:
    SpectrogramAdaptive() = default;
    SpectrogramAdaptive(const Coefficients &c);
};
