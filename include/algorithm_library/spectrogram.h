#pragma once
#include "interface/interface.h"

// spectrogram
//
// author: Kristian Timm Andersen

struct SpectrogramConfiguration
{
    using Input = I::Real;
    using Output = O::Real;

    struct Coefficients
    {
        int bufferSize = 1024; // input buffer size
        int nBands = 1025;     // number of frequency bands in output
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        int nonlinearity = 0;  // nonlinearity factor where left/right side of window is reduced by a factor of 2^nonlinearity, 0 = no nonlinearity
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nFolds, nonlinearity)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getValidFFTSize(int fftSize); // return valid FFT size larger or equal to fftSize

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time samples

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(c.nBands); } // power spectrogram

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && output.allFinite() && (output >= 0).all(); }
};

class Spectrogram : public Algorithm<SpectrogramConfiguration>
{
  public:
    Spectrogram() = default;
    Spectrogram(const Coefficients &c);
};
