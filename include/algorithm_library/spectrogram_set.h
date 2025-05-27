#pragma once
#include "algorithm_library/fft.h"
#include "interface/interface.h"

// A set of spectrograms supporting 1 channel input
//
// The spectrograms are processed in parallel with FFT size halving between each spectrogram and output power spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen

struct SpectrogramSetConfiguration
{
    using Input = I::Real;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size in the first filterbank
        int nBands = 2049;     // number of frequency bands in the first filterbank
        int nSpectrograms = 4; // each spectrogram halves the buffer size
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        int nonlinearity = 0;  // nonlinearity factor where left/right side of window is reduced by a factor of 2^nonlinearity, 0 = no nonlinearity
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nSpectrograms, nFolds, nonlinearity)
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
            int nBands = (c.nBands - 1) / nFrames + 1;
            output[i] = Eigen::ArrayXXf::Zero(nBands, nFrames);
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
            int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands) / nFrames;
            if (!FFTConfiguration::isFFTSizeValid(fftSize)) { return false; }
            int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
            if ((output[i].rows() != nBands) || (output[i].cols() != nFrames) || (!output[i].allFinite())) { return false; }
        }
        return true;
    }
};

// Analysis filterbank
class SpectrogramSet : public Algorithm<SpectrogramSetConfiguration>
{
  public:
    SpectrogramSet() = default;
    SpectrogramSet(const Coefficients &c);
};
