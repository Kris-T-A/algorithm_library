#pragma once
#include "interface/interface.h"

// Perceptual Spectral Analysis
//
// author: Kristian Timm Andersen

struct PerceptualSpectralAnalysisConfiguration
{
    using Input = I::Real;
    using Output = O::Real2D;

    struct Coefficients
    {
        int bufferSize = 4096; // input buffer size
        int nBands = 100;      // number of perceptual frequency bands in output
        float sampleRate = 48000.0f;
        bool spectralTilt = true; // apply spectral tilt to output
        int nSpectrograms = 4;    // each spectrogram halves the buffer size, so output contains 2^(nSpectrograms-1) frames
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        int nonlinearity = 0;  // nonlinearity factor where left/right side of window is reduced by a factor of 2^nonlinearity, 0 = no nonlinearity
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, sampleRate, spectralTilt, nSpectrograms, nFolds, nonlinearity)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getValidBufferSize(int bufferSize); // return valid buffer size larger or equal to bufferSize

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time samples

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) // perceptual spectral analysis output
    {
        return Eigen::ArrayXXf::Zero(c.nBands, (1 << (c.nSpectrograms - 1)));
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.rows() == c.nBands) && (output.cols() == (1 << (c.nSpectrograms - 1))) && output.allFinite();
    }
};

class PerceptualSpectralAnalysis : public Algorithm<PerceptualSpectralAnalysisConfiguration>
{
  public:
    PerceptualSpectralAnalysis() = default;
    PerceptualSpectralAnalysis(const Coefficients &c);
};
