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
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, sampleRate, spectralTilt)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getValidBufferSize(int bufferSize); // return valid buffer size larger or equal to bufferSize

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time samples

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) // perceptual spectral analysis output
    {
        return Eigen::ArrayXXf::Zero(c.nBands, 8);
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && output.allFinite() && (output.cols() == 8); }
};

class PerceptualSpectralAnalysis : public Algorithm<PerceptualSpectralAnalysisConfiguration>
{
  public:
    PerceptualSpectralAnalysis() = default;
    PerceptualSpectralAnalysis(const Coefficients &c);
};
