#pragma once
#include "interface/interface.h"

// Perceptual Spectral Attenuate
//
// Attenuate audio using linear amplitude gains on the logarithmic frequency scale of
// PerceptualSpectralAnalysis. Gains are held between adjacent band midpoints;
// the first and last bands extend to DC and Nyquist respectively.
// Gain column j describes signal interval [j * H, (j + 1) * H), where
// H = bufferSize / 2^(nSpectrograms - 1). Its analysis window is centred at j * H;
// overlapping windows smooth gain changes across neighbouring intervals.
// Total audio delay: 3 * bufferSize + 3 * bufferSize / 2^(nSpectrograms - 1) samples.
//
// author: Kristian Timm Andersen

struct PerceptualSpectralAttenuateConfiguration
{
    struct Input
    {
        I::Real signal; // time signal
        I::Real2D gain; // linear amplitude gain in [0, 1]: 0 is silence, 1 is unity
    };

    using Output = O::Real;

    struct Coefficients
    {
        int bufferSize = 4096; // input buffer size
        int nBands = 100;      // number of perceptual frequency bands in gain
        float sampleRate = 48000.0f;
        float frequencyMin = 20.f;    // minimum frequency (Hz)
        float frequencyMax = 20000.f; // maximum frequency (Hz)
        int nSpectrograms = 3;        // each spectrogram halves the buffer size, so gain contains 2^(nSpectrograms-1) frames
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, sampleRate, frequencyMin, frequencyMax, nSpectrograms)
    };

    struct Parameters
    { DEFINE_NO_TUNABLE_PARAMETERS };

    static int getValidBufferSize(int bufferSize); // return valid buffer size >= bufferSize for the default nSpectrograms

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXXf> initInput(const Coefficients &c)
    {
        Eigen::ArrayXf signal = Eigen::ArrayXf::Random(c.bufferSize); // time samples
        Eigen::ArrayXXf gain = Eigen::ArrayXXf::Random(c.nBands, 1LL << (c.nSpectrograms - 1)).abs();
        return std::make_tuple(signal, gain);
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(c.bufferSize); }

    static bool validInput(Input input, const Coefficients &c)
    {
        return (c.nSpectrograms > 0) && (c.nSpectrograms <= 30) && (input.signal.rows() == c.bufferSize) && (input.gain.rows() == c.nBands) &&
               (input.gain.cols() == (1LL << (c.nSpectrograms - 1))) && input.signal.allFinite() && input.gain.allFinite() && (input.gain >= 0.f).all() &&
               (input.gain <= 1.f).all();
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && output.allFinite(); }
};

class PerceptualSpectralAttenuate : public Algorithm<PerceptualSpectralAttenuateConfiguration>
{
  public:
    PerceptualSpectralAttenuate() = default;
    PerceptualSpectralAttenuate(const Coefficients &c);
};
