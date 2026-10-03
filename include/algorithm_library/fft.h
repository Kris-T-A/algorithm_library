#pragma once
#include "interface/interface.h"

// Real version of an FFT.
//
// This class has a public inverse FFT function.
// Supported FFT sizes are multiples of 32 and FFT class has public isFFTSizeValid() method.
// NOTE: This class will throw an exception if FFT size is not supported.
//
// author: Kristian Timm Andersen

struct FFTConfiguration
{
    using Input = I::Real2D;
    using Output = O::Complex2D;

    struct Coefficients
    {
        int fftSize = 512;
        DEFINE_TUNABLE_COEFFICIENTS(fftSize)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static inline int convertNBandsToFFTSize(int nBands) { return (nBands - 1) * 2; }
    static inline int convertFFTSizeToNBands(int fftSize) { return fftSize / 2 + 1; }

    static bool isFFTSizeValid(const int fftSize);
    static int getValidFFTSize(const int fftSize); // return valid fftSize that is equal or greater than fftSize

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.fftSize, 2); } // time samples. Number of channels is arbitrary

    static Eigen::ArrayXXcf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXcf::Zero(convertFFTSizeToNBands(c.fftSize), input.cols());
    } // frequency bins. Number of channels is arbitrary

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.fftSize) && (input.cols() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == convertFFTSizeToNBands(c.fftSize)) && (output.cols() > 0) && output.allFinite(); }

    // exception for constructing FFT with invalid FFT size
    class ExceptionFFT : public std::runtime_error
    {
      public:
        ExceptionFFT(int fftSize) : std::runtime_error("FFT size = " + std::to_string(fftSize) + " is not supported.") {}
    };
};

// Transform real time samples into their nonnegative-frequency complex spectrum;
// the inverse method reconstructs the real time samples.
class FFT : public Algorithm<FFTConfiguration>
{
  public:
    FFT() = default;
    FFT(const Coefficients &c);

    void inverse(I::Complex2D xFreq, O::Real2D yTime);
};
