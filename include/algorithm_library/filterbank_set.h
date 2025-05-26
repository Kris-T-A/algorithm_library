#pragma once
#include "algorithm_library/fft.h"
#include "algorithm_library/filterbank.h"
#include "interface/interface.h"

// A set of analysis filter banks supporting 1 channel input
//
// The filterbanks are processed in parallel with FFT size halving between each filterbank and output complex spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen

struct FilterbankSetAnalysisConfiguration
{
    using Input = I::Real;
    using Output = O::VectorComplex2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size in the first filterbank
        int nBands = 2049;     // number of frequency bands in the first filterbank
        int nFilterbanks = 4;  // each filterbank halves the buffer size
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nFilterbanks, nFolds)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time domain signal

    static std::vector<Eigen::ArrayXXcf> initOutput(Input input, const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXcf> output(c.nFilterbanks);
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;
            int nBands = (c.nBands - 1) / nFrames + 1;
            output[i] = Eigen::ArrayXXcf::Zero(nBands, nFrames);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && (input.cols() == 1) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != c.nFilterbanks) { return false; }
        for (auto i = 0; i < c.nFilterbanks; i++)
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
class FilterbankSetAnalysis : public Algorithm<FilterbankSetAnalysisConfiguration>
{
  public:
    FilterbankSetAnalysis() = default;
    FilterbankSetAnalysis(const Coefficients &c);

    float getDelaySamples() const;
};

// -------------------------------------- Filterbank Set Synthesis -----------------------------------------
struct FilterbankSetSynthesisConfiguration
{
    using Input = I::VectorComplex2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size in the first filterbank
        int nBands = 2049;     // number of frequency bands in the first filterbank
        int nFilterbanks = 4;  // each filterbank doubles the buffer size
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nFilterbanks, nFolds)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::vector<Eigen::ArrayXXcf> initInput(const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXcf> input(c.nFilterbanks);
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;
            int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands) / nFrames;
            int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
            input[i] = Eigen::ArrayXXcf::Random(nBands, nFrames);
        }
        return input;
    }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.bufferSize, c.nFilterbanks); }

    static bool validInput(Input input, const Coefficients &c)
    {
        if (static_cast<int>(input.size()) != c.nFilterbanks) { return false; }
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;
            int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands) / nFrames;
            if (!FFTConfiguration::isFFTSizeValid(fftSize)) { return false; }
            int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
            if ((input[i].rows() != nBands) || (input[i].cols() != nFrames) || (!input[i].allFinite())) { return false; }
        }
        return true;
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && (output.cols() == c.nFilterbanks) && output.allFinite(); }
};

// Synthesis filterbank
class FilterbankSetSynthesis : public Algorithm<FilterbankSetSynthesisConfiguration>
{
  public:
    FilterbankSetSynthesis() = default;
    FilterbankSetSynthesis(const Coefficients &c);

    float getDelaySamples() const;
};