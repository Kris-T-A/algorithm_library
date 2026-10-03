#pragma once
#include "algorithm_library/filterbank_set.h"
#include "filterbank/filterbank_single_channel.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// A set of filterbanks supporting 1 channel input
//
// author: Kristian Timm Andersen

struct FilterbankSetAnalysisZeropadConfiguration
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
            int nBands = c.nBands;
            output[i] = Eigen::ArrayXXcf::Zero(nBands, nFrames);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && (input.cols() == 1) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != c.nFilterbanks) { return false; }
        int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
        if (!FFTConfiguration::isFFTSizeValid(fftSize)) { return false; }
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;
            if ((output[i].rows() != nBands) || (output[i].cols() != nFrames) || (!output[i].allFinite())) { return false; }
        }
        return true;
    }
};

// create a set of single channel filterbanks zeropadded to same number of bands, but each new filterbank halving the bufferSize.
class FilterbankSetAnalysisZeropad : public AlgorithmImplementation<FilterbankSetAnalysisZeropadConfiguration, FilterbankSetAnalysisZeropad>
{
  public:
    FilterbankSetAnalysisZeropad(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, filterbanks([&c]() {
              std::vector<FilterbankAnalysisSingleChannel::Coefficients> cFB(c.nFilterbanks);
              for (auto i = 0; i < c.nFilterbanks; i++)
              {
                  // 2^-i the buffersize but keep number of bands
                  cFB[i].bufferSize = c.bufferSize / positivePow2(i);
                  cFB[i].nBands = c.nBands;
                  cFB[i].nFolds = c.nFolds;
                  cFB[i].nChannels = 1;
              }
              return cFB;
          }())
    {
        nBuffers.resize(C.nFilterbanks);
        bufferSizes.resize(C.nFilterbanks);
        nBuffers[0] = 1;
        bufferSizes[0] = C.bufferSize;
        // scale the window length of the filterbanks by 2^-i
        for (auto i = 1; i < C.nFilterbanks; i++)
        {
            const int factor = positivePow2(i);
            nBuffers[i] = factor;
            bufferSizes[i] = C.bufferSize / factor;

            // calculate analysis window if nBands was also scaled down, corresponding to a smaller window
            FilterbankAnalysisSingleChannel::Coefficients cFB = filterbanks[i].getCoefficients();
            cFB.nBands = (c.nBands - 1) / factor + 1;
            auto winSize = static_cast<int>(filterbanks[i].getWindow().size());
            Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Zero(winSize); // create a zeroed array of the same size as the original window
            int winSmallSize = winSize / factor;
            windowSmall.tail(winSmallSize) = FilterbankShared::getAnalysisWindow(cFB);
            filterbanks[i].setWindow(windowSmall);
        }
    }

    VectorAlgo<FilterbankAnalysisSingleChannel> filterbanks;
    DEFINE_MEMBER_ALGORITHMS(filterbanks)

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto iFB = 0; iFB < C.nFilterbanks; iFB++)
        {
            for (auto iSubFrame = 0; iSubFrame < nBuffers[iFB]; iSubFrame++)
            {
                filterbanks[iFB].process(input.segment(iSubFrame * bufferSizes[iFB], bufferSizes[iFB]), output[iFB].col(iSubFrame));
            }
        }
    }

    size_t getDynamicSizeVariables() const final { return 2 * sizeof(int) * C.nFilterbanks; }

    std::vector<int> bufferSizes;
    std::vector<int> nBuffers;
    friend BaseAlgorithm;
};


struct FilterbankSetSynthesisZeropadConfiguration
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
        int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;    
            input[i] = Eigen::ArrayXXcf::Random(nBands, nFrames);
        }
        return input;
    }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.bufferSize, c.nFilterbanks); }

    static bool validInput(Input input, const Coefficients &c)
    {
        if (static_cast<int>(input.size()) != c.nFilterbanks) { return false; }
        int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        if (!FFTConfiguration::isFFTSizeValid(fftSize)) { return false; }
        int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
        for (auto i = 0; i < c.nFilterbanks; i++)
        {
            int nFrames = 1 << i;
            if ((input[i].rows() != nBands) || (input[i].cols() != nFrames) || (!input[i].allFinite())) { return false; }
        }
        return true;
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && (output.cols() == c.nFilterbanks) && output.allFinite(); }
};

// Reconstruct one audio column per resolution from spectra on a shared FFT grid,
// using the matching reduced synthesis windows and overlap-add.
class FilterbankSetSynthesisZeropad : public AlgorithmImplementation<FilterbankSetSynthesisZeropadConfiguration, FilterbankSetSynthesisZeropad>
{
  public:
    FilterbankSetSynthesisZeropad(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, inverseFilterbanks([&c]() {
              std::vector<FilterbankAnalysisSingleChannel::Coefficients> cFB(c.nFilterbanks);
              for (auto i = 0; i < c.nFilterbanks; i++)
              {
                  // 2^-i the buffersize but keep number of bands
                  cFB[i].bufferSize = c.bufferSize / positivePow2(i);
                  cFB[i].nBands = c.nBands;
                  cFB[i].nFolds = c.nFolds;
                  cFB[i].nChannels = 1;
              }
              return cFB;
          }())
    {
        nBuffers.resize(C.nFilterbanks);
        bufferSizes.resize(C.nFilterbanks);
        nBuffers[0] = 1;
        bufferSizes[0] = C.bufferSize;
        for (auto i = 1; i < C.nFilterbanks; i++)
        {
            const int factor = positivePow2(i);
            nBuffers[i] = factor;
            bufferSizes[i] = C.bufferSize / factor;

            // calculate synthesis window if nBands was also scaled down, corresponding to a smaller window
            FilterbankAnalysisSingleChannel::Coefficients cFB = inverseFilterbanks[i].getCoefficients();
            cFB.nBands = (c.nBands - 1) / factor + 1;
            auto winSize = static_cast<int>(inverseFilterbanks[i].getWindow().size());
            Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Zero(winSize); // create a zeroed array of the same size as the original window
            int winSmallSize = winSize / factor;
            windowSmall.tail(winSmallSize) = FilterbankShared::getSynthesisWindow(cFB);
            inverseFilterbanks[i].setWindow(windowSmall);
        }
    }

    VectorAlgo<FilterbankSynthesisSingleChannel> inverseFilterbanks;
    DEFINE_MEMBER_ALGORITHMS(inverseFilterbanks)

  private:
    void processAlgorithm(Input input, Output output)
    {
        assert(output.rows() == C.bufferSize);
        assert(output.cols() == C.nFilterbanks);

        for (auto iFB = 0; iFB < C.nFilterbanks; iFB++)
        {
            for (auto iSubFrame = 0; iSubFrame < nBuffers[iFB]; iSubFrame++)
            {
                inverseFilterbanks[iFB].process(input[iFB].col(iSubFrame), output.col(iFB).segment(iSubFrame * bufferSizes[iFB], bufferSizes[iFB]));
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 2 * sizeof(int) * C.nFilterbanks; // dynamic size of bufferSizes and nBuffers
        return size;
    }

    std::vector<int> bufferSizes;
    std::vector<int> nBuffers;

    friend BaseAlgorithm;
};