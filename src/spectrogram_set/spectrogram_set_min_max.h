#pragma once
#include "framework/framework.h"
#include "spectrogram/spectrogram_min_max.h"
#include "utilities/fastonebigheader.h"

// A set of spectrograms supporting 1 channel input
//
// The spectrograms are processed in parallel with FFT size halving between each spectrogram and output power spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen

struct SpectrogramSetMinMaxConfiguration
{
    using Input = I::Real;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size in the first filterbank
        int nBands = 2049;     // number of frequency bands in the first filterbank
        int nSpectrograms = 4; // each spectrogram halves the buffer size
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nSpectrograms, nFolds)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time domain signal

    static std::vector<Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXf> output(2 * c.nSpectrograms);
        for (auto i = 0; i < c.nSpectrograms; i++)
        {
            int nFrames = 1 << i;
            int nBands = (c.nBands - 1) / nFrames + 1;
            output[i] = Eigen::ArrayXXf::Zero(nBands, nFrames);
            output[c.nSpectrograms + i] = Eigen::ArrayXXf::Zero(nBands, nFrames);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != 2 * c.nSpectrograms) { return false; }
        for (auto i = 0; i < c.nSpectrograms; i++)
        {
            int nFrames = 1 << i;
            int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands) / nFrames;
            if (!FFTConfiguration::isFFTSizeValid(fftSize)) { return false; }
            int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
            if ((output[i].rows() != nBands) || (output[i].cols() != nFrames) || (!output[i].allFinite()) || !((output[i] >= 0).all())) { return false; }
            if ((output[c.nSpectrograms + i].rows() != nBands) || (output[c.nSpectrograms + i].cols() != nFrames) || (!output[c.nSpectrograms + i].allFinite()) ||
                !((output[c.nSpectrograms + i] >= 0).all()))
            {
                return false;
            }
        }
        return true;
    }
};

class SpectrogramSetMinMax : public AlgorithmImplementation<SpectrogramSetMinMaxConfiguration, SpectrogramSetMinMax>
{
  public:
    SpectrogramSetMinMax(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .nSpectrograms = 4, .nFolds = 1})
        : BaseAlgorithm{c}, spectrograms([&c]() {
              std::vector<SpectrogramMinMax::Coefficients> cSG(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cSG[i].bufferSize = c.bufferSize / positivePow2(i);
                  cSG[i].nBands = (c.nBands - 1) / positivePow2(i) + 1;
                  cSG[i].nFolds = c.nFolds;
              }
              return cSG;
          }())
    {
        nBuffers.resize(C.nSpectrograms);
        bufferSizes.resize(C.nSpectrograms);
        nBuffers[0] = 1;
        bufferSizes[0] = C.bufferSize;
        float winScale = spectrograms[0].filterbanks[0].getWindow().sum();
        for (auto iSG = 1; iSG < C.nSpectrograms; iSG++)
        {
            nBuffers[iSG] = nBuffers[iSG - 1] * 2;
            bufferSizes[iSG] = bufferSizes[iSG - 1] / 2;
            for (auto iFB = 0; iFB < static_cast<int>(spectrograms[iSG].filterbanks.size()); iFB++)
            {
                Eigen::ArrayXf window = spectrograms[iSG].filterbanks[iFB].getWindow();
                window *= winScale / window.sum(); // scale the window to have the same DC value as the first filterbank
                spectrograms[iSG].filterbanks[iFB].setWindow(window);
            }
        }
    }

    VectorAlgo<SpectrogramMinMax> spectrograms;
    DEFINE_MEMBER_ALGORITHMS(spectrograms)

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto iSG = 0; iSG < C.nSpectrograms; iSG++)
        {
            for (auto iSubFrame = 0; iSubFrame < nBuffers[iSG]; iSubFrame++)
            {
                spectrograms[iSG].process(input.segment(iSubFrame * bufferSizes[iSG], bufferSizes[iSG]),
                                          {output[iSG].col(iSubFrame), output[C.nSpectrograms + iSG].col(iSubFrame)});
            }
        }
    }

    size_t getDynamicSizeVariables() const final { return 2 * sizeof(int) * C.nSpectrograms; }

    std::vector<int> bufferSizes;
    std::vector<int> nBuffers;
    friend BaseAlgorithm;
};
