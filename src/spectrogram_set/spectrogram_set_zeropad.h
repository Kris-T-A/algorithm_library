#pragma once
#include "framework/framework.h"
#include "spectrogram/spectrogram_nonlinear.h"
#include "utilities/fastonebigheader.h"

// A set of spectrograms supporting 1 channel input
//
// The spectrograms are processed in parallel with FFT size halving between each spectrogram and output power spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen

struct SpectrogramSetZeropadConfiguration
{
    using Input = I::Real;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int bufferSize = 1024; // buffer size in the first filterbank
        int nBands = 2049;     // number of frequency bands in all filterbanks
        int nSpectrograms = 4; // each spectrogram halves the buffer size
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        int nonlinearity = 1;  // nonlinearity
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
            output[i] = Eigen::ArrayXXf::Zero(c.nBands, nFrames);
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
            if ((output[i].rows() != c.nBands) || (output[i].cols() != nFrames) || (!output[i].allFinite()) || !((output[i] >= 0).all())) { return false; }
        }
        return true;
    }
};

class SpectrogramSetZeropad : public AlgorithmImplementation<SpectrogramSetZeropadConfiguration, SpectrogramSetZeropad>
{
  public:
    SpectrogramSetZeropad(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, spectrograms([&c]() {
              std::vector<SpectrogramNonlinear::Coefficients> cSG(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cSG[i].bufferSize = c.bufferSize / positivePow2(i);
                  cSG[i].nBands = c.nBands;
                  cSG[i].nFolds = c.nFolds;
                  cSG[i].nonlinearity = c.nonlinearity;
              }
              return cSG;
          }())
    {
        nBuffers.resize(C.nSpectrograms);
        bufferSizes.resize(C.nSpectrograms);
        nBuffers[0] = 1;
        bufferSizes[0] = C.bufferSize;
        float winScale = spectrograms[0].filterbanks[0].getWindow().sum();
        for (auto iFB = 0; iFB < static_cast<int>(spectrograms[0].filterbanks.size()); iFB++)
        {
            Eigen::ArrayXf window = spectrograms[0].filterbanks[iFB].getWindow();
            window *= winScale / window.sum();
            spectrograms[0].filterbanks[iFB].setWindow(window);
        }

        for (auto iSG = 1; iSG < C.nSpectrograms; iSG++)
        {
            nBuffers[iSG] = nBuffers[iSG - 1] * 2;
            bufferSizes[iSG] = bufferSizes[iSG - 1] / 2;
            for (auto iFB = 0; iFB < static_cast<int>(spectrograms[iSG].filterbanks.size()); iFB++)
            {
                setReducedWindow(iSG, iFB, positivePow2(iSG), winScale);
            }
        }
    }

    void setReducedWindow(int nSpectrogram, int nFilterbank, int stride, float winScale)
    {
        Eigen::ArrayXf window = spectrograms[nSpectrogram].filterbanks[nFilterbank].getWindow();
        auto winSize = static_cast<int>(window.size());
        Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Zero(winSize); // create a zeroed array of the same size as the original window
        int winSmallSize = winSize / stride;
        windowSmall.tail(winSmallSize) = Eigen::ArrayXf::Map(window.data(), winSmallSize, Eigen::InnerStride<>(stride));
        windowSmall *= winScale / windowSmall.sum();
        spectrograms[nSpectrogram].filterbanks[nFilterbank].setWindow(windowSmall);
    }

    VectorAlgo<SpectrogramNonlinear> spectrograms;
    DEFINE_MEMBER_ALGORITHMS(spectrograms)

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto iSG = 0; iSG < C.nSpectrograms; iSG++)
        {
            for (auto iSubFrame = 0; iSubFrame < nBuffers[iSG]; iSubFrame++)
            {
                spectrograms[iSG].process(input.segment(iSubFrame * bufferSizes[iSG], bufferSizes[iSG]), output[iSG].col(iSubFrame));
            }
        }
    }

    size_t getDynamicSizeVariables() const final { return 2 * sizeof(int) * C.nSpectrograms; }

    std::vector<int> bufferSizes;
    std::vector<int> nBuffers;
    friend BaseAlgorithm;
};
