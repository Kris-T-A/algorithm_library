#pragma once
#include "algorithm_library/spectrogram_set.h"
#include "framework/framework.h"
#include "spectrogram/spectrogram_nonlinear.h"
#include "utilities/fastonebigheader.h"

// A set of spectrograms supporting 1 channel input
//
// The spectrograms are processed in parallel with FFT size halving between each spectrogram and output power spectrograms are stored in a vector.
//
// author: Kristian Timm Andersen
class SpectrogramSetWOLA : public AlgorithmImplementation<SpectrogramSetConfiguration, SpectrogramSetWOLA>
{
  public:
    SpectrogramSetWOLA(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .nSpectrograms = 4, .nFolds = 1, .nonlinearity = 1})
        : BaseAlgorithm{c}, spectrograms([&c]() {
              std::vector<SpectrogramNonlinear::Coefficients> cSG(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cSG[i].bufferSize = c.bufferSize / positivePow2(i);
                  cSG[i].nBands = (c.nBands - 1) / positivePow2(i) + 1;
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
        float winScale = spectrograms[0].filterbanks[0].getWindow().abs2().sum();
        for (auto iSG = 1; iSG < C.nSpectrograms; iSG++)
        {
            nBuffers[iSG] = nBuffers[iSG - 1] * 2;
            bufferSizes[iSG] = bufferSizes[iSG - 1] / 2;
            for (auto iFB = 0; iFB < static_cast<int>(spectrograms[iSG].filterbanks.size()); iFB++)
            {
                Eigen::ArrayXf window = spectrograms[iSG].filterbanks[iFB].getWindow();
                window *= std::sqrt(winScale / window.abs2().sum()); // scale the window to have the same energy as the first filterbank
                spectrograms[iSG].filterbanks[iFB].setWindow(window);
            }
        }
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
