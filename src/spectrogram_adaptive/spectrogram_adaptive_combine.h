#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "framework/framework.h"
#include "spectrogram/spectrogram_filterbank.h"
#include "spectrogram_adaptive/spectrogram_adaptive_wola.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveCombine : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveCombine>
{
  public:
    SpectrogramAdaptiveCombine(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, spectrogramAdaptive({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds, .nonlinearity = 0}),
          spectrograms(2,                                                               // set 2 filterbanks
                       {.bufferSize = c.bufferSize / positivePow2(c.nSpectrograms - 1), // set to smallest buffer size
                        .nBands = c.nBands,
                        .nFolds = c.nFolds,
                        .nonlinearity = 0}) // nonlinearity is not used
    {
        assert(c.nSpectrograms > 0 && c.nBands > 0 && c.bufferSize > 0 && c.nFolds > 0);

        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        frameSize = c.bufferSize / nOutputFrames;
        spectrogramOut = spectrograms[0].initDefaultOutput();

        // calculate small window
        FilterbankConfiguration::Coefficients cFB = spectrograms[0].filterbank.getCoefficients();
        cFB.nBands = (c.nBands - 1) / positivePow2(c.nSpectrograms - 1) + 1; // adjust nBands to the smallest filterbank
        Eigen::ArrayXf windowSmall = FilterbankShared::getAnalysisWindow(cFB);
        int lengthSmall = windowSmall.size();

        // set last half of the first spectrogram's window to the last half of the small window
        Eigen::ArrayXf window = spectrograms[0].filterbank.getWindow();
        float winScale = window.abs2().sum();
        int length = window.size();
        window.segment(length / 2, lengthSmall / 2) =
            windowSmall.segment(lengthSmall / 2, lengthSmall / 2); // copy the last half of the small window to the middle of the large window
        window.tail((length - lengthSmall) / 2).setZero();         // zero out the last half of the large window
        window *= std::sqrt(winScale / window.abs2().sum());       // scale the large window to have the same energy as the small window
        spectrograms[0].filterbank.setWindow(window);

        // set first half of the second spectrogram's window to the first half of the small window
        window = spectrograms[1].filterbank.getWindow();
        window.segment((length - lengthSmall) / 2, lengthSmall / 2) =
            windowSmall.head(lengthSmall / 2);               // copy the first half of the small window to the middle of the large window
        window.head((length - lengthSmall) / 2).setZero();   // zero out the first half of the large window
        window *= std::sqrt(winScale / window.abs2().sum()); // scale the large window to have the same energy as the small window
        spectrograms[1].filterbank.setWindow(window);
    }

    SpectrogramAdaptiveWOLA spectrogramAdaptive;
    VectorAlgo<SpectrogramFilterbank> spectrograms;
    DEFINE_MEMBER_ALGORITHMS(spectrogramAdaptive, spectrograms)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        // process the adaptive spectrogram
        spectrogramAdaptive.process(input, output);

        for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
        {
            for (auto iSG = 0; iSG < static_cast<int>(spectrograms.size()); iSG++)
            {
                spectrograms[iSG].process(input.segment(iFrame * frameSize, frameSize), spectrogramOut);
                spectrogramOut = 10.f * spectrogramOut.max(1e-20f).log10(); // convert to dB
                output.col(iFrame) = output.col(iFrame).min(spectrogramOut);
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    int frameSize;
    Eigen::ArrayXXf spectrogramOut;

    friend BaseAlgorithm;
};