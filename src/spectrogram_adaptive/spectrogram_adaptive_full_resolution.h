#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"
#include "spectrogram/spectrogram_nonlinear.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveFullResolution : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveFullResolution>
{
  public:
    SpectrogramAdaptiveFullResolution(Coefficients c = Coefficients())
        : BaseAlgorithm{c},
          spectrograms(c.nSpectrograms,
                       {.bufferSize = c.bufferSize / positivePow2(c.nSpectrograms - 1), .nBands = c.nBands, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity}),
          filterMinMax({.filterLength = static_cast<int>(250 * FFTConfiguration::convertNBandsToFFTSize(c.nBands) / c.sampleRate), .nChannels = 1})
    {
        assert(c.nSpectrograms > 0 && c.nBands > 0 && c.bufferSize > 0 && c.nFolds > 0);

        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        frameSize = c.bufferSize / nOutputFrames;
        spectrogramOut = spectrograms[0].initDefaultOutput();

        // spectrogram 0
        Eigen::ArrayXf window = spectrograms[0].filterbanks[0].getWindow();
        float winScale = window.sum();
        for (auto iFB = 0; iFB < static_cast<int>(spectrograms[0].filterbanks.size()); iFB++)
        {
            Eigen::ArrayXf windowSmall = spectrograms[0].filterbanks[iFB].getWindow();
            windowSmall *= winScale / windowSmall.sum();
            spectrograms[0].filterbanks[iFB].setWindow(windowSmall);
        }

        // scale smaller spectrograms
        for (auto iSpectrogram = 1; iSpectrogram < c.nSpectrograms; iSpectrogram++)
        {

            for (auto iFilterbank = 0; iFilterbank < spectrograms[iSpectrogram].filterbanks.size(); iFilterbank++)
            {
                setReducedWindow(iSpectrogram, iFilterbank, positivePow2(iSpectrogram), winScale);
            }
        }

        minEnvelope.resize(spectrogramOut.rows());
        maxEnvelope.resize(spectrogramOut.rows());
        weight.resize(spectrogramOut.rows());
    }

    void setReducedWindow(int nSpectrogram, int nFilterbank, int stride, int winScale)
    {
        Eigen::ArrayXf window = spectrograms[nSpectrogram].filterbanks[nFilterbank].getWindow();
        int winSize = window.size();
        Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Zero(winSize); // create a zeroed array of the same size as the original window
        int winSmallSize = winSize / stride;
        windowSmall.segment((winSize - winSmallSize) / 2, winSmallSize) = Eigen::ArrayXf::Map(window.data(), winSmallSize, Eigen::InnerStride<>(stride));
        windowSmall *= winScale / windowSmall.sum();
        spectrograms[nSpectrogram].filterbanks[nFilterbank].setWindow(windowSmall);
    }

    VectorAlgo<SpectrogramNonlinear> spectrograms;
    FilterMinMaxLemire filterMinMax;
    DEFINE_MEMBER_ALGORITHMS(spectrograms, filterMinMax)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
        {
            // process each spectrogram
            spectrograms[0].process(input.segment(iFrame * frameSize, frameSize), spectrogramOut);
            output.col(iFrame) = 10 * spectrogramOut.max(1e-20f).log10(); // convert to dB
            for (auto iSpectrogram = 1; iSpectrogram < C.nSpectrograms; iSpectrogram++)
            {
                spectrograms[iSpectrogram].process(input.segment(iFrame * frameSize, frameSize), spectrogramOut);
                spectrogramOut = 10 * spectrogramOut.max(1e-20f).log10();    // convert to dB
                output.col(iFrame) = output.col(iFrame).min(spectrogramOut); // combine spectrograms by taking the minimum
            }
            filterMinMax.process(output.col(iFrame), {minEnvelope, maxEnvelope});
            weight = ((output.col(iFrame) - minEnvelope).max(1e-3f) / (maxEnvelope - minEnvelope).max(1e-3f)).abs2();
            // Here spectrogramOut contains the last spectrogram
            weight = weight.min(1.f - ((spectrogramOut - output.col(iFrame) - 35.f) / 70.f).min(1.f).max(0.f).unaryExpr([](float x) { return fasterpow(x, 0.5f); }));
            output.col(iFrame) += weight * (spectrogramOut - output.col(iFrame));
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        size += minEnvelope.getDynamicMemorySize();
        size += maxEnvelope.getDynamicMemorySize();
        size += weight.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    int frameSize;
    Eigen::ArrayXf spectrogramOut;
    Eigen::ArrayXf minEnvelope;
    Eigen::ArrayXf maxEnvelope;
    Eigen::ArrayXf weight;

    friend BaseAlgorithm;
};