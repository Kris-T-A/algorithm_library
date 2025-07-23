#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_min_max.h"
#include "utilities/fastonebigheader.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveZeropad : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveZeropad>
{
  public:
    SpectrogramAdaptiveZeropad(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds}), upscale([&c]() {
              std::vector<Upscale2DLinear::Coefficients> cUpscale(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cUpscale[i].factorHorizontal = positivePow2(c.nSpectrograms - 1 - i);
                  cUpscale[i].factorVertical = 1;
                  cUpscale[i].leftBoundaryExcluded = true;
              }
              return cUpscale;
          }()),
          filterMinMax({.filterLength = static_cast<int>(250 * FFTConfiguration::convertNBandsToFFTSize(c.nBands) / c.sampleRate), .nChannels = 1})
    {
        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        Eigen::ArrayXf inputFrame(c.bufferSize);
        spectrogramOut = spectrogramSet.initOutput(inputFrame);

        spectrogramRaw.resize(c.nSpectrograms);
        spectrogramRaw[0].resize(spectrogramOut[0].rows(), 2);                           // first spectrogram has 2 columns (current and previous frame)
        int delayRef = spectrogramSet.spectrograms[0].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
        for (auto i = 1; i < c.nSpectrograms; i++)
        {
            int bufferSize = c.bufferSize / positivePow2(i);
            int delay = spectrogramSet.spectrograms[i].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
            int nCols = 2 + (delayRef - delay) / bufferSize + positivePow2(i) - 1;        // 2 columns for current and previous frame, plus additional columns for the delay
            spectrogramRaw[i].resize(spectrogramOut[i].rows(), nCols);
        }
        spectrogramUpscaled.resize(c.nBands, nOutputFrames);

        // calculate the index of the spectrogram that has the closest frame size to the desired envelope frame size
        indexEnvelope = [&]() {
            const int desiredEnvelopeFrameSize = 0.064 * c.sampleRate; // 64 ms envelope
            int frameSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
            float dist = std::abs(frameSize - desiredEnvelopeFrameSize);
            int indexEnvelope = 0;
            for (auto i = 1; i < c.nSpectrograms; i++)
            {
                frameSize /= 2;
                float distNew = std::abs(frameSize - desiredEnvelopeFrameSize);
                if (distNew < dist)
                {
                    dist = distNew;
                    indexEnvelope = i;
                }
            }
            return indexEnvelope;
        }();

        minEnvelope.resize(spectrogramOut[indexEnvelope].rows());
        maxEnvelope.resize(spectrogramOut[indexEnvelope].rows());
        weight.resize(spectrogramOut[indexEnvelope].rows(), spectrogramOut[indexEnvelope].cols() + 1);
        weightUpscaled.resize(c.nBands, nOutputFrames);

        resetVariables();
    }

    SpectrogramSetMinMax spectrogramSet;
    VectorAlgo<Upscale2DLinear> upscale;
    FilterMinMaxLemire filterMinMax;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale, filterMinMax)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);

        spectrogramRaw[0].col(0) = spectrogramRaw[0].col(1);                   // copy prevous frame
        spectrogramRaw[0].col(1) = 10 * spectrogramOut[0].max(1e-20f).log10(); // convert power to dB;
        upscale[0].process(spectrogramRaw[0], output);

        for (auto iFB = 1; iFB < static_cast<int>(spectrogramOut.size()); iFB++)
        {
            const int newCols = spectrogramOut[iFB].cols();
            const int currentCols = spectrogramRaw[iFB].cols();
            const int shiftCols = currentCols - newCols;
            assert(shiftCols > 0);
            spectrogramRaw[iFB].leftCols(shiftCols) = spectrogramRaw[iFB].rightCols(shiftCols);    // copy prevous frames
            spectrogramRaw[iFB].rightCols(newCols) = 10 * spectrogramOut[iFB].max(1e-20f).log10(); // convert power to dB
            upscale[iFB].process(spectrogramRaw[iFB].leftCols(newCols + 1), spectrogramUpscaled);
            output = output.min(spectrogramUpscaled);
        }

        for (auto iFrame = 0; iFrame < spectrogramOut[indexEnvelope].cols() + 1; iFrame++)
        {
            filterMinMax.process(spectrogramRaw[indexEnvelope].col(iFrame), {minEnvelope, maxEnvelope});
            weight.col(iFrame) = ((spectrogramRaw[indexEnvelope].col(iFrame) - minEnvelope).max(1e-3f) / (maxEnvelope - minEnvelope).max(1e-3f)).abs2();
        }
        upscale[indexEnvelope].process(weight, weightUpscaled);

        // Here spectrogramUpscaled contains the upscaled spectrogram of the smallest frame size
        weightUpscaled = weightUpscaled.min(1.f - ((spectrogramUpscaled - output - 35.f) / 70.f).min(1.f).max(0.f).unaryExpr([](float x) { return fasterpow(x, 0.5f); }));
        output += weightUpscaled * (spectrogramUpscaled - output);
    }

    void resetVariables() final
    {
        for (auto &spectrogram : spectrogramRaw)
        {
            spectrogram.setZero();
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 0;
        for (auto i = 0; i < static_cast<int>(spectrogramOut.size()); i++)
        {
            size += spectrogramOut[i].getDynamicMemorySize();
            size += spectrogramRaw[i].getDynamicMemorySize();
        }
        size += spectrogramUpscaled.getDynamicMemorySize();
        size += minEnvelope.getDynamicMemorySize();
        size += maxEnvelope.getDynamicMemorySize();
        size += weight.getDynamicMemorySize();
        size += weightUpscaled.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    int indexEnvelope;
    std::vector<Eigen::ArrayXXf> spectrogramOut;
    std::vector<Eigen::ArrayXXf> spectrogramRaw;
    Eigen::ArrayXXf spectrogramUpscaled;
    Eigen::ArrayXf minEnvelope;
    Eigen::ArrayXf maxEnvelope;
    Eigen::ArrayXXf weight;
    Eigen::ArrayXXf weightUpscaled;

    friend BaseAlgorithm;
};