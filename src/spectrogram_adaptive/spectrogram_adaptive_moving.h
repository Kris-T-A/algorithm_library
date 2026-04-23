#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"
#include "moving_max_min/moving_max_min_horizontal.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_zeropad.h"
#include "utilities/fastonebigheader.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveMoving : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveMoving>
{
  public:
    SpectrogramAdaptiveMoving(Coefficients c = Coefficients())
        : BaseAlgorithm{c},
          spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity}),
          upscale({.factorHorizontal = 2, .factorVertical = 1, .leftBoundaryExcluded = true}),
          filterMinMax({.filterLength = static_cast<int>(250 * FFTConfiguration::convertNBandsToFFTSize(c.nBands) / c.sampleRate), .nChannels = 1}), //
          movingMaxMin([&c]() {
              std::vector<MovingMaxMinHorizontal::Coefficients> cMMM(c.nSpectrograms - 1);
              for (auto i = 0; i < c.nSpectrograms - 1; i++)
              {
                  cMMM[i].filterLength = positivePow2(i + 1);
                  cMMM[i].nChannels = c.nBands;
              }
              return cMMM;
          }())
    {
        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        Eigen::ArrayXf inputFrame(c.bufferSize);
        spectrograms = spectrogramSet.initOutput(inputFrame);

        spectrogramBuffer.resize(c.nSpectrograms - 1);
        int delayRef = spectrogramSet.spectrograms[0].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
        for (auto i = 1; i < c.nSpectrograms; i++)
        {
            int bufferSize = c.bufferSize / positivePow2(i);
            int delay = spectrogramSet.spectrograms[i].filterbanks[0].getFrameSize() / 2 / positivePow2(i); // delay is half the frame size
            int nCols = 1 + (delayRef - delay) / bufferSize + positivePow2(i) - 1;                          // 1 column for current, plus additional columns for the delay
            nCols -= positivePow2(i) - 1;                                                                   // remove delay due to movingMinMax
            spectrogramBuffer[i - 1].resize(spectrograms[i].rows(), nCols);
        }
        outputWithLeftBoundary.resize(c.nBands, nOutputFrames + 1);
        leftBoundaries.resize(c.nBands, c.nSpectrograms - 1);
        resetVariables();
    }

    SpectrogramSetZeropad spectrogramSet;
    Upscale2DLinear upscale;
    FilterMinMaxLemire filterMinMax;
    VectorAlgo<MovingMaxMinHorizontal> movingMaxMin;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale, filterMinMax, movingMaxMin)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrograms);

        output.col(0) = 10 * spectrograms[0].max(1e-20f).log10(); // convert power to dB;

        for (auto iFB = 0; iFB < C.nSpectrograms - 1; iFB++)
        {
            const int prevCols = positivePow2(iFB);
            const auto newCols = positivePow2(iFB + 1);
            const auto currentCols = static_cast<int>(spectrogramBuffer[iFB].cols());
            const int shiftCols = currentCols - newCols;
            assert(shiftCols > 0);

            outputWithLeftBoundary.col(0) = leftBoundaries.col(iFB);
            outputWithLeftBoundary.middleCols(1, prevCols) = output.leftCols(prevCols);
            // save next left boundary
            leftBoundaries.col(iFB) = output.col(prevCols - 1);
            // upscale previous output
            upscale.process(outputWithLeftBoundary.leftCols(prevCols + 1), output.leftCols(newCols));

            // update current spectrogram
            spectrogramBuffer[iFB].leftCols(shiftCols) = spectrogramBuffer[iFB].rightCols(shiftCols); // copy prevous frames
            movingMaxMin[iFB].process(spectrograms[iFB + 1], spectrograms[iFB + 1]);
            spectrogramBuffer[iFB].rightCols(newCols) = 10 * spectrograms[iFB + 1].max(1e-20f).log10(); // convert power to dB

            // combine previous and current spectrogram
            output.leftCols(newCols) = output.leftCols(newCols).min(spectrogramBuffer[iFB].leftCols(newCols));
        }
    }

    void resetVariables() final
    {
        leftBoundaries.setZero();
        for (auto &spectrogram : spectrogramBuffer)
        {
            spectrogram.setZero();
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 0;
        for (auto &s : spectrograms)
        {
            size += s.getDynamicMemorySize();
        }
        for (auto &sp : spectrogramBuffer)
        {
            size += sp.getDynamicMemorySize();
        }
        size += outputWithLeftBoundary.getDynamicMemorySize();
        size += leftBoundaries.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrograms;
    std::vector<Eigen::ArrayXXf> spectrogramBuffer;
    Eigen::ArrayXXf outputWithLeftBoundary;
    Eigen::ArrayXXf leftBoundaries;

    friend BaseAlgorithm;
};