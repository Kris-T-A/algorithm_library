#pragma once
#include "algorithm_library/spectrogram_adaptive_set.h"
#include "framework/framework.h"
#include "moving_max_min/moving_max_min_horizontal.h"
#include "moving_max_min/moving_max_min_vertical.h"
#include "scale_transform/log_scale.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_zeropad.h"
#include "utilities/fastonebigheader.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveSetMin : public AlgorithmImplementation<SpectrogramAdaptiveSetConfiguration, SpectrogramAdaptiveSetMin>
{
  public:
    SpectrogramAdaptiveSetMin(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, spectrogramSet({.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nSpectrograms = c.nSpectrograms, .nFolds = 1, .nonlinearity = 1}),
          upscale({.factorHorizontal = 2, .factorVertical = 1, .leftBoundaryExcluded = true}), movingMaxMinTime([&c]() {
              std::vector<MovingMaxMinHorizontal::Coefficients> cMMM(c.nSpectrograms - 1);
              for (auto i = 0; i < c.nSpectrograms - 1; i++)
              {
                  cMMM[i].filterLength = positivePow2(i + 1);
                  cMMM[i].nChannels = 2 * c.bufferSize + 1;
              }
              return cMMM;
          }()),
          movingMaxMinFreq({.filterLength = std::max(1, static_cast<int>(c.nBands / 500)), .nChannels = c.nBands}),
          logScale({.nInputs = 2 * c.bufferSize + 1,
                    .nOutputs = c.nBands,
                    .outputStart = c.frequencyMin,
                    .outputEnd = c.frequencyMax,
                    .inputEnd = c.sampleRate / 2,
                    .transformType = LogScale::Coefficients::LOGARITHMIC})
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
        outputWithLeftBoundary.resize(2 * c.bufferSize + 1, nOutputFrames + 1);
        leftBoundaries.resize(2 * c.bufferSize + 1, c.nSpectrograms - 1);

        if (c.spectralTilt) { spectralTiltVector = 10.f * (Eigen::ArrayXf::LinSpaced(2 * c.bufferSize + 1, 0.f, c.sampleRate / 2) / 1000.f).log10(); } // 3dB boost per octave
        else
        {
            spectralTiltVector.resize(0);
        }

        resetVariables();
    }

    SpectrogramSetZeropad spectrogramSet;
    Upscale2DLinear upscale;
    VectorAlgo<MovingMaxMinHorizontal> movingMaxMinTime;
    MovingMaxMinVertical movingMaxMinFreq;
    LogScale logScale;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale, movingMaxMinTime, movingMaxMinFreq, logScale)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrograms);

        spectrograms[0] = 10 * spectrograms[0].max(1e-20f).log10(); // convert power to dB;
        if (C.spectralTilt) { spectrograms[0] += spectralTiltVector; }
        logScale.process(spectrograms[0], output[0]);
        movingMaxMinFreq.process(output[0], output[0]);

        for (auto iFB = 0; iFB < C.nSpectrograms - 1; iFB++)
        {
            const int prevCols = positivePow2(iFB);
            const auto newCols = positivePow2(iFB + 1);
            const auto currentCols = static_cast<int>(spectrogramBuffer[iFB].cols());
            const int shiftCols = currentCols - newCols;
            assert(shiftCols > 0);

            // update current spectrogram
            spectrogramBuffer[iFB].leftCols(shiftCols) = spectrogramBuffer[iFB].rightCols(shiftCols); // copy prevous frames
            movingMaxMinTime[iFB].process(spectrograms[iFB + 1], spectrograms[iFB + 1]);
            spectrogramBuffer[iFB].rightCols(newCols) = 10 * spectrograms[iFB + 1].max(1e-20f).log10(); // convert power to dB
            if (C.spectralTilt) { spectrogramBuffer[iFB].rightCols(newCols).colwise() += spectralTiltVector; }

            outputWithLeftBoundary.col(0) = leftBoundaries.col(iFB);
            outputWithLeftBoundary.middleCols(1, prevCols) = spectrograms[iFB];
            // save next left boundary
            leftBoundaries.col(iFB) = spectrograms[iFB].col(prevCols - 1);
            // upscale previous output
            upscale.process(outputWithLeftBoundary.leftCols(prevCols + 1), spectrograms[iFB + 1]);

            // combine upscaled previous and current spectrogram
            spectrograms[iFB + 1] = spectrograms[iFB + 1].min(spectrogramBuffer[iFB].leftCols(newCols));

            logScale.process(spectrograms[iFB + 1], output[iFB + 1]);
            movingMaxMinFreq.process(output[iFB + 1], output[iFB + 1]);
        }
    }

    void resetVariables() final
    {
        leftBoundaries.setConstant(1e6f); // sentinel for "no prior frame" in dB: large enough that .min(buffer) always discards it, finite to avoid inf-inf NaN in upscale's vertical interp
        for (auto &spectrogram : spectrogramBuffer)
        {
            spectrogram.setConstant(1e6f); // history sentinel: shifted-in cols at frame 0 must lose the .min(buffer), same reason as leftBoundaries
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
        size += spectralTiltVector.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrograms;
    std::vector<Eigen::ArrayXXf> spectrogramBuffer;
    Eigen::ArrayXXf outputWithLeftBoundary;
    Eigen::ArrayXXf leftBoundaries;
    Eigen::ArrayXf spectralTiltVector;

    friend BaseAlgorithm;
};