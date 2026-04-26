#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "framework/framework.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_wola.h"
#include "utilities/fastonebigheader.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveEnvelope : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveEnvelope>
{
  public:
    SpectrogramAdaptiveEnvelope(Coefficients c = Coefficients())
        : BaseAlgorithm{c},
          spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity}),
          upscale([&c]() {
              std::vector<Upscale2DLinear::Coefficients> cUpscale(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cUpscale[i].factorHorizontal = positivePow2(c.nSpectrograms - 1 - i);
                  cUpscale[i].factorVertical = positivePow2(i);
                  cUpscale[i].leftBoundaryExcluded = true;
              }
              return cUpscale;
          }())
    {
        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        Eigen::ArrayXf inputFrame(c.bufferSize);
        spectrogramOut = spectrogramSet.initOutput(inputFrame);

        spectrogramRaw.resize(c.nSpectrograms);
        spectrogramRaw[0] = Eigen::ArrayXXf::Zero(spectrogramOut[0].rows(), 2);          // first spectrogram has 2 columns (current and previous frame)
        int delayRef = spectrogramSet.spectrograms[0].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
        for (auto i = 1; i < c.nSpectrograms; i++)
        {
            int bufferSize = c.bufferSize / positivePow2(i);
            int delay = spectrogramSet.spectrograms[i].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
            int nCols = 2 + (delayRef - delay) / bufferSize + positivePow2(i) - 1;        // 2 columns for current and previous frame, plus additional columns for the delay
            spectrogramRaw[i] = Eigen::ArrayXXf::Zero(spectrogramOut[i].rows(), nCols);
        }
        spectrogramUpscaled = Eigen::ArrayXXf::Zero(c.nBands, nOutputFrames);
        oldGain = Eigen::ArrayXf::Zero(c.nBands);
        int nFB = C.nSpectrograms - 2;
        int nRows = static_cast<int>(spectrogramRaw[nFB].rows());
        int nCols = static_cast<int>(spectrogramOut[nFB].cols()) + 1;
        maxValue.resize(nRows, nCols);
    }

    SpectrogramSetWOLA spectrogramSet;
    VectorAlgo<Upscale2DLinear> upscale;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);

        spectrogramRaw[0].col(0) = spectrogramRaw[0].col(1);                   // copy prevous frame
        spectrogramRaw[0].col(1) = 10 * spectrogramOut[0].max(1e-20f).log10(); // convert power to dB;
        upscale[0].process(spectrogramRaw[0], output);

        for (auto iFB = 1; iFB < static_cast<int>(spectrogramOut.size()); iFB++)
        {
            const int newCols = static_cast<int>(spectrogramOut[iFB].cols());
            const int currentCols = static_cast<int>(spectrogramRaw[iFB].cols());
            const int shiftCols = currentCols - newCols;
            assert(shiftCols > 0);
            spectrogramRaw[iFB].leftCols(shiftCols) = spectrogramRaw[iFB].rightCols(shiftCols);    // copy prevous frames
            spectrogramRaw[iFB].rightCols(newCols) = 10 * spectrogramOut[iFB].max(1e-20f).log10(); // convert power to dB
            upscale[iFB].process(spectrogramRaw[iFB].leftCols(newCols + 1), spectrogramUpscaled);
            output = output.min(spectrogramUpscaled);
        }

        int nFB = C.nSpectrograms - 2;
        int nRows = static_cast<int>(spectrogramRaw[nFB].rows());
        int nCols = static_cast<int>(spectrogramOut[nFB].cols()) + 1;
        int upRow = positivePow2(C.nSpectrograms - 1 - nFB); // 1
        int upCol = positivePow2(nFB);                       // 8
        maxValue(0, 0) = oldGain.head(1 + upCol / 2).maxCoeff();
        for (auto iBand = 1; iBand < nRows - 1; iBand++)
        {
            maxValue(iBand, 0) = oldGain.segment(1 + upCol / 2 + (iBand - 1) * upCol, upCol).maxCoeff();
        }
        maxValue(nRows - 1, 0) = oldGain.tail(1 + upCol / 2).maxCoeff();
        for (auto frame = 1; frame < nCols; frame++)
        {
            maxValue(0, frame) = output.block(0, (frame - 1) * upRow, 1 + upCol / 2, upRow).maxCoeff();
            for (auto iBand = 1; iBand < nRows - 1; iBand++)
            {
                maxValue(iBand, frame) = output.block(1 + upCol / 2 + (iBand - 1) * upCol, (frame - 1) * upRow, upCol, upRow).maxCoeff();
            }
            maxValue(nRows - 1, frame) = output.bottomRows(1 + upCol / 2).middleCols((frame - 1) * upRow, upRow).maxCoeff();
        }
        maxValue -= spectrogramRaw[nFB].leftCols(nCols); // convert power to dB
        upscale[nFB].process(maxValue, spectrogramUpscaled);
        oldGain = output.col(nOutputFrames - 1);
        output -= spectrogramUpscaled;
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
        size += oldGain.getDynamicMemorySize();
        size += maxValue.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrogramOut;
    std::vector<Eigen::ArrayXXf> spectrogramRaw;
    Eigen::ArrayXXf spectrogramUpscaled;
    Eigen::ArrayXf oldGain;
    Eigen::ArrayXXf maxValue;

    friend BaseAlgorithm;
};