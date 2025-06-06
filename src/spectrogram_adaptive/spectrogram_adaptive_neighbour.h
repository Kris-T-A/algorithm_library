#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "framework/framework.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_wola.h"
#include "utilities/fastonebigheader.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveNeighbour : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveNeighbour>
{
  public:
    SpectrogramAdaptiveNeighbour(Coefficients c = Coefficients())
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
    }

    SpectrogramSetWOLA spectrogramSet;
    VectorAlgo<Upscale2DLinear> upscale;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);

        // process spectrogram with smallest buffer size (last in the list)
        int newCols = spectrogramOut[C.nSpectrograms - 1].cols();
        int currentCols = spectrogramRaw[C.nSpectrograms - 1].cols();
        int shiftCols = currentCols - newCols;
        assert(shiftCols > 0);
        spectrogramRaw[C.nSpectrograms - 1].leftCols(shiftCols) = spectrogramRaw[C.nSpectrograms - 1].rightCols(shiftCols);      // copy prevous frames
        spectrogramRaw[C.nSpectrograms - 1].rightCols(newCols) = 10.f * spectrogramOut[C.nSpectrograms - 1].max(1e-20f).log10(); // convert power to dB
        upscale[C.nSpectrograms - 1].process(spectrogramRaw[C.nSpectrograms - 1].leftCols(newCols + 1), spectrogramUpscaled);
        output = spectrogramUpscaled;

        for (auto iFB = C.nSpectrograms - 2; iFB >= 0; iFB--)
        {
            const int newCols = spectrogramOut[iFB].cols();
            const int currentCols = spectrogramRaw[iFB].cols();
            const int shiftCols = currentCols - newCols;
            assert(shiftCols > 0);
            spectrogramRaw[iFB].leftCols(shiftCols) = spectrogramRaw[iFB].rightCols(shiftCols);      // copy prevous frames
            spectrogramRaw[iFB].rightCols(newCols) = 10.f * spectrogramOut[iFB].max(1e-20f).log10(); // convert power to dB

            Eigen::Map<Eigen::ArrayXXf> spec(spectrogramRaw[iFB].col(1).data(), spectrogramRaw[iFB].rows(), newCols); // spectrogramRaw[iFB].middleCols(1, newCols)
            for (int iCols = 0; iCols < newCols; iCols++)
            {
                float maxV = -200.f;
                int maxI = 0;
                bool upWards = true;
                for (auto iRow = 0; iRow < spectrogramRaw[iFB].rows(); iRow++)
                {
                    if (spectrogramRaw[iFB](iRow, iCols) > maxV)
                    {
                        if (!upWards)
                        {
                            float maxS =
                                spectrogramUpscaled(maxI * newCols, iCols * 8 / newCols + 1); // current value in upscaled current spectrogram. NOT the prevous spectrogram
                            upWards = true;
                        }
                        maxV = spectrogramRaw[iFB](iRow, iCols);
                        maxI = iRow;
                    }
                }
            }

            upscale[iFB].process(spectrogramRaw[iFB].leftCols(newCols + 1), spectrogramUpscaled);
            output = output.min(spectrogramUpscaled);
        }
        // output += lin2dB(std::min(10.f,scale));
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
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrogramOut;
    std::vector<Eigen::ArrayXXf> spectrogramRaw;
    Eigen::ArrayXXf spectrogramUpscaled;

    friend BaseAlgorithm;
};