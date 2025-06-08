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

            const int nRows = spectrogramRaw[iFB].rows();
            // first column is previous frame so start counting from index 1
            for (int iCols = 1; iCols < newCols + 1; iCols++)
            {

                Eigen::ArrayXf prevSpec(nRows);
                for (int iRow = 0; iRow < spectrogramRaw[iFB + 1].rows() - 1; iRow++)
                {
                    prevSpec(2 * iRow) = spectrogramRaw[iFB + 1](iRow, 2*iCols - 1);                                                            // copy previous frame
                    prevSpec(2 * iRow + 1) = 0.5 * (spectrogramRaw[iFB + 1](iRow, 2*iCols - 1) + spectrogramRaw[iFB + 1](iRow + 1, 2*iCols - 1)); // average with next frame
                }
                prevSpec(nRows - 1) = spectrogramRaw[iFB + 1](spectrogramRaw[iFB + 1].rows() - 1, 2*iCols - 1);

                Eigen::ArrayXf nextSpec(nRows);
                for (int iRow = 0; iRow < spectrogramRaw[iFB + 1].rows() - 1; iRow++)
                {
                    nextSpec(2 * iRow) = spectrogramRaw[iFB + 1](iRow, 2*iCols + 1);                                                            // copy previous frame
                    nextSpec(2 * iRow + 1) = 0.5 * (spectrogramRaw[iFB + 1](iRow, 2*iCols + 1) + spectrogramRaw[iFB + 1](iRow + 1, 2*iCols + 1)); // average with next frame
                }
                nextSpec(nRows - 1) = spectrogramRaw[iFB + 1](spectrogramRaw[iFB + 1].rows() - 1, 2*iCols + 1);

                int extremumI = 0;
                float extremumV = spectrogramRaw[iFB](extremumI, iCols);
                bool upWards = true;
                float specOld = extremumV;

                for (auto iRow = 1; iRow < nRows; iRow++)
                {
                    const float specCurrent = spectrogramRaw[iFB](iRow, iCols);
                    if (specCurrent > specOld)
                    {
                        if (!upWards)
                        {
                            if (nextSpec(extremumI) > extremumV && nextSpec(extremumI) > prevSpec(extremumI))
                            {
                                spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI) =
                                    spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI).min(nextSpec.segment(extremumI, iRow - extremumI));
                            }
                            else if (prevSpec(extremumI) > extremumV)
                            {
                                spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI) =
                                    spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI).min(prevSpec.segment(extremumI, iRow - extremumI));
                            }
                            upWards = true;
                            extremumV = specOld;
                            extremumI = iRow - 1;
                        }
                    }
                    else if (specCurrent < specOld)
                    {
                        if (upWards)
                        {
                            if (nextSpec(iRow - 1) > specOld && nextSpec(iRow - 1) > prevSpec(iRow - 1))
                            {
                                spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI) =
                                    spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI).min(nextSpec.segment(extremumI, iRow - extremumI));
                            }
                            else if (prevSpec(iRow - 1) > specOld)
                            {
                                spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI) =
                                    spectrogramRaw[iFB].col(iCols).segment(extremumI, iRow - extremumI).min(prevSpec.segment(extremumI, iRow - extremumI));
                            }
                            upWards = false;
                            extremumV = specOld;
                            extremumI = iRow - 1;
                        }
                    }
                    specOld = specCurrent;
                }
                if (upWards)
                {
                    if (nextSpec(nRows - 1) > specOld && nextSpec(nRows - 1) > prevSpec(nRows - 1))
                    {
                        spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI) =
                            spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI).min(nextSpec.segment(extremumI, nRows - extremumI));
                    }
                    else if (prevSpec(nRows - 1) > specOld)
                    {
                        spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI) =
                            spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI).min(prevSpec.segment(extremumI, nRows - extremumI));
                    }
                }
                else
                {
                    if (nextSpec(extremumI) > extremumV && nextSpec(extremumI) > prevSpec(extremumI))
                    {
                        spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI) =
                            spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI).min(nextSpec.segment(extremumI, nRows - extremumI));
                    }
                    else if (prevSpec(extremumI) > extremumV)
                    {
                        spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI) =
                            spectrogramRaw[iFB].col(iCols).segment(extremumI, nRows - extremumI).min(prevSpec.segment(extremumI, nRows - extremumI));
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