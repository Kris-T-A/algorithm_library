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
class SpectrogramAdaptiveMinMax : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveMinMax>
{
  public:
    SpectrogramAdaptiveMinMax(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds}), upscale([&c]() {
              std::vector<Upscale2DLinear::Coefficients> cUpscale(c.nSpectrograms);
              for (auto i = 0; i < c.nSpectrograms; i++)
              {
                  cUpscale[i].factorHorizontal = positivePow2(c.nSpectrograms - 1 - i);
                  cUpscale[i].factorVertical = positivePow2(i);
                  cUpscale[i].leftBoundaryExcluded = true;
              }
              return cUpscale;
          }()),
          filterMinLemire({.filterLength = static_cast<int>(250 * c.nBands / c.sampleRate), .nChannels = 1}),
          filterMaxLemire({.filterLength = static_cast<int>(250 * c.nBands / c.sampleRate), .nChannels = 1})
    {
        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        Eigen::ArrayXf inputFrame(c.bufferSize);
        spectrogramOut = spectrogramSet.initOutput(inputFrame);

        spectrogramMin.resize(c.nSpectrograms);
        spectrogramMax.resize(c.nSpectrograms);
        spectrogramMin[0] = Eigen::ArrayXXf::Zero(spectrogramOut[0].rows(), 2);          // first spectrogram has 2 columns (current and previous frame)
        spectrogramMax[0] = Eigen::ArrayXXf::Zero(spectrogramOut[0].rows(), 2);          // first spectrogram has 2 columns (current and previous frame)
        int delayRef = spectrogramSet.spectrograms[0].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
        for (auto i = 1; i < c.nSpectrograms; i++)
        {
            int bufferSize = c.bufferSize / positivePow2(i);
            int delay = spectrogramSet.spectrograms[i].filterbanks[0].getFrameSize() / 2; // delay is half the frame size
            int nCols = 2 + (delayRef - delay) / bufferSize + positivePow2(i) - 1;        // 2 columns for current and previous frame, plus additional columns for the delay
            spectrogramMin[i] = Eigen::ArrayXXf::Zero(spectrogramOut[i].rows(), nCols);
            spectrogramMax[i] = Eigen::ArrayXXf::Zero(spectrogramOut[i].rows(), nCols);
        }
        spectrogramMinUpscaled = Eigen::ArrayXXf::Zero(c.nBands, nOutputFrames);
        spectrogramMaxUpscaled = Eigen::ArrayXXf::Zero(c.nBands, nOutputFrames);
        envelopeScale = Eigen::ArrayXXf::Zero(spectrogramOut[c.nSpectrograms - 1].rows(), nOutputFrames + 1);
        oldGain = Eigen::ArrayXf::Zero(c.nBands);
        dynamicRange.resize(c.nBands, nOutputFrames);
        minGlobal.resize(c.nBands, nOutputFrames);
        maxGlobal.resize(c.nBands, nOutputFrames);
        minRange.resize(c.nBands);
        maxRange.resize(c.nBands);
    }

    SpectrogramSetMinMax spectrogramSet;
    VectorAlgo<Upscale2DLinear> upscale;
    FilterMinLemire filterMinLemire;
    FilterMaxLemire filterMaxLemire;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale, filterMinLemire, filterMaxLemire)

  private:
    void upscaleSpectrogram(const int iFilterbank)
    {
        const int newCols = spectrogramOut[iFilterbank].cols();
        const int currentCols = spectrogramMin[iFilterbank].cols();
        const int shiftCols = currentCols - newCols;
        assert(shiftCols > 0);
        spectrogramMin[iFilterbank].leftCols(shiftCols) = spectrogramMin[iFilterbank].rightCols(shiftCols);    // copy prevous frames
        spectrogramMin[iFilterbank].rightCols(newCols) = 10 * spectrogramOut[iFilterbank].max(1e-20f).log10(); // convert power to dB
        upscale[iFilterbank].process(spectrogramMin[iFilterbank].leftCols(newCols + 1), spectrogramMinUpscaled);

        spectrogramMax[iFilterbank].leftCols(shiftCols) = spectrogramMax[iFilterbank].rightCols(shiftCols);                      // copy prevous frames
        spectrogramMax[iFilterbank].rightCols(newCols) = 10 * spectrogramOut[C.nSpectrograms + iFilterbank].max(1e-20f).log10(); // convert power to dB
        upscale[iFilterbank].process(spectrogramMax[iFilterbank].leftCols(newCols + 1), spectrogramMaxUpscaled);
    }

    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);

        // start with spectrogram with the lowest resolution
        upscaleSpectrogram(C.nSpectrograms - 1);
        for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
        {
            filterMinLemire.process(spectrogramMinUpscaled.col(iFrame), minRange);
            filterMaxLemire.process(spectrogramMaxUpscaled.col(iFrame), maxRange);
            minGlobal.col(iFrame) = spectrogramMinUpscaled.col(iFrame);
            maxGlobal.col(iFrame) = spectrogramMaxUpscaled.col(iFrame);
            dynamicRange.col(iFrame) = maxRange - minRange; // calculate dynamic range
        }

        for (auto iFB = C.nSpectrograms - 2; iFB >= 0; iFB--)
        {
            upscaleSpectrogram(iFB);
            for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
            {
                filterMinLemire.process(spectrogramMinUpscaled.col(iFrame), minRange);
                filterMaxLemire.process(spectrogramMaxUpscaled.col(iFrame), maxRange);
                for (auto iBand = 0; iBand < C.nBands; iBand++)
                {
                    float range = maxRange(iBand) - minRange(iBand);
                    if (dynamicRange(iBand, iFrame) < range)
                    {
                        minGlobal(iBand, iFrame) =
                            std::min(minGlobal(iBand, iFrame), spectrogramMinUpscaled(iBand, iFrame)); // update minimum value if it is smaller than the current minimum
                        maxGlobal(iBand, iFrame) = std::max(maxGlobal(iBand, iFrame), spectrogramMaxUpscaled(iBand, iFrame));
                        dynamicRange(iBand, iFrame) = range; // update dynamic range if it is larger than the current range
                    }
                }
            }
        }

        for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
        {
            for (auto iBand = 0; iBand < C.nBands; iBand++)
            {
                float weight = fasterpow(std::max(std::min((maxGlobal(iBand, iFrame) - minGlobal(iBand, iFrame) - 10) / 20.f, 1.f), 0.f), 0.5f);
                output(iBand, iFrame) = weight * minGlobal(iBand, iFrame) + (1.f - weight) * maxGlobal(iBand, iFrame); // interpolate between min and max
            }
        }

        // // find max value in output
        // Eigen::ArrayXXf maxOutput = Eigen::ArrayXXf::Constant(spectrogramOut[C.nSpectrograms - 1].rows(), nOutputFrames, -1000.f); // set to very low value in dB
        // for (auto iFrame = 0; iFrame < nOutputFrames; iFrame++)
        // {
        //     maxOutput(0, iFrame) = std::max(maxOutput(0, iFrame), output.col(iFrame).head(1 + nOutputFrames / 2).maxCoeff());

        //     for (auto iBand = 1; iBand < static_cast<int>(maxOutput.rows()) - 1; iBand++)
        //     {
        //         maxOutput(iBand, iFrame) =
        //             std::max(maxOutput(iBand, iFrame), output.col(iFrame).segment(1 + nOutputFrames / 2 + (iBand - 1) * nOutputFrames, nOutputFrames).maxCoeff());
        //     }
        //     maxOutput(maxOutput.rows() - 1, iFrame) = std::max(maxOutput(maxOutput.rows() - 1, iFrame), output.col(iFrame).tail(1 + nOutputFrames / 2).maxCoeff());
        // }

        // envelopeScale.col(0) = envelopeScale.col(nOutputFrames);
        // // scale envelope to the difference between the minimum of the spectrograms and the maximum of the output
        // envelopeScale.rightCols(nOutputFrames) = minEnvelope - maxOutput;
        // Eigen::ArrayXXf envelopeUpscaled(output.rows(), output.cols());
        // upscale[C.nSpectrograms-1].process(envelopeScale, envelopeUpscaled);

        // // scale output with the envelope
        // output = output + envelopeUpscaled;

        // int nFB = C.nSpectrograms - 2;
        // int nRows = spectrogramRaw[nFB].rows();
        // int nCols = 8 + 1;
        // Eigen::ArrayXXf maxValue(nRows, nCols);
        // int upRow = 1;                 // 1
        // int upCol = positivePow2(nFB); // 8
        // maxValue(0, 0) = oldGain.head(1 + upCol / 2).maxCoeff();
        // for (auto iBand = 1; iBand < nRows - 1; iBand++)
        // {
        //     maxValue(iBand, 0) = oldGain.segment(1 + upCol / 2 + (iBand - 1) * upCol, upCol).maxCoeff();
        // }
        // maxValue(nRows - 1, 0) = oldGain.tail(1 + upCol / 2).maxCoeff();
        // for (auto frame = 1; frame < nCols; frame++)
        // {
        //     maxValue(0, frame) = output.block(0, (frame - 1) * upRow, 1 + upCol / 2, upRow).maxCoeff();
        //     for (auto iBand = 1; iBand < nRows - 1; iBand++)
        //     {
        //         maxValue(iBand, frame) = output.block(1 + upCol / 2 + (iBand - 1) * upCol, (frame - 1) * upRow, upCol, upRow).maxCoeff();
        //     }
        //     maxValue(nRows - 1, frame) = output.bottomRows(1 + upCol / 2).middleCols((frame - 1) * upRow, upRow).maxCoeff();
        // }
        // Upscale2DLinear ups({.factorHorizontal = 2, .factorVertical = 1, .leftBoundaryExcluded = true});
        // Eigen::ArrayXXf upsOut(maxValue.rows(), 8);
        // ups.process(spectrogramRaw[nFB].leftCols(9), upsOut);
        // maxValue -= upsOut; // convert power to dB

        // Upscale2DLinear ups2({.factorHorizontal = 1, .factorVertical = 4, .leftBoundaryExcluded = true});
        // ups2.process(maxValue, spectrogramUpscaled);
        // oldGain = output.col(nOutputFrames - 1);
        // output -= spectrogramUpscaled;
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 0;
        // for (auto i = 0; i < static_cast<int>(spectrogramOut.size()); i++)
        // {
        //     size += spectrogramOut[i].getDynamicMemorySize();
        //     size += spectrogramMin[i].getDynamicMemorySize();
        //     size += spectrogramMax[i].getDynamicMemorySize();
        // }
        // size += spectrogramMinUpscaled.getDynamicMemorySize();
        // size += spectrogramMaxUpscaled.getDynamicMemorySize();
        // size += dynamicRange.getDynamicMemorySize();
        // size += minRange.getDynamicMemorySize();
        // size += maxRange.getDynamicMemorySize();
        // size += envelopeScale.getDynamicMemorySize();
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrogramOut;
    std::vector<Eigen::ArrayXXf> spectrogramMax;
    std::vector<Eigen::ArrayXXf> spectrogramMin;
    Eigen::ArrayXXf spectrogramMinUpscaled;
    Eigen::ArrayXXf spectrogramMaxUpscaled;
    Eigen::ArrayXXf envelopeScale;
    Eigen::ArrayXf oldGain;
    Eigen::ArrayXXf dynamicRange;
    Eigen::ArrayXXf minGlobal;
    Eigen::ArrayXXf maxGlobal;
    Eigen::ArrayXf minRange;
    Eigen::ArrayXf maxRange;

    friend BaseAlgorithm;
};