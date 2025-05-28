#pragma once
#include "algorithm_library/spectrogram_adaptive.h"
#include "framework/framework.h"
#include "spectrogram_adaptive/upscale2d_linear.h"
#include "spectrogram_set/spectrogram_set_wola.h"

// Adaptive Spectrogram
//
// author: Kristian Timm Andersen
class SpectrogramAdaptiveWOLA : public AlgorithmImplementation<SpectrogramAdaptiveConfiguration, SpectrogramAdaptiveWOLA>
{
  public:
    SpectrogramAdaptiveWOLA(Coefficients c = Coefficients())
        : BaseAlgorithm{c},
          spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.nBands, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity})
    {
        nOutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
        Eigen::ArrayXf inputFrame(c.bufferSize);
        spectrogramOut = spectrogramSet.initOutput(inputFrame);
        spectrogramRaw.resize(spectrogramOut.size());
        for (auto i = 0; i < static_cast<int>(spectrogramOut.size()); i++)
        {
            spectrogramRaw[i] = Eigen::ArrayXXf::Zero(spectrogramOut[i].rows(), positivePow2(i + 1)); // +1 to keep the last previous frame
        }
        spectrogramUpscaled = Eigen::ArrayXXf::Zero(c.nBands, nOutputFrames);

        auto cUpscale = upscale.getCoefficients();
        cUpscale.resize(c.nSpectrograms);
        for (auto i = 0; i < c.nSpectrograms; i++)
        {
            cUpscale[i].factorHorizontal = positivePow2(c.nSpectrograms - 1 - i);
            cUpscale[i].factorVertical = positivePow2(i);
            cUpscale[i].leftBoundaryExcluded = true;
        }
        upscale.setCoefficients(cUpscale);
    }

    SpectrogramSetWOLA spectrogramSet;
    VectorAlgo<Upscale2DLinear> upscale;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, upscale)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);

        spectrogramRaw[0].col(0) = spectrogramRaw[0].col(1); // copy prevous frame
        spectrogramRaw[0].col(1) = spectrogramOut[0];
        upscale[0].process(spectrogramRaw[0], output);
        for (auto iFB = 1; iFB < static_cast<int>(spectrogramOut.size()); iFB++)
        {
            spectrogramRaw[iFB].leftCols(1 << iFB) = spectrogramRaw[iFB].rightCols(1 << iFB); // copy prevous frames
            spectrogramRaw[iFB].rightCols(spectrogramOut[iFB].cols()) = spectrogramOut[iFB];
            upscale[iFB].process(spectrogramRaw[iFB].leftCols(spectrogramOut[iFB].cols() + 1), spectrogramUpscaled);
            output = output.min(spectrogramUpscaled);
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
        return size;
    }

    int nOutputFrames;
    std::vector<Eigen::ArrayXXf> spectrogramOut;
    std::vector<Eigen::ArrayXXf> spectrogramRaw;
    Eigen::ArrayXXf spectrogramUpscaled;

    friend BaseAlgorithm;
};