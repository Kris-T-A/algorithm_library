#pragma once
#include "algorithm_library/perceptual_spectral_analysis.h"
#include "framework/framework.h"
#include "scale_transform/log_scale.h"
#include "spectrogram/spectrogram_set.h"
#include "utilities/fastonebigheader.h"

class PerceptualSpectrogram : public AlgorithmImplementation<PerceptualSpectralAnalysisConfiguration, PerceptualSpectrogram>
{
  public:
    PerceptualSpectrogram(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, spectrogramSet({.bufferSize = c.bufferSize, .nBands = c.bufferSize + 1, .algorithmType = SpectrogramSet::Coefficients::ADAPTIVE_HANN_8}),
          logScale({.nInputs = c.bufferSize + 1, .nOutputs = c.nBands, .indexEnd = c.sampleRate / 2, .transformType = LogScale::Coefficients::LOGARITHMIC})
    {
        spectrogramOut = spectrogramSet.initDefaultOutput();
    }

    SpectrogramSet spectrogramSet;
    LogScale logScale;
    DEFINE_MEMBER_ALGORITHMS(spectrogramSet, logScale)

    Eigen::ArrayXf getCornerFrequencies() const { return logScale.getCornerIndices(); }

    Eigen::ArrayXf getCenterFrequencies() const
    {
        Eigen::ArrayXf cornerFrequencies = getCornerFrequencies();
        return (cornerFrequencies.head(C.nBands) + cornerFrequencies.tail(C.nBands)) / 2;
    }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        spectrogramSet.process(input, spectrogramOut);
        logScale.process(spectrogramOut, output);
        output = output.max(1e-20f).unaryExpr(std::ref(energy2dB)); // energy2dB = 10*log10(x)
    }

    size_t getDynamicSizeVariables() const override
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf spectrogramOut;

    friend BaseAlgorithm;
};