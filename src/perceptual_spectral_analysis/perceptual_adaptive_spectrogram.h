#pragma once
#include "algorithm_library/perceptual_spectral_analysis.h"
#include "framework/framework.h"
#include "scale_transform/log_scale.h"
#include "spectrogram_adaptive/spectrogram_adaptive_full_resolution.h"
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include "utilities/fastonebigheader.h"

class PerceptualAdaptiveSpectrogram : public AlgorithmImplementation<PerceptualSpectralAnalysisConfiguration, PerceptualAdaptiveSpectrogram>
{
  public:
    PerceptualAdaptiveSpectrogram(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c},
          spectrogram({.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nSpectrograms = c.nSpectrograms, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity}),
          logScale({.nInputs = c.bufferSize + 1, .nOutputs = c.nBands, .indexEnd = c.sampleRate / 2, .transformType = LogScale::Coefficients::LOGARITHMIC})
    {
        spectrogramOut = spectrogram.initDefaultOutput();
        if (c.spectralTilt) { spectralTiltVector = Eigen::ArrayXf::LinSpaced(c.bufferSize + 1, 0.f, c.sampleRate / 2) / 1000.f; } // 3dB boost per octave
    }

    SpectrogramAdaptiveZeropad spectrogram;
    LogScale logScale;
    DEFINE_MEMBER_ALGORITHMS(spectrogram, logScale)

    Eigen::ArrayXf getCenterFrequencies() const { return logScale.getCenterFrequencies(); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        spectrogram.process(input, spectrogramOut);
        if (C.spectralTilt) { spectrogramOut.colwise() *= spectralTiltVector; }
        logScale.process(spectrogramOut, output);
    }

    size_t getDynamicSizeVariables() const override
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        size += spectralTiltVector.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf spectrogramOut;
    Eigen::ArrayXf spectralTiltVector;

    friend BaseAlgorithm;
};