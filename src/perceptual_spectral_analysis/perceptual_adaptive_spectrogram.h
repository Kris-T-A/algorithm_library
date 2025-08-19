#pragma once
#include "algorithm_library/perceptual_spectral_analysis.h"
#include "framework/framework.h"
#include "moving_max_min/moving_max_min_vertical.h"
#include "scale_transform/log_scale.h"
#include "spectrogram_adaptive/spectrogram_adaptive_moving.h"
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include "utilities/fastonebigheader.h"

class PerceptualAdaptiveSpectrogram : public AlgorithmImplementation<PerceptualSpectralAnalysisConfiguration, PerceptualAdaptiveSpectrogram>
{
  public:
    PerceptualAdaptiveSpectrogram(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, spectrogram({.bufferSize = c.bufferSize,
                                         .nBands = 2 * c.bufferSize + 1,
                                         .nSpectrograms = c.nSpectrograms,
                                         .nFolds = c.nFolds,
                                         .nonlinearity = c.nonlinearity,
                                         .sampleRate = c.sampleRate}),
          logScale(
              {.nInputs = 2 * c.bufferSize + 1, .nOutputs = c.nBands, .indexStart = 40, .indexEnd = c.sampleRate / 2, .transformType = LogScale::Coefficients::LOGARITHMIC}),
          movingMaxMin({.filterLength = std::max(1, static_cast<int>(c.nBands / 500)), .nChannels = c.nBands})
    {
        spectrogramOut = spectrogram.initDefaultOutput();
        if (c.spectralTilt) { spectralTiltVector = (Eigen::ArrayXf::LinSpaced(2 * c.bufferSize + 1, 0.f, c.sampleRate / 2) / 1000.f).log10(); } // 3dB boost per octave
        else { spectralTiltVector.resize(0); }
    }

    SpectrogramAdaptiveMoving spectrogram;
    LogScale logScale;
    MovingMaxMinVertical movingMaxMin;
    DEFINE_MEMBER_ALGORITHMS(spectrogram, logScale, movingMaxMin)

    Eigen::ArrayXf getCenterFrequencies() const { return logScale.getCenterFrequencies(); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        spectrogram.process(input, spectrogramOut);
        if (C.spectralTilt) { spectrogramOut.colwise() += spectralTiltVector; }
        logScale.process(spectrogramOut, output);
        movingMaxMin.process(output, output);
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