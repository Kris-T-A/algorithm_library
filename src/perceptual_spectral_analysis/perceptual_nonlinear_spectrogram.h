#pragma once
#include "algorithm_library/perceptual_spectral_analysis.h"
#include "framework/framework.h"
#include "scale_transform/log_scale.h"
#include "spectrogram/spectrogram_nonlinear.h"
#include "utilities/fastonebigheader.h"

class PerceptualNonlinearSpectrogram : public AlgorithmImplementation<PerceptualSpectralAnalysisConfiguration, PerceptualNonlinearSpectrogram>
{
  public:
    PerceptualNonlinearSpectrogram(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c},
          spectrogram({.bufferSize = c.bufferSize / positivePow2(c.nSpectrograms - 1), .nBands = 2 * c.bufferSize + 1, .nFolds = c.nFolds, .nonlinearity = c.nonlinearity}),
          logScale({.nInputs = 2 * c.bufferSize + 1,
                    .nOutputs = c.nBands,
                    .outputStart = c.frequencyMin,
                    .outputEnd = c.frequencyMax,
                    .inputEnd = c.sampleRate / 2,
                    .transformType = LogScale::Coefficients::LOGARITHMIC})
    {
        framePerBuffer = positivePow2(c.nSpectrograms - 1);
        frameSize = c.bufferSize / framePerBuffer;
        spectrogramOut = spectrogram.initDefaultOutput();
        if (c.spectralTilt) { spectralTiltVector = 10.f * (Eigen::ArrayXf::LinSpaced(2 * c.bufferSize + 1, 0.f, c.sampleRate / 2) / 1000.f).log10(); } // 3dB boost per octave
        else { spectralTiltVector.resize(0); }
    }

    SpectrogramNonlinear spectrogram;
    LogScale logScale;
    DEFINE_MEMBER_ALGORITHMS(spectrogram, logScale)

    Eigen::ArrayXf getCenterFrequencies() const { return logScale.getCenterFrequencies(); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto iBuffer = 0; iBuffer < framePerBuffer; ++iBuffer)
        {
            spectrogram.process(input.segment(iBuffer * frameSize, frameSize), spectrogramOut);
            spectrogramOut = 10.f * spectrogramOut.max(1e-20f).log10();
            if (C.spectralTilt) { spectrogramOut.colwise() += spectralTiltVector; }
            logScale.process(spectrogramOut, output.col(iBuffer));
        }
    }

    size_t getDynamicSizeVariables() const override
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        size += spectralTiltVector.getDynamicMemorySize();
        return size;
    }

    int framePerBuffer;
    int frameSize;
    Eigen::ArrayXXf spectrogramOut;
    Eigen::ArrayXf spectralTiltVector;

    friend BaseAlgorithm;
};