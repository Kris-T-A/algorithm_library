#pragma once
#include "algorithm_library/audio_attenuate.h"
#include "framework/framework.h"
#include "spectrogram/spectrogram_set.h"

// Adaptive audio attenuation algorithm. The gainSpectrogram is a matrix of size (hopSize * 2 + 1) x nFrames.
// The gainSpectrogram is used to attenuate the input audio signal. The attenuation is done by element-wise multiplication of the input audio signal and the gainSpectrogram.
class AudioAttenuateAdaptive : public AlgorithmImplementation<AudioAttenuateConfiguration, AudioAttenuateAdaptive>
{
  public:
    AudioAttenuateAdaptive(const Coefficients &c = Coefficients())
        : BaseAlgorithm(c), spectrogram({.bufferSize = c.hopSize, .nBands = 2 * c.hopSize + 1, .algorithmType = SpectrogramSet::Coefficients::ADAPTIVE_HANN_8})
    {}

    SpectrogramSet spectrogram;
    DEFINE_MEMBER_ALGORITHMS(spectrogram)

  private:
    void processAlgorithm(Input input, Output output) {}

    friend BaseAlgorithm;
};