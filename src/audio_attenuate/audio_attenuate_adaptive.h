#pragma once
#include "algorithm_library/audio_attenuate.h"
#include "decimate_gain.h"
#include "filterbank_set/filterbank_set_wola.h"
#include "framework/framework.h"

// Adaptive audio attenuation algorithm. The gainSpectrogram is a matrix of size (hopSize * 2 + 1) x nFrames.
// The gainSpectrogram is used to attenuate the input audio signal. The attenuation is done by element-wise multiplication of the input audio signal and the gainSpectrogram.
class AudioAttenuateAdaptive : public AlgorithmImplementation<AudioAttenuateConfiguration, AudioAttenuateAdaptive>
{
  public:
    AudioAttenuateAdaptive(const Coefficients &c = Coefficients())
        : BaseAlgorithm(c),
          filterbankAnalysis(
              {.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFilterbanks = 4, .filterbankType = FilterbankSetAnalysisConfiguration::Coefficients::HANN}),
          decimateGain({.nBands = 2 * c.bufferSize + 1})
    {
        spectrogramMultipleResolution = filterbankAnalysis.initDefaultOutput();
        gainMultipleResolution = decimateGain.initDefaultOutput();
    }

    FilterbankSetAnalysisWOLA filterbankAnalysis;
    DecimateGain decimateGain;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, decimateGain)

  private:
    void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input.audio, spectrogramMultipleResolution);
        decimateGain.process(input.gainSpectrogram, gainMultipleResolution);
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 0;
        for (auto &spectrogram : spectrogramMultipleResolution)
        {
            size += spectrogram.getDynamicMemorySize();
        }
        for (auto &gain : gainMultipleResolution)
        {
            size += gain.getDynamicMemorySize();
        }
        return size;
    }

    std::vector<Eigen::ArrayXXcf> spectrogramMultipleResolution;
    std::vector<Eigen::ArrayXXf> gainMultipleResolution;

    friend BaseAlgorithm;
};