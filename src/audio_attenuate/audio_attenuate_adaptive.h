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
          decimateGain({.nBands = 2 * c.bufferSize + 1}),
          filterbankSynthesis(
              {.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFilterbanks = 4, .filterbankType = FilterbankSetSynthesisConfiguration::Coefficients::HANN})
    {
        spectrogramMultipleResolution = filterbankAnalysis.initDefaultOutput();
        gainMultipleResolution = decimateGain.initDefaultOutput();
        gainOldMultipleResolution = decimateGain.initDefaultOutput();
    }

    FilterbankSetAnalysisWOLA filterbankAnalysis;
    DecimateGain decimateGain;
    FilterbankSetSynthesisWOLA filterbankSynthesis;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, decimateGain, filterbankSynthesis)

  private:
    void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input.audio, spectrogramMultipleResolution);
        decimateGain.process(input.gainSpectrogram, gainMultipleResolution);

        for (auto iFilterbank = 0; iFilterbank < 4; iFilterbank++)
        {
            spectrogramMultipleResolution[iFilterbank] *= gainOldMultipleResolution[iFilterbank];
        }
        gainOldMultipleResolution = gainMultipleResolution;

        filterbankSynthesis.process(spectrogramMultipleResolution, output);
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
        for (auto &gainOld : gainOldMultipleResolution)
        {
            size += gainOld.getDynamicMemorySize();
        }
        return size;
    }

    std::vector<Eigen::ArrayXXcf> spectrogramMultipleResolution;
    std::vector<Eigen::ArrayXXf> gainMultipleResolution;
    std::vector<Eigen::ArrayXXf> gainOldMultipleResolution;

    friend BaseAlgorithm;
};