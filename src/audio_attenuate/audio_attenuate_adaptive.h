#pragma once
#include "algorithm_library/audio_attenuate.h"
#include "audio_combine.h"
#include "decimate_gain.h"
#include "delay/circular_buffer.h"
#include "filterbank_set/filterbank_set_wola.h"
#include "framework/framework.h"

// Adaptive audio attenuation algorithm. The gainSpectrogram is a matrix of size (hopSize * 2 + 1) x nFrames.
// The gainSpectrogram is used to attenuate the input audio signal. The attenuation is done by element-wise multiplication of the input audio signal and the gainSpectrogram.
class AudioAttenuateAdaptive : public AlgorithmImplementation<AudioAttenuateConfiguration, AudioAttenuateAdaptive>
{
  public:
    AudioAttenuateAdaptive(const Coefficients &c = Coefficients())
        : BaseAlgorithm(c), bufferSizeSmall(c.bufferSize / Configuration::nGains),
          filterbankAnalysis({.bufferSize = c.bufferSize,
                              .nBands = 2 * c.bufferSize + 1,
                              .nFilterbanks = nFilterbanks,
                              .filterbankType = FilterbankSetAnalysisConfiguration::Coefficients::HANN}),
          decimateGain({.nBands = 2 * c.bufferSize + 1}), filterbankSynthesis({.bufferSize = c.bufferSize,
                                                                               .nBands = 2 * c.bufferSize + 1,
                                                                               .nFilterbanks = nFilterbanks,
                                                                               .filterbankType = FilterbankSetSynthesisConfiguration::Coefficients::HANN}),
          audioCombineMax({.bufferSize = bufferSizeSmall, .nChannels = nFilterbanks})

    {
        delay.resize(nFilterbanks - 1); // one less than the number of filterbanks since the first filterbank does not need a delay
        for (auto i = 0; i < nFilterbanks - 1; i++)
        {
            CircularBuffer::Coefficients cDelay;
            cDelay.delayLength = 3 * c.bufferSize - static_cast<int>(1.5f / positivePow2(i) * c.bufferSize);
            cDelay.nChannels = 1;
            delay[i].setCoefficients(cDelay);
        }
        spectrogramMultipleResolution = filterbankAnalysis.initDefaultOutput();
        gainMultipleResolution = decimateGain.initDefaultOutput();
        gainOldMultipleResolution = decimateGain.initDefaultOutput();
        delayedOutput = Eigen::ArrayXXf::Zero(bufferSizeSmall, nFilterbanks);
    }

    int bufferSizeSmall; // smallest buffer size. Initialize first so it's available in the constructor
    FilterbankSetAnalysisWOLA filterbankAnalysis;
    DecimateGain decimateGain;
    FilterbankSetSynthesisWOLA filterbankSynthesis;
    VectorAlgo<CircularBuffer> delay;
    AudioCombineMax audioCombineMax;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, decimateGain, filterbankSynthesis, delay, audioCombineMax)

  private:
    void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input.audio, spectrogramMultipleResolution);
        decimateGain.process(input.gainSpectrogram, gainMultipleResolution);

        for (auto iFilterbank = 0; iFilterbank < nFilterbanks; iFilterbank++)
        {
            spectrogramMultipleResolution[iFilterbank] *= gainOldMultipleResolution[iFilterbank];
        }
        gainOldMultipleResolution = gainMultipleResolution;

        filterbankSynthesis.process(spectrogramMultipleResolution, output);

        // for each small buffer size, delay and choose the signal with maximum power
        for (auto i = 0; i < Configuration::nGains; i++)
        {
            delayedOutput.col(0) = output.col(0).segment(i * bufferSizeSmall, bufferSizeSmall);
            for (auto iDelay = 1; iDelay < nFilterbanks; iDelay++)
            {
                delay[iDelay - 1].process(output.col(iDelay).segment(i * bufferSizeSmall, bufferSizeSmall), delayedOutput.col(iDelay));
            }
            audioCombineMax.process(delayedOutput, output.col(i)).segment(i * bufferSizeSmall, bufferSizeSmall);
        }
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
    Eigen::ArrayXXf delayedOutput;

    constexpr static int nFilterbanks = numberOfBits(Configuration::nGains); // Number of filterbanks: log2(8) = 4
    constexpr static int nGains2 = Configuration::nGains / 2;                // 8 / 2 = 4

    friend BaseAlgorithm;
};