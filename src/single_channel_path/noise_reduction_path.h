#pragma once
#include "algorithm_library/single_channel_path.h"
#include "dc_remover/dc_remover_first_order.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include "noise_reduction/noise_reduction_apriori.h"

class NoiseReductionPath : public AlgorithmImplementation<SingleChannelPathConfiguration, NoiseReductionPath>
{
  public:
    NoiseReductionPath(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, //
          filterbank({.nChannels = 1, .bufferSize = c.bufferSize, .nBands = FFTConfiguration::convertFFTSizeToNBands(4 * c.bufferSize), .nFolds = 1}),
          filterbankInverse({.nChannels = 1, .bufferSize = c.bufferSize, .nBands = FFTConfiguration::convertFFTSizeToNBands(4 * c.bufferSize), .nFolds = 1}),
          noiseReduction({.nBands = FFTConfiguration::convertFFTSizeToNBands(4 * c.bufferSize), .nChannels = 1, .filterbankRate = c.sampleRate / c.bufferSize}),
          dcRemover({.nChannels = 1, .sampleRate = c.sampleRate})
    {
        int nBands = FFTConfiguration::convertFFTSizeToNBands(4 * c.bufferSize);
        xTime.resize(c.bufferSize);
        xFreq.resize(nBands);
    }

    FilterbankAnalysisWOLA filterbank;
    FilterbankSynthesisWOLA filterbankInverse;
    NoiseReductionAPriori noiseReduction;
    DCRemoverFirstOrder dcRemover;

    DEFINE_MEMBER_ALGORITHMS(filterbank, filterbankInverse, noiseReduction, dcRemover)

    int getDelaySamples() const { return static_cast<int>(filterbank.getDelaySamples() + filterbankInverse.getDelaySamples()); }

  private:
    void processAlgorithm(Input input, Output output)
    {
        dcRemover.process(input, xTime);
        filterbank.process(xTime, xFreq);
        noiseReduction.process(xFreq, xFreq);
        filterbankInverse.process(xFreq, output);
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = xTime.getDynamicMemorySize();
        size += xFreq.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXf xTime;
    Eigen::ArrayXcf xFreq;

    friend BaseAlgorithm;
};