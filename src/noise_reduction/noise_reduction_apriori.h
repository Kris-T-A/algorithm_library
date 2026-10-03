#pragma once
#include "algorithm_library/noise_reduction.h"
#include "framework/framework.h"
#include "gain_calculation/gain_calculation_apriori.h"
#include "noise_estimation/noise_estimation_activity_detection.h"

// Estimate noise power using signal activity, derive a priori gains from the
// input-to-noise power ratio, and multiply them into the complex spectrum.
class NoiseReductionAPriori : public AlgorithmImplementation<NoiseReductionConfiguration, NoiseReductionAPriori>
{
  public:
    NoiseReductionAPriori(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, //
          noiseEstimation({.filterbankRate = c.filterbankRate, .nBands = c.nBands, .nChannels = c.nChannels}),
          gainCalculation({.nChannels = c.nChannels, .filterbankRate = c.filterbankRate, .nBands = c.nBands})
    {
        xFreq2.resize(c.nBands, c.nChannels);
        noisePow.resize(c.nBands, c.nChannels);
        gain.resize(c.nBands, c.nChannels);
    }

    NoiseEstimationActivityDetection noiseEstimation;
    GainCalculationApriori gainCalculation;
    DEFINE_MEMBER_ALGORITHMS(noiseEstimation, gainCalculation)

  private:
    void processAlgorithm(Input xFreq, Output yFreq)
    {
        xFreq2 = xFreq.abs2();
        noiseEstimation.process(xFreq2, noisePow);

        xFreq2 /= noisePow.max(1e-16f);
        gainCalculation.process(xFreq2, gain);

        yFreq = xFreq * gain;
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = xFreq2.getDynamicMemorySize();
        size += noisePow.getDynamicMemorySize();
        size += gain.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf xFreq2;
    Eigen::ArrayXXf noisePow;
    Eigen::ArrayXXf gain;

    friend BaseAlgorithm;
};