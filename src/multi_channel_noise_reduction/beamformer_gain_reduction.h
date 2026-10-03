#pragma once
#include "algorithm_library/multi_channel_noise_reduction.h"
#include "beamformer/beamformer_mvdr.h"
#include "framework/framework.h"
#include "gain_calculation/gain_calculation_apriori.h"

// Run MVDR beamforming, estimate the output signal-to-noise ratio from its
// signal and noise spectra, and apply smoothed a priori noise-reduction gains.
class BeamformerGainReduction : public AlgorithmImplementation<MultiChannelNoiseReductionConfiguration, BeamformerGainReduction>
{
  public:
    BeamformerGainReduction(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, //
          beamformer({.nChannels = c.nChannels, .filterbankRate = c.filterbankRate, .nBands = c.nBands}),
          gainCalculation({.nChannels = 1, .filterbankRate = c.filterbankRate, .nBands = c.nBands})
    {
        signalBeamformed.resize(c.nBands);
        noiseBeamformed.resize(c.nBands);
        snr.resize(c.nBands);
        gain.resize(c.nBands);
        resetVariables();
    }

    BeamformerMVDR beamformer;
    GainCalculationApriori gainCalculation;

    DEFINE_MEMBER_ALGORITHMS(beamformer, gainCalculation)

  private:
    void processAlgorithm(Input input, Output yFreq)
    {
        beamformer.process({input.xFreq, input.signalOfInterestFlag}, {signalBeamformed, noiseBeamformed});
        snr = signalBeamformed.abs2() / noiseBeamformed.abs2().max(1e-10f);
        gainCalculation.process(snr, gain);
        yFreq = signalBeamformed * gain;
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = signalBeamformed.getDynamicMemorySize();
        size += noiseBeamformed.getDynamicMemorySize();
        size += snr.getDynamicMemorySize();
        size += gain.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        signalBeamformed.setZero();
        noiseBeamformed.setZero();
        snr.setZero();
        gain.setOnes();
    }

    Eigen::ArrayXcf signalBeamformed;
    Eigen::ArrayXcf noiseBeamformed;
    Eigen::ArrayXf snr;
    Eigen::ArrayXf gain;

    friend BaseAlgorithm;
};