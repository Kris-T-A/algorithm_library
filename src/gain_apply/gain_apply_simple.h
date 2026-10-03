#pragma once
#include "algorithm_library/gain_apply.h"
#include "framework/framework.h"

// Smooth the product of global and per-channel gains for each time sample
// and multiply the input audio by the evolving gains.
class GainApplySimple : public AlgorithmImplementation<GainApplyConfiguration, GainApplySimple>
{
  public:
    GainApplySimple(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        resetVariables();
        onParametersChanged();
    }

    void setChannelGain(int channel, float gain)
    {
        if (channel < 0 || channel >= C.nChannels) { return; }
        channelGain(channel) = gain;
        targetGain(channel) = channelGain(channel) * P.gain;
    }

    float getChannelGain(int channel) const
    {
        if (channel < 0 || channel >= C.nChannels) { return 0.f; }
        return channelGain(channel);
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto channel = 0; channel < C.nChannels; channel++)
        {
            for (auto sample = 0; sample < input.rows(); sample++)
            {
                gainApplied(channel) += smoothingFactor * (targetGain(channel) - gainApplied(channel));
                output(sample, channel) = input(sample, channel) * gainApplied(channel);
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = targetGain.getDynamicMemorySize();
        size += channelGain.getDynamicMemorySize();
        size += gainApplied.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        channelGain = Eigen::ArrayXf::Constant(C.nChannels, 1.f);
        targetGain = Eigen::ArrayXf::Constant(C.nChannels, P.gain);
        gainApplied = Eigen::ArrayXf::Constant(C.nChannels, P.gain);
    }

    void onParametersChanged()
    {
        if (P.smoothingTConstant > 0.f)
        {
            smoothingFactor = static_cast<float>(1. - exp(-1. / (P.smoothingTConstant * C.sampleRate))); // calculation is double precision in case of large time constants
        }
        else { smoothingFactor = 1.f; }
        targetGain = channelGain * P.gain;
    }

    float smoothingFactor;
    Eigen::ArrayXf channelGain;
    Eigen::ArrayXf targetGain;
    Eigen::ArrayXf gainApplied;

    friend BaseAlgorithm;
};