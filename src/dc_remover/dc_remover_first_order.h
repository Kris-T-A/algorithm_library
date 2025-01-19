#pragma once
#include "algorithm_library/dc_remover.h"
#include "framework/framework.h"

// DC remover implemented as a first-order IIR filter:
// H(z) = (1 + a) / 2 * (1 - z^-1) / (1 - a * z^-1)
// where:
// a = (1 - sqrt(1-b^2))/b
// b = cos(2*pi*f/fs)
// f = cutoff frequency
// fs = sample rate
//
// author: Kristian Timm Andersen
class DCRemoverFirstOrder : public AlgorithmImplementation<DCRemoverConfiguration, DCRemoverFirstOrder>
{
  public:
    DCRemoverFirstOrder(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        delay.resize(c.nChannels);
        resetVariables();
        onParametersChanged();
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto channel = 0; channel < C.nChannels; channel++)
        {
            for (auto sample = 0; sample < input.rows(); sample++)
            {
                const float xi = coef1 * input(sample, channel);
                output(sample, channel) = xi + delay(channel);
                delay(channel) = coef0 * output(sample, channel) - xi;
            }
        }
    }

    void onParametersChanged()
    {
        const float cosf = static_cast<float>(std::cos(2 * 3.14159265358979323846 * P.cutoffFrequency / C.sampleRate));
        const float a = (1.f - std::sqrt(1.f - cosf * cosf)) / cosf;

        coef0 = a;
        coef1 = (1 + a) / 2;
    }

    void resetVariables() final { delay.setZero(); }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = delay.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXf delay;
    float coef0, coef1;

    friend BaseAlgorithm;
};