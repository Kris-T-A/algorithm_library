#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "framework/framework.h"

// Time-domain NLMS adaptive line enhancer used as a periodic-component canceller.
//
//   regressor:    u[n] = [x[n - delta], x[n - delta - 1], ..., x[n - delta - N + 1]]^T
//   prediction:   yhat[n] = w^T u[n]
//   residual:     e[n] = x[n] - yhat[n]
//   NLMS update:  w <- (1 - leakage) * w + (mu / (u^T u + reg)) * e[n] * u[n]
//
// The decorrelation delay delta makes the regressor uncorrelated with the broadband part of x[n],
// so the FIR converges only on periodic / narrowband content.
//
// author: Kristian Timm Andersen
class AdaptivePredictorNLMSTimeDomain : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorNLMSTimeDomain>
{
  public:
    AdaptivePredictorNLMSTimeDomain(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        bufferSize = c.filterLength + c.decorrelationDelay;
        weights.setZero(c.filterLength);
        delayBuffer.setZero(bufferSize);
        writeIndex = 0;
        onParametersChanged();
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        const int N = C.filterLength;
        const int delta = C.decorrelationDelay;
        const bool outputResidual = (C.outputMode == Coefficients::RESIDUAL);

        for (Eigen::Index n = 0; n < input.rows(); n++)
        {
            delayBuffer(writeIndex) = input(n);

            float yhat = 0.f;
            float uEnergy = 0.f;
            for (int k = 0; k < N; k++)
            {
                int idx = writeIndex - delta - k;
                if (idx < 0) { idx += bufferSize; }
                const float u = delayBuffer(idx);
                yhat += weights(k) * u;
                uEnergy += u * u;
            }

            const float e = input(n) - yhat;
            const float gain = mu / (uEnergy + P.regularization);
            const float decay = 1.f - leakage;
            for (int k = 0; k < N; k++)
            {
                int idx = writeIndex - delta - k;
                if (idx < 0) { idx += bufferSize; }
                weights(k) = decay * weights(k) + gain * e * delayBuffer(idx);
            }

            output(n) = outputResidual ? e : yhat;

            writeIndex++;
            if (writeIndex >= bufferSize) { writeIndex = 0; }
        }
    }

    void onParametersChanged()
    {
        // Map convergenceTimeMs -> step size mu.
        // Heuristic: tau in samples = convergenceTimeMs * sampleRate / 1000.
        // For NLMS, convergence time scales as N / mu, so mu ≈ N / tau, clipped for stability.
        const float tauSamples = std::max(1.f, P.convergenceTimeMs * 1e-3f * C.sampleRate);
        mu = static_cast<float>(C.filterLength) / tauSamples;
        if (mu > 0.5f) { mu = 0.5f; }
        if (mu < 1e-6f) { mu = 1e-6f; }
        leakage = mu * 1e-3f;
    }

    void resetVariables() final
    {
        weights.setZero();
        delayBuffer.setZero();
        writeIndex = 0;
    }

    size_t getDynamicSizeVariables() const final
    {
        return weights.getDynamicMemorySize() + delayBuffer.getDynamicMemorySize();
    }

    Eigen::ArrayXf weights;
    Eigen::ArrayXf delayBuffer;
    int bufferSize;
    int writeIndex;
    float mu;
    float leakage;

    friend BaseAlgorithm;
};
