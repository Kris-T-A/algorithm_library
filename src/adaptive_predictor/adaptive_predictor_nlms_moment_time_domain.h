#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "framework/framework.h"

// Time-domain NLMS with momentum adaptive step size. Same adaptive-q loop as the freq-domain
// momentum variant (see adaptive_predictor_nlms_moment.h) but per-sample on a single real FIR of
// N = filterLength taps. Updates happen every input sample rather than once per WOLA hop.
//
//   regressor:        u[n] = [x[n-delta], x[n-delta-1], ..., x[n-delta-N+1]]^T
//   LoopbackVar:      V_u   += lambda * (x[n]^2 - V_u)               (by stationarity)
//   micEst:           yhat = u^T w
//   error:            e    = x[n] - yhat
//   min-variance out: yOut = (e^2 < x[n]^2) ? e : x[n]                (fallback if filter worsens)
//   NearendVar:       V_n  += lambda * (max(e^2, alpha*yhat^2) - V_n)
//   Momentum step:    p = M + N * C
//                     q = p / (N * V_n + (N+2) * p * V_u + R)
//                     q = min(q, 1 / (N * V_u + R))                   (plain-NLMS stability cap)
//                     M = max((1 - q * V_u) * p, 0.01)
//   Filter update:    w[i] += q * e * u[i]
//   CoeffVar:         C   += lambda * ((w[0]-change)^2 - C)           (tap-0 change energy)
//
// Lambda for the variance trackers is a fixed function of sampleRate and N, chosen so the
// estimates average over ~0.1 s plus N-1 samples:
//   lambda = 1 - exp(-1 / (sampleRate * 0.1 + N - 1))
// convergenceTimeMs is ignored — the step size is fully data-driven through the momentum/q loop.
//
// author: Kristian Timm Andersen
class AdaptivePredictorNLMSMomentumTimeDomain : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorNLMSMomentumTimeDomain>
{
  public:
    AdaptivePredictorNLMSMomentumTimeDomain(Coefficients c = {.algorithmType = Coefficients::NLMS_MOMENTUM_TIME_DOMAIN}) : BaseAlgorithm{c}
    {
        bufferSize = c.filterLength + c.decorrelationDelay;
        weights.setZero(c.filterLength);
        delayBuffer.setZero(bufferSize);
        writeIndex = 0;

        loopbackVariance = kLoopbackVarianceInit;
        nearendVariance = kNearendVarianceInit;
        coefficientVariance = kCoefficientVarianceInit;
        momentum = kMomentumInit;

        lambdaSmooth = 1.f - std::exp(-1.f / (c.sampleRate * 0.005f + static_cast<float>(c.filterLength - 1)));
    }

  private:
    static constexpr float kNearendLimit = 0.0001f;
    static constexpr float kMomentumFloor = 0.01f;
    static constexpr float kLoopbackVarianceInit = 100.f;
    static constexpr float kNearendVarianceInit = 10.f;
    static constexpr float kCoefficientVarianceInit = 0.f;
    static constexpr float kMomentumInit = 1.f;

    void processAlgorithm(Input input, Output output)
    {
        const int N = C.filterLength;
        const int delta = C.decorrelationDelay;
        const bool outputResidual = (C.outputMode == Coefficients::RESIDUAL);
        const float lambda = lambdaSmooth;
        const float Nf = static_cast<float>(N);

        for (Eigen::Index n = 0; n < input.rows(); n++)
        {
            delayBuffer(writeIndex) = input(n);

            // LoopbackVariance tracks x[n]^2 — by stationarity, this matches the regressor power.
            const float xPower = input(n) * input(n);
            loopbackVariance += lambda * (xPower - loopbackVariance);

            // Filter-prediction.
            float yhat = 0.f;
            for (int k = 0; k < N; k++)
            {
                int idx = writeIndex - delta - k;
                if (idx < 0) { idx += bufferSize; }
                yhat += weights(k) * delayBuffer(idx);
            }
            const float e = input(n) - yhat;
            const float pNew = e * e;

            // Min-variance output fallback.
            const float pInput = xPower;
            const float yOut = (pNew < pInput) ? e : input(n);
            output(n) = outputResidual ? yOut : (input(n) - yOut);

            // NearendVariance with soft-floor at kNearendLimit * yhat^2.
            const float pFloor = yhat * yhat * kNearendLimit;
            nearendVariance += lambda * (std::max(pNew, pFloor) - nearendVariance);

            // Momentum & adaptive step.
            const float Vu = loopbackVariance;
            const float p = momentum + Nf * coefficientVariance;
            float q = p / (Nf * nearendVariance + (Nf + 2.f) * p * Vu + 1e-30f);
            const float qCap = 1.f / (Nf * Vu + 1e-30f);
            if (q > qCap) { q = qCap; }
            momentum = std::max((1.f - q * Vu) * p, kMomentumFloor);

            // NLMS-style filter update with adaptive step q. Track tap-0 change separately for C.
            const float W = q * e;
            int tap0idx = writeIndex - delta;
            if (tap0idx < 0) { tap0idx += bufferSize; }
            const float tap0change = W * delayBuffer(tap0idx);
            weights(0) += tap0change;
            for (int k = 1; k < N; k++)
            {
                int idx = writeIndex - delta - k;
                if (idx < 0) { idx += bufferSize; }
                weights(k) += W * delayBuffer(idx);
            }

            // CoefficientVariance tracks tap-0 change magnitude.
            coefficientVariance += lambda * (tap0change * tap0change - coefficientVariance);

            writeIndex++;
            if (writeIndex >= bufferSize) { writeIndex = 0; }
        }
    }

    void resetVariables() final
    {
        weights.setZero();
        delayBuffer.setZero();
        writeIndex = 0;
        loopbackVariance = kLoopbackVarianceInit;
        nearendVariance = kNearendVarianceInit;
        coefficientVariance = kCoefficientVarianceInit;
        momentum = kMomentumInit;
    }

    size_t getDynamicSizeVariables() const final { return weights.getDynamicMemorySize() + delayBuffer.getDynamicMemorySize(); }

    Eigen::ArrayXf weights;
    Eigen::ArrayXf delayBuffer;
    int bufferSize;
    int writeIndex;
    float lambdaSmooth;
    float loopbackVariance;
    float nearendVariance;
    float coefficientVariance;
    float momentum;

    friend BaseAlgorithm;
};
