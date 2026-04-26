#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "algorithm_library/fft.h"
#include "algorithm_library/filterbank.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include <complex>

// Frequency-domain NLMS with momentum adaptive step size. Same filterbank scaffolding as the
// plain NLMS / Kalman freq-domain variants: per-bin FIR of length kFilterLength complex taps
// running on delayed past bin frames from a circular history buffer.
//
// The difference is the step size. Plain NLMS uses mu / (|u|^2 + reg) with mu driven by
// convergenceTimeMs. Momentum NLMS derives the step q per bin from three tracked second-order
// statistics so the step automatically shrinks during nearend bursts and grows when the filter
// is still mis-matched. Adapted from the EchoCancellerNLMS::ProcessOn reference (see the ticket
// that introduced this variant) with loopback = delayed input:
//
//   regressor:        u_k[m] = [X[k, m-delta], X[k, m-delta-1], ..., X[k, m-delta-L+1]]^T
//   LoopbackVar:      V_u[k]   += lambda * (|X[k]|^2 - V_u[k])        (by stationarity)
//   micEst:           yhat = u_k^T * w_k
//   error:            e    = X[k] - yhat
//   min-variance out: yOut = (|e|^2 < |X|^2) ? e : X                  (fallback if filter worsens)
//   NearendVar:       V_n[k]  += lambda * (max(|e|^2, alpha*|yhat|^2) - V_n[k])
//                                (soft-floor alpha=kNearendLimit prevents lockout when e=0)
//   Momentum step:    p = M[k] + L * C[k]
//                     q = p / (L * V_n[k] + (L+2) * p * V_u[k] + eps)
//                     q = min(q, 1 / (L * V_u[k] + eps))
//                     M[k] = max((1 - q * V_u[k]) * p, 0.01)
//   Filter update:    w_k[i] += q * e * conj(u_k[i])
//   CoeffVar:         C[k]   += lambda * (|w_k[0]-change|^2 - C[k])   (tap-0 change energy)
//
// M is the momentum state, C is the coefficient-change variance; together they raise the effective
// step when the filter is changing a lot (early in convergence) and damp it once the filter settles.
// The q-cap keeps the step below the plain-NLMS stability limit 1/(L*V_u).
//
// Filterbank sizing and decorrelation-delay handling are identical to the other freq-domain
// variants; see adaptive_predictor_kalman_freq_domain.h for the full argument.
//
// Adaptation rate is fully data-driven through the momentum/step-size loop; there is no step-size
// knob exposed to the API. The smoothing factor lambda of the variance trackers is a fixed
// function of the filterbank rate (sampleRate/bufferSize) and kFilterLength:
//   FilterbankRate = sampleRate / bufferSize
//   lambda        = 1 - exp(-1 / (FilterbankRate * 0.1 + kFilterLength - 1))
// chosen so the variance estimates average over ~0.1 s plus kFilterLength-1 frames. convergenceTimeMs
// is ignored by this variant.
//
// author: Kristian Timm Andersen
class AdaptivePredictorNLMSMomentum : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorNLMSMomentum>
{
  public:
    AdaptivePredictorNLMSMomentum(Coefficients c = {.algorithmType = Coefficients::NLMS_MOMENTUM})
        : BaseAlgorithm{c}, analysisFilterbank(makeFilterbankCoefficients(c)), synthesisFilterbank(makeFilterbankCoefficients(c))
    {
        const FilterbankConfiguration::Coefficients fc = makeFilterbankCoefficients(c);
        bufferSize = fc.bufferSize;
        nBands = fc.nBands;
        delayFrames = std::max(2, (std::max(1, c.decorrelationDelay) + bufferSize - 1) / bufferSize);
        nHistory = delayFrames + kFilterLength - 1;

        inputHop.setZero(bufferSize);
        outputHop.setZero(bufferSize);
        X.setZero(nBands);
        E.setZero(nBands);
        binHistory.setZero(nBands, nHistory);
        filters.setZero(nBands, kFilterLength);

        loopbackVariance.setConstant(nBands, 10);   // 100
        nearendVariance.setConstant(nBands, 1);     // 10
        coefficientVariance.setConstant(nBands, 1); // 0
        momentums.setConstant(nBands, 1);           // 1

        binHistoryIndex = 0;
        samplesInBlock = 0;
        outputReadIndex = bufferSize;

        const float filterbankRate = c.sampleRate / static_cast<float>(bufferSize);
        lambdaSmooth = 1.f - std::exp(-1.f / (filterbankRate * 0.005f + static_cast<float>(kFilterLength - 1)));
    }

    FilterbankAnalysisWOLA analysisFilterbank;
    FilterbankSynthesisWOLA synthesisFilterbank;
    DEFINE_MEMBER_ALGORITHMS(analysisFilterbank, synthesisFilterbank)

  private:
    static constexpr int kFilterLength = 4;         // complex FIR taps per bin
    static constexpr float kNearendLimit = 0.0001f; // soft-floor factor for NearendVariance
    static constexpr float kMomentumFloor = 0.01f;  // lower bound on momentum state
    using Complex = std::complex<float>;

    static FilterbankConfiguration::Coefficients makeFilterbankCoefficients(const Coefficients &c)
    {
        FilterbankConfiguration::Coefficients fc;
        fc.nChannels = 1;
        const int fftSize = FFTConfiguration::getValidFFTSize(std::max(1, c.filterLength));
        fc.bufferSize = fftSize / 4;
        fc.nBands = fftSize / 2 + 1;
        fc.nFolds = 1;
        return fc;
    }

    void processAlgorithm(Input input, Output output)
    {
        const bool outputResidual = (C.outputMode == Coefficients::RESIDUAL);
        const float lambda = lambdaSmooth;
        constexpr int L = kFilterLength;
        constexpr float Lf = static_cast<float>(L);

        for (Eigen::Index n = 0; n < input.rows(); n++)
        {
            inputHop(samplesInBlock++) = input(n);

            if (samplesInBlock == bufferSize)
            {
                analysisFilterbank.process(inputHop, X);

                // Regressor slot offsets — tap 0 is the newest regressor (delayFrames hops back),
                // tap L-1 is oldest (delayFrames + L - 1 hops back).
                int regSlots[kFilterLength];
                for (int i = 0; i < L; i++)
                {
                    int idx = binHistoryIndex + (L - 1 - i);
                    if (idx >= nHistory) { idx -= nHistory; }
                    regSlots[i] = idx;
                }

                for (int k = 0; k < nBands; k++)
                {
                    // LoopbackVariance uses |X[k]|^2 — by stationarity, the regressor power tracks
                    // the input power, and using the current frame avoids a one-iteration lag.
                    const float xPower = std::norm(X(k));
                    loopbackVariance(k) += lambda * (xPower - loopbackVariance(k));

                    // Filter-prediction (micEst = convolution of filter with regressor taps).
                    Complex micEst{0.f, 0.f};
                    for (int i = 0; i < L; i++)
                    {
                        micEst += binHistory(k, regSlots[i]) * filters(k, i);
                    }
                    const Complex err = X(k) - micEst;
                    const float pNew = std::norm(err);

                    // Min-variance output fallback: if the filter made things worse at this bin,
                    // bypass it. yOut plays the role of the residual; micEst-or-0 is the prediction.
                    const float pInput = std::norm(X(k));
                    const Complex yOut = (pNew < pInput) ? err : X(k);
                    E(k) = outputResidual ? yOut : (X(k) - yOut);

                    // NearendVariance with soft-floor at kNearendLimit * |micEst|^2 — keeps the
                    // adaptive step alive when |e| dips transiently to zero on well-cancelled bins.
                    const float pFloor = std::norm(micEst) * kNearendLimit;
                    nearendVariance(k) += lambda * (std::max(pNew, pFloor) - nearendVariance(k));

                    // Momentum & adaptive step.
                    const float Vu = loopbackVariance(k);
                    const float p = momentums(k) + Lf * coefficientVariance(k);
                    float q = p / (Lf * nearendVariance(k) + (Lf + 2.f) * p * Vu + 1e-30f);
                    const float qCap = 1.f / (Lf * Vu + 1e-30f);
                    if (q > qCap) { q = qCap; }
                    momentums(k) = std::max((1.f - q * Vu) * p, kMomentumFloor);

                    // NLMS-style filter update with adaptive step q.
                    const Complex W = q * err;
                    const Complex tap0change = W * std::conj(binHistory(k, regSlots[0]));
                    filters(k, 0) += tap0change;
                    for (int i = 1; i < L; i++)
                    {
                        filters(k, i) += W * std::conj(binHistory(k, regSlots[i]));
                    }

                    // CoefficientVariance tracks how much the first tap is moving per update —
                    // proxy for "is the filter still converging".
                    const float tap0Energy = std::norm(tap0change);
                    coefficientVariance(k) += lambda * (tap0Energy - coefficientVariance(k));
                }

                // Overwrite oldest slot with the fresh frame.
                binHistory.col(binHistoryIndex) = X;
                binHistoryIndex++;
                if (binHistoryIndex >= nHistory) { binHistoryIndex = 0; }

                synthesisFilterbank.process(E, outputHop);
                outputReadIndex = 0;
                samplesInBlock = 0;
            }

            output(n) = (outputReadIndex < bufferSize) ? outputHop(outputReadIndex++) : 0.f;
        }
    }

    // // onParametersChanged intentionally omitted: lambda is a fixed function of the filterbank rate
    // // (computed in the constructor), and P.regularization is read directly inside processAlgorithm.
    // void onParametersChanged()
    // {
    //     // lambda = bufferSize / tauSamples gives ~tauSamples time constant on the variance
    //     // trackers when tauSamples >> bufferSize. Clamp to [1e-6, 1.0].
    //     const float tauSamples = std::max(1.f, P.convergenceTimeMs * 1e-3f * C.sampleRate);
    //     float lam = static_cast<float>(bufferSize) / tauSamples;
    //     if (lam > 1.f) { lam = 1.f; }
    //     if (lam < 1e-6f) { lam = 1e-6f; }
    //     lambdaSmooth = lam;
    // }

    void resetVariables() final
    {
        inputHop.setZero();
        outputHop.setZero();
        X.setZero();
        E.setZero();
        binHistory.setZero();
        filters.setZero();
        loopbackVariance.setConstant(100);
        nearendVariance.setConstant(10);
        coefficientVariance.setZero();
        momentums.setConstant(1);
        binHistoryIndex = 0;
        samplesInBlock = 0;
        outputReadIndex = bufferSize;
    }

    size_t getDynamicSizeVariables() const final
    {
        return inputHop.getDynamicMemorySize() + outputHop.getDynamicMemorySize() + X.size() * sizeof(Complex) + E.size() * sizeof(Complex) +
               binHistory.size() * sizeof(Complex) + filters.size() * sizeof(Complex) + loopbackVariance.size() * sizeof(float) + nearendVariance.size() * sizeof(float) +
               coefficientVariance.size() * sizeof(float) + momentums.size() * sizeof(float);
    }

    int bufferSize;
    int nBands;
    int delayFrames;
    int nHistory;
    int binHistoryIndex;
    int samplesInBlock;
    int outputReadIndex;
    float lambdaSmooth;

    Eigen::ArrayXf inputHop;            // (bufferSize)
    Eigen::ArrayXf outputHop;           // (bufferSize)
    Eigen::ArrayXcf X;                  // (nBands) measurement spectrum
    Eigen::ArrayXcf E;                  // (nBands) residual or prediction spectrum
    Eigen::ArrayXXcf binHistory;        // (nBands, nHistory) ring of past measurement frames
    Eigen::ArrayXXcf filters;           // (nBands, kFilterLength) per-bin FIR taps
    Eigen::ArrayXf loopbackVariance;    // (nBands) smoothed |regressor|^2
    Eigen::ArrayXf nearendVariance;     // (nBands) smoothed max(|error|^2, kNearendLimit*|micEst|^2)
    Eigen::ArrayXf coefficientVariance; // (nBands) smoothed |tap0 change|^2
    Eigen::ArrayXf momentums;           // (nBands) momentum state

    friend BaseAlgorithm;
};
