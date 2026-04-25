#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "algorithm_library/fft.h"
#include "algorithm_library/filterbank.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include <complex>

// Frequency-domain NLMS adaptive line enhancer built on top of
// FilterbankAnalysisWOLA + FilterbankSynthesisWOLA.
//
// Each hop (bufferSize samples), the analysis filterbank produces a complex bin frame X[k].
// Per bin, an AR(p) predictor with p = kNumTaps complex weights models the bin frame as a
// linear combination of past bin frames:
//
//   regressor:    u_k[m] = [X[k, m - delta], X[k, m - delta - 1], ..., X[k, m - delta - p + 1]]^T
//   prediction:   yhat   = u_k^T * w_k
//   residual:     e      = X[k, m] - yhat
//   NLMS update:  w_k <- (1 - leakage) * w_k + (mu / (u_k^H u_k + reg)) * conj(u_k) * e
//
// The conjugate on u_k comes from the Wirtinger gradient of |e|^2 with respect to w_k*.
//
// Filterbank sizing matches the Kalman freq-domain variant:
//   fftSize    = FFTConfiguration::getValidFFTSize(filterLength)
//   bufferSize = fftSize / 4                           (75% overlap, hop = fftSize/4)
//   nBands     = fftSize / 2 + 1
//
// Decorrelation delay: delayFrames = max(2, ceil(decorrelationDelay / bufferSize)) — same
// reasoning as the Kalman freq-domain variant: below 2 hops the regressor and target share
// more than 50% of their input samples through the WOLA overlap, which leaks broadband
// cross-correlation that NLMS would then spuriously cancel.
//
// Why p > 1 per bin: a single complex tap can only cancel one complex exponential per bin,
// and bins 0 (DC) and N/2 (Nyquist) are real-valued so a p=1 predictor collapses to a real
// AR(1) with vanishing suppression at low per-hop angles. See adaptive_predictor_kalman_freq_domain.h
// for the full argument — the same geometry applies here.
//
// Convergence-time mapping: in time-domain NLMS, mu = N / tauSamples drives an ~tauSamples
// time constant. Here the filter updates once per hop with p taps, so the analogous mapping
// is mu = p / tauHops = p * bufferSize / tauSamples, clipped to [1e-6, 0.5]. Leakage is a
// small multiple of mu to keep inactive bins' weights from drifting on noise alone.
//
// author: Kristian Timm Andersen
class AdaptivePredictorNLMSFreqDomain : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorNLMSFreqDomain>
{
  public:
    AdaptivePredictorNLMSFreqDomain(Coefficients c = {.algorithmType = Coefficients::NLMS_FREQ_DOMAIN})
        : BaseAlgorithm{c}, analysisFilterbank(makeFilterbankCoefficients(c)), synthesisFilterbank(makeFilterbankCoefficients(c))
    {
        const FilterbankConfiguration::Coefficients fc = makeFilterbankCoefficients(c);
        bufferSize = fc.bufferSize;
        nBands = fc.nBands;
        delayFrames = std::max(2, (std::max(1, c.decorrelationDelay) + bufferSize - 1) / bufferSize);
        nHistory = delayFrames + kNumTaps - 1;

        inputHop.setZero(bufferSize, 1);
        outputHop.setZero(bufferSize, 1);
        X.setZero(nBands, 1);
        E.setZero(nBands, 1);
        binHistory.setZero(nBands, nHistory);
        weights.setZero(nBands, kNumTaps);

        binHistoryIndex = 0;
        samplesInBlock = 0;
        outputReadIndex = bufferSize;

        onParametersChanged();
    }

    FilterbankAnalysisWOLA analysisFilterbank;
    FilterbankSynthesisWOLA synthesisFilterbank;
    DEFINE_MEMBER_ALGORITHMS(analysisFilterbank, synthesisFilterbank)

  private:
    static constexpr int kNumTaps = 4;
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
        const float muLocal = mu;
        const float decay = 1.f - leakage;
        const float R = P.regularization;

        for (Eigen::Index n = 0; n < input.rows(); n++)
        {
            inputHop(samplesInBlock++, 0) = input(n);

            if (samplesInBlock == bufferSize)
            {
                analysisFilterbank.process(inputHop, X);

                // Regressor slot offsets (newest tap first) from the oldest-slot pointer.
                // binHistory[binHistoryIndex] holds the oldest frame; taps walk forward.
                int regSlots[kNumTaps];
                for (int i = 0; i < kNumTaps; i++)
                {
                    int idx = binHistoryIndex + (kNumTaps - 1 - i);
                    if (idx >= nHistory) { idx -= nHistory; }
                    regSlots[i] = idx;
                }

                for (int k = 0; k < nBands; k++)
                {
                    Complex u[kNumTaps];
                    for (int i = 0; i < kNumTaps; i++)
                    {
                        u[i] = binHistory(k, regSlots[i]);
                    }

                    Complex yhat{0.f, 0.f};
                    float uEnergy = 0.f;
                    for (int i = 0; i < kNumTaps; i++)
                    {
                        yhat += u[i] * weights(k, i);
                        uEnergy += std::norm(u[i]);
                    }
                    const Complex e = X(k, 0) - yhat;
                    const float gain = muLocal / (uEnergy + R);

                    for (int i = 0; i < kNumTaps; i++)
                    {
                        weights(k, i) = decay * weights(k, i) + gain * std::conj(u[i]) * e;
                    }

                    E(k, 0) = outputResidual ? e : yhat;
                }

                // Overwrite the oldest slot with the freshly measured bin frame and advance.
                binHistory.col(binHistoryIndex) = X.col(0);
                binHistoryIndex++;
                if (binHistoryIndex >= nHistory) { binHistoryIndex = 0; }

                synthesisFilterbank.process(E, outputHop);
                outputReadIndex = 0;
                samplesInBlock = 0;
            }

            output(n) = (outputReadIndex < bufferSize) ? outputHop(outputReadIndex++, 0) : 0.f;
        }
    }

    void onParametersChanged()
    {
        const float tauSamples = std::max(1.f, P.convergenceTimeMs * 1e-3f * C.sampleRate);
        float bs = static_cast<float>(bufferSize);
        float m = static_cast<float>(kNumTaps) * bs * bs / tauSamples; // experimentally tuned
        if (m > 1.0f) { m = 1.0f; }
        if (m < 1e-6f) { m = 1e-6f; }
        mu = m;
        leakage = mu * 1e-5f;
    }

    void resetVariables() final
    {
        inputHop.setZero();
        outputHop.setZero();
        X.setZero();
        E.setZero();
        binHistory.setZero();
        weights.setZero();
        binHistoryIndex = 0;
        samplesInBlock = 0;
        outputReadIndex = bufferSize;
    }

    size_t getDynamicSizeVariables() const final
    {
        return inputHop.getDynamicMemorySize() + outputHop.getDynamicMemorySize() + X.size() * sizeof(Complex) + E.size() * sizeof(Complex) +
               binHistory.size() * sizeof(Complex) + weights.size() * sizeof(Complex);
    }

    int bufferSize;
    int nBands;
    int delayFrames;
    int nHistory;
    int binHistoryIndex;
    int samplesInBlock;
    int outputReadIndex;
    float mu;
    float leakage;

    Eigen::ArrayXXf inputHop;    // (bufferSize, 1)
    Eigen::ArrayXXf outputHop;   // (bufferSize, 1)
    Eigen::ArrayXXcf X;          // (nBands, 1) measurement spectrum
    Eigen::ArrayXXcf E;          // (nBands, 1) residual or prediction spectrum
    Eigen::ArrayXXcf binHistory; // (nBands, nHistory) ring of past measurement frames
    Eigen::ArrayXXcf weights;    // (nBands, kNumTaps) per-bin AR(p) weights

    friend BaseAlgorithm;
};
