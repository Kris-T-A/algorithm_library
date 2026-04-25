#pragma once
#include "algorithm_library/adaptive_predictor.h"
#include "algorithm_library/fft.h"
#include "algorithm_library/filterbank.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include <complex>
#include <vector>

// Frequency-domain random-walk Kalman adaptive line enhancer built on top of
// FilterbankAnalysisWOLA + FilterbankSynthesisWOLA.
//
// Each hop (bufferSize samples), the analysis filterbank produces a complex bin frame X[k].
// Per bin, an AR(p) predictor with p = kNumTaps complex weights models the bin frame as a
// linear combination of past bin frames. Under stationarity and with a proper WOLA window,
// bins are approximately decorrelated so the joint weight covariance is approximately
// block-diagonal; each bin k is its own p-dim complex Kalman with state w_k (length p) and
// Hermitian covariance P_k (p x p):
//
//   regressor:    u_k[m] = [X[k, m - delta], X[k, m - delta - 1], ..., X[k, m - delta - p + 1]]^T
//   measurement:  X[k, m] = u_k[m]^T * w_k[m] + r_k[m],     r_k ~ CN(0, R)
//   state model:  w_k[m+1] = w_k[m] + v_k[m],               v_k ~ CN(0, Q*I)
//
// Update per hop, per bin:
//   P += Q*I
//   yhat = u^T * w
//   e    = X[k] - yhat                       (innovation)
//   Pu   = P * conj(u)
//   S    = real(u^T * Pu) + R                (S is real-positive since P is Hermitian)
//   K    = Pu / S
//   w   += K * e
//   P   -= K * Pu^H                          (rank-1 Hermitian update; Pu Pu^H / S)
//
// Output is synthesized from e (residual) or yhat (prediction) via the synthesis filterbank.
//
// Why p > 1:
//   1. Bins 0 (DC) and N/2 (Nyquist) are real-valued (FFT of real input). A p=1 complex
//      predictor collapses to a real AR(1), which can only suppress a real sinusoid at per-hop
//      angle omega by a factor of sin(omega); for a low-frequency tone leaked into DC this
//      leaves essentially no suppression. AR(p>=2) on a real sequence can fully cancel a
//      single real sinusoid (AR(2) has complex-conjugate poles at the sinusoid's frequency).
//   2. A single complex coefficient per bin can only cancel one complex exponential. Multiple
//      leaked tones or a chirp sweeping past a bin need higher order to cancel.
//
// Filterbank sizing (unchanged from the p=1 version):
//   fftSize    = FFTConfiguration::getValidFFTSize(filterLength)
//   bufferSize = fftSize / 4                           (75% overlap, hop = fftSize/4)
//   nBands     = fftSize / 2 + 1
//
// Decorrelation delay: delayFrames = max(2, ceil(decorrelationDelay / bufferSize)). The
// minimum of 2 hops keeps the newest regressor tap from sharing more than 50% time samples
// with the target frame; combined with Hann-on-Hann windowing this drops the WOLA-induced
// cross-correlation between target and regressor below ~17%, below what the Kalman can
// exploit at the default regularization. Without it, white noise gets spuriously suppressed
// by ~8 dB because 75% overlap leaves 60%+ correlation between adjacent bin frames. The
// freshest regressor tap is delayFrames hops back; successive taps extend one hop each.
//
// author: Kristian Timm Andersen
class AdaptivePredictorKalmanFreqDomain : public AlgorithmImplementation<AdaptivePredictorConfiguration, AdaptivePredictorKalmanFreqDomain>
{
  public:
    AdaptivePredictorKalmanFreqDomain(Coefficients c = {.algorithmType = Coefficients::KALMAN_FREQ_DOMAIN})
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
        const float initP = 1e-4f;
        covariance.assign(nBands, initP * CovMat::Identity());

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
    using WeightVec = Eigen::Matrix<Complex, kNumTaps, 1>;
    using CovMat = Eigen::Matrix<Complex, kNumTaps, kNumTaps>;

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
        const float Q = processNoise;
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
                    WeightVec u;
                    for (int i = 0; i < kNumTaps; i++)
                    {
                        u(i) = binHistory(k, regSlots[i]);
                    }

                    CovMat &Pk = covariance[k];
                    Pk.diagonal().array() += Q;

                    const WeightVec Pu = Pk * u.conjugate();

                    Complex yhat{0.f, 0.f};
                    Complex Sc{0.f, 0.f};
                    for (int i = 0; i < kNumTaps; i++)
                    {
                        yhat += u(i) * weights(k, i);
                        Sc += u(i) * Pu(i);
                    }
                    const Complex e = X(k, 0) - yhat;
                    const float S = Sc.real() + R;
                    const WeightVec K = Pu / S;

                    for (int i = 0; i < kNumTaps; i++)
                    {
                        weights(k, i) += K(i) * e;
                    }

                    // Rank-1 Hermitian update: P -= K * Pu^H = (Pu * Pu^H) / S
                    Pk.noalias() -= K * Pu.adjoint();

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
        const float ratio = static_cast<float>(bufferSize) / tauSamples;
        const float bs = static_cast<float>(bufferSize);
        float q = P.regularization * ratio * ratio * bs * bs * bs; // experimentally tuned for bufferSize = 32. TODO: test if other bufferSizes have good performance
        if (q < 1e-20f) { q = 1e-20f; }
        if (q > 1e-2f) { q = 1e-2f; }
        processNoise = q;
    }

    void resetVariables() final
    {
        inputHop.setZero();
        outputHop.setZero();
        X.setZero();
        E.setZero();
        binHistory.setZero();
        weights.setZero();
        const float initP = 1e-4f;
        for (auto &Pk : covariance)
        {
            Pk = initP * CovMat::Identity();
        }
        binHistoryIndex = 0;
        samplesInBlock = 0;
        outputReadIndex = bufferSize;
    }

    size_t getDynamicSizeVariables() const final
    {
        return inputHop.getDynamicMemorySize() + outputHop.getDynamicMemorySize() + X.size() * sizeof(Complex) + E.size() * sizeof(Complex) +
               binHistory.size() * sizeof(Complex) + weights.size() * sizeof(Complex) + covariance.size() * sizeof(CovMat);
    }

    int bufferSize;
    int nBands;
    int delayFrames;
    int nHistory;
    int binHistoryIndex;
    int samplesInBlock;
    int outputReadIndex;
    float processNoise;

    Eigen::ArrayXXf inputHop;       // (bufferSize, 1)
    Eigen::ArrayXXf outputHop;      // (bufferSize, 1)
    Eigen::ArrayXXcf X;             // (nBands, 1) measurement spectrum
    Eigen::ArrayXXcf E;             // (nBands, 1) residual or prediction spectrum
    Eigen::ArrayXXcf binHistory;    // (nBands, nHistory) ring of past measurement frames
    Eigen::ArrayXXcf weights;       // (nBands, kNumTaps) per-bin AR(p) weights
    std::vector<CovMat> covariance; // (nBands) per-bin Hermitian covariance

    friend BaseAlgorithm;
};
