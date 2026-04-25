#pragma once
#include "interface/interface.h"

// Adaptive predictor for removing (or isolating) the periodic component of a single-channel
// audio signal in real time. Topologically an adaptive line enhancer (ALE): the regressor is a
// delayed copy of the input, so the filter can only learn periodic / narrowband content.
//
// The public interface is algorithm-agnostic — algorithmType selects the underlying estimator:
//   NLMS_TIME_DOMAIN   — time-domain normalized LMS, O(N) per sample, cheapest.
//   KALMAN_TIME_DOMAIN — time-domain full-matrix random-walk Kalman, O(N^2) per sample; tracks
//                        both nonstationary signals and correlated regressors better than NLMS.
//   NLMS_FREQ_DOMAIN   — frequency-domain NLMS built on FilterbankAnalysisWOLA/Synthesis, a
//                        per-bin complex NLMS with p taps; cheapest freq-domain variant,
//                        introduces analysis+synthesis latency.
//   KALMAN_FREQ_DOMAIN — frequency-domain Kalman built on FilterbankAnalysisWOLA/Synthesis, one
//                        per-bin complex Kalman with p taps; cheapest of the time-domain-quality
//                        variants at large N, introduces analysis+synthesis latency.
//   NLMS_MOMENTUM       — frequency-domain NLMS with momentum-based adaptive step; ignores
//                         convergenceTimeMs (fully data-driven step size).
//   NLMS_MOMENTUM_TIME_DOMAIN — time-domain analogue of NLMS_MOMENTUM: same per-sample momentum/q
//                         loop on a single real FIR of N taps. No analysis/synthesis latency.
//
// Parameters are intent-based (convergenceTimeMs, regularization) so each implementation maps
// them to its own native rate-of-change knobs (NLMS: step size + leakage; Kalman: process-noise
// covariance) without breaking the API.
//
// author: Kristian Timm Andersen

struct AdaptivePredictorConfiguration
{
    using Input = I::Real;
    using Output = O::Real;

    struct Coefficients
    {
        float sampleRate = 48000.f;
        int filterLength = 128;      // time-domain variants: predictor order N (number of FIR taps).
                                     // freq-domain variant: WOLA frame size (rounded up to a valid
                                     // FFT size). Hop is frameSize/2 (50% overlap).
        int decorrelationDelay = 32; // Delta in samples; >= 1. Chosen so that during a moderate
                                     // frequency step the residual magnitude stays close to the
                                     // input magnitude (ratio ~1); shorter delays let the predictor
                                     // partially cancel transients, longer delays let it
                                     // anti-correlate with the new signal and amplify the residual.
                                     // Freq-domain variant rounds up to at least 2 hops so the
                                     // regressor/target share <50% of their input samples, below
                                     // what the Kalman can exploit given the regularization —
                                     // avoids the overlap-induced spurious white-noise suppression.

        enum AlgorithmType { NLMS_TIME_DOMAIN, KALMAN_TIME_DOMAIN, NLMS_FREQ_DOMAIN, KALMAN_FREQ_DOMAIN, NLMS_MOMENTUM, NLMS_MOMENTUM_TIME_DOMAIN };
        AlgorithmType algorithmType = NLMS_TIME_DOMAIN;

        enum OutputMode { RESIDUAL, PREDICTION };
        OutputMode outputMode = RESIDUAL;

        DEFINE_TUNABLE_ENUM(AlgorithmType, {{NLMS_TIME_DOMAIN, "NLMS time domain"},
                                            {KALMAN_TIME_DOMAIN, "Kalman time domain"},
                                            {NLMS_FREQ_DOMAIN, "NLMS frequency domain"},
                                            {KALMAN_FREQ_DOMAIN, "Kalman frequency domain"},
                                            {NLMS_MOMENTUM, "NLMS momentum frequency domain"},
                                            {NLMS_MOMENTUM_TIME_DOMAIN, "NLMS momentum time domain"}})
        DEFINE_TUNABLE_ENUM(OutputMode, {{RESIDUAL, "Residual"}, {PREDICTION, "Prediction"}})
        DEFINE_TUNABLE_COEFFICIENTS(sampleRate, filterLength, decorrelationDelay, algorithmType, outputMode)
    };

    struct Parameters
    {
        float convergenceTimeMs = 50.f; // approximate adaptation time constant in milliseconds
        float regularization = 1e-6f;   // small positive value preventing instability under near-silent inputs
        DEFINE_TUNABLE_PARAMETERS(convergenceTimeMs, regularization)
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(100); } // arbitrary length

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(input.rows()); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() > 0); }
};

class AdaptivePredictor : public Algorithm<AdaptivePredictorConfiguration>
{
  public:
    AdaptivePredictor() = default;
    AdaptivePredictor(const Coefficients &c);
};
