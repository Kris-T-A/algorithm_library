#pragma once
#include "interface/interface.h"

// Adaptive predictor for removing (or isolating) the periodic component of a single-channel
// audio signal in real time. Topologically an adaptive line enhancer (ALE): the regressor is a
// delayed copy of the input, so the filter can only learn periodic / narrowband content.
//
// The public interface is algorithm-agnostic — algorithmType selects the underlying estimator
// (currently NLMS, time domain). Parameters are intent-based (convergenceTimeMs, regularization)
// so that future estimators (e.g. Kalman) can be added without an API break.
//
// author: Kristian Timm Andersen

struct AdaptivePredictorConfiguration
{
    using Input = I::Real;
    using Output = O::Real;

    struct Coefficients
    {
        float sampleRate = 48000.f;
        int filterLength = 64;       // predictor order N (number of FIR taps)
        int decorrelationDelay = 1;  // Delta in samples; >= 1

        enum AlgorithmType { NLMS_TIME_DOMAIN, KALMAN_TIME_DOMAIN };
        AlgorithmType algorithmType = NLMS_TIME_DOMAIN;

        enum OutputMode { RESIDUAL, PREDICTION };
        OutputMode outputMode = RESIDUAL;

        DEFINE_TUNABLE_ENUM(AlgorithmType, {{NLMS_TIME_DOMAIN, "NLMS time domain"}, {KALMAN_TIME_DOMAIN, "Kalman time domain"}})
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
