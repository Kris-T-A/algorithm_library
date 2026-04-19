#include "adaptive_predictor/adaptive_predictor_kalman_time_domain.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

namespace
{

ArrayXf makeNoisySinusoid(int nSamples, float sampleRate, float freqHz, float toneAmp, float noiseAmp, unsigned seed)
{
    srand(seed);
    ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
    ArrayXf t = ArrayXf::LinSpaced(nSamples, 0.f, static_cast<float>(nSamples - 1) / sampleRate);
    ArrayXf tone = (2.f * static_cast<float>(M_PI) * freqHz * t).sin() * toneAmp;
    return tone + noise;
}

// Phase-continuous tone that steps from f1 (n < nStep) to f2 (n >= nStep), plus uniform noise.
ArrayXf makeStepFrequencySinusoid(int nSamples, float sampleRate, float f1, float f2, int nStep, float toneAmp, float noiseAmp, unsigned seed)
{
    srand(seed);
    ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
    ArrayXf signal(nSamples);
    double phase = 0.0;
    const double twoPi = 2.0 * 3.14159265358979323846;
    for (int n = 0; n < nSamples; n++)
    {
        const double f = (n < nStep) ? static_cast<double>(f1) : static_cast<double>(f2);
        signal(n) = toneAmp * static_cast<float>(std::sin(phase)) + noise(n);
        phase += twoPi * f / static_cast<double>(sampleRate);
        if (phase > twoPi) { phase -= twoPi; }
    }
    return signal;
}

float meanSquareTail(const ArrayXf &x, int tailSamples) { return x.tail(tailSamples).square().mean(); }

} // namespace

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(AdaptivePredictorKalman, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AdaptivePredictorKalmanTimeDomain>()); }

// Convergence test: feed sinusoid + noise. After convergence the residual at the tone frequency
// should be much weaker than the input energy (the periodic tone is removed, broadband noise remains).
TEST(AdaptivePredictorKalman, ConvergesAndReducesPeriodicEnergy)
{
    const float fs = 48000.f;
    const float toneFreq = 1000.f;
    const float toneAmp = 1.f;
    const float noiseAmp = 0.1f;
    const int nSamples = 24000; // 0.5s

    AdaptivePredictorKalmanTimeDomain::Coefficients c;
    c.sampleRate = fs;
    c.filterLength = 32;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanTimeDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanTimeDomain algo(c);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 20.f;
    algo.setParameters(p);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, toneFreq, toneAmp, noiseAmp, 0);
    ArrayXf output(nSamples);
    algo.process(input, output);

    const int tail = nSamples / 4;
    const float inputPowerTail = meanSquareTail(input, tail);
    const float residualPowerTail = meanSquareTail(output, tail);

    fmt::print("Input power (tail): {}\n", inputPowerTail);
    fmt::print("Residual power (tail): {}\n", residualPowerTail);

    EXPECT_LT(residualPowerTail * 10.f, inputPowerTail);
}

// Output-mode complementarity: residual + prediction ≈ input, sample-by-sample,
// when both predictors are run with identical coefficients/parameters on identical input.
TEST(AdaptivePredictorKalman, ResidualPlusPredictionEqualsInput)
{
    const float fs = 48000.f;
    const int nSamples = 4000;

    AdaptivePredictorKalmanTimeDomain::Coefficients cR;
    cR.sampleRate = fs;
    cR.filterLength = 16;
    cR.decorrelationDelay = 1;
    cR.outputMode = AdaptivePredictorKalmanTimeDomain::Coefficients::RESIDUAL;

    AdaptivePredictorKalmanTimeDomain::Coefficients cP = cR;
    cP.outputMode = AdaptivePredictorKalmanTimeDomain::Coefficients::PREDICTION;

    AdaptivePredictorKalmanTimeDomain algoR(cR);
    AdaptivePredictorKalmanTimeDomain algoP(cP);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, 500.f, 1.f, 0.2f, 42);
    ArrayXf residual(nSamples), prediction(nSamples);

    algoR.process(input, residual);
    algoP.process(input, prediction);

    ArrayXf reconstructed = residual + prediction;
    float err = (reconstructed - input).abs().maxCoeff();
    fmt::print("Max |residual + prediction - input| = {}\n", err);
    EXPECT_LT(err, 1e-5f);
}

// Parameter-change responsiveness + Kalman-specific nonstationary tracking:
// Signal steps from f1 to f2 midway. Fast convergence re-converges quickly after the step;
// slow convergence does not within the signal duration. Measure residual in the tail.
TEST(AdaptivePredictorKalman, ConvergenceTimeAffectsTracking)
{
    const float fs = 48000.f;
    const float f1 = 1000.f, f2 = 1200.f;
    const int nSamples = 48000;  // 1s
    const int nStep = 24000;     // step at 0.5s
    const float toneAmp = 1.f, noiseAmp = 0.05f;

    AdaptivePredictorKalmanTimeDomain::Coefficients c;
    c.sampleRate = fs;
    c.filterLength = 32;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanTimeDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanTimeDomain algo(c);

    ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, f1, f2, nStep, toneAmp, noiseAmp, 0);
    ArrayXf outputFast(nSamples), outputSlow(nSamples);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 10.f; // fast
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputFast);

    p.convergenceTimeMs = 5000.f; // slow
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputSlow);

    // Tail (last quarter): fast should have re-converged after the step, slow should not.
    const int tail = nSamples / 4;
    float tailFast = outputFast.tail(tail).square().mean();
    float tailSlow = outputSlow.tail(tail).square().mean();
    fmt::print("Tail residual: fast={}, slow={}\n", tailFast, tailSlow);
    EXPECT_LT(tailFast, tailSlow);
}

// Absolute tracking: with a reasonable convergenceTimeMs, residual re-converges within a generous
// window after a frequency step. Exercises the reason Kalman is here (Q > 0).
TEST(AdaptivePredictorKalman, TracksFrequencyStep)
{
    const float fs = 48000.f;
    const float f1 = 1000.f, f2 = 1200.f;
    const int nSamples = 48000;
    const int nStep = 24000;
    const float toneAmp = 1.f, noiseAmp = 0.05f;

    AdaptivePredictorKalmanTimeDomain::Coefficients c;
    c.sampleRate = fs;
    c.filterLength = 32;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanTimeDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanTimeDomain algo(c);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 20.f;
    algo.setParameters(p);

    ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, f1, f2, nStep, toneAmp, noiseAmp, 0);
    ArrayXf output(nSamples);
    algo.process(input, output);

    // Pre-step converged residual (last quarter before the step).
    const int preLen = nSamples / 4;
    const int preStart = nStep - preLen;
    float prePower = output.segment(preStart, preLen).square().mean();

    // Post-step converged residual (last quarter of the signal, well after the step).
    const int postLen = nSamples / 4;
    const int postStart = nSamples - postLen;
    float postPower = output.segment(postStart, postLen).square().mean();

    fmt::print("Pre-step residual power: {}\n", prePower);
    fmt::print("Post-step residual power: {}\n", postPower);

    // Post-step converged power should be comparable to pre-step — within a loose 3x factor.
    EXPECT_LT(postPower, 3.f * prePower);
}
