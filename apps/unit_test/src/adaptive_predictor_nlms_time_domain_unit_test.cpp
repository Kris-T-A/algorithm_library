#include "adaptive_predictor/adaptive_predictor_nlms_time_domain.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

namespace
{

// Helper: generate a noisy sinusoid input (periodic + broadband noise).
ArrayXf makeNoisySinusoid(int nSamples, float sampleRate, float freqHz, float toneAmp, float noiseAmp, unsigned seed)
{
    srand(seed);
    ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
    ArrayXf t = ArrayXf::LinSpaced(nSamples, 0.f, static_cast<float>(nSamples - 1) / sampleRate);
    ArrayXf tone = (2.f * static_cast<float>(M_PI) * freqHz * t).sin() * toneAmp;
    return tone + noise;
}

float meanSquareTail(const ArrayXf &x, int tailSamples)
{
    return x.tail(tailSamples).square().mean();
}

} // namespace

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(AdaptivePredictor, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AdaptivePredictorNLMSTimeDomain>()); }

// Convergence test: feed sinusoid + noise. Steady-state residual at the tone frequency
// should be much weaker than the input at the tone frequency (residual energy < input energy
// after convergence, because the periodic tone is removed).
TEST(AdaptivePredictor, ConvergesAndReducesPeriodicEnergy)
{
    const float fs = 48000.f;
    const float toneFreq = 1000.f;
    const float toneAmp = 1.f;
    const float noiseAmp = 0.1f;
    const int nSamples = 24000; // 0.5s

    AdaptivePredictorNLMSTimeDomain::Coefficients c;
    c.sampleRate = fs;
    c.filterLength = 32;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorNLMSTimeDomain::Coefficients::RESIDUAL;
    AdaptivePredictorNLMSTimeDomain algo(c);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 20.f;
    algo.setParameters(p);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, toneFreq, toneAmp, noiseAmp, 0);
    ArrayXf output(nSamples);
    algo.process(input, output);

    // After convergence, residual should be dominated by noise (~ noiseAmp^2 / 3 for uniform Random)
    // and the tone (amp 1, contributes 0.5 power) should be largely removed.
    const int tail = nSamples / 4; // last quarter = converged
    const float inputPowerTail = meanSquareTail(input, tail);
    const float residualPowerTail = meanSquareTail(output, tail);

    fmt::print("Input power (tail): {}\n", inputPowerTail);
    fmt::print("Residual power (tail): {}\n", residualPowerTail);

    // Tone alone contributes ~0.5 to the input power; noise contributes ~noiseAmp^2/3 ≈ 0.0033.
    // Residual should drop by at least 10x (10 dB).
    EXPECT_LT(residualPowerTail * 10.f, inputPowerTail);
}

// Output-mode complementarity: residual + prediction ≈ input, sample-by-sample,
// when both predictors are run with identical coefficients/parameters on identical input.
TEST(AdaptivePredictor, ResidualPlusPredictionEqualsInput)
{
    const float fs = 48000.f;
    const int nSamples = 4000;

    AdaptivePredictorNLMSTimeDomain::Coefficients cR;
    cR.sampleRate = fs;
    cR.filterLength = 16;
    cR.decorrelationDelay = 1;
    cR.outputMode = AdaptivePredictorNLMSTimeDomain::Coefficients::RESIDUAL;

    AdaptivePredictorNLMSTimeDomain::Coefficients cP = cR;
    cP.outputMode = AdaptivePredictorNLMSTimeDomain::Coefficients::PREDICTION;

    AdaptivePredictorNLMSTimeDomain algoR(cR);
    AdaptivePredictorNLMSTimeDomain algoP(cP);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, 500.f, 1.f, 0.2f, 42);
    ArrayXf residual(nSamples), prediction(nSamples);

    algoR.process(input, residual);
    algoP.process(input, prediction);

    ArrayXf reconstructed = residual + prediction;
    float err = (reconstructed - input).abs().maxCoeff();
    fmt::print("Max |residual + prediction - input| = {}\n", err);
    EXPECT_LT(err, 1e-5f);
}

// Parameter-change responsiveness: changing convergenceTimeMs changes adaptation speed.
TEST(AdaptivePredictor, ConvergenceTimeAffectsAdaptation)
{
    const float fs = 48000.f;
    const int nSamples = 4800; // 100ms

    AdaptivePredictorNLMSTimeDomain::Coefficients c;
    c.sampleRate = fs;
    c.filterLength = 32;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorNLMSTimeDomain::Coefficients::RESIDUAL;
    AdaptivePredictorNLMSTimeDomain algo(c);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, 1000.f, 1.f, 0.1f, 7);
    ArrayXf outputFast(nSamples), outputSlow(nSamples);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 5.f;
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputFast);

    p.convergenceTimeMs = 500.f;
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputSlow);

    // Early in the signal, the slow predictor hasn't converged so its residual still contains
    // most of the tone energy; the fast predictor's residual is much smaller.
    const int head = nSamples / 4;
    float earlyFast = outputFast.head(head).square().mean();
    float earlySlow = outputSlow.head(head).square().mean();
    fmt::print("Early residual power: fast={}, slow={}\n", earlyFast, earlySlow);
    EXPECT_LT(earlyFast, earlySlow);
}
