#include "adaptive_predictor/adaptive_predictor_kalman_freq_domain.h"
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

ArrayXf makeStepFrequencySinusoid(int nSamples, float sampleRate, float f1, float f2, int nStep, float toneAmp, float noiseAmp, unsigned seed)
{
    srand(seed);
    ArrayXf noise = ArrayXf::Random(nSamples) * noiseAmp;
    ArrayXf signal(nSamples);
    double phase = 0.0;
    const double twoPi = 2.0 * M_PI;
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

TEST(AdaptivePredictorKalmanFreq, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AdaptivePredictorKalmanFreqDomain>()); }

// Convergence: feed sinusoid + noise. The tail residual should be much weaker than the input,
// because the periodic content has been predicted away. Use a long signal so the freq-domain
// variant has time to accumulate frames and converge per bin.
TEST(AdaptivePredictorKalmanFreq, ConvergesAndReducesPeriodicEnergy)
{
    const float fs = 48000.f;
    const float toneFreq = 1000.f;
    const float toneAmp = 1.f;
    const float noiseAmp = 0.1f;
    const int nSamples = 48000; // 1s

    AdaptivePredictorKalmanFreqDomain::Coefficients c;
    c.sampleRate = fs;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanFreqDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanFreqDomain algo(c);

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

// Output-mode complementarity: residual + prediction = input (shifted by the WOLA analysisFilterbank +
// synthesisFilterbank latency). Compare the tail of (residual + prediction) against the head of the input.
TEST(AdaptivePredictorKalmanFreq, ResidualPlusPredictionEqualsInput)
{
    const float fs = 48000.f;
    const int nSamples = 8000;

    AdaptivePredictorKalmanFreqDomain::Coefficients cR;
    cR.sampleRate = fs;
    cR.decorrelationDelay = 1;
    cR.outputMode = AdaptivePredictorKalmanFreqDomain::Coefficients::RESIDUAL;

    AdaptivePredictorKalmanFreqDomain::Coefficients cP = cR;
    cP.outputMode = AdaptivePredictorKalmanFreqDomain::Coefficients::PREDICTION;

    AdaptivePredictorKalmanFreqDomain algoR(cR);
    AdaptivePredictorKalmanFreqDomain algoP(cP);

    ArrayXf input = makeNoisySinusoid(nSamples, fs, 500.f, 1.f, 0.2f, 42);
    ArrayXf residual(nSamples), prediction(nSamples);

    algoR.process(input, residual);
    algoP.process(input, prediction);

    // Latency is deterministic but depends on windows; measure by cross-correlating the sum
    // against the input and picking the best offset.
    ArrayXf reconstructed = residual + prediction;

    // Search a reasonable latency range: up to 4 * filterLength samples (analysisFilterbank + synthesisFilterbank + hop).
    const int maxLatency = 4 * 64;
    int bestLatency = -1;
    float bestErr = std::numeric_limits<float>::max();
    for (int lat = 0; lat <= maxLatency; lat++)
    {
        const int len = nSamples - lat;
        if (len <= 0) { break; }
        float err = (reconstructed.tail(len) - input.head(len)).abs().mean();
        if (err < bestErr)
        {
            bestErr = err;
            bestLatency = lat;
        }
    }

    fmt::print("Best latency = {} samples, mean abs err = {}\n", bestLatency, bestErr);
    EXPECT_LT(bestErr, 1e-4f);
}

// Convergence-time parameter responsiveness: frequency step mid-signal; fast convergence
// re-adapts, slow one doesn't.
TEST(AdaptivePredictorKalmanFreq, ConvergenceTimeAffectsTracking)
{
    const float fs = 48000.f;
    const float f1 = 1000.f, f2 = 1200.f;
    const int nSamples = 96000; // 2s
    const int nStep = 48000;
    const float toneAmp = 1.f, noiseAmp = 0.05f;

    AdaptivePredictorKalmanFreqDomain::Coefficients c;
    c.sampleRate = fs;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanFreqDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanFreqDomain algo(c);

    ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, f1, f2, nStep, toneAmp, noiseAmp, 0);
    ArrayXf outputFast(nSamples), outputSlow(nSamples);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 20.f;
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputFast);

    p.convergenceTimeMs = 500000.f; // labeled slow enough that even the aggressive-Q freq-domain mapping can't fully adapt within the test window
    algo.setParameters(p);
    algo.reset();
    algo.process(input, outputSlow);

    const int tail = nSamples / 4;
    float tailFast = outputFast.tail(tail).square().mean();
    float tailSlow = outputSlow.tail(tail).square().mean();
    fmt::print("Tail residual: fast={}, slow={}\n", tailFast, tailSlow);
    EXPECT_LT(tailFast, tailSlow);
}

// Frequency-step tracking: with a reasonable convergenceTimeMs, the residual recovers after the
// step within a loose upper bound (freq-domain specific motivation — nonstationary tracking).
TEST(AdaptivePredictorKalmanFreq, TracksFrequencyStep)
{
    const float fs = 48000.f;
    const float f1 = 1000.f, f2 = 1200.f;
    const int nSamples = 96000;
    const int nStep = 48000;
    const float toneAmp = 1.f, noiseAmp = 0.05f;

    AdaptivePredictorKalmanFreqDomain::Coefficients c;
    c.sampleRate = fs;
    c.decorrelationDelay = 1;
    c.outputMode = AdaptivePredictorKalmanFreqDomain::Coefficients::RESIDUAL;
    AdaptivePredictorKalmanFreqDomain algo(c);

    auto p = algo.getParameters();
    p.convergenceTimeMs = 20.f;
    algo.setParameters(p);

    ArrayXf input = makeStepFrequencySinusoid(nSamples, fs, f1, f2, nStep, toneAmp, noiseAmp, 0);
    ArrayXf output(nSamples);
    algo.process(input, output);

    const int preLen = nSamples / 4;
    const int preStart = nStep - preLen;
    float prePower = output.segment(preStart, preLen).square().mean();

    const int postLen = nSamples / 4;
    const int postStart = nSamples - postLen;
    float postPower = output.segment(postStart, postLen).square().mean();

    fmt::print("Pre-step residual power: {}\n", prePower);
    fmt::print("Post-step residual power: {}\n", postPower);

    EXPECT_LT(postPower, 3.f * prePower);
}
