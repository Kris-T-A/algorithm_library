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

} // namespace

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(AdaptivePredictorKalman, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AdaptivePredictorKalmanTimeDomain>()); }

// Output-mode complementarity: residual + prediction ≈ input, sample-by-sample,
// when both predictors are run with identical coefficients/parameters on identical input.
TEST(AdaptivePredictorKalman, ResidualPlusPredictionEqualsInput)
{
    const float fs = 48000.f;
    const int nSamples = 4000;

    AdaptivePredictorKalmanTimeDomain::Coefficients cR;
    cR.sampleRate = fs;
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
