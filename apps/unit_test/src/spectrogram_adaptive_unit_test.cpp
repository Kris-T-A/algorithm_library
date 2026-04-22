#include "spectrogram_adaptive/spectrogram_adaptive_envelope.h"
#include "spectrogram_adaptive/spectrogram_adaptive_full_resolution.h"
#include "spectrogram_adaptive/spectrogram_adaptive_moving.h"
#include "spectrogram_adaptive/spectrogram_adaptive_upscale.h"
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(SpectrogramAdaptive, InterfaceEnvelope) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveEnvelope>()); }

TEST(SpectrogramAdaptive, InterfaceFullResolution) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveFullResolution>()); }

TEST(SpectrogramAdaptive, InterfaceUpscale) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveUpscale>()); }

TEST(SpectrogramAdaptive, InterfaceZeropad) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveZeropad>()); }

TEST(SpectrogramAdaptive, InterfaceMoving) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveMoving>()); }

// description: test nFrames is a static function and that it returns correct number of frames.
TEST(SpectrogramAdaptive, getNFrames)
{
    SpectrogramAdaptive spec;
    auto c = spec.getCoefficients();
    const int noutputFrames = positivePow2(c.nSpectrograms - 1); // 2^(nSpectrograms-1) frames
    const int bufferSize = c.bufferSize;
    const int nFrames = 10;
    const int nSamples = bufferSize * nFrames;

    ArrayXf signal(nSamples);
    signal.setRandom();

    ArrayXXf output(c.nBands, noutputFrames * nFrames + 1); // add one extra frame than needed
    output.setZero();                                       // important to set to zero, since we are checking last frame is zero in success criteria
    for (auto frame = 0; frame < nFrames; frame++)
    {
        spec.process(signal.segment(frame * bufferSize, bufferSize), output.middleCols(noutputFrames * frame, noutputFrames));
    }

    bool criteria =
        (!output.leftCols(noutputFrames * nFrames).isZero()) && (output.rightCols(1).isZero()); // test criteria that all nFrames are non-zero and last frame is zero
    fmt::print("Criteria: {}\n", criteria);
    EXPECT_TRUE(criteria);
}

TEST(SpectrogramAdaptive, SinePeakLocation)
{
    SpectrogramAdaptiveMoving::Coefficients c; // defaults: bufferSize=1024, nBands=2049, nSpectrograms=3, sampleRate=16000
    SpectrogramAdaptiveMoving algo(c);

    const int nOutputFrames = 1 << (c.nSpectrograms - 1);
    const int nBlocks = 16;
    const int nSamples = c.bufferSize * nBlocks;

    const float freqHz = 1000.0f;
    const int expectedBin = static_cast<int>(std::round(freqHz * 2.0f * (c.nBands - 1) / c.sampleRate));

    ArrayXf signal(nSamples);
    for (int n = 0; n < nSamples; ++n)
    {
        signal(n) = std::sin(2.0 * M_PI * freqHz * n / c.sampleRate);
    }

    ArrayXXf output(c.nBands, nOutputFrames);
    const int warmupBlocks = 4;
    for (int b = 0; b < warmupBlocks; ++b)
    {
        algo.process(signal.segment(b * c.bufferSize, c.bufferSize), output);
    }

    int correctFrames = 0;
    for (int b = warmupBlocks; b < nBlocks; ++b)
    {
        algo.process(signal.segment(b * c.bufferSize, c.bufferSize), output);
        ASSERT_TRUE(output.allFinite());
        for (int f = 0; f < nOutputFrames; ++f)
        {
            Eigen::Index maxIdx;
            output.col(f).maxCoeff(&maxIdx);
            if (std::abs(static_cast<int>(maxIdx) - expectedBin) <= 1) { ++correctFrames; }
        }
    }
    const int totalFrames = (nBlocks - warmupBlocks) * nOutputFrames;
    EXPECT_GE(correctFrames, static_cast<int>(0.9 * totalFrames)) << "Peak bin correct in " << correctFrames << "/" << totalFrames << " frames";
}
