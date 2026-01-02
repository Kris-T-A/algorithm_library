#include "perceptual_spectral_analysis/perceptual_adaptive_spectrogram.h"
#include "perceptual_spectral_analysis/perceptual_nonlinear_spectrogram.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>

using namespace Eigen;

//--------------------------------------------- TEST CASES ---------------------------------------------

TEST(PerceptualSpectralAnalysis, InterfaceAdaptive) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<PerceptualAdaptiveSpectrogram>()); }

TEST(PerceptualSpectralAnalysis, InterfaceNonlinear) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<PerceptualNonlinearSpectrogram>()); }

// test the public interface can run without crashing and gives valid output
TEST(PerceptualSpectralAnalysis, ProcessPublic)
{
    PerceptualSpectralAnalysis algo;
    auto input = algo.initInput();
    auto output = algo.initOutput(input);
    algo.process(input, output);
    fmt::print("Input size: {}x{}\n", input.rows(), input.cols());
    fmt::print("Output size: {}x{}\n", output.rows(), output.cols());
    EXPECT_TRUE(algo.validOutput(output));
}

TEST(PerceptualSpectralAnalysis, GetFrequencies)
{
    PerceptualAdaptiveSpectrogram algo;
    auto centerFrequencies = algo.getCenterFrequencies();
    fmt::print("Center frequencies: {}\n", (centerFrequencies * 100).round() / 100);
}

TEST(PerceptualSpectralAnalysis, MemorySize)
{
    PerceptualAdaptiveSpectrogram algo;
    auto c = algo.getCoefficients();
    c.sampleRate = 48000.f;
    c.bufferSize = .032 * c.sampleRate; // set buffer size to 32 ms
    c.nBands = 1025;
    c.frequencyMin = 40.f;
    c.frequencyMax = 20000.f;
    c.spectralTilt = true;
    c.nSpectrograms = 4;
    c.nFolds = 1;
    c.nonlinearity = 1;
    algo.setCoefficients(c);

    auto sizeStatic = algo.getStaticSize();
    fmt::print("Static size variables: {} bytes\n", sizeStatic);
    auto sizeDynamic = algo.getDynamicSize();
    fmt::print("Dynamic size variables: {} bytes\n", sizeDynamic);
}