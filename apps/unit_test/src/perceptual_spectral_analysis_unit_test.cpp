#include "perceptual_spectral_analysis/perceptual_spectrogram.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>

using namespace Eigen;

//--------------------------------------------- TEST CASES ---------------------------------------------

TEST(PerceptualSpectralAnalysis, Interface) { 
    int dummy = 1;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<PerceptualSpectrogram>()); 
    EXPECT_TRUE(dummy);
}

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
    PerceptualSpectrogram algo;
    auto cornerFrequencies = algo.getCornerFrequencies();
    auto centerFrequencies = algo.getCenterFrequencies();
    fmt::print("Corner frequencies: {}\n", (cornerFrequencies * 100).round() / 100);
    fmt::print("Center frequencies: {}\n", (centerFrequencies * 100).round() / 100);
    EXPECT_TRUE(cornerFrequencies.size() == centerFrequencies.size() + 1);
}