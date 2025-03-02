#include "perceptual_spectral_analysis/perceptual_spectrogram.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(PerceptualSpectralAnalysis, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<PerceptualSpectrogram>()); }

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