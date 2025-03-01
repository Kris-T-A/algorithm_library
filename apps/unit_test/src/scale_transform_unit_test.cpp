#include "scale_transform/mel_scale.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(ScaleTransform, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<MelScale>()); }

// description: check that get corner frequencies work
TEST(ScaleTransform, getters)
{
    ScaleTransform algo;
    auto c = algo.getCoefficients();
    ArrayXf cornerFreqs = algo.getCornerIndices();
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    fmt::print("Mel Corner frequencies (Hz): {}\n", cornerFreqs);

    fmt::print("Setting new sample rate...\n");

    c.indexEnd = 48000 / 2;
    algo.setCoefficients(c);
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    cornerFreqs = algo.getCornerIndices();
    fmt::print("Mel Corner frequencies (Hz): {}\n", cornerFreqs);
}

// process MelScale with an input of ones, and invert the output. Check the inverse is equal to input.
TEST(ScaleTransform, processInverse)
{
    MelScale algo;
    auto c = algo.getCoefficients();

    Eigen::ArrayXf input(c.nInputs);
    input.setOnes();
    Eigen::ArrayXf output(c.nOutputs);
    Eigen::ArrayXf inverse(c.nInputs);

    algo.process(input, output);
    algo.inverse(output, inverse);

    float error = (input - inverse).abs2().sum() / input.abs2().sum();
    fmt::print("Test error: {}\n", error);

    EXPECT_LT(error, 1e-10f);
}