#include "scale_transform/log_scale.h"
#include "scale_transform/mel_scale.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(ScaleTransform, InterfaceMel) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<MelScale>()); }

TEST(ScaleTransform, InterfaceLog) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<LogScale>()); }

// description: check that get corner frequencies work
TEST(ScaleTransform, getters)
{
    ScaleTransform algo;
    auto c = algo.getCoefficients();

    c.transformType = c.MEL;
    algo.setCoefficients(c);
    ArrayXf cornerFreqs = algo.getCornerIndices();
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    fmt::print("{} Mel Corner frequencies (Hz): {}\n", cornerFreqs.size(), cornerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    cornerFreqs = algo.getCornerIndices();
    fmt::print("{} Logarithmic Corner frequencies (Hz): {}\n", cornerFreqs.size(), (cornerFreqs * 1000).round() / 1000); // round to 3 decimals

    fmt::print("\nSetting new sample rate...\n");

    c.indexEnd = 48000 / 2;
    c.transformType = c.MEL;
    algo.setCoefficients(c);
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    cornerFreqs = algo.getCornerIndices();
    fmt::print("{} Mel Corner frequencies (Hz): {}\n", cornerFreqs.size(), cornerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    cornerFreqs = algo.getCornerIndices();
    fmt::print("{} Logarithmic Corner frequencies (Hz): {}\n", cornerFreqs.size(), (cornerFreqs * 1000).round() / 1000); // round to 3 decimals
}

// process ScaleTransform with an input of ones, and invert the output. Check the inverse is equal to input (This is only true due to the simple input and not in general)
TEST(ScaleTransform, processInverse)
{
    ScaleTransform algo;
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