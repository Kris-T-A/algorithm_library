#include "scale_transform/log_scale.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>
#include <iostream>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(ScaleTransform, InterfaceLog) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<LogScale>()); }

// description: check that get corner frequencies work
TEST(ScaleTransform, getters)
{
    ScaleTransform algo;
    auto c = algo.getCoefficients();

    c.transformType = c.MEL;
    algo.setCoefficients(c);
    ArrayXf centerFreqs = algo.getCenterFrequencies();
    fmt::print("Sample rate: {} Hz\n", c.inputEnd * 2);
    fmt::print("{} Mel Center frequencies (Hz): {}\n", centerFreqs.size(), centerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    centerFreqs = algo.getCenterFrequencies();
    fmt::print("{} Logarithmic Center frequencies (Hz): {}\n", centerFreqs.size(), (centerFreqs * 1000).round() / 1000); // round to 3 decimals

    fmt::print("\nSetting new sample rate...\n");

    c.inputEnd = 48000 / 2;
    c.outputEnd = c.inputEnd;
    c.transformType = c.MEL;
    algo.setCoefficients(c);
    fmt::print("Sample rate: {} Hz\n", c.inputEnd * 2);
    centerFreqs = algo.getCenterFrequencies();
    fmt::print("{} Mel Center frequencies (Hz): {}\n", centerFreqs.size(), centerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    centerFreqs = algo.getCenterFrequencies();
    fmt::print("{} Logarithmic Center frequencies (Hz): {}\n", centerFreqs.size(), (centerFreqs)); // round to 3 decimals
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

    if (error > 1e-3f)
    {
        std::cout << "Input: " << input.transpose() << std::endl;
        std::cout << "Output: " << output.transpose() << std::endl;
        std::cout << "Inverse: " << inverse.transpose() << std::endl;
    }
    EXPECT_LT(error, 1e-3f);
}
