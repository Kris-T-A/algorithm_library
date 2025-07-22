#include "scale_transform/log_scale.h"
#include "scale_transform/mel_scale.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ranges.h>
#include <iostream>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(ScaleTransform, InterfaceMel) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<MelScale>()); }

TEST(ScaleTransform, InterfaceLog) 
{ 
    bool dummy = false;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<LogScale>(dummy)); 
}

// description: check that get corner frequencies work
TEST(ScaleTransform, getters)
{
    ScaleTransform algo;
    auto c = algo.getCoefficients();

    c.transformType = c.MEL;
    algo.setCoefficients(c);
    ArrayXf centerFreqs = algo.getCenterFrequencies();
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    fmt::print("{} Mel Center frequencies (Hz): {}\n", centerFreqs.size(), centerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    centerFreqs = algo.getCenterFrequencies();
    fmt::print("{} Logarithmic Center frequencies (Hz): {}\n", centerFreqs.size(), (centerFreqs * 1000).round() / 1000); // round to 3 decimals

    fmt::print("\nSetting new sample rate...\n");

    c.indexEnd = 48000 / 2;
    c.transformType = c.MEL;
    algo.setCoefficients(c);
    fmt::print("Sample rate: {} Hz\n", c.indexEnd * 2);
    centerFreqs = algo.getCenterIndices();
    fmt::print("{} Mel Center frequencies (Hz): {}\n", centerFreqs.size(), centerFreqs);

    c.transformType = c.LOGARITHMIC;
    algo.setCoefficients(c);
    centerFreqs = algo.getCenterFrequencies();
    fmt::print("{} Logarithmic Center frequencies (Hz): {}\n", centerFreqs.size(), (centerFreqs * 1000).round() / 1000); // round to 3 decimals
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

TEST(ScaleTransform, processASize)
{

    auto c = ScaleTransformConfiguration::Coefficients();
    c.nInputs = 2 * 1024 + 1;
    c.nOutputs = 200;
    c.indexEnd = 48000 / 2;
    c.transformType = c.LOGARITHMIC;
    LogScale algo(c);

    Eigen::ArrayXXf input = Eigen::ArrayXXf::Ones(c.nInputs, 8).abs2();
    Eigen::ArrayXXf output = algo.initOutput(input);

    fmt::print("Input size: {} x {}\n", input.rows(), input.cols());
    fmt::print("Output size: {} x {}\n", output.rows(), output.cols());
    // fmt::print("indexEnd: {}\n", algo.indexEnd);
    // fmt::print("Corner indices: {}\n", algo.getCornerIndices());
    // fmt::print("Corner indices size: {}\n", algo.getCornerIndices().size());
    // fmt::print("nSmallBins: {}\n", algo.nSmallBins);
    // fmt::print("indexStart: {}\n", algo.indexStart);
    // fmt::print("nInputsSum: {}\n", algo.nInputsSum);
    // fmt::print("nInputsSum cumulative sum: {}\n", algo.nInputsSum.sum());
    // fmt::print("binsWeight: {}\n", algo.binsWeight);

    algo.process(input, output);

    std::cout << "Output:\n" << output << std::endl;
}