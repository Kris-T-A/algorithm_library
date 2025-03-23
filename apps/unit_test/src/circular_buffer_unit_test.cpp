#include "delay/circular_buffer.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(CircularBuffer, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<CircularBuffer>()); }

// Test that input is correctly delayed when the delay amount is larger than the input
TEST(CircularBuffer, DelayLargerThanInput)
{
    CircularBuffer::Coefficients c;
    int inputSize = 128;
    int delayFactor = 3;
    c.delayLength = inputSize * delayFactor;
    c.nChannels = 1;
    CircularBuffer algo(c);
    Eigen::ArrayXXf input = Eigen::ArrayXf::Random(inputSize, 1);
    Eigen::ArrayXXf inputZero = Eigen::ArrayXf::Zero(inputSize, 1);
    Eigen::ArrayXXf output = algo.initOutput(input);

    // send some zeros in to put the delay buffer into a state where it is not reset
    for (auto i = 0; i < 3; i++) { algo.process(inputZero, output); }

    // send the input through the delay
    algo.process(input, output);

    // send zeros in until we expect the input to come out
    for (auto i = 0; i < delayFactor-1; i++)
    {
        algo.process(inputZero, output);
        float error = output.abs().sum();
        if (error != 0.f)
        {
            fmt::print("Error in iteration {}: {}\n", i, error);
            EXPECT_TRUE(error == 0.f);
        }
    }
    // get final output
    algo.process(inputZero, output);

    // check result
    float error = (output - input).abs().sum();
    fmt::print("Error: {}\n", error);
    EXPECT_TRUE(error < 1e-10);
}