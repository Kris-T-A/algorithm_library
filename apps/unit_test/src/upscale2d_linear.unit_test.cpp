#include "spectrogram/upscale2d_linear.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <iostream>
using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(Upscale2DLinear, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<Upscale2DLinear>()); }

// print output of upscaling. This test requires manual inspection to verify output
TEST(Upscale2DLinear, printOutput)
{
    auto c = Upscale2DLinear::Coefficients();
    c.factorHorizontal = 3;
    c.factorVertical = 5;
    Upscale2DLinear upscale(c);

    Eigen::ArrayXXf input(3, 3);
    for (auto i = 0; i < 3; i++)
    {
        for (auto j = 0; j < 3; j++)
        {
            input(i, j) = static_cast<float>(i + j);
        }
    }
    auto output = upscale.initOutput(input);
    upscale.process(input, output);

    std::cout << "input: \n" << input << "\n";
    std::cout << "output: \n" << output << "\n";
}