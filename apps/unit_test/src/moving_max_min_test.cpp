#include "moving_max_min/moving_max_min_horizontal.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(MovingMaxMin, InterfaceHorizontal) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<MovingMaxMinHorizontal>()); }

// plot example
TEST(MovingMaxMin, plotResult)
{
    MovingMaxMinHorizontal::Coefficients c;
    c.filterLength = 3;
    c.nChannels = 2;

    MovingMaxMinHorizontal algo(c);

    Eigen::ArrayXXf input = Eigen::ArrayXXf::Random(c.nChannels, 10);
    Eigen::ArrayXXf output = algo.initOutput(input);

    algo.process(input, output);

    std::cout << "Input:\n" << input << "\n\n";
    std::cout << "Output:\n" << output << "\n";
}
