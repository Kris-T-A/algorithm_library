#include "dc_remover/dc_remover_first_order.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(DCRemover, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<DCRemoverFirstOrder>()); }

TEST(DCRemover, changecutoff)
{
    auto c = DCRemoverFirstOrder::Coefficients();
    c.nChannels = 1;
    DCRemoverFirstOrder dcRemover(c);

    auto p = dcRemover.getParameters();
    p.cutoffFrequency = 100.f;
    dcRemover.setParameters(p);

    int n = 1000;
    Eigen::ArrayXf input(n), output(n), output2(n);
    srand(0); // initialize to get consistent results
    input.setRandom();

    dcRemover.process(input, output);

    p.cutoffFrequency = 10000.f;
    dcRemover.setParameters(p);
    dcRemover.process(input, output2);
    dcRemover.process(input, output2);

    float difference = (output - output2).abs2().sum() / output.abs2().sum();
    fmt::print("Relative difference: {}\n", difference);
    EXPECT_TRUE(difference > .4f);
}
