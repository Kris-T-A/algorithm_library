#include "audio_attenuate/audio_attenuate_adaptive.h"
#include "audio_attenuate/decimate_gain.h"
#include "fmt/ranges.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(AudioAttenuate, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AudioAttenuateAdaptive>()); }

TEST(DecimateGain, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<DecimateGain>()); }

TEST(AudioCombine, InterfaceMaxn) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AudioCombineMax>()); }

TEST(AudioCombine, InterfaceMin) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<AudioCombineMin>()); }

// print small example output
TEST(DecimateGain, PrintOutput)
{
    DecimateGain::Coefficients c;
    c.nBands = 9;
    DecimateGain decimateGain(c);
    Eigen::ArrayXXf input = decimateGain.initInput();
    std::vector<Eigen::ArrayXXf> output = decimateGain.initDefaultOutput();
    decimateGain.process(input, output);

    std::cout << "Input: \n" << input << "\n";
    std::cout << "Output 0: \n" << output[0] << "\n";
    std::cout << "Output 1: \n" << output[1] << "\n";
    std::cout << "Output 2: \n" << output[2] << "\n";
    std::cout << "Output 3: \n" << output[3] << "\n";
}