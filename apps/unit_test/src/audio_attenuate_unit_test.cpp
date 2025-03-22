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

// pass an impulse through the algorithm to check if it comes out unchanged
// This test is currently not passing!
TEST(AudioAttenuate, ImpulseTest)
{
    // initialize algorithm
    AudioAttenuateAdaptive::Coefficients c;
    c.bufferSize = 1024;
    c.sampleRate = 48000;
    AudioAttenuate algo(c);

    int nFrames = 10; // number of input buffer sizes to send through algorithm
    Eigen::ArrayXf input(nFrames * c.bufferSize), output(nFrames * c.bufferSize);
    input.setZero();
    input(0) = 1; // impulse at the beginning of the input signal

    Eigen::ArrayXf inputFrame, outputFrame;
    Eigen::ArrayXXf gainSpectrogram;
    std::tie(inputFrame, gainSpectrogram) = algo.initInput();
    gainSpectrogram.setOnes();
    outputFrame = algo.initDefaultOutput();

    for (int i = 0; i < nFrames; i++)
    {
        inputFrame = input.segment(i * c.bufferSize, c.bufferSize);
        algo.process({inputFrame, gainSpectrogram}, outputFrame);
        output.segment(i * c.bufferSize, c.bufferSize) = outputFrame;
    }
    fmt::print("Output: {}\n", output);
}