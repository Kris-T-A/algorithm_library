#include "filterbank/filterbank_single_channel.h"
#include "unit_test.h"
#include "gtest/gtest.h"

TEST(FilterbankSingleChannel, InterfaceAnalysisSingleChannel) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<FilterbankAnalysisSingleChannel>()); }

TEST(FilterbankSingleChannel, InterfaceSynthesisSingleChannel) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<FilterbankSynthesisSingleChannel>()); }

// send an impulse through analysis and synthesis filterbank and check output is a delayed version of the input
TEST(FilterbankSingleChannel, Reconstruct)
{
    // setup filterbanks
    FilterbankAnalysisSingleChannel::Coefficients cAnalysis;
    cAnalysis.bufferSize = 1024;
    cAnalysis.nFolds = 1;
    cAnalysis.nBands = 2 * cAnalysis.bufferSize + 1;
    cAnalysis.nChannels = 1;
    FilterbankAnalysisSingleChannel filterbank(cAnalysis);

    FilterbankSynthesisSingleChannel::Coefficients cSynthesis;
    cSynthesis.bufferSize = cAnalysis.bufferSize;
    cSynthesis.nFolds = 1;
    cSynthesis.nBands = cAnalysis.nBands;
    cSynthesis.nChannels = cAnalysis.nChannels;
    FilterbankSynthesisSingleChannel filterbankInverse(cSynthesis);

    // define input/outputs
    int frameFactor = 10;
    Eigen::ArrayXf input(frameFactor * cAnalysis.bufferSize), output(frameFactor * cAnalysis.bufferSize);
    auto impulseDelay = static_cast<int>(frameFactor * 1.5);
    input.setZero();
    input(impulseDelay) = 1.0f;
    output.setZero();

    // process input
    Eigen::ArrayXXcf xFreq = filterbank.initDefaultOutput();
    for (auto i = 0; i < frameFactor; i++)
    {
        filterbank.process(input.segment(i * cAnalysis.bufferSize, cAnalysis.bufferSize), xFreq);
        filterbankInverse.process(xFreq, output.segment(i * cAnalysis.bufferSize, cAnalysis.bufferSize));
    }

    auto delay = static_cast<int>(filterbank.getDelaySamples() + filterbankInverse.getDelaySamples() - cAnalysis.bufferSize);
    fmt::print("Buffer size: {}\n", cAnalysis.bufferSize);
    fmt::print("Delay: {}\n", delay);
    fmt::print("Output impulse value: {}\n", output(impulseDelay + delay));

    // calculate reconstruction error
    Eigen::ArrayXf outputError = output;
    outputError(impulseDelay + delay) -= 1.0f; // subtract impulse from output
    float error = 10 * std::log10(outputError.abs2().mean());
    float errorMax = 10 * std::log10(outputError.abs2().maxCoeff());
    fmt::print("Mean error: {} dB\n", error);
    fmt::print("Max error: {} dB\n", errorMax);
    EXPECT_LT(error, -150);
    EXPECT_LT(errorMax, -130);
}