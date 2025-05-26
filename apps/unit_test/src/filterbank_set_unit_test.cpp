#include "filterbank_set/filterbank_set_wola.h"
#include "fmt/ranges.h"
#include "unit_test.h"
#include "utilities/fastonebigheader.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(FilterbankSetAnalysis, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<FilterbankSetAnalysisWOLA>()); }

TEST(FilterbankSetSynthesis, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<FilterbankSetSynthesisWOLA>()); }

// Interface test with coefficients.filterbankType = WOLA
TEST(FilterbankSetAnalysis, InterfaceWOLA)
{
    FilterbankSetAnalysisWOLA algo;
    auto c = algo.getCoefficients();
    c.nFolds = 2; // set filterbankType to WOLA
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<FilterbankSetAnalysisWOLA>(c));
}

// process random signal through analysis and synthesis, and verify that all outputs are the same as the input
TEST(FilterbankSetAnalysis, Reconstruction)
{
    FilterbankSetAnalysisWOLA filterbank;
    FilterbankSetSynthesisWOLA inverseFilterbank;

    auto c = filterbank.getCoefficients();
    int nFrames = 10;
    int nSamples = nFrames * c.bufferSize;
    ArrayXf input = ArrayXf::Random(nSamples);
    ArrayXXf output = ArrayXXf::Zero(nSamples, c.nFilterbanks);

    auto filterbankOut = filterbank.initOutput(input.head(c.bufferSize));

    for (auto frame = 0; frame < nFrames; frame++)
    {
        filterbank.process(input.segment(frame * c.bufferSize, c.bufferSize), filterbankOut);
        inverseFilterbank.process(filterbankOut, output.middleRows(frame * c.bufferSize, c.bufferSize));
    }

    // delay is equal to FFT size - buffer size. For this type of filterbanks that corresponds to 2 * groupdelay - bufferSize
    std::vector<int> delays(c.nFilterbanks);
    for (auto i = 0; i < c.nFilterbanks; i++)
    {
        delays[i] = static_cast<int>(std::round(filterbank.filterbanks[i].getDelaySamples())) * 2 - c.bufferSize / positivePow2(i);
        Eigen::ArrayXf outputDelayCompensated = output.col(i).segment(delays[i], nSamples - delays[0]);
        float error = (outputDelayCompensated.head(nSamples - delays[0]) - input.head(nSamples - delays[0])).abs2().mean();
        fmt::print("Filterbank {} with delay: {} and Error: {}\n", i, delays[i], error);
        EXPECT_TRUE(error < 1e-10f);
    }
}

// print delay
TEST(FilterbankSetAnalysis, MeasureDelay)
{
    FilterbankSetAnalysisWOLA filterbankSet;
    auto c = filterbankSet.getCoefficients();

    std::vector<int> delays(c.nFilterbanks);
    for (auto i = 0; i < c.nFilterbanks; i++)
    {
        delays[i] = 2 * static_cast<int>(std::round(filterbankSet.filterbanks[i].getDelaySamples())) - c.bufferSize / positivePow2(i);
    }

    fmt::print("Delays: {} samples\n", fmt::join(delays, ", "));
}