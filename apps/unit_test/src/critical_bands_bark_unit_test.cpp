#include "critical_bands/critical_bands_bark.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <fmt/ostream.h>

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(CriticalBandsBark, InterfaceSum) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<CriticalBandsBarkSum>()); }

TEST(CriticalBandsBark, InterfaceMax) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<CriticalBandsBarkMax>()); }

TEST(CriticalBandsBark, InterfaceMean) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<CriticalBandsBarkMean>()); }

// description: check that get number of critical bands, center frequencies and corner frequencies work
TEST(CriticalBandsBark, getters)
{
    float sampleRate = 44.1e3;
    int nCritBands = CriticalBandsConfiguration::getNCriticalBands(sampleRate);
    fmt::print("Sample rate: {} Hz\n", sampleRate);
    fmt::print("Number of critical bands: {}\n", nCritBands);

    ArrayXf centerFreqs = CriticalBandsConfiguration::getCenterFrequencies(sampleRate);
    fmt::print("Center frequencies (Hz): {}\n", fmt::streamed(centerFreqs));

    ArrayXf cornerFreqs = CriticalBandsConfiguration::getCornerFrequencies(sampleRate);
    fmt::print("Corner frequencies (Hz): {}\n", fmt::streamed(cornerFreqs));

    fmt::print("Setting new sample rate...\n");

    sampleRate = 16000;
    nCritBands = CriticalBandsConfiguration::getNCriticalBands(sampleRate);
    fmt::print("Sample rate: {} Hz\n", sampleRate);
    fmt::print("Number of critical bands: {}\n", nCritBands);

    centerFreqs = CriticalBandsConfiguration::getCenterFrequencies(sampleRate);
    fmt::print("Center frequencies (Hz): {}\n", fmt::streamed(centerFreqs));

    cornerFreqs = CriticalBandsConfiguration::getCornerFrequencies(sampleRate);
    fmt::print("Corner frequencies (Hz): {}\n", fmt::streamed(cornerFreqs));
}

TEST(CriticalBandsBark, inverse)
{
    auto c = CriticalBandsBarkSum::Coefficients();
    CriticalBandsBarkSum critBands(c);
    const int nFFTBands = 257;
    ArrayXf input = ArrayXf::Random(nFFTBands).abs2();
    ArrayXf output = critBands.initOutput(input);
    critBands.process(input, output);
    critBands.inverse(output, input);
}