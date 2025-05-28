#include "spectrogram_adaptive/spectrogram_adaptive_wola.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(SpectrogramAdaptive, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveWOLA>()); }

// description: test nFrames is a static function and that it returns correct number of frames.
TEST(SpectrogramAdaptive, getNFrames)
{
    SpectrogramAdaptive spec;
    auto c = spec.getCoefficients();
    const int bufferSize = c.bufferSize;
    const int nFrames = 10;
    const int nSamples = bufferSize * nFrames;

    ArrayXf signal(nSamples);
    signal.setRandom();

    ArrayXXf output(c.nBands, 8 * nFrames + 1); // add one extra frame than needed
    output.setZero();                           // important to set to zero, since we are checking last frame is zero in success criteria
    for (auto frame = 0; frame < nFrames; frame++)
    {
        spec.process(signal.segment(frame * bufferSize, bufferSize), output.middleCols(8 * frame, 8));
    }

    bool criteria = (!output.leftCols(8 * nFrames).isZero()) && (output.rightCols(1).isZero()); // test criteria that all nFrames are non-zero and last frame is zero
    fmt::print("Criteria: {}\n", criteria);
    EXPECT_TRUE(criteria);
}
