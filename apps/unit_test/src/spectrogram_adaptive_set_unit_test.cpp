#include "perceptual_spectral_analysis/perceptual_adaptive_spectrogram.h"
#include "spectrogram_adaptive_set/spectrogram_adaptive_set_min.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

TEST(SpectrogramAdaptiveSet, InterfaceMin)
{
    bool dummy = true;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveSetMin>(dummy));
}

// SpectrogramAdaptiveSetMin internally fixes nFolds=1 and nonlinearity=1 in its SpectrogramSetZeropad
// and adds the spectral tilt onto the linear-band spectrogram at every stage, before logScale and
// vertical movingMaxMin produce its per-stage outputs. PerceptualAdaptiveSpectrogram runs the same
// underlying SpectrogramAdaptiveMoving pipeline (tilt added after the recursion) and then logScale +
// vertical movingMaxMin once on the finest-resolution stage. With matching outward coefficients and
// nFolds=1, nonlinearity=1, the set's last entry must equal the perceptual output sample-for-sample.
template <typename Coefficients> static void runComparison(const Coefficients &cSet)
{
    PerceptualAdaptiveSpectrogram::Coefficients cRef;
    cRef.bufferSize = cSet.bufferSize;
    cRef.nBands = cSet.nBands;
    cRef.sampleRate = cSet.sampleRate;
    cRef.frequencyMin = cSet.frequencyMin;
    cRef.frequencyMax = cSet.frequencyMax;
    cRef.spectralTilt = cSet.spectralTilt;
    cRef.nSpectrograms = cSet.nSpectrograms;
    cRef.nFolds = 1;
    cRef.nonlinearity = 1;

    SpectrogramAdaptiveSetMin algoSet(cSet);
    PerceptualAdaptiveSpectrogram algoRef(cRef);

    auto outputSet = algoSet.initDefaultOutput();
    ArrayXXf outputRef = algoRef.initDefaultOutput();

    const int lastIndex = cSet.nSpectrograms - 1;
    const int nFrames = 8;
    srand(42);
    for (int frame = 0; frame < nFrames; frame++)
    {
        ArrayXf input = ArrayXf::Random(cSet.bufferSize);

        algoSet.process(input, outputSet);
        algoRef.process(input, outputRef);

        ASSERT_EQ(outputSet[lastIndex].rows(), outputRef.rows());
        ASSERT_EQ(outputSet[lastIndex].cols(), outputRef.cols());
        EXPECT_TRUE(outputSet[lastIndex].isApprox(outputRef))
            << "Mismatch at frame " << frame << "\nmax abs diff: " << (outputSet[lastIndex] - outputRef).abs().maxCoeff();
    }
}

TEST(SpectrogramAdaptiveSet, LastOutputMatchesPerceptualAdaptiveSpectrogram)
{
    SpectrogramAdaptiveSetMin::Coefficients cSet;
    cSet.bufferSize = 1024;
    cSet.nBands = 100;
    cSet.nSpectrograms = 3;
    cSet.sampleRate = 48000.f;
    cSet.spectralTilt = true;
    cSet.frequencyMin = 20.f;
    cSet.frequencyMax = 20000.f;
    runComparison(cSet);
}

TEST(SpectrogramAdaptiveSet, LastOutputMatchesPerceptualAdaptiveSpectrogramNoTilt)
{
    SpectrogramAdaptiveSetMin::Coefficients cSet;
    cSet.bufferSize = 512;
    cSet.nBands = 64;
    cSet.nSpectrograms = 4;
    cSet.sampleRate = 16000.f;
    cSet.spectralTilt = false;
    cSet.frequencyMin = 50.f;
    cSet.frequencyMax = 8000.f;
    runComparison(cSet);
}
