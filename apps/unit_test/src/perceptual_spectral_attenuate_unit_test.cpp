#include "audio_attenuate/audio_attenuate_adaptive.h"
#include "perceptual_spectral_attenuate/perceptual_attenuate_max.h"
#include "scale_transform/log_scale.h"
#include "unit_test.h"
#include "gtest/gtest.h"
#include <limits>

// Run the framework interface checks for configuration, processing, allocation, reset,
// serialization and setters. Pass when every interface check succeeds.
TEST(PerceptualSpectralAttenuate, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<PerceptualAttenuateMax>()); }

// Process generated input through the public class. Pass when the input is valid
// and the output contains bufferSize finite samples.
TEST(PerceptualSpectralAttenuate, ProcessPublic)
{
    PerceptualSpectralAttenuate algo;
    auto [signal, gain] = algo.initInput();
    auto output = algo.initDefaultOutput();
    ASSERT_TRUE(algo.validInput({signal, gain}));
    algo.process({signal, gain}, output);
    EXPECT_TRUE(algo.validOutput(output));
}

// Process an impulse with uniform linear gains of 1 and 0.1 using one to five resolutions.
// Pass when the summed absolute error against an impulse at the expected delay,
// with amplitude 1 or 0.1 respectively, is below 1e-5 for every configuration.
TEST(PerceptualSpectralAttenuate, ReconstructionAndLinearGain)
{
    for (int nSpectrograms = 1; nSpectrograms <= 5; ++nSpectrograms)
    {
        for (float linearGain : {1.f, 0.1f})
        {
            SCOPED_TRACE(nSpectrograms);
            SCOPED_TRACE(linearGain);
            PerceptualSpectralAttenuate::Coefficients c;
            c.bufferSize = 256;
            c.nBands = 32;
            c.nSpectrograms = nSpectrograms;
            PerceptualSpectralAttenuate algo(c);
            const int nFrames = 1 << (nSpectrograms - 1);
            Eigen::ArrayXXf gain = Eigen::ArrayXXf::Constant(c.nBands, nFrames, linearGain);
            Eigen::ArrayXf signal = Eigen::ArrayXf::Zero(12 * c.bufferSize);
            Eigen::ArrayXf output = signal;
            const int impulseIndex = 2 * c.bufferSize + 17;
            const int expectedIndex = impulseIndex + 3 * c.bufferSize + 3 * c.bufferSize / nFrames;
            signal(impulseIndex) = 1.f;
            for (int i = 0; i < 12; ++i)
            {
                algo.process({signal.segment(i * c.bufferSize, c.bufferSize), gain}, output.segment(i * c.bufferSize, c.bufferSize));
            }
            Eigen::ArrayXf expected = Eigen::ArrayXf::Zero(output.size());
            expected(expectedIndex) = linearGain;
            EXPECT_LT((output - expected).abs().sum(), 1e-5f);
        }
    }
}

// Measure latency by locating the output peak of a unit impulse with unity gains.
// Pass when the measured input-to-output shift equals 3 * bufferSize plus three
// smallest hops for both buffer sizes and every resolution count from one to five.
TEST(PerceptualSpectralAttenuate, Latency)
{
    for (int bufferSize : {256, 4096})
    {
        for (int nSpectrograms = 1; nSpectrograms <= 5; ++nSpectrograms)
        {
            SCOPED_TRACE(bufferSize);
            SCOPED_TRACE(nSpectrograms);
            PerceptualAttenuateMax::Coefficients c;
            c.bufferSize = bufferSize;
            c.nSpectrograms = nSpectrograms;
            PerceptualAttenuateMax algo(c);
            const int nFrames = 1 << (nSpectrograms - 1);
            const int expectedDelay = 3 * bufferSize + 3 * bufferSize / nFrames;
            const int impulseIndex = bufferSize + 17;
            Eigen::ArrayXf signal = Eigen::ArrayXf::Zero(10 * bufferSize);
            Eigen::ArrayXf output = Eigen::ArrayXf::Zero(signal.size());
            Eigen::ArrayXXf gain = Eigen::ArrayXXf::Ones(c.nBands, nFrames);
            signal(impulseIndex) = 1.f;
            for (int i = 0; i < 10; ++i)
            {
                algo.process({signal.segment(i * bufferSize, bufferSize), gain}, output.segment(i * bufferSize, bufferSize));
            }
            Eigen::Index outputPeakIndex;
            const float peakAmplitude = output.abs().maxCoeff(&outputPeakIndex);
            ASSERT_GT(peakAmplitude, 0.9f);
            const int measuredDelay = static_cast<int>(outputPeakIndex) - impulseIndex;
            EXPECT_EQ(measuredDelay, expectedDelay);
            fmt::print("bufferSize={}, nSpectrograms={}: measured latency={} samples ({:.3f} ms)\n", bufferSize, nSpectrograms, measuredDelay,
                       1000.f * measuredDelay / c.sampleRate);
        }
    }
}

// Apply a decreasing gain sequence to isolated impulses at several positions.
// Each gain column describes its input interval; the corresponding WOLA frame
// is centred at the interval's start. Calculate the expected impulse amplitude
// directly from squared Hann weights, temporal minimum gains and the maximum
// across resolutions. Pass when the complete output matches that independently
// calculated delayed impulse to 2e-5.
TEST(PerceptualSpectralAttenuate, GainSignalAlignment)
{
    for (int nSpectrograms = 1; nSpectrograms <= 5; ++nSpectrograms)
    {
        PerceptualSpectralAttenuate::Coefficients c;
        c.bufferSize = 256;
        c.nBands = 32;
        c.nSpectrograms = nSpectrograms;
        const int nFrames = 1 << (nSpectrograms - 1);
        const int smallestHop = c.bufferSize / nFrames;
        const auto linearGain = [](int frame) { return frame < 0 ? 1.0 : std::max(0.1, 1.0 - 0.015 * frame); };
        for (int offset : {0, smallestHop / 2, c.bufferSize - smallestHop / 2})
        {
            SCOPED_TRACE(nSpectrograms);
            SCOPED_TRACE(offset);
            PerceptualSpectralAttenuate algo(c);
            const int impulseIndex = 3 * c.bufferSize + offset;
            Eigen::ArrayXf signal = Eigen::ArrayXf::Zero(12 * c.bufferSize);
            Eigen::ArrayXf output = Eigen::ArrayXf::Zero(12 * c.bufferSize);
            Eigen::ArrayXXf gain(c.nBands, nFrames);
            signal(impulseIndex) = 1.f;
            for (int buffer = 0; buffer < 12; ++buffer)
            {
                for (int frame = 0; frame < nFrames; ++frame)
                {
                    gain.col(frame).setConstant(static_cast<float>(linearGain(buffer * nFrames + frame)));
                }
                algo.process({signal.segment(buffer * c.bufferSize, c.bufferSize), gain}, output.segment(buffer * c.bufferSize, c.bufferSize));
            }

            double expectedAmplitude = 0.0;
            for (int resolution = 0; resolution < nSpectrograms; ++resolution)
            {
                const int hop = c.bufferSize / (1 << resolution);
                const int framesPerHop = hop / smallestHop;
                double amplitude = 0.0;
                // Four overlapping windows contain this impulse. Their analysis
                // and synthesis products are Hann^2 / 1.5 (unity reconstruction).
                for (int frame = impulseIndex / hop; frame < impulseIndex / hop + 4; ++frame)
                {
                    const int windowIndex = impulseIndex - (frame - 3) * hop;
                    const double hann = 0.5 * (1.0 - std::cos(2.0 * std::acos(-1.0) * windowIndex / (4 * hop)));
                    // The window centre is (frame - 1) * hop. A decreasing
                    // sequence's minimum is the last gain in that hop's interval.
                    const int gainFrame = (frame - 1) * framesPerHop + framesPerHop - 1;
                    amplitude += hann * hann / 1.5 * linearGain(gainFrame);
                }
                expectedAmplitude = std::max(expectedAmplitude, amplitude);
            }
            Eigen::ArrayXf expected = Eigen::ArrayXf::Zero(output.size());
            expected(impulseIndex + 3 * c.bufferSize + 3 * smallestHop) = static_cast<float>(expectedAmplitude);
            EXPECT_LT((output - expected).abs().sum(), 2e-5f);
        }
    }
}

// Process three buffers at each resolution count from one to five with Eigen
// allocations disabled. Pass when processing triggers no allocation assertion
// in the debug build with EIGEN_RUNTIME_NO_MALLOC enabled.
TEST(PerceptualSpectralAttenuate, NoAllocationsAcrossResolutions)
{
    for (int nSpectrograms = 1; nSpectrograms <= 5; ++nSpectrograms)
    {
        SCOPED_TRACE(nSpectrograms);
        PerceptualSpectralAttenuate::Coefficients c;
        c.bufferSize = 256;
        c.nBands = 32;
        c.nSpectrograms = nSpectrograms;
        PerceptualSpectralAttenuate algo(c);
        auto [signal, gain] = algo.initInput();
        auto output = algo.initDefaultOutput();
        Eigen::internal::set_is_malloc_allowed(false);
        for (int i = 0; i < 3; ++i)
        {
            algo.process({signal, gain}, output);
        }
        Eigen::internal::set_is_malloc_allowed(true);
    }
}

// Compare four-resolution attenuation against AudioAttenuateAdaptive using the
// same random audio and time-constant linear perceptual-band gains expanded to
// FFT-bin gains. Prime gain histories with silent buffers because the algorithms
// delay time-varying gains differently. Pass when the maximum absolute sample
// difference is below 1e-6 in all 12 audio buffers.
TEST(PerceptualSpectralAttenuate, MatchesAudioAttenuateWithExpandedGains)
{
    PerceptualSpectralAttenuate::Coefficients c;
    c.bufferSize = 256;
    c.nBands = 32;
    c.nSpectrograms = 4;
    PerceptualSpectralAttenuate algo(c);
    AudioAttenuateAdaptive reference({.bufferSize = c.bufferSize, .timeOversampling = c.nSpectrograms});
    LogScale logScale({.nInputs = 2 * c.bufferSize + 1,
                       .nOutputs = c.nBands,
                       .outputStart = c.frequencyMin,
                       .outputEnd = c.frequencyMax,
                       .inputEnd = c.sampleRate / 2,
                       .transformType = LogScale::Coefficients::LOGARITHMIC});
    Eigen::ArrayXXf expandedGain(2 * c.bufferSize + 1, 8);
    auto output = algo.initDefaultOutput();
    auto referenceOutput = reference.initDefaultOutput();
    Eigen::ArrayXXf gain = Eigen::ArrayXf::Random(c.nBands).abs().replicate(1, 8);
    logScale.inverse(gain, expandedGain);
    const Eigen::ArrayXf silence = Eigen::ArrayXf::Zero(c.bufferSize);
    for (int i = 0; i < 4; ++i)
    {
        algo.process({silence, gain}, output);
        reference.process({silence, expandedGain}, referenceOutput);
    }
    for (int i = 0; i < 12; ++i)
    {
        Eigen::ArrayXf signal = Eigen::ArrayXf::Random(c.bufferSize);
        algo.process({signal, gain}, output);
        reference.process({signal, expandedGain}, referenceOutput);
        EXPECT_LT((output - referenceOutput).abs().maxCoeff(), 1e-6f);
    }
}

// Process five buffers, reset, then compare against a fresh instance receiving
// the same input. Pass when all samples match exactly for eight subsequent buffers.
TEST(PerceptualSpectralAttenuate, ResetMatchesFreshInstance)
{
    PerceptualSpectralAttenuate::Coefficients c;
    c.bufferSize = 256;
    c.nBands = 32;
    PerceptualSpectralAttenuate algo(c), fresh(c);
    auto [signal, gain] = algo.initInput();
    auto output = algo.initDefaultOutput();
    auto expected = fresh.initDefaultOutput();
    for (int i = 0; i < 5; ++i)
    {
        algo.process({signal, gain}, output);
    }
    algo.reset();
    for (int i = 0; i < 8; ++i)
    {
        algo.process({signal, gain}, output);
        fresh.process({signal, gain}, expected);
        EXPECT_EQ((output - expected).abs().maxCoeff(), 0.f);
    }
}

// Check gain validation with generated valid gains, then a gain above unity,
// negative infinity and NaN. Pass when only the generated input is accepted.
TEST(PerceptualSpectralAttenuate, RejectsInvalidGain)
{
    PerceptualSpectralAttenuate algo;
    auto [signal, gain] = algo.initInput();
    ASSERT_TRUE(algo.validInput({signal, gain}));
    gain(0, 0) = 1.1f;
    EXPECT_FALSE(algo.validInput({signal, gain}));
    gain(0, 0) = -std::numeric_limits<float>::infinity();
    EXPECT_FALSE(algo.validInput({signal, gain}));
    gain(0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(algo.validInput({signal, gain}));
}

// Request valid buffer sizes for 256, 257, 4096 and 4097 samples. Pass when each
// result is at least the requested size, supports its largest FFT and constructs
// a valid algorithm with the default resolution count.
TEST(PerceptualSpectralAttenuate, ValidBufferSize)
{
    for (int requested : {256, 257, 4096, 4097})
    {
        const int bufferSize = PerceptualSpectralAttenuate::Configuration::getValidBufferSize(requested);
        EXPECT_GE(bufferSize, requested);
        EXPECT_TRUE(FFTConfiguration::isFFTSizeValid(4 * bufferSize));
        PerceptualSpectralAttenuate::Coefficients c;
        c.bufferSize = bufferSize;
        PerceptualSpectralAttenuate algo(c);
        EXPECT_TRUE(algo.isConfigurationValid());
    }
}
