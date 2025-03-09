#pragma once
#include "algorithm_library/bandsplit_downsample.h"
#include "delay/circular_buffer.h"
#include "framework/framework.h"
#include "iir_filter/iir_filter_2nd_order.h"

// Bandsplit and downsample by a factor of 3 using Chebyshev Type II filter with 80dB attenuation.
//
// author: Kristian Timm Andersen
class BandsplitDownsampleChebyshev : public AlgorithmImplementation<BandsplitDownsampleConfiguration, BandsplitDownsampleChebyshev>
{
  public:
    BandsplitDownsampleChebyshev(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, filterLowpass({.nChannels = c.nChannels, .nSos = 5}), filterHighpass({.nChannels = c.nChannels, .nSos = 5}),
          delay({.nChannels = c.nChannels, .delayLength = delayHF})
    {
        if ((static_cast<int>(c.nSamples / 3) * 3) != c.nSamples) { throw BandsplitDownsampleConfiguration::ExceptionBandsplitDownsample(c); }
        buffer.resize(c.nSamples, c.nChannels);
        resetVariables();

        filterLowpass.setFilter(getLowpassCoefficients(), getLowpassGain());
        filterHighpass.setFilter(getHighpassCoefficients(), getHighpassGain());
    }

    IIRFilterCascaded filterLowpass;
    IIRFilterCascaded filterHighpass;
    CircularBuffer delay;

    DEFINE_MEMBER_ALGORITHMS(filterLowpass, filterHighpass, delay)

  private:
    void processAlgorithm(Input input, Output output)
    {
        // lowpass and downsample by factor of 3
        filterLowpass.process(input, buffer);
        output.downsampled = buffer(Eigen::seq(0, C.nSamples - 1, 3), Eigen::indexing::all); // downsample by factor of 3

        // highpass and delay
        filterHighpass.process(input, buffer);
        delay.process(buffer, output.highpass);
    }

    void resetVariables() final { buffer.setZero(); }

    size_t getDynamicSizeVariables() const final { return buffer.getDynamicMemorySize(); }

    constexpr static int delayHF = 11; // delay of highpass filter signal
    Eigen::ArrayXXf buffer;

    static Eigen::ArrayXXf getLowpassCoefficients();
    static float getLowpassGain();
    static Eigen::ArrayXXf getHighpassCoefficients();
    static float getHighpassGain();

    friend BaseAlgorithm;
};

class CombineBandsplitDownsampleChebyshev : public AlgorithmImplementation<CombineBandsplitDownsampleConfiguration, CombineBandsplitDownsampleChebyshev>
{
  public:
    CombineBandsplitDownsampleChebyshev(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}, filterLowpass({.nChannels = c.nChannels, .nSos = 5})
    {
        buffer.resize(3 * c.nSamples, c.nChannels);
        resetVariables();

        filterLowpass.setFilter(getLowpassCoefficients(), getLowpassGain());
    }

    IIRFilterCascaded filterLowpass;

    DEFINE_MEMBER_ALGORITHMS(filterLowpass)

  private:
    void processAlgorithm(Input input, Output output)
    {
        // upsample by a factor of 3 and lowpass
        buffer(Eigen::seq(0, 3 * C.nSamples - 1, 3), Eigen::indexing::all) = 3.f * input.downsampled;
        filterLowpass.process(buffer, output);
        // sum with highpass
        output += input.highpass;
    }

    void resetVariables() final { buffer.setZero(); }

    size_t getDynamicSizeVariables() const final { return buffer.getDynamicMemorySize(); }

    static Eigen::ArrayXXf getLowpassCoefficients();
    static float getLowpassGain();

    Eigen::ArrayXXf buffer;

    friend BaseAlgorithm;
};