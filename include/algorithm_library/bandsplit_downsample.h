#pragma once
#include "interface/interface.h"

// Bandsplit and downsample by a factor of 3.
struct BandsplitDownsampleConfiguration
{
    using Input = I::Real2D;

    struct Output
    {
        O::Real2D downsampled;
        O::Real2D highpass;
    };

    struct Coefficients
    {
        int nChannels = 2;
        int nSamples = 129; // must be factor of 3
        enum ResamplingType { K48HZ_TO_K16HZ };
        DEFINE_TUNABLE_ENUM(ResamplingType, {{K48HZ_TO_K16HZ, "48kHz to 16kHz"}})
        ResamplingType resamplingType = ResamplingType::K48HZ_TO_K16HZ;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, nSamples, resamplingType)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    // exception for constructing BandsplitDownsample with unsupported Configuration
    class ExceptionBandsplitDownsample : public std::runtime_error
    {
      public:
        ExceptionBandsplitDownsample(const Coefficients &c)
            : std::runtime_error(std::string("\nThis configuration is not supported:\nnumber of input samples = ") + std::to_string(c.nSamples) + "\n")
        {}
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nSamples, c.nChannels); }

    static std::tuple<Eigen::ArrayXXf, Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        return std::make_tuple(Eigen::ArrayXXf::Zero(c.nSamples / 3, c.nChannels), Eigen::ArrayXXf::Zero(c.nSamples, c.nChannels));
    }

    static bool validInput(Input input, const Coefficients &c)
    {
        return (input.rows() == c.nSamples) && (c.nSamples % 3 == 0) && (input.cols() == c.nChannels) && input.allFinite();
    }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.downsampled.rows() * 3 == output.highpass.rows()) && (output.downsampled.cols() == c.nChannels) && output.downsampled.allFinite() &&
               (output.highpass.rows() == c.nSamples) && (output.highpass.cols() == c.nChannels) && output.highpass.allFinite();
    }
};

class BandsplitDownsample : public Algorithm<BandsplitDownsampleConfiguration>
{
  public:
    BandsplitDownsample() = default;
    BandsplitDownsample(const Coefficients &c);
};

// ------------------------------------------------------------------------------------------------

// Combine two signals, one downsampled and one highpass filtered. The input signals are assumed to be the output signals from BandsplitDownsample.
struct CombineBandsplitDownsampleConfiguration
{
    struct Input
    {
        I::Real2D downsampled;
        I::Real2D highpass;
    };

    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        int nSamples = 43;
        enum ResamplingType { K16HZ_TO_K48HZ };
        DEFINE_TUNABLE_ENUM(ResamplingType, {{K16HZ_TO_K48HZ, "16kHz to 48kHz"}})
        ResamplingType resamplingType = ResamplingType::K16HZ_TO_K48HZ;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, nSamples, resamplingType)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXXf, Eigen::ArrayXXf> initInput(const Coefficients &c)
    {
        return std::make_tuple(Eigen::ArrayXXf::Random(c.nSamples, c.nChannels),
                               Eigen::ArrayXXf::Random(3 * c.nSamples, c.nChannels)); // arbitrary number of samples, but highpass must be 3 times longer than downsampled
    }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(3 * c.nSamples, c.nChannels); }

    static bool validInput(Input input, const Coefficients &c)
    {
        return (input.downsampled.rows() == c.nSamples) && (input.downsampled.cols() == c.nChannels) && input.downsampled.allFinite() &&
               (input.highpass.rows() == (3 * input.downsampled.rows())) && (input.highpass.cols() == c.nChannels) && input.highpass.allFinite();
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == 3 * c.nSamples) && (output.cols() == c.nChannels) && output.allFinite(); }
};

class CombineBandsplitDownsample : public Algorithm<CombineBandsplitDownsampleConfiguration>
{
  public:
    CombineBandsplitDownsample() = default;
    CombineBandsplitDownsample(const Coefficients &c);
};