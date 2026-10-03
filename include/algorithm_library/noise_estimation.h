#pragma once
#include "interface/interface.h"

// noise estimation based on an activity detector. The more activity in the power spectra, the more the output is smoothed.
//
// author: Kristian Timm Andersen
struct NoiseEstimationConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        float filterbankRate = 125.f;
        int nBands = 257;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterbankRate, nChannels, nBands)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // power spectrum

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nBands, c.nChannels); } // noise power spectrum estimate

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input.cols() == c.nChannels) && (input >= 0.f).all(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && (output.cols() == c.nChannels) && (output >= 0.f).all(); }
};

// Track background noise power in each frequency band, reducing adaptation
// when the activity detector indicates a signal is present.
class NoiseEstimation : public Algorithm<NoiseEstimationConfiguration>
{
  public:
    NoiseEstimation() = default;
    NoiseEstimation(const Coefficients &c);
};

// noise estimation based on an activity detector. This version has an additional output activity detector for each frequency bin
//
// author: Kristian Timm Andersen
struct NoiseEstimationActivityConfiguration
{
    using Input = I::Real2D;
    struct Output
    {
        O::Real2D powerNoise;
        O::Real2D activity;
    };

    struct Coefficients
    {
        float filterbankRate = 125.f;
        int nBands = 257;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterbankRate, nChannels, nBands)
    };

    struct Parameters
    {
        float smoothingTConstant = .072f;
        DEFINE_TUNABLE_PARAMETERS(smoothingTConstant)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // power spectrum

    static auto initOutput(Input input, const Coefficients &c) { return std::make_tuple(Eigen::ArrayXXf(c.nBands, c.nChannels), Eigen::ArrayXXf(c.nBands, c.nChannels)); }

    static auto validInput(Input input, const Coefficients &c) { return (input >= 0.f).all() && (input.rows() == c.nBands) && (input.cols() == c.nChannels); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.powerNoise.rows() == c.nBands) && (output.powerNoise.cols() == c.nChannels) && (output.powerNoise >= 0.f).all() &&
               (output.activity.rows() == c.nBands) && (output.activity.cols() == c.nChannels) && (output.activity >= 0.f).all();
    }
};

// Estimate background noise power and return the signal-activity probability
// for each frequency band and channel.
class NoiseEstimationActivity : public Algorithm<NoiseEstimationActivityConfiguration>
{
  public:
    NoiseEstimationActivity() = default;
    NoiseEstimationActivity(const Coefficients &c);
};

// noise estimation based on an activity detector. This version has an additional output activity detector as a single boolean value
//
// author: Kristian Timm Andersen
struct NoiseEstimationActivityFusedConfiguration
{
    using Input = I::Real2D;
    struct Output
    {
        O::Real2D powerNoise;
        O::Boolean activity;
    };

    struct Coefficients
    {
        float filterbankRate = 125.f;
        int nBands = 257;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterbankRate, nChannels, nBands)
    };

    struct Parameters
    {
        float smoothingTConstant = .072f;
        DEFINE_TUNABLE_PARAMETERS(smoothingTConstant)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // power spectrum

    static auto initOutput(Input input, const Coefficients &c) { return std::make_tuple(Eigen::ArrayXXf::Zero(c.nBands, c.nChannels), bool()); }

    static auto validInput(Input input, const Coefficients &c) { return (input >= 0.f).all() && (input.rows() == c.nBands) && (input.cols() == c.nChannels); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.powerNoise.rows() == c.nBands) && (output.powerNoise.cols() == c.nChannels) && (output.powerNoise >= 0.f).all();
    }
};

// Estimate background noise power and combine activity across bands and channels
// into a single signal-present flag.
class NoiseEstimationActivityFused : public Algorithm<NoiseEstimationActivityFusedConfiguration>
{
  public:
    NoiseEstimationActivityFused() = default;
    NoiseEstimationActivityFused(const Coefficients &c);
};
