#pragma once
#include "interface/interface.h"

// Acticity detection calculated in the frequency domain based on noise estimation and activity detector.
//
// author: Kristian Timm Andersen

struct ActivityDetectionConfiguration
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
        float smoothingTConstant = .072f; // how much to smooth the noise estimation
        DEFINE_TUNABLE_PARAMETERS(smoothingTConstant)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // power spectrum

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nBands, c.nChannels); } // activity between 0 and 1

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input.cols() == c.nChannels) && (input >= 0.f).all(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.rows() == c.nBands) && (output.cols() == c.nChannels) && (output >= 0).all() && (output <= 1.f).all();
    }
};

// Estimate signal activity probabilities for each frequency band and channel
// while tracking background noise power.
class ActivityDetection : public Algorithm<ActivityDetectionConfiguration>
{
  public:
    ActivityDetection() = default;
    ActivityDetection(const Coefficients &c);
};

// ActivityDetectionFused is a version of ActivityDetection where the output is combined into a single boolean value
struct ActivityDetectionFusedConfiguration
{
    using Input = I::Real2D;
    using Output = O::Boolean;

    struct Coefficients
    {
        float filterbankRate = 125.f;
        int nBands = 257;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterbankRate, nChannels, nBands)
    };

    struct Parameters
    {
        float activityThreshold = 1.f; // parameter to set how aggresive the detection is
        DEFINE_TUNABLE_PARAMETERS(activityThreshold)
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nBands, c.nChannels).abs2(); } // power spectrum

    static bool initOutput(Input input, const Coefficients &c) { return false; } // activity flag

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input.cols() == c.nChannels) && (input >= 0).all(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return true; // bool value is always valid
    }
};

// Combine frequency-band activity and channel power into one signal-activity flag
// while tracking background noise power.
class ActivityDetectionFused : public Algorithm<ActivityDetectionFusedConfiguration>
{
  public:
    ActivityDetectionFused() = default;
    ActivityDetectionFused(const Coefficients &c);
};
