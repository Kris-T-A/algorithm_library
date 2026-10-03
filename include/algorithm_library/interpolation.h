#pragma once
#include "interface/interface.h"

// Interpolation method.
// Input is a size 4 array and FractionalDelay is
// in the interval[0, 1[ between Input[1] and Input[2].
//
// author: Kristian Timm Andersen

// -----------------------------  interpolation of one sample --------------------------------------------

struct InterpolationSampleConfiguration
{
    struct Input
    {
        I::Real4 samples;
        I::Float fractionalDelay; // fractional index to interpolate. It must be between 0.0 - 1.0
    };

    using Output = O::Float;

    struct Coefficients
    {
        DEFINE_NO_TUNABLE_COEFFICIENTS
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::Array4f, float> initInput(const Coefficients &c)
    {
        Eigen::Array4f samples = Eigen::Array4f::Random(); // size 4 array samples. Interpolation is done between samples 1 and 2
        float fractionalDelay = 0.3f;                      // fractional index to interpolate. It must be between 0.0 - 1.0
        return std::make_tuple(samples, fractionalDelay);
    }

    static float initOutput(Input input, const Coefficients &c) { return 0.f; } // output is a single float value (interpolated value

    static bool validInput(Input input, const Coefficients &c)
    {
        return (input.samples.size() == 4) && input.samples.allFinite() && (input.fractionalDelay >= 0.f) && (input.fractionalDelay <= 1.f);
    }

    static bool validOutput(Output output, const Coefficients &c) { return std::isfinite(output); }
};

// Estimate one value between the two middle samples of a four-sample neighborhood
// using cubic interpolation.
class InterpolationSample : public Algorithm<InterpolationSampleConfiguration>
{
  public:
    InterpolationSample() = default;
    InterpolationSample(const Coefficients &c);
};

// -----------------------------  interpolation of array ------------------------------------------------------

struct InterpolationConfiguration
{
    struct Input
    {
        I::Real samples;
        I::Real fractionalIndices; // fractional indices to interpolate. Since 4 points are needed to do cubic interpolation it is not possible to interpolate indices smaller
                                   // than 1.0 and larger than samples.size()-2.0
    };

    using Output = O::Real;

    struct Coefficients
    {
        DEFINE_NO_TUNABLE_COEFFICIENTS
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> initInput(const Coefficients &c)
    {
        int nSamples = 100;
        Eigen::ArrayXf samples = Eigen::ArrayXf::Random(nSamples);                                          // samples. Number of samples can be arbitrary
        Eigen::ArrayXf fractionalIndices = Eigen::ArrayXf::Random(nSamples).abs() * (nSamples - 3.f) + 1.f; // indices must be between 1.0 and samples.size()-2.0
        return std::make_tuple(samples, fractionalIndices);
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(input.fractionalIndices.size()); } // interpolated samples

    static bool validInput(Input input, const Coefficients &c)
    {
        if (input.fractionalIndices.size() < 1) return false; // if not at least one value
        float minFractional = input.fractionalIndices(0);
        float maxFractional = input.fractionalIndices(input.fractionalIndices.size() - 1);
        if (minFractional < 1.0 || maxFractional > input.samples.size() - 2.0) { return false; } // if not in the interval [1.0, samples.size()-2.0]
        return input.samples.size() > 3 && input.samples.allFinite();                            // if not at least 4 samples
    }

    static bool validOutput(Output output, const Coefficients &c) { return output.allFinite(); }
};

// Evaluate cubic interpolation at arbitrary fractional indices of an input array.
class Interpolation : public Algorithm<InterpolationConfiguration>
{
  public:
    Interpolation() = default;
    Interpolation(const Coefficients &c);
};

// -----------------------------  interpolation of array with constant fractionalDelay. This is faster ------------------------------------------------------

struct InterpolationConstantConfiguration
{
    using Input = I::Real;
    using Output = O::Real;

    struct Coefficients
    {
        float fractionalDelay = 0.5;
        DEFINE_TUNABLE_COEFFICIENTS(fractionalDelay)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(100); } // interpolation samples. Number of samples can be arbitrary

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(input.size() - 3); } // interpolated samples

    static bool validInput(Input input, const Coefficients &c) { return (input.size() > 3) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return output.allFinite(); }
};

// Interpolate a sequence using a fixed fractional delay and precomputed cubic weights.
class InterpolationConstant : public Algorithm<InterpolationConstantConfiguration>
{
  public:
    InterpolationConstant() = default;
    InterpolationConstant(const Coefficients &c);
};
