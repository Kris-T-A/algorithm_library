
#pragma once
#include "interface/interface.h"

// StreamingMinMax finds the minimum and maximum value over a
// window with length C.Length for each new sample. It requires on
// average no more than 3 comparisons per sample. The algorithm uses 2
// double-ended queues for the minimum and maximum indices. A delay line
// is also used internally since in a true streaming application you need
// to be able to call the algorithm succesively with new frames (or just 1 new sample),
// and you are not guaranteed that the input frame is as long as C.Length.
// To be able to preallocate, the queues have been implemented as circular buffers.
//
// A symmetric version of StreamingMinMax has been implemented below called "Filter".
// It might be necessary to call the public ResetInitialValues function before Process,
// if certain initial conditions are required. Also versions that only find Min/Max have been implemented.
//
// ref: Daniel Lemire, STREAMING MAXIMUM - MINIMUM FILTER USING NO MORE THAN THREE COMPARISONS PER ELEMENT
//
// author : Kristian Timm Andersen

struct BaseFilterMinMaxConfiguration
{
    using Input = I::Real2D;

    struct Output
    {
        O::Real2D minValue;
        O::Real2D maxValue;
    };
    struct Coefficients
    {
        int filterLength = 100;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterLength, nChannels)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(1000, c.nChannels); } // time samples. Number of samples can be arbitrary

    static std::tuple<Eigen::ArrayXXf, Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        return std::make_tuple(Eigen::ArrayXXf::Zero(input.rows(), c.nChannels), Eigen::ArrayXXf::Zero(input.rows(), c.nChannels));
    } // minimum and maximum time samples. Number of samples can be arbitrary

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() > 0) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        bool flag = (output.minValue.rows() > 0) && (output.minValue.cols() == c.nChannels) && output.minValue.allFinite();
        flag &= (output.maxValue.rows() > 0) && (output.maxValue.cols() == c.nChannels) && output.maxValue.allFinite();
        return flag;
    }
};

struct FilterMinMaxConfiguration : public BaseFilterMinMaxConfiguration
{};
struct StreamingMinMaxConfiguration : public BaseFilterMinMaxConfiguration
{};

// Track the minimum and maximum over a trailing window independently for each channel,
// retaining history between input blocks.
class StreamingMinMax : public Algorithm<StreamingMinMaxConfiguration>
{
  public:
    StreamingMinMax() = default;
    StreamingMinMax(const Coefficients &c);

    // It might be necessary to call resetInitialValues function before process if certain initial conditions are required.
    void resetInitialValue(float inputOld);
    void resetInitialValue(I::Real inputOld);
};

// Compute minimum and maximum envelopes over a centered window in each input block,
// extending its first and last samples at the boundaries.
class FilterMinMax : public Algorithm<FilterMinMaxConfiguration>
{
  public:
    FilterMinMax() = default;
    FilterMinMax(const Coefficients &c);

    // It might be necessary to call resetInitialValues function before process if certain initial conditions are required.
    void resetInitialValue(const float inputOld);
    void resetInitialValue(I::Real inputOld);
};

// ---------------------------------------------------------------------------------------------------------------------
// Extremum configuration finds either min or max value and is a cheaper version of minmax configuration
struct BaseFilterExtremumConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int filterLength = 100;
        int nChannels = 2;
        DEFINE_TUNABLE_COEFFICIENTS(filterLength, nChannels)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(1000, c.nChannels); } // time samples. Number of samples can be arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXf::Zero(input.rows(), c.nChannels);
    } // extremum time samples. Number of samples can be arbitrary

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() > 0) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() > 0) && (output.cols() == c.nChannels) && output.allFinite(); }
};

struct FilterMaxConfiguration : public BaseFilterExtremumConfiguration
{};
struct FilterMinConfiguration : public BaseFilterExtremumConfiguration
{};
struct StreamingMaxConfiguration : public BaseFilterExtremumConfiguration
{};
struct StreamingMinConfiguration : public BaseFilterExtremumConfiguration
{};

// Track the maximum over a trailing window for each channel, retaining history
// between input blocks.
class StreamingMax : public Algorithm<StreamingMaxConfiguration>
{
  public:
    StreamingMax() = default;
    StreamingMax(const Coefficients &c);
    // It might be necessary to call the public ResetInitialValues function before Process,
    // if certain initial conditions are required.
    void resetInitialValue(const float inputOld);
    void resetInitialValue(I::Real inputOld);
};

// Track the minimum over a trailing window for each channel, retaining history
// between input blocks.
class StreamingMin : public Algorithm<StreamingMinConfiguration>
{
  public:
    StreamingMin() = default;
    StreamingMin(const Coefficients &c);
    // It might be necessary to call the public ResetInitialValues function before Process,
    // if certain initial conditions are required.
    void resetInitialValue(const float inputOld);
    void resetInitialValue(I::Real inputOld);
};

// Apply a centered maximum filter to each channel, extending the endpoint samples
// at the boundaries of the input block.
class FilterMax : public Algorithm<FilterMaxConfiguration>
{
  public:
    FilterMax() = default;
    FilterMax(const Coefficients &c);

    // It might be necessary to call resetInitialValues function before process if certain initial conditions are required.
    void resetInitialValue(const float inputOld);
    void resetInitialValue(I::Real inputOld);
};

// Apply a centered minimum filter to each channel, extending the endpoint samples
// at the boundaries of the input block.
class FilterMin : public Algorithm<FilterMinConfiguration>
{
  public:
    FilterMin() = default;
    FilterMin(const Coefficients &c);

    // It might be necessary to call resetInitialValues function before process if certain initial conditions are required.
    void resetInitialValue(const float inputOld);
    void resetInitialValue(I::Real inputOld);
};