#pragma once
#include "interface/interface.h"

struct DelayConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 2;
        int delayLength = 10;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, delayLength)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(100, c.nChannels); } // time samples. Number of samples can be arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXXf::Zero(input.rows(), c.nChannels);
    } // time samples. Number of samples can be arbitrary

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() > 0) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() > 0) && (output.cols() == c.nChannels) && output.allFinite(); }
};

// Delay each input channel by the configured number of samples using a circular buffer.
class Delay : public Algorithm<DelayConfiguration>
{
  public:
    Delay() = default;
    Delay(const Coefficients &c);

    void push(Input input);
    void pop(Output output);

    float get(int index, int channel = 0);
    float get(float index, int channel = 0);
};