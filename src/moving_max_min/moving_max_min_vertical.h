#pragma once
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"

// moving max min filter
// The algorithm computes the moving maximum and then minimum of the 2D input over the vertical axis
// The delay of the algorithm is filterLength - 1
//
// author: Kristian Timm Andersen
struct MovingMaxMinVerticalConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int filterLength = 5; // number of samples to filter over the first dimension
        int nChannels = 100;  // length of first dimension
        DEFINE_TUNABLE_COEFFICIENTS(filterLength, nChannels);
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nChannels, 1000); } // arbitrary number of inputs

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nChannels, input.cols()); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nChannels) && (input.cols() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nChannels) && (output.cols() > 0) && output.allFinite(); }
};

class MovingMaxMinVertical : public AlgorithmImplementation<MovingMaxMinVerticalConfiguration, MovingMaxMinVertical>
{
  public:
    MovingMaxMinVertical(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c},                                            //
          filterMin({.filterLength = c.filterLength, .nChannels = 1}), //
          filterMax({.filterLength = c.filterLength, .nChannels = 1})
    {
        maxOut.resize(c.nChannels);
    }

    FilterMinLemire filterMin;
    FilterMaxLemire filterMax;
    DEFINE_MEMBER_ALGORITHMS(filterMin, filterMax)

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto iCol = 0; iCol < input.cols(); iCol++)
        {
            filterMax.process(input.col(iCol), maxOut);
            filterMin.process(maxOut, output.col(iCol));
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = maxOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXf maxOut;
    friend BaseAlgorithm;
};