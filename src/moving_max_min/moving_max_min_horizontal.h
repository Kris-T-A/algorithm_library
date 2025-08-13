#pragma once
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"

// moving max min filter
// The algorithm computes the moving maximum and then minimum of the 2D input over the horizontal axis
//
// author: Kristian Timm Andersen
struct MovingMaxMinHorizontalConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int filterLength = 5; // number of samples to filter over the second dimension
        int nChannels = 100;  // first dimension
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

class MovingMaxMinHorizontal : public AlgorithmImplementation<MovingMaxMinHorizontalConfiguration, MovingMaxMinHorizontal>
{
  public:
    MovingMaxMinHorizontal(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        maxOut.resize(c.nChannels, c.filterLength);
        minOut.resize(c.nChannels, c.filterLength);
        resetVariables();
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto sample = 0; sample < input.cols(); sample++)
        {
            maxOut.col(counter) = input.col(sample);
            minOut.col(counter) = maxOut.col(0);
            for (auto iMax = 1; iMax < C.filterLength; iMax++)
            {
                minOut.col(counter) = minOut.col(counter).max(maxOut.col(iMax));
            }
            output.col(sample) = minOut.col(0);
            for (auto iMin = 1; iMin < C.filterLength; iMin++)
            {
                output.col(sample) = output.col(sample).min(minOut.col(iMin));
            }
            counter++;
            if (counter >= C.filterLength) { counter = 0; }
        }
    }

    void resetVariables() final
    {
        maxOut.setConstant(-std::numeric_limits<float>::infinity());
        minOut.setConstant(std::numeric_limits<float>::infinity());
        counter = 0;
    }
    size_t getDynamicSizeVariables() const final
    {
        size_t size = maxOut.getDynamicMemorySize();
        size += minOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf maxOut;
    Eigen::ArrayXXf minOut;
    int counter;
    friend BaseAlgorithm;
};