#pragma once
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"

// moving max min filter
// The algorithm computes the moving maximum and then minimum of the 2D input over the vertical axis
// The delay of the algorithm is filterLength - 1
//
// author: Kristian Timm Andersen
struct MovingMaxMinVerticalConfiguration2
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int filterLength = 5; // number of samples to filter over the first dimension
        int nChannels = 1000; // length of first dimension
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

class MovingMaxMinVertical2 : public AlgorithmImplementation<MovingMaxMinVerticalConfiguration2, MovingMaxMinVertical2>
{
  public:
    MovingMaxMinVertical2(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        assert(c.filterLength > 0);
        maxOut.resize(c.filterLength);
        minOut.resize(c.filterLength);
        counter = 0;
        wHalf = (c.filterLength - 1) / 2 + 1;
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto iCol = 0; iCol < input.cols(); iCol++)
        {
            maxOut.setConstant(input(0, iCol));
            minOut.setConstant(input(0, iCol));
            counter = 0;
            // this output is discarded and is only used to update internal values
            int iRow = 1;
            for (; iRow < wHalf; iRow++)
            {
                maxOut(counter) = input(iRow, iCol);
                minOut(counter) = maxOut.maxCoeff();
                counter++;
                if (counter >= C.filterLength) { counter = 0; }
            }
            // This is the shifted filter, which creates a symmetric window
            int iOut = 0;
            for (; iRow < input.rows(); iRow++, iOut++)
            {
                maxOut(counter) = input(iRow, iCol);
                minOut(counter) = maxOut.maxCoeff();
                output(iOut, iCol) = minOut.minCoeff();
                counter++;
                if (counter >= C.filterLength) { counter = 0; }
            }
            for (; iOut < input.rows(); iOut++)
            {
                maxOut(counter) = input(iRow - 1, iCol);
                minOut(counter) = maxOut.maxCoeff();
                output(iOut, iCol) = minOut.minCoeff();
                counter++;
                if (counter >= C.filterLength) { counter = 0; }
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = maxOut.getDynamicMemorySize();
        size += minOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXf maxOut;
    Eigen::ArrayXf minOut;
    int counter = 0;
    int wHalf;
    friend BaseAlgorithm;
};