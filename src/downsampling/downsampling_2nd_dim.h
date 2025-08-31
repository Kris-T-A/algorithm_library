#pragma once
#include "framework/framework.h"
#include "utilities/functions.h"

struct Downsampling2ndDimConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int ratio = 4;    // downsampling ratio
        int nBands = 100; // size of 1st dimension
        DEFINE_TUNABLE_COEFFICIENTS(ratio, nBands)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(c.nBands, 100);
    } // arbitrary number of time samples, but must be a multiple of C.ratio

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nBands, static_cast<int>(input.cols() / c.ratio)); }

    static bool validInput(Input input, const Coefficients &c)
    {
        return (input.rows() == c.nBands) && (input.cols() % c.ratio == 0) && input.allFinite();
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nBands) && output.allFinite(); }
};

// input is a matrix and downsampling is applied to the second dimension
class DownSampling2ndDim : public AlgorithmImplementation<Downsampling2ndDimConfiguration, DownSampling2ndDim>
{
  public:
    static constexpr int filterOversampling = 4; // how much filter is oversampled compared to downsampling ratio

    DownSampling2ndDim(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        windowLowpass = hann(filterOversampling * c.ratio);
        windowLowpass /= windowLowpass.sum(); // normalize window
        timeBuffer.resize(c.nBands, (filterOversampling - 1) * c.ratio);
        resetVariables();
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        const int downsampledCols = static_cast<int>(input.cols() / C.ratio);
        for (auto iCol = 0; iCol < downsampledCols; ++iCol)
        {
            output.col(iCol) = windowLowpass(0) * input.col(C.ratio * iCol); // first sample
            for (int iRatio = 1; iRatio < C.ratio; ++iRatio)
            {
                output.col(iCol) += windowLowpass(iRatio) * input.col(C.ratio * iCol + iRatio);
            }
            for (auto iBuffer = 0; iBuffer < timeBuffer.cols(); ++iBuffer)
            {
                // Apply the window function to the time buffer
                output.col(iCol) += windowLowpass(iBuffer + C.ratio) * timeBuffer.col(iBuffer);
            }
            timeBuffer.rightCols((filterOversampling - 2) * C.ratio) = timeBuffer.leftCols((filterOversampling - 2) * C.ratio);
            timeBuffer.leftCols(C.ratio) = input.middleCols(C.ratio * iCol, C.ratio);
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = windowLowpass.getDynamicMemorySize();
        size += timeBuffer.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final { timeBuffer.setZero(); }

    Eigen::ArrayXf windowLowpass;
    Eigen::ArrayXXf timeBuffer;

    friend BaseAlgorithm;
};