#pragma once
#include "algorithm_library/interface/interface.h"
#include "framework/framework.h"

// Upscale a 2D float array with a factor in horizontal and vertical direction
//
// author: Kristian Timm Andersen

struct Upscale2DConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int factorHorizontal = 4;
        int factorVertical = 4;
        bool leftBoundaryExcluded = false;
        DEFINE_TUNABLE_COEFFICIENTS(factorHorizontal, factorVertical, leftBoundaryExcluded)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(100, 100); // arbitrary number of samples
    }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c)
    {
        int nRows = c.factorVertical * (static_cast<int>(input.rows()) - 1) + 1;
        int nCols = c.factorHorizontal * (static_cast<int>(input.cols()) - 1) + 1;
        nCols -= c.leftBoundaryExcluded ? 1 : 0;
        return Eigen::ArrayXXf::Zero(nRows, nCols);
    }

    static bool validInput(Input input, const Coefficients &c) { return input.allFinite() && (input.rows() > 0) && (input.cols() > 0); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return output.allFinite() && (output.rows() >= c.factorVertical) && (output.cols() >= c.factorHorizontal);
    }
};

// upscale a 2D float array using linear interpolation
//
// author: Kristian Timm Andersen
class Upscale2DLinear : public AlgorithmImplementation<Upscale2DConfiguration, Upscale2DLinear>
{
  public:
    Upscale2DLinear(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        interpolateVertical = Eigen::VectorXf::LinSpaced(c.factorVertical, 1, 1.f / c.factorVertical);
        interpolateHorizontal = Eigen::VectorXf::LinSpaced(c.factorHorizontal, 1, 1.f / c.factorHorizontal);
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        const int cols = static_cast<int>(input.cols());
        const int colsm1 = cols - 1;
        const int rows = static_cast<int>(input.rows());
        const int rowsm1 = rows - 1;

        int startCol = 0; // start from column 0
        if (C.leftBoundaryExcluded == true)
        {
            startCol = 1; // exclude column 0

            // In this case we write the upscaled vertical output for the first input column into the first column.
            // linear interpolation across rows
            for (auto iRow = 0; iRow < rowsm1; iRow++)
            {
                float diff0 = input(iRow, 0) - input(iRow + 1, 0);
                for (auto i = 0; i < C.factorVertical; i++)
                {
                    output(i + iRow * C.factorVertical, 0) = diff0 * interpolateVertical(i) + input(iRow + 1, 0);
                }
            }
            output(rowsm1 * C.factorVertical, 0) = input(rowsm1, 0);
        }

        for (auto iCol = startCol; iCol < cols; iCol++)
        {
            // linear interpolation across rows
            for (auto iRow = 0; iRow < rowsm1; iRow++)
            {
                for (auto i = 0; i < C.factorVertical; i++)
                {
                    output(i + iRow * C.factorVertical, iCol * C.factorHorizontal - startCol) =
                        (input(iRow, iCol) - input(iRow + 1, iCol)) * interpolateVertical(i) + input(iRow + 1, iCol);
                }
            }
            output(rowsm1 * C.factorVertical, iCol * C.factorHorizontal - startCol) = input(rowsm1, iCol);
        }

        // linear interpolation across first col
        for (auto iHor = 1 + startCol; iHor < C.factorHorizontal; iHor++)
        {
            output.col(iHor - startCol) = output.col(0) * interpolateHorizontal(iHor);
        }
        if ((C.leftBoundaryExcluded == true) && (C.factorHorizontal > 1)) { output.col(0) *= interpolateHorizontal(1); }

        for (auto iCol = 1; iCol < colsm1; iCol++)
        {
            for (auto iHor = 1; iHor < C.factorHorizontal; iHor++)
            {
                // linear interpolation across cols
                output.col(iCol * C.factorHorizontal + iHor - startCol) = output.col(iCol * C.factorHorizontal - startCol) * interpolateHorizontal(iHor);
                // backwards linear interpolation across cols
                output.col(iCol * C.factorHorizontal - iHor - startCol) += output.col(iCol * C.factorHorizontal + iHor - startCol);
            }
        }
        // backwards linear interpolation across last col
        for (auto iHor = 1; iHor < C.factorHorizontal; iHor++)
        {
            output.col((colsm1 - 1) * C.factorHorizontal + iHor - startCol) +=
                output.col(colsm1 * C.factorHorizontal - startCol) * interpolateHorizontal(C.factorHorizontal - iHor);
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = interpolateVertical.getDynamicMemorySize();
        size += interpolateHorizontal.getDynamicMemorySize();
        return size;
    }

    Eigen::VectorXf interpolateVertical;
    Eigen::VectorXf interpolateHorizontal;
    friend BaseAlgorithm;
};