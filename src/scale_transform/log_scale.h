#pragma once
#include "algorithm_library/scale_transform.h"
#include "framework/framework.h"
#include "interpolation/interpolation_cubic.h"
#include "utilities/fastonebigheader.h"
#include <iostream>
// Logarithmic Scale
//
// author: Kristian Timm Andersen

class LogScale : public AlgorithmImplementation<ScaleTransformConfiguration, LogScale>
{
  public:
    LogScale(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        float minFreq = 20;                                                  // minimum corner frequency in Hz
        float freqPerBin = static_cast<float>(c.indexEnd) / (c.nInputs - 1); // frequency difference between two adjacent bins
        // linear scale from corresponding minFreq to log10(nInputs)
        Eigen::ArrayXf linLogs = Eigen::ArrayXf::LinSpaced(c.nOutputs, std::log10(1.f + minFreq / freqPerBin), std::log10(static_cast<float>(c.nInputs)));
        // logarithmic scale from 0 to nInputs-1. Corresponds to index of center bins in the input array (size: nOutputs)
        Eigen::ArrayXf centerBins = (Eigen::ArrayXf::Constant(c.nOutputs, 10).pow(linLogs) - 1.f);

        // count number of output bins that has width smaller or equal to 1 (corresponds to upsampling)
        // these output bins will be calculated using linear interpolation
        nLinearBins = 0; // number of bins calculated using linear interpolation
        while (((centerBins(nLinearBins + 1) - centerBins(nLinearBins) <= 1.f) && (nLinearBins < c.nOutputs - 1)) || (centerBins(nLinearBins) < 1.f))
        {
            nLinearBins++;
        }
        indexStart = centerBins.head(nLinearBins).cast<int>();
        fractionLinear = centerBins.head(nLinearBins) - indexStart.cast<float>(); // fraction for linear interpolation

        // count number of output bins that has width smaller or equal to 2 (corresponds to downsampling by a factor of 2 or less)
        // these output bins will be calculated using cubic interpolation
        int nSum = nLinearBins; // number of bins calculated using linear and cubic interpolation
        while (((centerBins(nSum + 1) - centerBins(nSum) <= 2.f) || (centerBins(nSum) < 2.f)) && (nSum < c.nOutputs - 2))
        {
            nSum++;
        }
        nCubicBins = nSum - nLinearBins; // number of cubic bins
        fractionCubic = centerBins.segment(nLinearBins, nCubicBins);

        // count number of output bins that has width larger than 2 (corresponds to downsampling by a factor larger than 2)
        // these output bins will be calculated using triangular interpolation
        nTriangularBins = c.nOutputs - nSum; // number of triangular bins
        fractionTriangular = centerBins.segment(nSum, nTriangularBins);
        distanceTriangular = centerBins.segment(nSum, nTriangularBins) - centerBins.segment(nSum - 1, nTriangularBins);
    }

    InterpolationCubic interpolationCubic;
    DEFINE_MEMBER_ALGORITHMS(interpolationCubic)

    inline void inverse(I::Real2D xPower, O::Real2D yPower)
    {
        // assert(xPower.rows() == C.nOutputs);

        // for (auto channel = 0; channel < xPower.cols(); channel++)
        // {
        //     for (auto i = 0; i < C.nOutputs; i++)
        //     {
        //         yPower.block(indexStart(i), channel, nInputsSum(i), 1).setConstant(xPower(i, channel));
        //     }
        //     yPower.block(indexEnd, channel, C.nInputs - indexEnd, 1).setConstant(xPower(C.nOutputs - 1, channel));
        // }
    }

    Eigen::ArrayXf getCenterIndices() const
    {
        Eigen::ArrayXf array(C.nOutputs);
        array.head(nLinearBins) = indexStart.cast<float>() + fractionLinear; // linear interpolation
        array.segment(nLinearBins, nCubicBins) = fractionCubic; // cubic interpolation
        array.segment(nLinearBins + nCubicBins, nTriangularBins) = fractionTriangular; // triangular interpolation
        return array;
    }

    Eigen::ArrayXf getCenterFrequencies() const
    {
        return getCenterIndices() * (C.indexEnd / (C.nInputs - 1)); // convert indices to frequencies
    }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto channel = 0; channel < input.cols(); channel++)
        {
            // linear interpolation
            for (auto i = 0; i < nLinearBins; i++)
            {
                // weighted sum
                output(i, channel) = (1.f - fractionLinear(i)) * input(indexStart(i), channel) + fractionLinear(i) * input(indexStart(i) + 1, channel);
            }

            // cubic interpolation
            interpolationCubic.process({input.col(channel), fractionCubic}, output.col(channel).segment(nLinearBins, nCubicBins));

            // triangular interpolation
            for (auto i = 0; i < nTriangularBins - 1; i++)
            {
                int iStart = std::ceil(fractionTriangular(i) - distanceTriangular(i));
                int iMid = std::ceil(fractionTriangular(i));
                int iEnd = std::ceil(fractionTriangular(i) + distanceTriangular(i + 1));
                output(i + nLinearBins + nCubicBins, channel) = -10000; // initialize to a very low value
                for (auto iBin = iStart; iBin < iMid; iBin++)
                {
                    float weightdB = energy2dB(1.f - (fractionTriangular(i) - iBin) / distanceTriangular(i));
                    output(i + nLinearBins + nCubicBins, channel) = std::max(output(i + nLinearBins + nCubicBins, channel), input(iBin, channel) + weightdB);
                }
                for (auto iBin = iMid; iBin < iEnd; iBin++)
                {
                    float weightdB = energy2dB(1.f - (iBin - fractionTriangular(i)) / distanceTriangular(i + 1));
                    output(i + nLinearBins + nCubicBins, channel) = std::max(output(i + nLinearBins + nCubicBins, channel), input(iBin, channel) + weightdB);
                }
            }
            int iStart = std::ceil(fractionTriangular(nTriangularBins - 1) - distanceTriangular(nTriangularBins - 1));
            int iMid = std::ceil(fractionTriangular(nTriangularBins - 1));
            output(nTriangularBins - 1 + nLinearBins + nCubicBins, channel) = -10000; // initialize to a very low value
            for (auto iBin = iStart; iBin < iMid; iBin++)
            {
                float weightdB = energy2dB(1.f - (fractionTriangular(nTriangularBins - 1) - iBin) / distanceTriangular(nTriangularBins - 1));
                output(nTriangularBins - 1 + nLinearBins + nCubicBins, channel) =
                    std::max(output(nTriangularBins - 1 + nLinearBins + nCubicBins, channel), input(iBin, channel) + weightdB);
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = indexStart.getDynamicMemorySize();
        size += fractionLinear.getDynamicMemorySize();
        size += fractionCubic.getDynamicMemorySize();
        size += fractionTriangular.getDynamicMemorySize();
        size += distanceTriangular.getDynamicMemorySize();
        return size;
    }

  public:
    int nLinearBins, nCubicBins, nTriangularBins; // number of bins calculated using linear, cubic and triangular interpolation
    Eigen::ArrayXi indexStart;
    Eigen::ArrayXf fractionLinear;
    Eigen::ArrayXf fractionCubic;
    Eigen::ArrayXf fractionTriangular;
    Eigen::ArrayXf distanceTriangular;

    friend BaseAlgorithm;
};