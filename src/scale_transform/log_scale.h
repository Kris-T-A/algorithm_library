#pragma once
#include "algorithm_library/scale_transform.h"
#include "framework/framework.h"
#include "interpolation/interpolation_cubic.h"
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
        // logarithmic scale from 0 to nInputs-1. Corresponds to index of corner bins in the input array (size: nOutputs)
        Eigen::ArrayXf centerBins = (Eigen::ArrayXf::Constant(c.nOutputs, 10).pow(linLogs) - 1.f);
        Eigen::ArrayXf freqs = freqPerBin * centerBins; // convert center bins to frequencies

        std::cout << "freqs: " << freqs.transpose() << std::endl;
        std::cout << "freqs.size(): " << freqs.size() << std::endl;
        std::cout << "centerBins: " << centerBins.transpose() << std::endl;
        std::cout << "centerBins.size(): " << centerBins.size() << std::endl;

        // count number of output bins that has width smaller or equal to 1 (corresponds to upsampling)
        // these output bins will be calculated using linear interpolation
        nLinearBins = 0; // number of bins calculated using linear interpolation
        while (((centerBins(nLinearBins + 1) - centerBins(nLinearBins) <= 1.f) && (nLinearBins < c.nOutputs - 1)) || (centerBins(nLinearBins) < 1.f))
        {
            nLinearBins++;
        }
        fractionLinear = centerBins.head(nLinearBins) - centerBins.head(nLinearBins).floor(); // fraction for linear interpolation
        indexStart = centerBins.head(nLinearBins).cast<int>();

        // count number of output bins that has width smaller or equal to 2 (corresponds to downsampling by a factor of 2 or less)
        // these output bins will be calculated using cubic interpolation
        int nSum = nLinearBins; // number of bins calculated using linear and cubic interpolation
        while (((centerBins(nSum + 1) - centerBins(nSum) <= 2.f) || (centerBins(nSum) < 2.f)) && (nSum < c.nOutputs - 2))
        {
            nSum++;
        }
        nCubicBins = nSum - nLinearBins; // number of cubic bins
        fractionCubic = centerBins.segment(nLinearBins, nCubicBins);
    }

    InterpolationCubic interpolationCubic;
    DEFINE_MEMBER_ALGORITHMS(interpolationCubic)

    // inline void inverse(I::Real2D xPower, O::Real2D yPower)
    // {
    //     assert(xPower.rows() == C.nOutputs);

    //     for (auto channel = 0; channel < xPower.cols(); channel++)
    //     {
    //         for (auto i = 0; i < C.nOutputs; i++)
    //         {
    //             yPower.block(indexStart(i), channel, nInputsSum(i), 1).setConstant(xPower(i, channel));
    //         }
    //         yPower.block(indexEnd, channel, C.nInputs - indexEnd, 1).setConstant(xPower(C.nOutputs - 1, channel));
    //     }
    // }

    // // Indices corresponds to frequencies if indexEnd is half the sample rate
    // Eigen::ArrayXf getCornerIndices() const
    // {
    //     Eigen::ArrayXf array(C.nOutputs + 1);
    //     array.head(nLinearBins) = (1 - binsWeight) * indexStart.head(nLinearBins).cast<float>() + binsWeight * (indexStart.head(nLinearBins) + 1).cast<float>();
    //     array.segment(nLinearBins, C.nOutputs - nLinearBins) = indexStart.segment(nLinearBins, C.nOutputs - nLinearBins).cast<float>();
    //     array.tail(1) = static_cast<float>(indexEnd);
    //     array *= C.indexEnd / (C.nInputs - 1);
    //     return array;
    // }

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
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = indexStart.getDynamicMemorySize();
        size += fractionLinear.getDynamicMemorySize();
        return size;
    }

  public:
    int nLinearBins, nCubicBins;
    Eigen::ArrayXi indexStart;
    Eigen::ArrayXf fractionLinear;
    Eigen::ArrayXf fractionCubic;

    friend BaseAlgorithm;
};