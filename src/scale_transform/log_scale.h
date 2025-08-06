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
        // use double precision in the calculation of centerBins to ensure accuracy in log and pow conversions
        double freqPerBin = static_cast<double>(c.indexEnd) / (c.nInputs - 1);                                // frequency difference between two adjacent bins
        double scale = c.transformType == Coefficients::TransformType::LOGARITHMIC ? 1.0 : freqPerBin / 700; // scale = 1 for logarithmic scale, freqPerBin / 700 for Mel scale
        double minLog = std::log10(1.0 + scale * c.indexStart / freqPerBin);                                 // minimum log value
        double maxLog = std::log10(1.0 + scale * c.indexEnd / freqPerBin);                                   // maximum log value
        // linear scale from corresponding indexStart to indexEnd
        Eigen::ArrayXd linLogs = Eigen::ArrayXd::LinSpaced(c.nOutputs, minLog, maxLog); // linear spaced center indices in logarithmic domain
        // logarithmic scale corresponding to index of center bins in the input array (size: nOutputs)
        Eigen::ArrayXf centerBins = linLogs.unaryExpr([&scale](double x) { return (std::pow(10, x) - 1.0)/scale; }).cast<float>();

        // count number of output bins that has width smaller or equal to 1 (corresponds to upsampling)
        // these output bins will be calculated using linear interpolation
        nLinearBins = 0; // number of bins calculated using linear interpolation
        while ((nLinearBins < c.nOutputs - 1) && ((centerBins(nLinearBins + 1) - centerBins(nLinearBins) <= 1.f) || (centerBins(nLinearBins) < 1.f)))
        {
            nLinearBins++;
        }
        indexStart = centerBins.head(nLinearBins).cast<int>();
        fractionLinear = centerBins.head(nLinearBins) - indexStart.cast<float>(); // fraction for linear interpolation

        // count number of output bins that has width smaller or equal to 2 (corresponds to downsampling by a factor of 2 or less)
        // these output bins will be calculated using cubic interpolation
        int nSum = nLinearBins; // number of bins calculated using linear and cubic interpolation
        while ((nSum < c.nOutputs - 2) && ((centerBins(nSum + 1) - centerBins(nSum) <= 2.f) || (centerBins(nSum) < 2.f)))
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

    inline void inverse(I::Real2D x, O::Real2D y)
    {
        assert(x.rows() == C.nOutputs);
        Eigen::ArrayXf indices = getCenterIndices();
        Eigen::ArrayXf cornerIndices(C.nOutputs + 1);
        cornerIndices << 0, ((indices.head(C.nOutputs - 1) + indices.tail(C.nOutputs - 1)) / 2).round(), static_cast<float>(C.nInputs); // corner indices for the output bins
        Eigen::ArrayXf diff = cornerIndices.tail(C.nOutputs) - cornerIndices.head(C.nOutputs);                       // get difference between adjacent indices

        for (auto channel = 0; channel < x.cols(); channel++)
        {
            for (auto i = 0; i < C.nOutputs; i++)
            {

                y.col(channel).segment(static_cast<size_t>(cornerIndices(i)), diff(i)).setConstant(x(i, channel));
            }
        }
    }

    Eigen::ArrayXf getCenterIndices() const
    {
        Eigen::ArrayXf array(C.nOutputs);
        array.head(nLinearBins) = indexStart.cast<float>() + fractionLinear;           // linear interpolation
        array.segment(nLinearBins, nCubicBins) = fractionCubic;                        // cubic interpolation
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
                auto iStart = static_cast<int>(std::ceil(fractionTriangular(i) - distanceTriangular(i)));
                auto iMid = static_cast<int>(std::ceil(fractionTriangular(i)));
                auto iEnd = static_cast<int>(std::ceil(fractionTriangular(i) + distanceTriangular(i + 1)));
                auto dist = static_cast<float>(iMid - iStart);
                output(i + nLinearBins + nCubicBins, channel) = input(iMid, channel); // initialize to iMid since it always has full weight
                for (auto iBin = iStart; iBin < iMid; iBin++)
                {
                    float weightdB = energy2dB(1.f - (iMid - iBin) / dist);
                    output(i + nLinearBins + nCubicBins, channel) = std::max(output(i + nLinearBins + nCubicBins, channel), input(iBin, channel) + weightdB);
                }
                dist = static_cast<float>(iEnd - iMid);
                for (auto iBin = iMid + 1; iBin < iEnd; iBin++)
                {
                    float weightdB = energy2dB(1.f - (iBin - iMid) / dist);
                    output(i + nLinearBins + nCubicBins, channel) = std::max(output(i + nLinearBins + nCubicBins, channel), input(iBin, channel) + weightdB);
                }
            }
            auto iStart = static_cast<int>(std::ceil(fractionTriangular(nTriangularBins - 1) - distanceTriangular(nTriangularBins - 1)));
            auto iMid = static_cast<int>(std::ceil(fractionTriangular(nTriangularBins - 1)));
            auto dist = static_cast<float>(iMid - iStart);
            output(nTriangularBins - 1 + nLinearBins + nCubicBins, channel) = input(iMid, channel); // initialize to iMid since it always has full weight
            for (auto iBin = iStart; iBin < iMid; iBin++)
            {
                float weightdB = energy2dB(1.f - (iMid - iBin) / dist);
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