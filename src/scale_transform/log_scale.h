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
        assert(c.outputEnd <= c.inputEnd);
        // use double precision in the calculation of centerBins to ensure accuracy in log and pow conversions
        double scale = c.transformType == Coefficients::TransformType::LOGARITHMIC ? 1.0 : 1.0 / 700.0; // scale = 1 for logarithmic scale, 1 / 700 for Mel scale
        double minLog = std::log10(1.0 + scale * c.outputStart);                                        // minimum log value
        double maxLog = std::log10(1.0 + scale * c.outputEnd);                                          // maximum log value
        // linear scale from corresponding outputStart to outputEnd
        Eigen::ArrayXd linLogs = Eigen::ArrayXd::LinSpaced(c.nOutputs, minLog, maxLog); // linear spaced center indices in logarithmic domain
        // logarithmic scale corresponding to index of center bins in the input array (size: nOutputs)
        const double freqPerBin = scale * static_cast<double>(c.inputEnd) / (c.nInputs - 1); // frequency difference between two adjacent bins multiplied by scale
        Eigen::ArrayXf centerBins = linLogs.unaryExpr([&freqPerBin](double x) { return (std::pow(10, x) - 1.0) / freqPerBin; }).cast<float>();

        // count number of output bins that has width smaller or equal to 1 (corresponds to upsampling)
        // these output bins will be calculated using linear interpolation
        nLinearBins = 0; // number of bins calculated using linear interpolation
        while ((nLinearBins < c.nOutputs - 1) && ((centerBins(nLinearBins + 1) - centerBins(nLinearBins) <= 1.f) || (centerBins(nLinearBins) < 1.f)))
        {
            nLinearBins++;
        }
        outputStart = centerBins.head(nLinearBins).cast<int>();
        fractionLinear = centerBins.head(nLinearBins) - outputStart.cast<float>(); // fraction for linear interpolation

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

        // Precompute triangular weights matrix (dB) — replaces the runtime energy2dB calls.
        // Rows i ∈ [0, nTriangularBins), cols j ∈ [0, nInputs). Value = 10*log10(linear weight) for j inside the
        // triangular window centered at fractionTriangular(i); -inf otherwise. The mid-bin is given full weight (0 dB).
        triangularWeights = Eigen::ArrayXXf::Constant(nTriangularBins, c.nInputs, -std::numeric_limits<float>::infinity());
        for (int i = 0; i < nTriangularBins; ++i)
        {
            const int iMid = static_cast<int>(std::ceil(fractionTriangular(i)));
            const int iStart = static_cast<int>(std::ceil(fractionTriangular(i) - distanceTriangular(i)));
            const int iEnd = (i < nTriangularBins - 1)
                                 ? static_cast<int>(std::ceil(fractionTriangular(i) + distanceTriangular(i + 1)))
                                 : iMid; // last bin has no right half
            triangularWeights(i, iMid) = 0.0f; // full linear weight = 1.0 → 0 dB
            const float distLeft = static_cast<float>(iMid - iStart);
            for (int iBin = iStart; iBin < iMid; ++iBin)
            {
                const float linWeight = 1.0f - (iMid - iBin) / distLeft;
                triangularWeights(i, iBin) = 10.0f * std::log10(linWeight);
            }
            if (iEnd > iMid)
            {
                const float distRight = static_cast<float>(iEnd - iMid);
                for (int iBin = iMid + 1; iBin < iEnd; ++iBin)
                {
                    const float linWeight = 1.0f - (iBin - iMid) / distRight;
                    triangularWeights(i, iBin) = 10.0f * std::log10(linWeight);
                }
            }
        }
    }

    InterpolationCubic interpolationCubic;
    DEFINE_MEMBER_ALGORITHMS(interpolationCubic)

    inline void inverse(I::Real2D x, O::Real2D y)
    {
        assert(x.rows() == C.nOutputs);
        Eigen::ArrayXf indices = getCenterIndices();
        Eigen::ArrayXf cornerIndices(C.nOutputs + 1);
        cornerIndices << 0, ((indices.head(C.nOutputs - 1) + indices.tail(C.nOutputs - 1)) / 2).round(), static_cast<float>(C.nInputs); // corner indices for the output bins
        Eigen::ArrayXf diff = cornerIndices.tail(C.nOutputs) - cornerIndices.head(C.nOutputs); // get difference between adjacent indices

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
        array.head(nLinearBins) = outputStart.cast<float>() + fractionLinear;          // linear interpolation
        array.segment(nLinearBins, nCubicBins) = fractionCubic;                        // cubic interpolation
        array.segment(nLinearBins + nCubicBins, nTriangularBins) = fractionTriangular; // triangular interpolation
        return array;
    }

    Eigen::ArrayXf getCenterFrequencies() const
    {
        return getCenterIndices() * (C.outputEnd / (C.nInputs - 1)); // convert indices to frequencies
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
                output(i, channel) = (1.f - fractionLinear(i)) * input(outputStart(i), channel) + fractionLinear(i) * input(outputStart(i) + 1, channel);
            }

            // cubic interpolation
            interpolationCubic.process({input.col(channel), fractionCubic}, output.col(channel).segment(nLinearBins, nCubicBins));

            // triangular interpolation (precomputed weights)
            for (int i = 0; i < nTriangularBins; ++i)
            {
                output(i + nLinearBins + nCubicBins, channel) = (input.col(channel) + triangularWeights.row(i).transpose()).maxCoeff();
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = outputStart.getDynamicMemorySize();
        size += fractionLinear.getDynamicMemorySize();
        size += fractionCubic.getDynamicMemorySize();
        size += fractionTriangular.getDynamicMemorySize();
        size += distanceTriangular.getDynamicMemorySize();
        size += triangularWeights.getDynamicMemorySize();
        return size;
    }

  public:
    int nLinearBins, nCubicBins, nTriangularBins; // number of bins calculated using linear, cubic and triangular interpolation
    Eigen::ArrayXi outputStart;
    Eigen::ArrayXf fractionLinear;
    Eigen::ArrayXf fractionCubic;
    Eigen::ArrayXf fractionTriangular;
    Eigen::ArrayXf distanceTriangular;
    Eigen::ArrayXXf triangularWeights; // dense (nTriangularBins, nInputs); -inf outside each triangular window

    friend BaseAlgorithm;
};