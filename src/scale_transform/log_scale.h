#pragma once
#include "algorithm_library/scale_transform.h"
#include "framework/framework.h"

// Logarithmic Scale
//
// author: Kristian Timm Andersen

class LogScale : public AlgorithmImplementation<ScaleTransformConfiguration, LogScale>
{
  public:
    LogScale(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        float lowFreqLog = static_cast<float>(c.indexEnd) / c.nInputs;
        float highFreqLog = std::log10(1.f + c.indexEnd / lowFreqLog);
        Eigen::ArrayXf linLogs = Eigen::ArrayXf::LinSpaced(c.nOutputs + 1, 0, highFreqLog);
        Eigen::ArrayXf freqs = lowFreqLog * (Eigen::ArrayXf::Constant(c.nOutputs + 1, 10).pow(linLogs) - 1.f);
        Eigen::ArrayXf cornerBinsFloat = ((c.nInputs - 1) / c.indexEnd * freqs);

        Eigen::ArrayXf cornerBinsFloatDiff = cornerBinsFloat.tail(c.nOutputs) - cornerBinsFloat.head(c.nOutputs);
        nSmallBins = (cornerBinsFloatDiff <= 1.f).cast<int>().sum();                            // number of bins that has width smaller or equal to 1
        binsWeight = cornerBinsFloat.head(nSmallBins) - cornerBinsFloat.head(nSmallBins).floor(); // weight for the bins with width smaller or equal to 1

        Eigen::ArrayXi cornerBins =
            cornerBinsFloat.round().cast<int>(); // corner bins, including 0 and nInputs. Use round() instead of floor() to ensure increasing mel bin sizes
        indexStart.resize(c.nOutputs);
        indexStart.head(nSmallBins) = cornerBinsFloat.head(nSmallBins).floor().cast<int>();
        indexStart.tail(c.nOutputs - nSmallBins) = cornerBins.segment(nSmallBins, c.nOutputs - nSmallBins);
        nInputsSum.resize(c.nOutputs);
        nInputsSum.head(nSmallBins) = Eigen::ArrayXi::Constant(nSmallBins, 1);
        nInputsSum.tail(c.nOutputs - nSmallBins) =
            (cornerBins.segment(nSmallBins + 1, c.nOutputs - nSmallBins) - indexStart.segment(nSmallBins, c.nOutputs - nSmallBins)).cwiseMax(1);
        indexEnd = indexStart(c.nOutputs - 1) + nInputsSum(c.nOutputs - 1);
    }

    inline void inverse(I::Real2D xPower, O::Real2D yPower)
    {
        assert(xPower.rows() == C.nOutputs);

        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            for (auto i = 0; i < C.nOutputs; i++)
            {
                yPower.block(indexStart(i), channel, nInputsSum(i), 1).setConstant(xPower(i, channel));
            }
            yPower.block(indexEnd, channel, C.nInputs - indexEnd, 1).setConstant(xPower(C.nOutputs - 1, channel));
        }
    }

    // Indices corresponds to frequencies if indexEnd is half the sample rate
    Eigen::ArrayXf getCornerIndices() const
    {
        Eigen::ArrayXf array(C.nOutputs + 1);
        array.head(nSmallBins) = (1 - binsWeight) * indexStart.head(nSmallBins).cast<float>() + binsWeight * (indexStart.head(nSmallBins) + 1).cast<float>();
        array.segment(nSmallBins, C.nOutputs - nSmallBins) = indexStart.segment(nSmallBins, C.nOutputs - nSmallBins).cast<float>();
        array.tail(1) = static_cast<float>(indexEnd);
        array *= C.indexEnd / (C.nInputs - 1);
        return array;
    }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto channel = 0; channel < input.cols(); channel++)
        {
            for (auto i = 0; i < nSmallBins; i++)
            {
                // weighted sum
                output(i, channel) = (1.f - binsWeight(i)) * input(indexStart(i), channel) + binsWeight(i) * input(indexStart(i) + 1, channel);
            }
            for (auto i = nSmallBins; i < C.nOutputs; i++)
            {
                // max
                output(i, channel) = input.block(indexStart(i), channel, nInputsSum(i), 1).maxCoeff();
            }
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = indexStart.getDynamicMemorySize();
        size += nInputsSum.getDynamicMemorySize();
        size += binsWeight.getDynamicMemorySize();
        return size;
    }

    int indexEnd, nSmallBins;
    Eigen::ArrayXi indexStart;
    Eigen::ArrayXi nInputsSum;
    Eigen::ArrayXf binsWeight;

    friend BaseAlgorithm;
};