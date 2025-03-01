#pragma once
#include "algorithm_library/scale_transform.h"
#include "framework/framework.h"

// Mel Scale
//
// author: Kristian Timm Andersen

class MelScale : public AlgorithmImplementation<ScaleTransformConfiguration, MelScale>
{
  public:
    MelScale(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        float highFreqMel = 2595 * std::log10(1 + (c.indexEnd) / 700);                                    // convert Hz to Mel
        Eigen::ArrayXf mels = Eigen::ArrayXf::LinSpaced(c.nOutputs + 1, 0, highFreqMel);                  // linear spaced corner indices in Mel domain
        Eigen::ArrayXf freqs = 700 * (Eigen::ArrayXf::Constant(c.nOutputs + 1, 10).pow(mels / 2595) - 1); // convert Mel to Hz
        Eigen::ArrayXi cornerBins = ((c.nInputs - 1) / c.indexEnd * freqs)
                                        .round()
                                        .cast<int>(); // corner bins, including 0 and nInputs. Use round() instead of floor() to ensure increasing mel bin sizes
        indexStart = cornerBins.head(c.nOutputs);
        nInputsSum = (cornerBins.segment(1, c.nOutputs) - indexStart).cwiseMax(1);
        indexEnd = indexStart(c.nOutputs - 1) + nInputsSum(c.nOutputs - 1);
    }

    inline void inverse(I::Real2D xPower, O::Real2D yPower)
    {
        assert(xPower.rows() == C.nOutputs);

        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            yPower.block(0, channel, indexStart(0), 1).setConstant(xPower(0, channel) / nInputsSum(0));
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
        array.head(C.nOutputs) = indexStart.cast<float>();
        array.tail(1) = static_cast<float>(indexEnd);
        array *= C.indexEnd / (C.nInputs - 1);
        return array;
    }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto channel = 0; channel < input.cols(); channel++)
        {
            for (auto i = 0; i < C.nOutputs; i++)
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
        return size;
    }

    int indexEnd;
    Eigen::ArrayXi indexStart;
    Eigen::ArrayXi nInputsSum;

    friend BaseAlgorithm;
};
