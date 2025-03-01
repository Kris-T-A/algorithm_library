#pragma once
#include "interface/interface.h"

// Transform a linear scale to a choice of non-linear scale.
//
// author: Kristian Timm Andersen

struct ScaleTransformConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int nInputs = 513;     // input size
        int nOutputs = 40;     // output size
        float indexEnd = 8000; // the input indices goes from 0 to indexEnd. For instance, 8 kHz for 16 kHz sample rate on a frequency scale
        enum TransformType { MEL, LOGARITHMIC };
        TransformType transformType = MEL; // choose type of scale transformation
        DEFINE_TUNABLE_ENUM(TransformType, {{MEL, "Mel Scale"}, {LOGARITHMIC, "Logarithmic Scale"}})
        DEFINE_TUNABLE_COEFFICIENTS(nInputs, nOutputs, indexEnd, transformType)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nInputs, 2).abs2(); } // number of channels is arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nOutputs, input.cols()); } // transformed scale

    static bool validInput(Input input, const Coefficients &c) { return (input.cols() > 0) && (input.rows() == c.nInputs) && (input >= 0).all(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.cols() > 0) && (output.rows() == c.nOutputs) && (output >= 0).all(); }
};

class ScaleTransform : public Algorithm<ScaleTransformConfiguration>
{
  public:
    ScaleTransform() = default;
    ScaleTransform(const Coefficients &c);

    Eigen::ArrayXf getCornerIndices() const; // returns the corner indices of the transformed scale. The length of the returned array is nOutputs + 1, the first element is 0
    // and the last element is C.indexEnd

    void inverse(I::Real2D input, O::Real2D output); // inverse transform of the scale. Note that this will not be exact due to the nature of the transformation
};
