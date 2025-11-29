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
        int nInputs = 50;       // input size of linear scale
        int nOutputs = 10;      // output size of transformed scale
        float outputStart = 20; // the output indices goes from outputStart to outputEnd. For instance, 20 Hz for a frequency scale
        float outputEnd = 8000; // the output indices goes from outputStart to outputEnd. For instance, 8 kHz for 16 kHz sample rate on a frequency scale
        float inputEnd = 8000;  // the input indices goes from 0 to inputEnd. For instance, 8 kHz for 16 kHz sample rate on a frequency scale
        enum TransformType { MEL, LOGARITHMIC };
        TransformType transformType = LOGARITHMIC; // choose type of scale transformation
        DEFINE_TUNABLE_ENUM(TransformType, {{MEL, "Mel Scale"}, {LOGARITHMIC, "Logarithmic Scale"}})
        DEFINE_TUNABLE_COEFFICIENTS(nInputs, nOutputs, outputStart, outputEnd, inputEnd, transformType)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return 100.f * Eigen::ArrayXXf::Random(c.nInputs, 2); } // number of channels is arbitrary

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nOutputs, input.cols()); } // transformed scale

    static bool validInput(Input input, const Coefficients &c) { return (input.cols() > 0) && (input.rows() == c.nInputs); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.cols() > 0) && (output.rows() == c.nOutputs); }
};

class ScaleTransform : public Algorithm<ScaleTransformConfiguration>
{
  public:
    ScaleTransform() = default;
    ScaleTransform(const Coefficients &c);

    Eigen::ArrayXf getCenterIndices() const;     // returns the center indices of the transformed scale. The length of the returned array is nOutputs
    Eigen::ArrayXf getCenterFrequencies() const; // returns the center frequencies of the transformed scale. The length of the returned array is nOutputs

    void inverse(I::Real2D input, O::Real2D output); // inverse transform of the scale. Note that this will not be exact due to the nature of the transformation
};
