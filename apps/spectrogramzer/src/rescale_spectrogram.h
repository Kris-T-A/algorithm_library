#pragma once
#include "scale_transform/log_scale.h"

// apect ratio = 0.5625 corresponds to a 16:9 aspect ratio
Eigen::ArrayXXf rescaleSpectrogram(const Eigen::Ref<const Eigen::ArrayXXf> &spec, float sampleRate, float aspectRatio = 0.5625)
{
    int width = spec.cols();
    int height = spec.rows();
    int heightNew = static_cast<int>(width * aspectRatio);

    LogScale::Coefficients c;
    c.nInputs = height;
    c.nOutputs = heightNew;
    c.outputStart = 20;           // 20 Hz
    c.outputEnd = sampleRate / 2; // Nyquist frequency
    c.inputEnd = sampleRate / 2;  // Nyquist frequency
    c.transformType = LogScale::Coefficients::TransformType::LOGARITHMIC;

    LogScale rescaler(c);

    Eigen::ArrayXXf rescaledSpec = Eigen::ArrayXXf::Zero(heightNew, width);
    rescaler.process(spec, rescaledSpec);

    return rescaledSpec;
}