#include "scale_transform/log_scale.h"

DEFINE_ALGORITHM_CONSTRUCTOR(ScaleTransform, LogScale, ScaleTransformConfiguration)

Eigen::ArrayXf ScaleTransform::getCenterIndices() const { return static_cast<LogScaleSingleBufferImpl *>(pimpl.get())->algo.getCenterIndices(); }

Eigen::ArrayXf ScaleTransform::getCenterFrequencies() const { return static_cast<LogScaleSingleBufferImpl *>(pimpl.get())->algo.getCenterFrequencies(); }

void ScaleTransform::inverse(I::Real2D input, O::Real2D output) { static_cast<LogScaleSingleBufferImpl *>(pimpl.get())->algo.inverse(input, output); }