#include "scale_transform/mel_scale.h"

DEFINE_ALGORITHM_CONSTRUCTOR(ScaleTransform, MelScale, ScaleTransformConfiguration)

Eigen::ArrayXf ScaleTransform::getCornerIndices() const { return static_cast<MelScaleSingleBufferImpl *>(pimpl.get())->algo.getCornerIndices(); }