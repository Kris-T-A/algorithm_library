#include "scale_transform/log_scale.h"
#include "scale_transform/mel_scale.h"

using MelScaleImpl = Implementation<MelScale, ScaleTransformConfiguration>;
using LogScaleImpl = Implementation<LogScale, ScaleTransformConfiguration>;

template <>
void Algorithm<ScaleTransformConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.transformType == c.MEL) { pimpl = std::make_unique<MelScaleImpl>(c); }
    else { pimpl = std::make_unique<LogScaleImpl>(c); }
}

ScaleTransform::ScaleTransform(const Coefficients &c) : Algorithm<ScaleTransformConfiguration>(c) {}

Eigen::ArrayXf ScaleTransform::getCornerIndices() const
{
    if (getCoefficients().transformType == Coefficients::MEL) { return static_cast<MelScaleImpl *>(pimpl.get())->algo.getCornerIndices(); }
    else { return static_cast<LogScaleImpl *>(pimpl.get())->algo.getCornerIndices(); }
}

void ScaleTransform::inverse(I::Real2D input, O::Real2D output)
{
    if (getCoefficients().transformType == Coefficients::MEL) { static_cast<MelScaleImpl *>(pimpl.get())->algo.inverse(input, output); }
    else { static_cast<LogScaleImpl *>(pimpl.get())->algo.inverse(input, output); }
}