#include "spectrogram_adaptive/spectrogram_adaptive_moving.h"
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"

using ZeropadImpl = Implementation<SpectrogramAdaptiveZeropad, SpectrogramAdaptiveConfiguration>;
using MovingImpl = Implementation<SpectrogramAdaptiveMoving, SpectrogramAdaptiveConfiguration>;

template <>
void Algorithm<SpectrogramAdaptiveConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.method == SpectrogramAdaptiveConfiguration::Coefficients::ZEROPAD)
    {
        pimpl = std::make_unique<ZeropadImpl>(c);
    }
    else
    {
        pimpl = std::make_unique<MovingImpl>(c);
    }
}

SpectrogramAdaptive::SpectrogramAdaptive(const Coefficients &c) : Algorithm<SpectrogramAdaptiveConfiguration>(c) {}
