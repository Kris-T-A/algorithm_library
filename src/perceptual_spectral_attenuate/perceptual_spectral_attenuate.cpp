#include "perceptual_spectral_attenuate/perceptual_attenuate_max.h"

DEFINE_ALGORITHM_CONSTRUCTOR(PerceptualSpectralAttenuate, PerceptualAttenuateMax, PerceptualSpectralAttenuateConfiguration)

int PerceptualSpectralAttenuateConfiguration::getValidBufferSize(int bufferSize)
{
    // With three resolutions, the smallest FFT size equals the largest hop size.
    return FFTConfiguration::getValidFFTSize(bufferSize);
}
