#include "spectrogram/spectrogram_filterbank.h"
#include "spectrogram/spectrogram_nonlinear.h"

using FilterbankImpl = Implementation<SpectrogramFilterbank, SpectrogramConfiguration>;
using NonlinearImpl = Implementation<SpectrogramNonlinear, SpectrogramConfiguration>;

template <>
void Algorithm<SpectrogramConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.nonlinearity == 0) { pimpl = std::make_unique<FilterbankImpl>(c); }
    else { pimpl = std::make_unique<NonlinearImpl>(c); }
}

Spectrogram::Spectrogram(const Coefficients &c) : Algorithm<SpectrogramConfiguration>(c) {}

int SpectrogramConfiguration::getValidFFTSize(int fftSize) { return FFTConfiguration::getValidFFTSize(fftSize); }