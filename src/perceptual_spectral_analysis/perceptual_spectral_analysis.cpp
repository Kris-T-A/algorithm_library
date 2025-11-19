#include "perceptual_spectral_analysis/perceptual_adaptive_spectrogram.h"
#include "perceptual_spectral_analysis/perceptual_nonlinear_spectrogram.h"

using AdaptiveImpl = Implementation<PerceptualAdaptiveSpectrogram, PerceptualSpectralAnalysisConfiguration>;
using NonlinearImpl = Implementation<PerceptualNonlinearSpectrogram, PerceptualSpectralAnalysisConfiguration>;

template <>
void Algorithm<PerceptualSpectralAnalysisConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.method == PerceptualSpectralAnalysisConfiguration::Coefficients::ADAPTIVE) { pimpl = std::make_unique<AdaptiveImpl>(c); }
    else {
        pimpl = std::make_unique<NonlinearImpl>(c);
    }
}

PerceptualSpectralAnalysis::PerceptualSpectralAnalysis(const Coefficients &c) : Algorithm<PerceptualSpectralAnalysisConfiguration>(c) {}
