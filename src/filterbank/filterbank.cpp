#include "filterbank/filterbank_single_channel.h"
#include "filterbank/filterbank_wola.h"

template <>
void Algorithm<FilterbankAnalysisConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.nChannels == 1) { pimpl = std::make_unique<Implementation<FilterbankAnalysisSingleChannel, FilterbankAnalysisConfiguration>>(c); }
    else { pimpl = std::make_unique<Implementation<FilterbankAnalysisWOLA, FilterbankAnalysisConfiguration>>(c); }
}

FilterbankAnalysis::FilterbankAnalysis(const Coefficients &c) : Algorithm<FilterbankAnalysisConfiguration>(c) {}

template <>
void Algorithm<FilterbankSynthesisConfiguration>::setImplementation(const Coefficients &c)
{
    if (c.nChannels == 1) { pimpl = std::make_unique<Implementation<FilterbankSynthesisSingleChannel, FilterbankSynthesisConfiguration>>(c); }
    else { pimpl = std::make_unique<Implementation<FilterbankSynthesisWOLA, FilterbankSynthesisConfiguration>>(c); }
}

FilterbankSynthesis::FilterbankSynthesis(const Coefficients &c) : Algorithm<FilterbankSynthesisConfiguration>(c) {}

float FilterbankAnalysis::getDelaySamples() const
{
    if (getCoefficients().nChannels == 1)
    {
        return static_cast<Implementation<FilterbankAnalysisSingleChannel, FilterbankAnalysisConfiguration> *>(pimpl.get())->algo.getDelaySamples();
    }
    else { return static_cast<Implementation<FilterbankAnalysisWOLA, FilterbankAnalysisConfiguration> *>(pimpl.get())->algo.getDelaySamples(); }
}

float FilterbankSynthesis::getDelaySamples() const
{
    if (getCoefficients().nChannels == 1)
    {
        return static_cast<Implementation<FilterbankSynthesisSingleChannel, FilterbankSynthesisConfiguration> *>(pimpl.get())->algo.getDelaySamples();
    }
    else { return static_cast<Implementation<FilterbankSynthesisWOLA, FilterbankSynthesisConfiguration> *>(pimpl.get())->algo.getDelaySamples(); }
}

namespace FilterbankShared
{

bool isCoefficientsValid(const FilterbankConfiguration::Coefficients &c)
{
    if (c.nFolds < 1) { return false; }
    return true;
}

Eigen::ArrayXf getAnalysisWindow(const FilterbankConfiguration::Coefficients &c)
{
    const int frameSize = FilterbankConfiguration::calculateFrameSize(c);
    Eigen::ArrayXf window = (c.nFolds > 1) ? sinc(frameSize, 2) * kaiser(frameSize, 10) : hann(frameSize);
    return window;
}

Eigen::ArrayXf getSynthesisWindow(const FilterbankConfiguration::Coefficients &c)
{
    Eigen::ArrayXf window;
    const int frameSize = FilterbankConfiguration::calculateFrameSize(c);
    const int fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);

    if (c.nFolds > 1) { window = kaiser(frameSize, 14); }
    else
    {
        if ((fftSize / c.bufferSize) <= 2) { window = Eigen::ArrayXf::Ones(frameSize); }
        window = hann(frameSize);
    }

    // scale synthesis window to give unit output
    Eigen::ArrayXf windowSum = Eigen::ArrayXf::Zero(c.bufferSize);
    Eigen::ArrayXf windowProd = window * getAnalysisWindow(c);
    for (auto i = 0; i < frameSize / c.bufferSize; i++)
    {
        windowSum += windowProd.segment(i * c.bufferSize, c.bufferSize);
    }
    window /= windowSum.mean();
    return window;
}

// calculate delay as the group delay at 0 Hz of the prototype window:  Group Delay(z) = Real{ FFT{window * ramp} / FFT{window} }
float getDelaySamples(I::Real window)
{
    Eigen::ArrayXf ramp = Eigen::ArrayXf::LinSpaced(window.size(), 0, static_cast<float>(window.size() - 1));
    return (window * ramp).sum() / (window.sum() + 1e-12f);
}
}; // namespace FilterbankShared