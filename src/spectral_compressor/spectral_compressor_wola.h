#pragma once
#include "algorithm_library/spectral_compressor.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// Spectral compressor using a weighted overlap-add (WOLA) filter bank.
//
// author: Kristian Timm Andersen
class SpectralCompressorWOLA : public AlgorithmImplementation<SpectralCompressorConfiguration, SpectralCompressorWOLA>
{
  public:
    SpectralCompressorWOLA(Coefficients c = Coefficients())
        : BaseAlgorithm{c}, filterbank(convertToFilterbankCoefficients(c)), filterbankInverse(convertToFilterbankInverseCoefficients(c))
    {
        Eigen::ArrayXf window = FilterbankShared::getAnalysisWindow(convertToFilterbankCoefficients(c));
        sumWindow = lin2dB(window.sum() / 2.f);                      // scaled so sine wave with amplitude 1 gives threshold level
        energyWindow = lin2dB(std::sqrt(window.abs2().sum()) * 8.f); // energyWindow scaled to give approximately similar threshold level to sumWindow

        nBands = FFTReal::Configuration::convertFFTSizeToNBands(c.bufferSize * 4);

        filterbankOut.resize(nBands, c.nChannels);
        energy.resize(nBands);
        gain.resize(nBands, c.nChannels);

        resetVariables();
        onParametersChanged();
    }

    FilterbankAnalysisWOLA filterbank;
    FilterbankSynthesisWOLA filterbankInverse;

    DEFINE_MEMBER_ALGORITHMS(filterbank, filterbankInverse)

    float getDelaySamples() const { return static_cast<float>(C.bufferSize * 3); }

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        filterbank.process(input, filterbankOut);
        for (auto channel = 0; channel < C.nChannels; channel++)
        {
            energy = filterbankOut.col(channel).abs2();
            for (auto band = 0; band < nBands; band++)
            {
                float energydB = energy2dB(energy(band) + 1e-20f); // 10*log10(x)
                float gainBanddB = energydB > threshold ? ratioOffset - ratioScale * energydB : 0.f;
                float gainBand = dB2lin(gainBanddB); // 10^(x/20)
                gain(band, channel) += gainBand > gain(band, channel) ? gainUpLambda * (gainBand - gain(band, channel)) : gainDownLambda * (gainBand - gain(band, channel));
            }
        }
        filterbankOut *= gain;
        filterbankInverse.process(filterbankOut, output);
    }

    void onParametersChanged()
    {
        threshold = P.thresholdDB;
        if (P.thresholdMode == Parameters::ThresholdMode::ENERGY_VALUE) { threshold += energyWindow; }
        else { threshold += sumWindow; }
        ratioScale = 1.f - 1.f / P.ratio;
        ratioOffset = ratioScale * threshold;
        gainUpLambda = 1.f - fasterexp(-1.f / (C.sampleRate / C.bufferSize * P.upTConstant));
        gainDownLambda = 1.f - fasterexp(-1.f / (C.sampleRate / C.bufferSize * P.downTConstant));
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = filterbankOut.getDynamicMemorySize();
        size += energy.getDynamicMemorySize();
        size += gain.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        filterbankOut.setZero();
        energy.setZero();
        gain.setOnes();
    }

    FilterbankAnalysisWOLA::Coefficients convertToFilterbankCoefficients(const Coefficients &c)
    {
        FilterbankAnalysisWOLA::Coefficients cFilterbank;
        cFilterbank.bufferSize = c.bufferSize;
        cFilterbank.nChannels = c.nChannels;
        cFilterbank.nBands = FFTReal::Configuration::convertFFTSizeToNBands(c.bufferSize * 4);
        cFilterbank.nFolds = 1;
        return cFilterbank;
    }

    FilterbankSynthesisWOLA::Coefficients convertToFilterbankInverseCoefficients(const Coefficients &c)
    {
        FilterbankSynthesisWOLA::Coefficients cFilterbank;
        cFilterbank.bufferSize = c.bufferSize;
        cFilterbank.nChannels = c.nChannels;
        cFilterbank.nBands = FFTReal::Configuration::convertFFTSizeToNBands(c.bufferSize * 4);
        cFilterbank.nFolds = 1;
        return cFilterbank;
    }

    int nBands;
    float threshold;
    float energyWindow;
    float sumWindow;
    float ratioScale;
    float ratioOffset;
    float gainUpLambda;
    float gainDownLambda;
    Eigen::ArrayXXcf filterbankOut;
    Eigen::ArrayXf energy;
    Eigen::ArrayXXf gain;

    friend BaseAlgorithm;
};