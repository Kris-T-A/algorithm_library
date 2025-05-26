#pragma once
#include "algorithm_library/interface/interface.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"

// Spectral selector between a number of inputs streams by choosing the one with minimum power.
//
// author: Kristian Timm Andersen

struct SpectralSelectorConfiguration
{
    using Input = I::Real2D;

    using Output = O::Real2D;

    struct Coefficients
    {
        int nChannels = 6; // number of columns in input is nChannels
        int nStreams = 3;  // number of columns in output is nChannels / nStreams
        int bufferSize = 256;
        DEFINE_TUNABLE_COEFFICIENTS(nChannels, nStreams, bufferSize)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getNOutputChannels(const Coefficients &c) { return c.nChannels / c.nStreams; }

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.bufferSize, c.nChannels); }

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.bufferSize, getNOutputChannels(c)); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && (input.cols() == c.nChannels) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.bufferSize) && (output.cols() == getNOutputChannels(c)) && output.allFinite(); }
};

class SpectralSelector : public AlgorithmImplementation<SpectralSelectorConfiguration, SpectralSelector>
{
  public:
    SpectralSelector(Coefficients c = Coefficients())
        : BaseAlgorithm{c},
          filterbank({.nChannels = c.nChannels, .bufferSize = c.bufferSize, .nBands = FFTConfiguration::convertFFTSizeToNBands(c.bufferSize * 4), .nFolds = 1}),
          filterbankInverse({.nChannels = Configuration::getNOutputChannels(c),
                             .bufferSize = c.bufferSize,
                             .nBands = FFTConfiguration::convertFFTSizeToNBands(c.bufferSize * 4),
                             .nFolds = 1})
    {
        nBands = FFTConfiguration::convertFFTSizeToNBands(c.bufferSize * 4);
        nOutputChannels = Configuration::getNOutputChannels(c);
        filterbankOut.resize(nBands, c.nChannels);
        powerMin.resize(nBands, nOutputChannels);
        outputFreq.resize(nBands, nOutputChannels);
        resetVariables();
    }

    FilterbankAnalysisWOLA filterbank;
    FilterbankSynthesisWOLA filterbankInverse;
    DEFINE_MEMBER_ALGORITHMS(filterbank, filterbankInverse)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        filterbank.process(input, filterbankOut);
        outputFreq = filterbankOut.leftCols(nOutputChannels);
        powerMin = outputFreq.abs2();
        for (auto stream = 1; stream < C.nStreams; stream++)
        {
            for (auto channel = 0; channel < nOutputChannels; channel++)
            {
                for (auto band = 0; band < nBands; band++)
                {
                    float power = std::norm(filterbankOut(band, channel + nOutputChannels * stream));
                    if (power < powerMin(band, channel))
                    {
                        powerMin(band, channel) = power;
                        outputFreq(band, channel) = filterbankOut(band, channel + nOutputChannels * stream);
                    }
                }
            }
        }
        filterbankInverse.process(outputFreq, output);
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = filterbankOut.getDynamicMemorySize();
        size += powerMin.getDynamicMemorySize();
        size += outputFreq.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        filterbankOut.setZero();
        powerMin.setZero();
        outputFreq.setZero();
    }

    int nBands;
    int nOutputChannels;
    Eigen::ArrayXXcf filterbankOut;
    Eigen::ArrayXXcf outputFreq;
    Eigen::ArrayXXf powerMin;

    friend BaseAlgorithm;
};