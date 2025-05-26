#include "algorithm_library/spectrogram.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"

// Spectrogram implemented using a weighted overlap-add (WOLA) filter bank.
//
// author: Kristian Timm Andersen
class SpectrogramFilterbank : public AlgorithmImplementation<SpectrogramConfiguration, SpectrogramFilterbank>
{
  public:
    SpectrogramFilterbank(Coefficients c = Coefficients()) : BaseAlgorithm{c}, filterbank(convertToFilterbankCoefficients(c))
    {
        assert(c.algorithmType == c.HANN || c.algorithmType == c.WOLA);
        filterbankOut.resize(c.nBands);
    }

    FilterbankAnalysisWOLA filterbank;
    DEFINE_MEMBER_ALGORITHMS(filterbank)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        filterbank.process(input, filterbankOut);
        output = filterbankOut.abs2();
    }

    size_t getDynamicSizeVariables() const final { return filterbankOut.getDynamicMemorySize(); }

    FilterbankAnalysisWOLA::Coefficients convertToFilterbankCoefficients(const Coefficients &c)
    {
        FilterbankAnalysisWOLA::Coefficients cFilterbank;
        cFilterbank.bufferSize = c.bufferSize;
        cFilterbank.nChannels = 1;
        cFilterbank.nBands = c.nBands;
        switch (c.algorithmType)
        {
        default: // Hann is the default case
        case Coefficients::HANN: cFilterbank.nFolds = 1; break;
        case Coefficients::WOLA: cFilterbank.nFolds = 2; break;
        }
        return cFilterbank;
    }

    Eigen::ArrayXcf filterbankOut;

    friend BaseAlgorithm;
};