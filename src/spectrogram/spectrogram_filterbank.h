#include "algorithm_library/spectrogram.h"
#include "filterbank/filterbank_single_channel.h"
#include "framework/framework.h"

// Spectrogram implemented using a single channel weighted overlap-add (WOLA) filter bank.
//
// author: Kristian Timm Andersen
class SpectrogramFilterbank : public AlgorithmImplementation<SpectrogramConfiguration, SpectrogramFilterbank>
{
  public:
    SpectrogramFilterbank(Coefficients c = Coefficients()) : BaseAlgorithm{c}, filterbank({.nChannels = 1, .bufferSize = c.bufferSize, .nBands = c.nBands, .nFolds = c.nFolds})
    {
        assert(c.nonlinearity == 0); // this implementation does not support nonlinearity
        assert(c.nBands > 0 && c.bufferSize > 0 && c.nFolds > 0);
        filterbankOut.resize(c.nBands);
    }

    FilterbankAnalysisSingleChannel filterbank;
    DEFINE_MEMBER_ALGORITHMS(filterbank)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        filterbank.process(input, filterbankOut);
        output = filterbankOut.abs2();
    }

    bool isCoefficientsValid() const final { return (C.nBands > 0) && (C.bufferSize > 0) && (C.nFolds > 0) && (C.nonlinearity == 0); }

    size_t getDynamicSizeVariables() const final { return filterbankOut.getDynamicMemorySize(); }

    Eigen::ArrayXcf filterbankOut;

    friend BaseAlgorithm;
};