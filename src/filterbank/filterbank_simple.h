#pragma once
#include "algorithm_library/filterbank.h"
#include "fft/fft_real.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include "utilities/functions.h"

// --------------------------------------------------- FilterbankAnalysisSimple ---------------------------------------------------------
// FilterbankAnalysisSimple is a class that implements the simplest possible DFT-based filterbank. It only supports the configuration:
// nChannels = 1
// nBands = (bufferSize*4)/2+1
// nFolds = 1
//
// author: Kristian Timm Andersen

class FilterbankAnalysisSimple : public AlgorithmImplementation<FilterbankAnalysisConfiguration, FilterbankAnalysisSimple>
{
  public:
    FilterbankAnalysisSimple(Coefficients c = {.nChannels = 1, .bufferSize = 128, .nBands = 257, .nFolds = 1})
        : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        if ((c.nChannels != 1) || (c.nBands != c.bufferSize * 2 + 1) || (c.nFolds != 1)) { throw Configuration::ExceptionFilterbank("FilterbankAnalysisSimple", c); }
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getAnalysisWindow(c);
        overlap = fftSize - C.bufferSize;
        fftBuffer.resize(fftSize);
        timeBuffer.resize(fftSize);

        resetVariables();
    }

    FFTReal fft;
    DEFINE_MEMBER_ALGORITHMS(fft)

    int getFFTSize() const { return fftSize; }
    int getNBands() const { return fftSize / 2 + 1; }
    int getFrameSize() const { return fftSize; }

    float getDelaySamples() const { return FilterbankShared::getDelaySamples(window); }

  private:
    inline void processAlgorithm(Input xTime, Output yFreq)
    {
        timeBuffer.head(overlap) = timeBuffer.tail(overlap);
        timeBuffer.tail(C.bufferSize) = xTime.col(0);
        fftBuffer = timeBuffer * window;
        fft.process(fftBuffer, yFreq);
    }

    bool isCoefficientsValid() const final { return FilterbankShared::isCoefficientsValid(C); }

    size_t getDynamicSizeVariables() const final
    {
        auto size = window.getDynamicMemorySize();
        size += fftBuffer.getDynamicMemorySize();
        size += timeBuffer.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        fftBuffer.setZero();
        timeBuffer.setZero();
    }

    int fftSize, overlap;
    Eigen::ArrayXf window, fftBuffer;
    Eigen::ArrayXf timeBuffer;

    friend BaseAlgorithm;
};

// --------------------------------------------------- FilterbankSynthesis ----------------------------------------------------------------

class FilterbankSynthesisSimple : public AlgorithmImplementation<FilterbankSynthesisConfiguration, FilterbankSynthesisSimple>
{
  public:
    FilterbankSynthesisSimple(Coefficients c = {.nChannels = 1, .bufferSize = 128, .nBands = 257, .nFolds = 1})
        : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        if ((c.nChannels != 1) || (c.nBands != c.bufferSize * 2 + 1) || (c.nFolds != 1)) { throw Configuration::ExceptionFilterbank("FilterbankSynthesisSimple", c); }
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getSynthesisWindow(c);
        overlap = fftSize - C.bufferSize;
        fftBuffer.resize(fftSize);
        timeBuffer.resize(fftSize);

        resetVariables();
    }

    FFTReal fft;
    DEFINE_MEMBER_ALGORITHMS(fft)

    int getFFTSize() const { return fftSize; }
    int getNBands() const { return fftSize / 2 + 1; }

    float getDelaySamples() const { return FilterbankShared::getDelaySamples(window); }

  private:
    inline void processAlgorithm(Input xFreq, Output yTime)
    {
        fft.inverse(xFreq, fftBuffer);
        timeBuffer += fftBuffer * window;
        yTime = timeBuffer.head(C.bufferSize);
        timeBuffer.head(overlap) = timeBuffer.tail(overlap);
        timeBuffer.tail(C.bufferSize) = 0.f;
    }

    bool isCoefficientsValid() const final { return FilterbankShared::isCoefficientsValid(C); }

    size_t getDynamicSizeVariables() const final
    {
        auto size = window.getDynamicMemorySize();
        size += fftBuffer.getDynamicMemorySize();
        size += timeBuffer.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        fftBuffer.setZero();
        timeBuffer.setZero();
    }

    int fftSize, overlap;
    Eigen::ArrayXf window, fftBuffer;
    Eigen::ArrayXf timeBuffer;

    friend BaseAlgorithm;
};
