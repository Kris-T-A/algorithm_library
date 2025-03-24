#pragma once
#include "algorithm_library/filterbank.h"
#include "fft/fft_real.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"
#include "utilities/functions.h"

// --------------------------------------------------- FilterbankAnalysisSingleChannel ---------------------------------------------------------
// FilterbankAnalysisSingleChannel is a class that implements the analysis part of a single-channel DFT filterbank.
// The output will have the size nBands x nFrames, where nFrames = input.size() / bufferSize.

class FilterbankAnalysisSingleChannel : public AlgorithmImplementation<FilterbankAnalysisConfiguration, FilterbankAnalysisSingleChannel>
{
  public:
    FilterbankAnalysisSingleChannel(Coefficients c = {.nChannels = 1, .bufferSize = 128, .nBands = 257, .filterbankType = Coefficients::HANN})
        : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        if (c.nChannels != 1) { throw Configuration::ExceptionFilterbank("FilterbankAnalysisSingleChannel", c); }
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getAnalysisWindow(c);
        frameSize = static_cast<int>(window.size());
        overlap = frameSize - C.bufferSize;
        nFolds = static_cast<int>(std::ceil(static_cast<float>(frameSize) / fftSize));
        maxSize = fftSize * nFolds;
        fftBuffer.resize(maxSize);
        timeBuffer.resize(frameSize);

        resetVariables();
    }

    FFTReal fft;
    DEFINE_MEMBER_ALGORITHMS(fft)

    int getFFTSize() const { return fftSize; }
    int getNBands() const { return fftSize / 2 + 1; }
    int getFrameSize() const { return frameSize; }

    float getDelaySamples() const { return FilterbankShared::getDelaySamples(window); }

    // set window if length equals frameSize
    void setWindow(I::Real win)
    {
        if (win.size() == frameSize)
        {
            C.filterbankType = C.USER_DEFINED; // force overwrite
            window = win;
        }
    }

    Eigen::ArrayXf getWindow() const { return window; }

  private:
    inline void processAlgorithm(Input xTime, Output yFreq)
    {
        timeBuffer.head(overlap) = timeBuffer.tail(overlap);
        timeBuffer.tail(C.bufferSize) = xTime.col(0).head(C.bufferSize);
        fftBuffer.head(frameSize) = timeBuffer * window;
        for (auto j = 1; j < nFolds; j++)
        {
            fftBuffer.head(fftSize) += fftBuffer.segment(j * fftSize, fftSize);
        }
        fft.process(fftBuffer.head(fftSize), yFreq);
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

    int frameSize, fftSize;
    Eigen::ArrayXf window, fftBuffer;
    Eigen::ArrayXf timeBuffer;
    int overlap, nFolds, maxSize;

    friend BaseAlgorithm;
};

// --------------------------------------------------- FilterbankSynthesis ----------------------------------------------------------------

class FilterbankSynthesisSingleChannel : public AlgorithmImplementation<FilterbankSynthesisConfiguration, FilterbankSynthesisSingleChannel>
{
  public:
    FilterbankSynthesisSingleChannel(Coefficients c = {.nChannels = 1, .bufferSize = 128, .nBands = 257, .filterbankType = Coefficients::HANN})
        : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        if (c.nChannels != 1) { throw Configuration::ExceptionFilterbank("FilterbankSynthesisSingleChannel", c); }
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getSynthesisWindow(c);
        frameSize = static_cast<int>(window.size());
        overlap = frameSize - C.bufferSize;
        nFolds = static_cast<int>(std::ceil(static_cast<float>(frameSize) / fftSize));
        maxSize = fftSize * nFolds;
        fftBuffer.resize(maxSize);
        timeBuffer.resize(frameSize);

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
        fft.inverse(xFreq, fftBuffer.head(fftSize));
        for (auto j = 1; j < nFolds; j++)
        {
            fftBuffer.segment(j * fftSize, fftSize) = fftBuffer.head(fftSize);
        }
        timeBuffer += fftBuffer.tail(frameSize) * window;

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

    int fftSize, frameSize;
    Eigen::ArrayXf window, fftBuffer;
    Eigen::ArrayXf timeBuffer;
    int overlap, nFolds, maxSize;

    friend BaseAlgorithm;
};
