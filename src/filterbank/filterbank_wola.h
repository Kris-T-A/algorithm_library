#pragma once
#include "algorithm_library/filterbank.h"
#include "fft/fft_real.h"
#include "framework/framework.h"
#include "utilities/functions.h"

// ------------------------------------- get filterbank configurations (defined in filterbank.cpp) -------------------------------------

namespace FilterbankShared
{
bool isCoefficientsValid(const FilterbankConfiguration::Coefficients &c);

Eigen::ArrayXf getAnalysisWindow(const FilterbankConfiguration::Coefficients &c);

Eigen::ArrayXf getSynthesisWindow(const FilterbankConfiguration::Coefficients &c);

float getDelaySamples(I::Real window);
}; // namespace FilterbankShared

// --------------------------------------------------- FilterbankAnalysis ----------------------------------------------------------------

class FilterbankAnalysisWOLA : public AlgorithmImplementation<FilterbankAnalysisConfiguration, FilterbankAnalysisWOLA>
{
  public:
    FilterbankAnalysisWOLA(Coefficients c = Coefficients()) : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getAnalysisWindow(c);
        frameSize = static_cast<int>(window.size());
        overlap = frameSize - C.bufferSize;
        fftBuffer.resize(frameSize);
        timeBuffer.resize(frameSize, c.nChannels);

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
        if (win.size() == frameSize) { window = win; }
    }

    Eigen::ArrayXf getWindow() const { return window; }

  private:
    inline void processAlgorithm(Input xTime, Output yFreq)
    {
        timeBuffer.topRows(overlap) = timeBuffer.bottomRows(overlap);
        timeBuffer.bottomRows(C.bufferSize) = xTime;
        for (auto channel = 0; channel < C.nChannels; channel++)
        {
            fftBuffer = timeBuffer.col(channel) * window;
            for (auto j = 1; j < C.nFolds; j++)
            {
                fftBuffer.head(fftSize) += fftBuffer.segment(j * fftSize, fftSize);
            }
            fft.process(fftBuffer.head(fftSize), yFreq.col(channel));
        }
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
    Eigen::ArrayXXf timeBuffer;
    int overlap;

    friend BaseAlgorithm;
};

// --------------------------------------------------- FilterbankSynthesis ----------------------------------------------------------------

class FilterbankSynthesisWOLA : public AlgorithmImplementation<FilterbankSynthesisConfiguration, FilterbankSynthesisWOLA>
{
  public:
    FilterbankSynthesisWOLA(Coefficients c = Coefficients()) : BaseAlgorithm{c}, fft({FFTConfiguration::convertNBandsToFFTSize(c.nBands)})
    {
        fftSize = FFTConfiguration::convertNBandsToFFTSize(c.nBands);
        window = FilterbankShared::getSynthesisWindow(c);
        frameSize = static_cast<int>(window.size());
        overlap = frameSize - C.bufferSize;
        fftBuffer.resize(frameSize);
        timeBuffer.resize(frameSize, C.nChannels);

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
        for (auto channel = 0; channel < C.nChannels; channel++)
        {
            fft.inverse(xFreq.col(channel), fftBuffer.head(fftSize));
            for (auto j = 1; j < C.nFolds; j++)
            {
                fftBuffer.segment(j * fftSize, fftSize) = fftBuffer.head(fftSize);
            }
            timeBuffer.col(channel) += (fftBuffer * window);
        }
        yTime = timeBuffer.topRows(C.bufferSize);
        timeBuffer.topRows(overlap) = timeBuffer.bottomRows(overlap);
        timeBuffer.bottomRows(C.bufferSize) = 0.f;
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
    Eigen::ArrayXXf timeBuffer;
    int overlap;

    friend BaseAlgorithm;
};
