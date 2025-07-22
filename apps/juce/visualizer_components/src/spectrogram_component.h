#pragma once
#include "algorithm_library/log_scale.h"
#include "algorithm_library/spectrogram.h"
#include "utilities/fastonebigheader.h"
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_core/juce_core.h>
#include <juce_gui_basics/juce_gui_basics.h>

class SpectrogramComponent : public juce::Component, juce::Timer
{
  public:
    SpectrogramComponent(float sampleRateNew)
        : sampleRate(sampleRateNew), nFramesOut(8), bufferSize(getBufferSize(sampleRate)), nBands(getNBands(bufferSize)), nMels(getNMels(sampleRate)),
          scalePlot(16000.f / (bufferSize * bufferSize)),
          spectrogram({.bufferSize = bufferSize, .nBands = nBands, .algorithmType = SpectrogramConfiguration::Coefficients::ADAPTIVE_HANN_8}),
          LogScale({.nMels = nMels, .nBands = nBands, .sampleRate = sampleRate}),
          spectrogramImage(juce::Image::RGB, nSpectrogramFrames, nMels, true, juce::SoftwareImageType())
    {
        circularBuffer = Eigen::ArrayXf::Zero(getcircularBufferSize(bufferSize, sampleRate));
        writeBufferIndex.store(0);
        readBufferIndex.store(0);
        bufferIn = Eigen::ArrayXf::Zero(bufferSize);
        spectrogramOut = Eigen::ArrayXXf::Zero(nBands, nFramesOut);
        spectrogramMel = Eigen::ArrayXXf::Zero(nMels, nFramesOut);

        // startTimerHz(60);
        setSize(750, 500);
    }

    void prepareToPlay(int expectedBufferSize, float sampleRateNew)
    {
        if (sampleRateNew != sampleRate)
        {
            sampleRate = sampleRateNew;
            bufferSize = getBufferSize(sampleRate);
            nBands = getNBands(bufferSize);
            nMels = getNMels(sampleRate);
            scalePlot = 16000.f / (bufferSize * bufferSize);

            auto c = spectrogram.getCoefficients();
            c.bufferSize = bufferSize;
            c.nBands = nBands;
            spectrogram.setCoefficients(c);

            auto cLog = LogScale.getCoefficients();
            cLog.nBands = nBands;
            cLog.nMels = nMels;
            cLog.sampleRate = sampleRate;
            LogScale.setCoefficients(cLog);

            spectrogramOut = Eigen::ArrayXXf::Zero(nBands, nFramesOut);
            bufferIn = Eigen::ArrayXf::Zero(bufferSize);
            spectrogramMel = Eigen::ArrayXXf::Zero(nMels, nFramesOut);

            spectrogramImage = juce::Image(juce::Image::RGB, nSpectrogramFrames, nMels, true, juce::SoftwareImageType());

            repaint(); // remove old plot
        }

        int circularBufferSize = getcircularBufferSize(expectedBufferSize, sampleRate);
        if (circularBufferSize != circularBuffer.size())
        {
            circularBuffer = Eigen::ArrayXf::Zero(circularBufferSize);
            writeBufferIndex.store(0);
            readBufferIndex.store(0);
        }
    }

    // push data into circular buffer. This method is likely to be called from the main audio thread and should not do any heavy calculations or GUI work
    void pushSamples(I::Real buffer)
    {
        int index = writeBufferIndex.load();
        int size = static_cast<int>(buffer.size());
        const int sizeCircular = static_cast<int>(circularBuffer.size());
        if (size > sizeCircular) // fallback if given more samples than we can handle
        {
            index = 0;
            size = sizeCircular;
        }

        const int size1 = std::min(sizeCircular - index, size);
        const int size2 = size - size1;
        circularBuffer.segment(index, size1) = buffer.head(size1);
        circularBuffer.head(size2) = buffer.tail(size2);

        int indexNew = index + size;
        if (indexNew >= sizeCircular) { indexNew -= sizeCircular; }
        writeBufferIndex.store(indexNew);
    }

    void reset()
    {
        circularBuffer.setZero();
        writeBufferIndex.store(0);
        readBufferIndex.store(0);
        spectrogram.reset();
        framePlot = 0;
    }

    // read from circular buffer and calculate spectrogram. This method is called from message thread and is not time critical
    void timerCallback() override
    {

        int startIndex = readBufferIndex.load();
        int endIndex = writeBufferIndex.load();
        const int sizeCircularBuffer = static_cast<int>(circularBuffer.size());
        int length = endIndex - startIndex;
        if (length < 0) { length += sizeCircularBuffer; }

        const int nFrames = length / bufferSize;
        for (auto i = 0; i < nFrames; i++)
        {
            endIndex = startIndex + bufferSize;
            const int size1 = bufferSize - std::max(0, endIndex - sizeCircularBuffer);
            const int size2 = bufferSize - size1;
            bufferIn.head(size1) = circularBuffer.segment(startIndex, size1);
            bufferIn.tail(size2) = circularBuffer.head(size2);
            startIndex = endIndex;
            if (startIndex >= circularBuffer.size()) { startIndex -= sizeCircularBuffer; }

            spectrogram.process(bufferIn, spectrogramOut);
            LogScale.process(spectrogramOut, spectrogramMel);
            for (auto iFrameOut = 0; iFrameOut < nFramesOut; iFrameOut++)
            {
                for (auto y = 1; y < nMels; ++y)
                {
                    auto level = juce::jmap(std::max(energy2dB(spectrogramMel(y, iFrameOut) * scalePlot + 1e-20f), -60.f), -60.f, 0.f, 0.0f, 1.0f);
                    spectrogramImage.setPixelAt(framePlot, nMels - 1 - y, juce::Colour::fromHSV(level, 1.0f, level, 1.0f));
                }
                repaint(framePlot * getLocalBounds().getWidth() / nSpectrogramFrames, 0, 1, getLocalBounds().getHeight());
                framePlot++;
                if (framePlot >= nSpectrogramFrames) { framePlot = 0; }
            }
        }
        readBufferIndex.store(startIndex);
    }

    void paint(juce::Graphics &g) override { g.drawImage(spectrogramImage, getLocalBounds().toFloat()); }

    void stopPlot() { stopTimer(); }
    void startPlot() { startTimerHz(60); }

  private:
    // bufferSize is around 100ms and half the number of samples in the FFT with 50% overlap
    static int getBufferSize(float sampleRate) { return SpectrogramConfiguration::getValidFFTSize(static_cast<int>(2 * sampleRate * 0.1f * 8)) / 2 / 8; }

    static int getNBands(int bufferSize) { return bufferSize + 1; }

    static int getNMels(float sampleRate) { return static_cast<int>(.1f * 2595 * std::log10(1 + (sampleRate / 2) / 700)); }

    // circular buffer size is max of 500ms and 8x the (expected) buffer size
    static int getcircularBufferSize(int expectedBufferSize, float sampleRate)
    {
        int size = 8 * getBufferSize(sampleRate);
        size = std::max(size, static_cast<int>(sampleRate * 0.5f));
        size = std::max(size, 8 * expectedBufferSize);
        return size;
    }

    float sampleRate;
    int nFramesOut;
    int bufferSize;
    int nBands;
    int nMels;
    float scalePlot;
    Eigen::ArrayXf circularBuffer;
    std::atomic<int> writeBufferIndex;
    std::atomic<int> readBufferIndex;
    Spectrogram spectrogram;
    LogScale logScale;
    Eigen::ArrayXf bufferIn;
    Eigen::ArrayXXf spectrogramOut;
    Eigen::ArrayXXf spectrogramMel;
    juce::Image spectrogramImage;
    constexpr static int nSpectrogramFrames = 3000; // number of time frames in spectrogram image
    int framePlot = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramComponent)
};
