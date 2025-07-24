#pragma once
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include "save_spectrogram.h"
#include "rescale_spectrogram.h"

void spectrogramAdaptiveZeropadProcess(const float* inputPtr, float sampleRate, const std::string &outputName, int bufferSize, int nBands, int nFolds, int nonlinearity, int nFrames)
{

    auto c = SpectrogramAdaptiveConfiguration::Coefficients();
    c.nSpectrograms = 3;
    c.bufferSize = bufferSize;
    c.nBands = nBands;
    c.nFolds = nFolds;
    c.nonlinearity = nonlinearity;

    SpectrogramAdaptiveZeropad spectrogram(c);

    Eigen::Map<const Eigen::ArrayXf> input(inputPtr, bufferSize * nFrames);

    // Allocate output matrix
    const int nOutputFrames = positivePow2(c.nSpectrograms - 1);
    Eigen::ArrayXXf spec(nBands, nFrames * nOutputFrames);

    // Process the audio file
    for (int iFrame = 0; iFrame < nFrames; iFrame++)
    {
        spectrogram.process(input.segment(iFrame * bufferSize, bufferSize), spec.middleCols(iFrame * nOutputFrames, nOutputFrames));
    }

    // rescale the spectrogram to a 16:9 aspect ratio
    spec = rescaleSpectrogram(spec, sampleRate);
    
    // save the spectrogram
    saveSpectrogram(spec, outputName);
}