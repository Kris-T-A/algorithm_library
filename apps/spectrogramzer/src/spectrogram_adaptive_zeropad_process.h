#pragma once
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include "save_spectrogram.h"

void spectrogramAdaptiveZeropadProcess(const float* inputPtr, const std::string &outputName, int bufferSize, int nBands, int nFolds, int nonlinearity, int nFrames)
{

    auto c = SpectrogramAdaptiveConfiguration::Coefficients();
    c.bufferSize = bufferSize;
    c.nBands = nBands;
    c.nFolds = nFolds;
    c.nonlinearity = nonlinearity;

    SpectrogramAdaptiveZeropad spectrogram(c);

    Eigen::Map<const Eigen::ArrayXf> input(inputPtr, bufferSize * nFrames);

    // Allocate output matrix
    Eigen::ArrayXXf spec(nBands, nFrames * 8);

    // Process the audio file
    for (int iFrame = 0; iFrame < nFrames; iFrame++)
    {
        spectrogram.process(input.segment(iFrame * bufferSize, bufferSize), spec.middleCols(iFrame * 8, 8));
    }

    // save the spectrogram
    saveSpectrogram(spec, outputName);
}