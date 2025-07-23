#pragma once
#include "algorithm_library/spectrogram.h"
#include "save_spectrogram.h"
#include "rescale_spectrogram.h"

void spectrogramProcess(const float* inputPtr, float sampleRate, const std::string &outputName, int bufferSize, int nBands, int nFolds, int nonlinearity, int nFrames)
{

    auto c = Spectrogram::Coefficients();
    c.bufferSize = bufferSize;
    c.nBands = nBands;
    c.nFolds = nFolds;
    c.nonlinearity = nonlinearity;

    Spectrogram spectrogram(c);

    Eigen::Map<const Eigen::ArrayXf> input(inputPtr, bufferSize * nFrames);

    // Allocate output matrix
    Eigen::ArrayXXf spec(nBands, nFrames);

    // Process the audio file
    for (int iFrame = 0; iFrame < nFrames; iFrame++)
    {
        spectrogram.process(input.segment(iFrame * bufferSize, bufferSize), spec.col(iFrame));
    }

    // convert to dB scale
    spec = 10.f * spec.max(1e-20).log10();
    
    // rescale the spectrogram to a 16:9 aspect ratio
    spec = rescaleSpectrogram(spec, sampleRate);
    // save the spectrogram
    saveSpectrogram(spec, outputName);
}