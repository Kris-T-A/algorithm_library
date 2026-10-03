#pragma once
#include "interface/interface.h"

// Algorithm to attenuate an entire audio signal using a gain spectrogram. The gain spectrogram is a matrix of size (hopSize * 2 + 1) x nFrames.
//
//
// audio: The input audio signal. The input audio signal must have a size equal to bufferSize
// gain spectrogram: The gain spectrogram is a matrix of size nBands x 2^(timeOversampling - 1). The gain spectrogram is used to attenuate the input audio signal.
// nBands: The number of frequency bands in the gain spectrogram. nBands = bufferSize * 2 + 1
//
// author: Kristian Timm Andersen

struct AudioAttenuateConfiguration
{
    struct Input
    {
        I::Real audio;             // input audio signal. The input audio signal must have a size equal to bufferSize
        I::Real2D gainSpectrogram; // gain attenuation matrix: nBands x nFrames, where nBands = (bufferSize * 2 + 1), nFrames = 2^(timeOversampling - 1)
    };

    using Output = O::Real;

    struct Coefficients
    {
        int bufferSize = 1024;    // bufferSize is equal to bufferSize of the input (largest filterbank)
        int timeOversampling = 3; // oversampling factor of gainSpectrogram: nFrames = 2^(timeOversampling - 1)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, timeOversampling)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXXf> initInput(const Coefficients &c)
    {
        Eigen::ArrayXf inputAudio = Eigen::ArrayXf::Random(c.bufferSize); // audio samples
        const int nFrames = std::pow(2, c.timeOversampling - 1);
        Eigen::ArrayXXf gainSpectrogram = Eigen::ArrayXXf::Random(c.bufferSize * 2 + 1, nFrames).abs(); // gain between 0 and 1
        return std::make_tuple(inputAudio, gainSpectrogram);
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXf::Zero(c.bufferSize); } // time samples. Same number of samples as input audio

    static bool validInput(Input input, const Coefficients &c)
    {
        const int nFrames = std::pow(2, c.timeOversampling - 1);
        return input.audio.allFinite() && (input.audio.size() == c.bufferSize) && (input.gainSpectrogram.rows() == (c.bufferSize * 2 + 1)) &&
               (input.gainSpectrogram.cols() == nFrames) && (c.timeOversampling > 0) && (input.gainSpectrogram >= 0.f).all() && (input.gainSpectrogram <= 1.f).all();
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.size() == c.bufferSize) && output.allFinite(); }
};

// Apply a gain spectrogram to audio at several time and frequency resolutions,
// then align and combine the attenuated signals into one output.
class AudioAttenuate : public Algorithm<AudioAttenuateConfiguration>
{
  public:
    AudioAttenuate() = default;
    AudioAttenuate(const Coefficients &c);
};