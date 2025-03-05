#pragma once
#include "interface/interface.h"

// Algorithm to attenuate an entire audio signal using a gain spectrogram. The gain spectrogram is a matrix of size (hopSize * 2 + 1) x nFrames.
//
//
// audio: The input audio signal. The input audio signal must have a size that is a multiple of the hop size.
// gain spectrogram: The gain spectrogram is a matrix of size nBands x nFrames. The gain spectrogram is used to attenuate the input audio signal.
// startFrame: The frame to start processing. The frame size is equal to the hop size divided by 8.
// nBands: The number of frequency bands in the gain spectrogram. nBands = hopSize * 2 + 1
// nFrames: The number of frames in the gain spectrogram, which is equal to the number of columns in the gain spectrogram matrix.
// frame size: The length of the signal that a gain spectrogram is applied to. frameSize = hopSize / 8
// hop size: The length between each gain calculation processing. The input audio signal must have a size that is a multiple of the hop size.
//
// author: Kristian Timm Andersen

struct AudioAttenuateConfiguration
{
    struct Input
    {
        I::Real audio;             // input audio signal, must be a multiple of the hop size
        I::Real2D gainSpectrogram; // gain attenuation matrix: nBands x nFrames, where nBands = (hopSize * 2 + 1) and nFrames is the number of frames
        I::Int startFrame;         // frame to start processing, frameSize = hopSize / 8
    };

    using Output = O::Real;

    struct Coefficients
    {
        float sampleRate = 48000;
        int hopSize = 1024; // hopsize is equal to bufferSize of the largest filterbank, hopSize = Largest bufferSize = 8 * frameSize = Largest FFT size / 4
        DEFINE_TUNABLE_COEFFICIENTS(sampleRate, hopSize)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXXf, int> initInput(const Coefficients &c)
    {
        Eigen::ArrayXf inputAudio = Eigen::ArrayXf::Random(48000);                              // number of audio samples can be arbitrary
        Eigen::ArrayXXf gainSpectrogram = Eigen::ArrayXXf::Random(c.hopSize * 2 + 1, 20).abs(); // gain between 0 and 1. Arbitrary number of frames
        int iFrame = 3;                                                                         // frame to start applying gain. Must be larger or equal to 0
        return std::make_tuple(inputAudio, gainSpectrogram, iFrame);
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXf::Zero(input.audio.size());
    } // time samples. Same number of samples as input audio

    static bool validInput(Input input, const Coefficients &c)
    {
        float nHops = static_cast<float>(input.audio.size()) / c.hopSize; // number of hops
        int nFrames = static_cast<int>(nHops) * 8;                        // number of frames
        return (nHops == std::round(nHops)) && input.audio.allFinite() && (input.gainSpectrogram.rows() == (c.hopSize * 2 + 1)) &&
               ((input.gainSpectrogram.cols() + input.startFrame) <= nFrames) && (input.gainSpectrogram >= 0.f).all() && (input.gainSpectrogram <= 1.f).all() &&
               (input.startFrame >= 0);
    }

    static bool validOutput(Output output, const Coefficients &c) { return (output.size() > 0) && output.allFinite(); }
};

class AudioAttenuate : public Algorithm<AudioAttenuateConfiguration>
{
  public:
    AudioAttenuate() = default;
    AudioAttenuate(const Coefficients &c);
};