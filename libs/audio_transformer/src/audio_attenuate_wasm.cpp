#include <nmmintrin.h>
#include "audio_attenuate/audio_attenuate_adaptive.h"
#include "spectrogram/spectrogram_set.h"
#include <emscripten/bind.h>


using namespace emscripten;

extern "C"
{

    namespace
    {
    constexpr int FRAME_DELAY = 3;     // Number of frames that produce zero output
    constexpr int GAINS_PER_FRAME = 8; // number of gains per frame
    }                                  // namespace

    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis(const float *input, float *output, const int length, const int bufferSize)
    {
        // Validate input parameters
        if (!input || !output) { return; }
        if (length <= 0 || bufferSize <= 0 ) { return; }

        // Create default configuration
        SpectrogramConfiguration::Coefficients c;
        c.bufferSize = bufferSize;
        c.nBands = 2 * bufferSize + 1;
        c.algorithmType = SpectrogramConfiguration::Coefficients::ADAPTIVE_HANN_8;

        // Create instance of SpectrogramSet
        SpectrogramSet spectrogram(c);

        // derived values
        const int nFrames = length / bufferSize;
        const int nGainFrames = 8 * nFrames;
        constexpr int outputDelay = FRAME_DELAY - 1;

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXXf> outputSpectrogram(output, c.nBands, nGainFrames);
        
        // Process audio
        for (int i = 0; i < outputDelay; i++)
        {
            spectrogram.process(inputAudio.segment(i * bufferSize, bufferSize), outputSpectrogram.middleCols(i * GAINS_PER_FRAME, GAINS_PER_FRAME));
        }

        // main process loop
        for (auto iFrame = outputDelay; iFrame < nFrames; iFrame++)
        {
            const int iOutputFrame = iFrame - outputDelay;
            spectrogram.process(inputAudio.segment(iFrame * bufferSize, bufferSize), outputSpectrogram.middleCols(iOutputFrame * GAINS_PER_FRAME, GAINS_PER_FRAME));
        }

        // get last output frames (input is zeroed for each frame)
        Eigen::ArrayXf inputZeros = Eigen::ArrayXf::Zero(bufferSize);
        for (int i = 0; i < outputDelay; i++)
        {
            spectrogram.process(inputZeros, outputSpectrogram.middleCols((nFrames - outputDelay + i) * GAINS_PER_FRAME, GAINS_PER_FRAME));
        }
    }


    /**
     * Process audio using attenuation
     * @param input Input audio buffer (must be 16-byte aligned)
     * @param gainSpectrogram Gain spectrogram matrix (nBands x nGainFrames)
     * @param output Output audio buffer (must be pre-allocated)
     * @param length Total length of input/output buffers in samples
     * @param bufferSize Size of processing buffer (must divide length evenly)
     *
     * @note nBands = 2 * bufferSize + 1
     * @note nFrames = length / bufferSize
     * @note nGainFrames = 8 * nFrames
     */
    EMSCRIPTEN_KEEPALIVE
    void audio_attenuate(const float *input, const float *gainSpectrogram, float *output, const int length, const int bufferSize)
    {
        // Validate input parameters
        if (!input || !gainSpectrogram || !output) { return; }
        if (length <= 0 || bufferSize <= 0 ) { return; }

        // Create default configuration
        AudioAttenuateConfiguration::Coefficients c;
        c.bufferSize = bufferSize;

        // Create instance of AudioAttenuate
        AudioAttenuateAdaptive attenuator(c);

        // derived values
        const int nFrames = length / bufferSize;
        const int nGainFrames = 8 * nFrames;
        const int nBands = 2 * bufferSize + 1;

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXf> outputAudio(output, length);
        Eigen::Map<const Eigen::ArrayXXf> gainSpectrogramMatrix(gainSpectrogram, nBands, nGainFrames);

        // Process audio

        // Initial frames give zero output
        for (int i = 0; i < FRAME_DELAY; i++)
        {
            attenuator.process({inputAudio.segment(i * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(i * GAINS_PER_FRAME, GAINS_PER_FRAME)},
                               outputAudio.segment(0, bufferSize));
        }

        // main process loop
        for (auto iFrame = FRAME_DELAY; iFrame < nFrames; iFrame++)
        {
            const int iOutputFrame = iFrame - FRAME_DELAY;
            attenuator.process({inputAudio.segment(iFrame * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(iFrame * GAINS_PER_FRAME, GAINS_PER_FRAME)},
                               outputAudio.segment(iOutputFrame * bufferSize, bufferSize));
        }

        // get last output frames (input is repeated for each frame)
        for (int i = 0; i < FRAME_DELAY; i++)
        {
            attenuator.process(
                {inputAudio.segment((nFrames - 1) * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols((nFrames - 1) * GAINS_PER_FRAME, GAINS_PER_FRAME)},
                outputAudio.segment((nFrames - FRAME_DELAY + i) * bufferSize, bufferSize));
        }
    }
}
