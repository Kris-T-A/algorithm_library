#include "audio_attenuate/audio_attenuate_adaptive.h"
#include "spectrogram_adaptive/spectrogram_adaptive_wola.h"
#include <emscripten/bind.h>
#include <nmmintrin.h>

using namespace emscripten;

extern "C"
{

    namespace
    {
    constexpr int ANALYSIS_DELAY = 2;                // number of buffers delay in analysis
    constexpr int OUTPUT_DELAY = 2 * ANALYSIS_DELAY; // number of buffers delay in output
    constexpr int BUFFER_DELAY = OUTPUT_DELAY - 1;   // Number of buffers that produce zero output
    constexpr int FRAMES_PER_BUFFER = 8;             // number of frames per buffer
    } // namespace

    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis(const float *input, const int bufferSize, const int nBuffers, float *output)
    {
        // Validate input parameters
        if (!input || !output) { return; }
        if (bufferSize <= 0 || nBuffers <= 0) { return; }

        // Create default configuration
        SpectrogramAdaptiveConfiguration::Coefficients c;
        c.bufferSize = bufferSize;
        c.nBands = 2 * bufferSize + 1;
        c.nFolds = 1;
        c.nonlinearity = 1;
        c.nSpectrograms = std::log2(FRAMES_PER_BUFFER) + 1; // number of spectrograms to produce, each halving the buffer size

        // Create instance of SpectrogramSet
        SpectrogramAdaptiveWOLA spectrogram(c);

        // derived values
        const int length = bufferSize * nBuffers;         // total length of input in samples. It is callers responsibility to ensure that input is at least this long
        const int nFrames = FRAMES_PER_BUFFER * nBuffers; // number of buffers in gain spectrogram

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXXf> outputSpectrogram(output, c.nBands, nFrames);

        // Process audio
        for (int iBuffer = 0; iBuffer < nBuffers; iBuffer++)
        {
            spectrogram.process(inputAudio.segment(iBuffer * bufferSize, bufferSize), outputSpectrogram.middleCols(iBuffer * FRAMES_PER_BUFFER, FRAMES_PER_BUFFER));
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
        if (length <= 0 || bufferSize <= 0) { return; }

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
        for (int i = 0; i < BUFFER_DELAY; i++)
        {
            attenuator.process({inputAudio.segment(i * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(i * FRAMES_PER_BUFFER, FRAMES_PER_BUFFER)},
                               outputAudio.segment(0, bufferSize));
        }

        // main process loop
        for (auto iFrame = BUFFER_DELAY; iFrame < nFrames; iFrame++)
        {
            const int iOutputFrame = iFrame - BUFFER_DELAY;
            attenuator.process({inputAudio.segment(iFrame * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(iFrame * FRAMES_PER_BUFFER, FRAMES_PER_BUFFER)},
                               outputAudio.segment(iOutputFrame * bufferSize, bufferSize));
        }

        // get last output frames (input is repeated for each frame)
        for (int i = 0; i < BUFFER_DELAY; i++)
        {
            attenuator.process(
                {inputAudio.segment((nFrames - 1) * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols((nFrames - 1) * FRAMES_PER_BUFFER, FRAMES_PER_BUFFER)},
                outputAudio.segment((nFrames - BUFFER_DELAY + i) * bufferSize, bufferSize));
        }
    }
}
