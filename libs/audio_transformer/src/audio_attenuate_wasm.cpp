#include "algorithm_library/perceptual_spectral_analysis.h"
#include "audio_attenuate/audio_attenuate_adaptive.h"
#include "spectrogram_adaptive/spectrogram_adaptive_zeropad.h"
#include <emscripten/emscripten.h>
#include <nmmintrin.h>

extern "C"
{

    namespace
    {
    constexpr int ANALYSIS_DELAY = 2;                // number of buffers delay in analysis
    constexpr int OUTPUT_DELAY = 2 * ANALYSIS_DELAY; // number of buffers delay in output
    constexpr int BUFFER_DELAY = OUTPUT_DELAY - 1;   // Number of buffers that produce zero output
    constexpr int FRAMES_PER_BUFFER = 4;             // number of frames per buffer
    }                                                // namespace

    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis(const float *input, const int bufferSize, const int nBuffers, float sampleRate, float *output, bool spectralTilt)
    {
        // Validate input parameters
        if (!input || !output) { return; }
        if (bufferSize <= 0 || nBuffers <= 0) { return; }

        // Create default configuration
        PerceptualSpectralAnalysisConfiguration::Coefficients c;
        c.spectralTilt = spectralTilt;
        // SpectrogramAdaptiveConfiguration::Coefficients c;
        c.bufferSize = bufferSize;
        c.nBands = 2 * bufferSize + 1;
        c.nFolds = 2;
        c.nonlinearity = 1;
        c.sampleRate = sampleRate;
        c.nSpectrograms = std::log2(FRAMES_PER_BUFFER) + 1; // number of spectrograms to produce, each halving the buffer size

        // Create instance of SpectrogramSet
        PerceptualSpectralAnalysis spectrogram(c);
        // SpectrogramAdaptiveZeropad spectrogram(c);

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
     * Create a stateful spectrogram analyzer instance
     * @param bufferSize Size of processing buffer
     * @param sampleRate Sample rate in Hz
     * @return Pointer to SpectrogramAdaptiveZeropad instance (managed by JavaScript)
     */
    EMSCRIPTEN_KEEPALIVE
    SpectrogramAdaptiveZeropad *create_audio_spectral_analysis(const int bufferSize, float sampleRate)
    {
        // Validate input parameters
        if (bufferSize <= 0 || sampleRate <= 0) { return nullptr; }

        // Create configuration
        SpectrogramAdaptiveConfiguration::Coefficients c;
        c.bufferSize = bufferSize;
        c.nBands = 2 * bufferSize + 1;
        c.nFolds = 1;
        c.nonlinearity = 1;
        c.sampleRate = sampleRate;
        c.nSpectrograms = std::log2(FRAMES_PER_BUFFER) + 1; // number of spectrograms to produce, each halving the buffer size

        // Create and return new instance
        return new SpectrogramAdaptiveZeropad(c);
    }

    /**
     * Process audio using a stateful spectrogram analyzer
     * @param analyzer Pointer to SpectrogramAdaptiveZeropad instance
     * @param input Input audio buffer
     * @param nBuffers Number of buffers to process
     * @param output Output spectrogram matrix (nBands x nFrames)
     *
     * @note nBands = 2 * bufferSize + 1
     * @note nFrames = FRAMES_PER_BUFFER * nBuffers
     */
    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis_stateful(SpectrogramAdaptiveZeropad *analyzer, const float *input, const int nBuffers, float *output)
    {
        // Validate input parameters
        if (!analyzer || !input || !output) { return; }

        const int bufferSize = analyzer->getCoefficients().bufferSize;
        const int nBands = analyzer->getCoefficients().nBands;

        // Calculate derived values
        const int length = bufferSize * nBuffers;
        const int nFrames = FRAMES_PER_BUFFER * nBuffers;

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXXf> outputSpectrogram(output, nBands, nFrames);

        // Process audio
        for (int iBuffer = 0; iBuffer < nBuffers; iBuffer++)
        {
            analyzer->process(inputAudio.segment(iBuffer * bufferSize, bufferSize), outputSpectrogram.middleCols(iBuffer * FRAMES_PER_BUFFER, FRAMES_PER_BUFFER));
        }
    }

    /**
     * Destroy a spectrogram analyzer instance
     * @param analyzer Pointer to SpectrogramAdaptiveZeropad instance to destroy
     */
    EMSCRIPTEN_KEEPALIVE
    void destroy_audio_spectral_analysis(SpectrogramAdaptiveZeropad *analyzer) { delete analyzer; }

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
        const int nGainFrames = FRAMES_PER_BUFFER * nFrames;
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
