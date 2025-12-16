#include "algorithm_library/convert_rgba.h"
#include "algorithm_library/perceptual_spectral_analysis.h"
#include "audio_attenuate/audio_attenuate_adaptive.h"
#include <emscripten/emscripten.h>
#include <nmmintrin.h>

extern "C"
{

    namespace
    {
    constexpr int ANALYSIS_DELAY = 2;                // number of buffers delay in analysis
    constexpr int OUTPUT_DELAY = 2 * ANALYSIS_DELAY; // number of buffers delay in output
    constexpr int BUFFER_DELAY = OUTPUT_DELAY - 1;   // Number of buffers that produce zero output
    } // namespace

    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis(const float *input, const int bufferSize, const int nBuffers, float sampleRate, float *output, bool spectralTilt, int framesPerBuffer)
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
        c.nFolds = 1; // number of folds for spectral tilt
        c.nonlinearity = 1;
        c.sampleRate = sampleRate;
        c.nSpectrograms = std::log2(framesPerBuffer) + 1; // number of spectrograms to produce, each halving the buffer size

        // Create instance of SpectrogramSet
        PerceptualSpectralAnalysis spectrogram(c);

        // derived values
        const int length = bufferSize * nBuffers;       // total length of input in samples. It is callers responsibility to ensure that input is at least this long
        const int nFrames = framesPerBuffer * nBuffers; // number of buffers in gain spectrogram

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXXf> outputSpectrogram(output, c.nBands, nFrames);

        // Process audio

        // Initial frames give zero output
        for (int iBuffer = 0; iBuffer < ANALYSIS_DELAY; iBuffer++)
        {
            spectrogram.process(inputAudio.segment(iBuffer * bufferSize, bufferSize), outputSpectrogram.leftCols(framesPerBuffer));
        }

        // processing buffers
        int oBuffer = 0;
        for (int iBuffer = ANALYSIS_DELAY; iBuffer < nBuffers; iBuffer++, oBuffer++)
        {
            spectrogram.process(inputAudio.segment(iBuffer * bufferSize, bufferSize), outputSpectrogram.middleCols(oBuffer * framesPerBuffer, framesPerBuffer));
        }

        // final frames to give output
        Eigen::ArrayXf inputZero = Eigen::ArrayXf::Zero(bufferSize);
        for (; oBuffer < nBuffers; oBuffer++)
        {
            spectrogram.process(inputZero, outputSpectrogram.middleCols(oBuffer * framesPerBuffer, framesPerBuffer));
        }
    }

    /**
     * Create a stateful spectrogram analyzer instance
     * @param bufferSize Size of processing buffer
     * @param sampleRate Sample rate in Hz
     * @return Pointer to SpectrogramAdaptiveZeropad instance (managed by JavaScript)
     */
    EMSCRIPTEN_KEEPALIVE
    PerceptualSpectralAnalysis *create_audio_spectral_analysis(const int bufferSize, const int nBands, const float sampleRate, const float frequencyMin,
                                                               const float frequencyMax, const bool spectralTilt, const int framesPerBuffer, int method)
    {
        // Validate input parameters
        if (bufferSize <= 0 || sampleRate <= 0) { return nullptr; }

        // Create configuration
        PerceptualSpectralAnalysisConfiguration::Coefficients c;
        c.spectralTilt = spectralTilt;
        c.bufferSize = bufferSize;
        c.nBands = nBands;
        c.nFolds = 1;
        c.nonlinearity = 1;
        c.sampleRate = sampleRate;
        c.frequencyMin = frequencyMin;
        c.frequencyMax = frequencyMax;
        c.nSpectrograms = std::log2(framesPerBuffer) + 1; // number of spectrograms to produce, each halving the buffer size
        if (method == 0) { c.method = PerceptualSpectralAnalysisConfiguration::Coefficients::ADAPTIVE; }
        else
        {
            c.method = PerceptualSpectralAnalysisConfiguration::Coefficients::NONLINEAR;
        }

        // Create and return new instance
        return new PerceptualSpectralAnalysis(c);
    }

    /**
     * Process audio using a stateful spectrogram analyzer
     * @param analyzer Pointer to PerceptualSpectralAnalysis instance
     * @param input Input audio buffer
     * @param nBuffers Number of buffers to process
     * @param output Output spectrogram matrix (nBands x nFrames)
     *
     * @note nBands = 2 * bufferSize + 1
     * @note nFrames = framesPerBuffer * nBuffers
     */
    EMSCRIPTEN_KEEPALIVE
    void audio_spectral_analysis_stateful(PerceptualSpectralAnalysis *analyzer, const float *input, const int nBuffers, float *output)
    {
        // Validate input parameters
        if (!analyzer || !input || !output) { return; }

        const PerceptualSpectralAnalysisConfiguration::Coefficients c = analyzer->getCoefficients();
        const int bufferSize = c.bufferSize;
        const int nBands = c.nBands;
        const int framesPerBuffer = positivePow2(c.nSpectrograms - 1);

        // Calculate derived values
        const int length = bufferSize * nBuffers;
        const int nFrames = framesPerBuffer * nBuffers;

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXXf> outputSpectrogram(output, nBands, nFrames);

        // Process audio
        for (int iBuffer = 0; iBuffer < nBuffers; iBuffer++)
        {
            analyzer->process(inputAudio.segment(iBuffer * bufferSize, bufferSize), outputSpectrogram.middleCols(iBuffer * framesPerBuffer, framesPerBuffer));
        }
    }

    /**
     * Destroy a spectrogram analyzer instance
     * @param analyzer Pointer to PerceptualSpectralAnalysis instance to destroy
     */
    EMSCRIPTEN_KEEPALIVE
    void destroy_audio_spectral_analysis(PerceptualSpectralAnalysis *analyzer) { delete analyzer; }

    EMSCRIPTEN_KEEPALIVE
    void getMinMaxTimeValues(const float *input, const int nBands, const int nFrames, float *minValues, float *maxValues)
    {
        Eigen::Map<const Eigen::ArrayXXf> inputSpectrogram(input, nBands, nFrames);
        Eigen::Map<Eigen::ArrayXf> minVals(minValues, nFrames);
        Eigen::Map<Eigen::ArrayXf> maxVals(maxValues, nFrames);
        minVals = inputSpectrogram.colwise().minCoeff().transpose();
        maxVals = inputSpectrogram.colwise().maxCoeff().transpose();
    }

    EMSCRIPTEN_KEEPALIVE
    void getMinMaxValues(const float *input, const int nBands, const int nFrames, float *minValues, float *maxValues)
    {
        Eigen::Map<const Eigen::ArrayXXf> inputSpectrogram(input, nBands, nFrames);
        *minValues = inputSpectrogram.minCoeff();
        *maxValues = inputSpectrogram.maxCoeff();
    }

    EMSCRIPTEN_KEEPALIVE
    int get_n_frequency_bands(PerceptualSpectralAnalysis *analyzer)
    {
        if (!analyzer) { return 0; }
        const PerceptualSpectralAnalysisConfiguration::Coefficients &c = analyzer->getCoefficients();
        return c.nBands;
    }

    EMSCRIPTEN_KEEPALIVE
    void convert_rgba(const float *input, const int width, const int height, uint8_t alpha, int method, uint8_t *output)
    {
        // Validate input parameters
        if (!input || !output) { return; }
        if (width <= 0 || height <= 0) { return; }

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXXf> inputImage(input, height, width);
        Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> outputImage(output, 4 * height, width);

        // Create converter instance
        ConvertRGBA::Coefficients c;
        c.alpha = alpha;
        switch (method)
        {
        case 0: c.colorScale = ConvertRGBA::Coefficients::OCEAN; break;
        case 1: c.colorScale = ConvertRGBA::Coefficients::PARULA; break;
        case 2: c.colorScale = ConvertRGBA::Coefficients::VIRIDIS; break;
        case 3: c.colorScale = ConvertRGBA::Coefficients::MAGMA; break;
        case 4: c.colorScale = ConvertRGBA::Coefficients::PLASMA; break;
        default: c.colorScale = ConvertRGBA::Coefficients::PARULA;
        }
        ConvertRGBA converter(c);

        // Perform conversion
        converter.process(inputImage, outputImage);
    }

    EMSCRIPTEN_KEEPALIVE
    void scale_and_convert_rgba(const float *input, const int width, const int height, uint8_t alpha, int method, uint8_t *output, float scaleMin, float scaleMax)
    {
        // Validate input parameters
        if (!input || !output) { return; }
        if (width <= 0 || height <= 0) { return; }

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXXf> inputImage(input, height, width);
        Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>> outputImage(output, 4 * width, height);

        // Create converter instance
        ConvertRGBA::Coefficients c;
        c.alpha = alpha;
        switch (method)
        {
        case 0: c.colorScale = ConvertRGBA::Coefficients::OCEAN; break;
        case 1: c.colorScale = ConvertRGBA::Coefficients::PARULA; break;
        case 2: c.colorScale = ConvertRGBA::Coefficients::VIRIDIS; break;
        case 3: c.colorScale = ConvertRGBA::Coefficients::MAGMA; break;
        case 4: c.colorScale = ConvertRGBA::Coefficients::PLASMA; break;
        default: c.colorScale = ConvertRGBA::Coefficients::PARULA;
        }
        ConvertRGBA converter(c);

        // scale and transform to row-major with flip
        float denominator = std::max(scaleMax - scaleMin, 1e-6f);
        Eigen::ArrayXXf scaledImage = (inputImage.transpose() - scaleMin).max(0.f).min(scaleMax) / denominator;

        // Perform conversion
        converter.process(scaledImage, outputImage);
    }

    // Get color values for a list of normalized floats (0.0 to 1.0)
    EMSCRIPTEN_KEEPALIVE
    void getColorValues(float *colorValues, int length, uint8_t alpha, int method, uint8_t *outputValues)
    {
        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputColors(colorValues, length);
        Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>> outputColors(outputValues, 4 * length);

        // Create converter instance
        ConvertRGBA::Coefficients c;
        c.alpha = alpha;
        switch (method)
        {
        case 0: c.colorScale = ConvertRGBA::Coefficients::OCEAN; break;
        case 1: c.colorScale = ConvertRGBA::Coefficients::PARULA; break;
        case 2: c.colorScale = ConvertRGBA::Coefficients::VIRIDIS; break;
        case 3: c.colorScale = ConvertRGBA::Coefficients::MAGMA; break;
        case 4: c.colorScale = ConvertRGBA::Coefficients::PLASMA; break;
        default: c.colorScale = ConvertRGBA::Coefficients::PARULA;
        }
        ConvertRGBA converter(c);

        // Get color values
        converter.process(inputColors, outputColors);
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
    void audio_attenuate(const float *input, const float *gainSpectrogram, float *output, const int length, const int bufferSize, int framesPerBuffer)
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
        const int nGainFrames = framesPerBuffer * nFrames;
        const int nBands = 2 * bufferSize + 1;

        // Map raw pointers to Eigen arrays
        Eigen::Map<const Eigen::ArrayXf> inputAudio(input, length);
        Eigen::Map<Eigen::ArrayXf> outputAudio(output, length);
        Eigen::Map<const Eigen::ArrayXXf> gainSpectrogramMatrix(gainSpectrogram, nBands, nGainFrames);

        // Process audio

        // Initial frames give zero output
        for (int i = 0; i < BUFFER_DELAY; i++)
        {
            attenuator.process({inputAudio.segment(i * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(i * framesPerBuffer, framesPerBuffer)},
                               outputAudio.segment(0, bufferSize));
        }

        // main process loop
        for (auto iFrame = BUFFER_DELAY; iFrame < nFrames; iFrame++)
        {
            const int iOutputFrame = iFrame - BUFFER_DELAY;
            attenuator.process({inputAudio.segment(iFrame * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols(iFrame * framesPerBuffer, framesPerBuffer)},
                               outputAudio.segment(iOutputFrame * bufferSize, bufferSize));
        }

        // get last output frames (input is repeated for each frame)
        for (int i = 0; i < BUFFER_DELAY; i++)
        {
            attenuator.process(
                {inputAudio.segment((nFrames - 1) * bufferSize, bufferSize), gainSpectrogramMatrix.middleCols((nFrames - 1) * framesPerBuffer, framesPerBuffer)},
                outputAudio.segment((nFrames - BUFFER_DELAY + i) * bufferSize, bufferSize));
        }
    }
}
