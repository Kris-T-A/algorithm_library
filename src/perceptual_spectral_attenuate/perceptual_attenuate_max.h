#pragma once
#include "algorithm_library/perceptual_spectral_attenuate.h"
#include "audio_attenuate/audio_combine.h"
#include "delay/circular_buffer.h"
#include "filterbank_set/filterbank_set_wola.h"
#include "framework/framework.h"

// Attenuate audio in parallel WOLA filterbanks with successively halved hop sizes.
// Expand linear perceptual-band gains to FFT-bin gains, then
// take the minimum gain over each resolution's frequency and time groups.
// Synthesize the attenuated spectra, align the resolutions' delays and combine
// them by selecting the maximum spectral power per bin.
// Total latency is three largest hops plus three smallest hops. All working
// buffers are allocated during construction so processing requires no allocation.
class PerceptualAttenuateMax : public AlgorithmImplementation<PerceptualSpectralAttenuateConfiguration, PerceptualAttenuateMax>
{
  public:
    PerceptualAttenuateMax(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, nFrames(positivePow2(c.nSpectrograms - 1)), bufferSizeSmall(c.bufferSize / nFrames),
          filterbankAnalysis({.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFilterbanks = c.nSpectrograms, .nFolds = 1}),
          filterbankSynthesis({.bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFilterbanks = c.nSpectrograms, .nFolds = 1}),
          audioCombineMax({.bufferSize = bufferSizeSmall, .nChannels = c.nSpectrograms})
    {
        delay.resize(c.nSpectrograms - 1);
        for (int i = 0; i < c.nSpectrograms - 1; ++i)
        {
            delay[i].setCoefficients({.nChannels = 1, .delayLength = 3 * c.bufferSize - 3 * (c.bufferSize / positivePow2(i + 1))});
        }
        spectrogramMultipleResolution = filterbankAnalysis.initDefaultOutput();
        gainOldMultipleResolution.resize(c.nSpectrograms);
        for (int i = 0; i < c.nSpectrograms; ++i)
        {
            gainOldMultipleResolution[i].resize(spectrogramMultipleResolution[i].rows(), spectrogramMultipleResolution[i].cols());
        }
        outputSet = filterbankSynthesis.initDefaultOutput();
        delayedOutput.resize(bufferSizeSmall, c.nSpectrograms);
        gainExpanded.resize(2 * c.bufferSize + 1, nFrames);

        // Same band midpoint boundaries as LogScale::inverse, computed once.
        const double minLog = std::log10(1.0 + c.frequencyMin);
        const double maxLog = std::log10(1.0 + c.frequencyMax);
        const double freqPerBin = static_cast<double>(c.sampleRate) / (4 * c.bufferSize);
        const Eigen::ArrayXd logs = Eigen::ArrayXd::LinSpaced(c.nBands, minLog, maxLog);
        const Eigen::ArrayXf centerBins = logs.unaryExpr([freqPerBin](double x) { return (std::pow(10.0, x) - 1.0) / freqPerBin; }).cast<float>();
        bandBoundaries.resize(c.nBands + 1);
        bandBoundaries(0) = 0;
        for (int i = 1; i < c.nBands; ++i)
        {
            bandBoundaries(i) = static_cast<int>(std::round((centerBins(i - 1) + centerBins(i)) / 2.f));
        }
        bandBoundaries(c.nBands) = gainExpanded.rows();
        resetVariables();
    }

    int nFrames;
    int bufferSizeSmall;
    FilterbankSetAnalysisWOLA filterbankAnalysis;
    FilterbankSetSynthesisWOLA filterbankSynthesis;
    VectorAlgo<CircularBuffer> delay;
    AudioCombineMax audioCombineMax;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, filterbankSynthesis, delay, audioCombineMax)

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input.signal, spectrogramMultipleResolution);
        for (int i = 0; i < C.nBands; ++i)
        {
            for (int frame = 0; frame < nFrames; ++frame)
            {
                gainExpanded.col(frame).segment(bandBoundaries(i), bandBoundaries(i + 1) - bandBoundaries(i)).setConstant(input.gain(i, frame));
            }
        }

        for (int i = 0; i < C.nSpectrograms; ++i)
        {
            const int frequencyGroup = positivePow2(i);
            const int timeGroup = nFrames / frequencyGroup;
            auto &gain = gainOldMultipleResolution[i];
            auto &spectrum = spectrogramMultipleResolution[i];
            // Each analysis window is centred one resolution hop before its
            // input interval. The first frame therefore uses the last gain
            // from the previous buffer; later frames use this buffer's gains.
            spectrum.col(0) *= gain.col(gain.cols() - 1);
            for (int frame = 0; frame < frequencyGroup; ++frame)
            {
                for (int bin = 0; bin < gain.rows() - 1; ++bin)
                {
                    gain(bin, frame) = gainExpanded.block(bin * frequencyGroup, frame * timeGroup, frequencyGroup, timeGroup).minCoeff();
                }
                // Nyquist is a single bin, rather than a full frequency group.
                gain(gain.rows() - 1, frame) = gainExpanded.row(gainExpanded.rows() - 1).segment(frame * timeGroup, timeGroup).minCoeff();
            }
            for (int frame = 1; frame < frequencyGroup; ++frame) { spectrum.col(frame) *= gain.col(frame - 1); }
        }
        filterbankSynthesis.process(spectrogramMultipleResolution, outputSet);
        for (int frame = 0; frame < nFrames; ++frame)
        {
            const int offset = frame * bufferSizeSmall;
            delayedOutput.col(0) = outputSet.col(0).segment(offset, bufferSizeSmall);
            for (int i = 1; i < C.nSpectrograms; ++i)
            {
                delay[i - 1].process(outputSet.col(i).segment(offset, bufferSizeSmall), delayedOutput.col(i));
            }
            audioCombineMax.process(delayedOutput, output.segment(offset, bufferSizeSmall));
        }
    }

    void resetVariables() final
    {
        for (auto &gain : gainOldMultipleResolution) { gain.setOnes(); }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = gainExpanded.getDynamicMemorySize() + bandBoundaries.getDynamicMemorySize();
        size += outputSet.getDynamicMemorySize() + delayedOutput.getDynamicMemorySize();
        for (const auto &spectrum : spectrogramMultipleResolution) { size += spectrum.getDynamicMemorySize(); }
        for (const auto &gain : gainOldMultipleResolution) { size += gain.getDynamicMemorySize(); }
        return size;
    }

    std::vector<Eigen::ArrayXXcf> spectrogramMultipleResolution;
    std::vector<Eigen::ArrayXXf> gainOldMultipleResolution;
    Eigen::ArrayXXf gainExpanded;
    Eigen::ArrayXi bandBoundaries;
    Eigen::ArrayXXf outputSet;
    Eigen::ArrayXXf delayedOutput;

    friend BaseAlgorithm;
};
