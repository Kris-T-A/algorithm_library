#pragma once
#include "algorithm_library/interface/interface.h"
#include "filterbank/filterbank_wola.h"
#include "framework/framework.h"

// Combine a number of audio channels to one channel by taking the min/max power of each frequency band. The input audio signal is a matrix of size bufferSize x
// nChannels and the FFT size is 4 * bufferSize. The output audio signal is a vector of size bufferSize.
//
// author: Kristian Timm Andersen

struct AudioCombineConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real;

    struct Coefficients
    {
        int bufferSize = 128;
        int nChannels = 4;
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nChannels)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(c.bufferSize, c.nChannels); // audio samples
    }

    static Eigen::ArrayXf initOutput(Input input, const Coefficients &c)
    {
        return Eigen::ArrayXf::Zero(c.bufferSize); // time samples. Same number of samples as input audio
    }

    static bool validInput(Input input, const Coefficients &c) { return input.allFinite() && (input.rows() == c.bufferSize) && (input.cols() == c.nChannels); }

    static bool validOutput(Output output, const Coefficients &c) { return output.allFinite() && (output.size() == c.bufferSize); }
};

class AudioCombineMax : public AlgorithmImplementation<AudioCombineConfiguration, AudioCombineMax>
{
  public:
    AudioCombineMax(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, filterbankAnalysis({.nChannels = c.nChannels, .bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFolds = 1}),
          filterbankSynthesis({.nChannels = 1, .bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFolds = 1})
    {
        nBands = 2 * c.bufferSize + 1;
        spectrogramOut.resize(nBands, c.nChannels);
        spectrogramPower.resize(nBands);
    }

    FilterbankAnalysisWOLA filterbankAnalysis;
    FilterbankSynthesisWOLA filterbankSynthesis;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, filterbankSynthesis)

  private:
    void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input, spectrogramOut);
        spectrogramPower = spectrogramOut.col(0).abs2();
        for (auto channel = 1; channel < C.nChannels; channel++)
        {
            for (auto i = 0; i < nBands; i++)
            {
                const float power = std::norm(spectrogramOut(i, channel));
                if (power > spectrogramPower(i))
                {
                    spectrogramPower(i) = power;
                    spectrogramOut(i, 0) = spectrogramOut(i, channel);
                }
            }
        }
        filterbankSynthesis.process(spectrogramOut.col(0), output);
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        size += spectrogramPower.getDynamicMemorySize();
        return size;
    }

    int nBands;
    Eigen::ArrayXXcf spectrogramOut;
    Eigen::ArrayXf spectrogramPower;

    friend BaseAlgorithm;
};

class AudioCombineMin : public AlgorithmImplementation<AudioCombineConfiguration, AudioCombineMin>
{
  public:
    AudioCombineMin(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c}, filterbankAnalysis({.nChannels = c.nChannels, .bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFolds = 1}),
          filterbankSynthesis({.nChannels = 1, .bufferSize = c.bufferSize, .nBands = 2 * c.bufferSize + 1, .nFolds = 1})
    {
        nBands = 2 * c.bufferSize + 1;
        spectrogramOut.resize(nBands, c.nChannels);
        spectrogramPower.resize(nBands);
    }

    FilterbankAnalysisWOLA filterbankAnalysis;
    FilterbankSynthesisWOLA filterbankSynthesis;
    DEFINE_MEMBER_ALGORITHMS(filterbankAnalysis, filterbankSynthesis)

  private:
    void processAlgorithm(Input input, Output output)
    {
        filterbankAnalysis.process(input, spectrogramOut);
        spectrogramPower = spectrogramOut.col(0).abs2();
        for (auto channel = 1; channel < C.nChannels; channel++)
        {
            for (auto i = 0; i < nBands; i++)
            {
                const float power = std::norm(spectrogramOut(i, channel));
                if (power < spectrogramPower(i))
                {
                    spectrogramPower(i) = power;
                    spectrogramOut(i, 0) = spectrogramOut(i, channel);
                }
            }
        }
        filterbankSynthesis.process(spectrogramOut.col(0), output);
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = spectrogramOut.getDynamicMemorySize();
        size += spectrogramPower.getDynamicMemorySize();
        return size;
    }

    int nBands;
    Eigen::ArrayXXcf spectrogramOut;
    Eigen::ArrayXf spectrogramPower;

    friend BaseAlgorithm;
};