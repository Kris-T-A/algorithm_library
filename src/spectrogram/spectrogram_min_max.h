#pragma once
#include "filterbank/filterbank_single_channel.h"
#include "framework/framework.h"

// spectrogram that outputs both the minimum and maximum power in each frequency band
//
// author: Kristian Timm Andersen

struct SpectrogramMinMaxConfiguration
{
    using Input = I::Real;
    struct Output
    {
        O::Real minPower;
        O::Real maxPower;
    };

    struct Coefficients
    {
        int bufferSize = 1024; // input buffer size
        int nBands = 1025;     // number of frequency bands in output
        int nFolds = 1;        // number of folds: frameSize = nFolds * 2 * (nBands - 1)
        DEFINE_TUNABLE_COEFFICIENTS(bufferSize, nBands, nFolds)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.bufferSize); } // time samples

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> initOutput(Input input, const Coefficients &c)
    {
        return {Eigen::ArrayXf::Zero(c.nBands), Eigen::ArrayXf::Zero(c.nBands)};
    } // power spectrograms

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.bufferSize) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.minPower.rows() == c.nBands) && (output.maxPower.rows() == c.nBands) && output.minPower.allFinite() && output.maxPower.allFinite() &&
               (output.minPower >= 0).all() && (output.maxPower >= 0).all();
    }
};

class SpectrogramMinMax : public AlgorithmImplementation<SpectrogramMinMaxConfiguration, SpectrogramMinMax>
{
  public:
    SpectrogramMinMax(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .nFolds = 1})
        : BaseAlgorithm{c}, filterbanks(3, {.nChannels = 1, .bufferSize = c.bufferSize, .nBands = c.nBands, .nFolds = c.nFolds})
    {
        filterbankOut.resize(c.nBands);

        // set windows
        const int frameSize = filterbanks[0].getFrameSize();
        const int frameSizeSmall = frameSize / 2;
        Eigen::ArrayXf window = filterbanks[0].getWindow();
        const float winScale = window.sum();
        Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Map(window.data(), frameSizeSmall, Eigen::InnerStride<2>());

        // assymetric window on left side
        window.head((frameSize - frameSizeSmall) / 2).setZero();
        window.segment((frameSize - frameSizeSmall) / 2, frameSizeSmall / 2) = windowSmall.head(frameSizeSmall / 2);
        window *= winScale / window.sum();
        filterbanks[1].setWindow(window);

        // assymetric window on right side
        window = filterbanks[0].getWindow();
        window.tail((frameSize - frameSizeSmall) / 2).setZero();
        window.segment(frameSize / 2, frameSizeSmall / 2) = windowSmall.tail(frameSizeSmall / 2);
        window *= winScale / window.sum();
        filterbanks[2].setWindow(window);
    }

    VectorAlgo<FilterbankAnalysisSingleChannel> filterbanks;
    DEFINE_MEMBER_ALGORITHMS(filterbanks)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        filterbanks[0].process(input, filterbankOut);
        output.minPower = filterbankOut.abs2();
        output.maxPower = output.minPower; // initialize max with min
        filterbanks[1].process(input, filterbankOut);
        for (auto iBand = 0; iBand < C.nBands; iBand++)
        {
            float power = std::norm(filterbankOut(iBand));
            output.minPower(iBand) = std::min(output.minPower(iBand), power);
            output.maxPower(iBand) = std::max(output.maxPower(iBand), power);
        }
        filterbanks[2].process(input, filterbankOut);
        for (auto iBand = 0; iBand < C.nBands; iBand++)
        {
            float power = std::norm(filterbankOut(iBand));
            output.minPower(iBand) = std::min(output.minPower(iBand), power);
            output.maxPower(iBand) = std::max(output.maxPower(iBand), power);
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = filterbankOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXcf filterbankOut;

    friend BaseAlgorithm;
};