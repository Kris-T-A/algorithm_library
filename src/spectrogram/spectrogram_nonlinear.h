#pragma once
#include "algorithm_library/spectrogram.h"
#include "filterbank/filterbank_single_channel.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// Spectrogram implemented as a nonlinear combination of several standard spectrograms. The criteria used for selecting the best time/frequency bin is the minimum power.
//
// author: Kristian Timm Andersen
class SpectrogramNonlinear : public AlgorithmImplementation<SpectrogramConfiguration, SpectrogramNonlinear>
{
  public:
    SpectrogramNonlinear(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .nFolds = 1, .nonlinearity = 1})
        : BaseAlgorithm{c}, filterbanks(
                                [&c]() {
                                    if (c.nonlinearity == 0) { return 1; } // if not nonlinear, use only one filterbank for the standard spectrogram
                                    return 3;                              // 3 filterbanks used for nonlinear combination
                                }(),
                                {.nChannels = 1, .bufferSize = c.bufferSize, .nBands = c.nBands, .nFolds = c.nFolds})
    {
        assert(c.nonlinearity >= 0);

        filterbankOut.resize(c.nBands);

        if (c.nonlinearity > 0)
        {
            // set windows
            const int stride = positivePow2(c.nonlinearity);
            const int frameSize = filterbanks[0].getFrameSize();
            const int frameSizeSmall = frameSize / stride;
            Eigen::ArrayXf window = filterbanks[0].getWindow();
            const float winScale = window.abs2().sum();
            Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Map(window.data(), frameSizeSmall, Eigen::InnerStride<>(stride));

            // assymetric window on left side
            window.head((frameSize - frameSizeSmall) / 2).setZero();
            window.segment((frameSize - frameSizeSmall) / 2, frameSizeSmall / 2) = windowSmall.head(frameSizeSmall / 2);
            window *= std::sqrt(winScale / window.abs2().sum());
            filterbanks[1].setWindow(window);

            // assymetric window on right side
            window = filterbanks[0].getWindow();
            window.tail((frameSize - frameSizeSmall) / 2).setZero();
            window.segment(frameSize / 2, frameSizeSmall / 2) = windowSmall.tail(frameSizeSmall / 2);
            window *= std::sqrt(winScale / window.abs2().sum());
            filterbanks[2].setWindow(window);
        }
    }

    VectorAlgo<FilterbankAnalysisSingleChannel> filterbanks;
    DEFINE_MEMBER_ALGORITHMS(filterbanks)

  private:
    void inline processAlgorithm(Input input, Output output)
    {
        // process linear filterbank and update output with the minimum power
        filterbanks[0].process(input, filterbankOut);
        output = filterbankOut.abs2();
        if (C.nonlinearity > 0) // process non linear filterbanks
        {
            filterbanks[1].process(input, filterbankOut);
            output = output.min(filterbankOut.abs2());
            filterbanks[2].process(input, filterbankOut);
            output = output.min(filterbankOut.abs2());
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