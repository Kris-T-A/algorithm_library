#include "algorithm_library/spectrogram.h"
#include "filterbank/filterbank_single_channel.h"
#include "framework/framework.h"

// Spectrogram implemented as a nonlinear combination of several standard spectrograms. The criteria used for selecting the best time/frequency bin is the minimum power.
//
// author: Kristian Timm Andersen
class SpectrogramNonlinear : public AlgorithmImplementation<SpectrogramConfiguration, SpectrogramNonlinear>
{
  public:
    SpectrogramNonlinear(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .algorithmType = Coefficients::ADAPTIVE_HANN_8})
        : BaseAlgorithm{c}, filterbank0({initializeFilterbanks(c)}), filterbank1({initializeFilterbanks(c)}), filterbank2({initializeFilterbanks(c)}),
          filterbank3({initializeFilterbanks(c)})
    {
        assert((c.algorithmType == Coefficients::ADAPTIVE_HANN_8) || (c.algorithmType == Coefficients::ADAPTIVE_WOLA_8));

        bufferSizeSmall = C.bufferSize / 8; // algorithm is hardcoded to process in buffersize of this size
        // set windows
        int frameSize = filterbank0.getFrameSize();
        Eigen::ArrayXf window = filterbank0.getWindow();
        float sqrtPower = std::sqrt(window.abs2().sum());

        // half the frame size and set window to every 2nd value of full window
        int frameSizeSmall = FFTConfiguration::getValidFFTSize(frameSize / 2);
        Eigen::ArrayXf windowSmall = Eigen::ArrayXf::Zero(frameSize);
        windowSmall.segment((frameSize - frameSizeSmall) / 2, frameSizeSmall) =
            Eigen::ArrayXf::Map(window.data(), frameSizeSmall, Eigen::InnerStride<>(frameSize / frameSizeSmall));
        windowSmall *= sqrtPower / std::sqrt(windowSmall.abs2().sum());
        filterbank1.setWindow(windowSmall);

        // half the frame size and set window to every 4th value of full window
        frameSizeSmall = FFTConfiguration::getValidFFTSize(frameSize / 4);
        windowSmall = Eigen::ArrayXf::Zero(frameSize);
        windowSmall.segment((frameSize - frameSizeSmall) / 2, frameSizeSmall) =
            Eigen::ArrayXf::Map(window.data(), frameSizeSmall, Eigen::InnerStride<>(frameSize / frameSizeSmall));
        windowSmall *= sqrtPower / std::sqrt(windowSmall.abs2().sum());
        filterbank2.setWindow(windowSmall);

        // half the frame size and set window to every 8th value of full window
        frameSizeSmall = FFTConfiguration::getValidFFTSize(frameSize / 8);
        windowSmall = Eigen::ArrayXf::Zero(frameSize);
        windowSmall.segment((frameSize - frameSizeSmall) / 2, frameSizeSmall) =
            Eigen::ArrayXf::Map(window.data(), frameSizeSmall, Eigen::InnerStride<>(frameSize / frameSizeSmall));
        windowSmall *= sqrtPower / std::sqrt(windowSmall.abs2().sum());
        filterbank3.setWindow(windowSmall);

        filterbankOut.resize(c.nBands);
    }

    FilterbankAnalysisSingleChannel filterbank0;
    FilterbankAnalysisSingleChannel filterbank1;
    FilterbankAnalysisSingleChannel filterbank2;
    FilterbankAnalysisSingleChannel filterbank3;
    DEFINE_MEMBER_ALGORITHMS(filterbank0, filterbank1, filterbank2, filterbank3)

  private:
    FilterbankAnalysisWOLA::Coefficients initializeFilterbanks(const Coefficients &c)
    {
        auto cFilterbank = FilterbankAnalysisWOLA::Coefficients();
        cFilterbank.nChannels = 1;
        cFilterbank.bufferSize = c.bufferSize / 8;
        cFilterbank.nBands = c.nBands;
        if (c.algorithmType == Coefficients::ADAPTIVE_HANN_8)
        {
            cFilterbank.nFolds = 1; // sets correct window size, but values are overwritten in constructor
        }
        else if (c.algorithmType == Coefficients::ADAPTIVE_WOLA_8)
        {
            cFilterbank.nFolds = 2; // sets correct window size, but values are overwritten in constructor
        }
        return cFilterbank;
    }

    void inline processAlgorithm(Input input, Output output)
    {
        for (auto frame = 0; frame < 8; frame++)
        {
            filterbank0.process(input.segment(frame * bufferSizeSmall, bufferSizeSmall), filterbankOut);
            output.col(frame) = filterbankOut.abs2();

            filterbank1.process(input.segment(frame * bufferSizeSmall, bufferSizeSmall), filterbankOut);
            output.col(frame) = output.col(frame).min(filterbankOut.abs2());

            filterbank2.process(input.segment(frame * bufferSizeSmall, bufferSizeSmall), filterbankOut);
            output.col(frame) = output.col(frame).min(filterbankOut.abs2());

            filterbank3.process(input.segment(frame * bufferSizeSmall, bufferSizeSmall), filterbankOut);
            output.col(frame) = output.col(frame).min(filterbankOut.abs2());
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = filterbankOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXcf filterbankOut;
    int bufferSizeSmall;

    friend BaseAlgorithm;
};