#pragma once
#include "algorithm_library/spectrogram.h"
#include "filterbank_set/filterbank_set_wola.h"
#include "framework/framework.h"
#include "spectrogram/upscale2d_linear.h"
#include "utilities/fastonebigheader.h"

// Spectrogram implemented as a combination of a set of filterbanks spectrograms. The criteria used for selecting the best time/frequency bin is the minimum power.
//
// author: Kristian Timm Andersen
class SpectrogramSet : public AlgorithmImplementation<SpectrogramConfiguration, SpectrogramSet>
{
  public:
    SpectrogramSet(Coefficients c = {.bufferSize = 1024, .nBands = 1025, .algorithmType = Coefficients::ADAPTIVE_HANN_8})
        : BaseAlgorithm{c}, filterbankSet(initializeFilterbankSet(c))
    {
        assert((c.algorithmType == Coefficients::ADAPTIVE_HANN_8) || (c.algorithmType == Coefficients::ADAPTIVE_WOLA_8));

        float windowPower = filterbankSet.filterbanks[0].getWindow().abs2().sum();
        for (auto iFilterbank = 1; iFilterbank < nFilterbanks; iFilterbank++)
        {
            Eigen::ArrayXf window = filterbankSet.filterbanks[iFilterbank].getWindow();
            window *= std::sqrt(windowPower / window.abs2().sum());
            filterbankSet.filterbanks[iFilterbank].setWindow(window);
        }

        Eigen::ArrayXf inputFrame(c.bufferSize);
        filterbankOut = filterbankSet.initOutput(inputFrame);
        spectrogramRaw.resize(filterbankOut.size());
        for (auto i = 0; i < static_cast<int>(filterbankOut.size()); i++)
        {
            spectrogramRaw[i] = Eigen::ArrayXXf::Zero(filterbankOut[i].rows(), positivePow2(i + 1)); // +1 to keep the last previous frame
        }
        spectrogramUpscaled = Eigen::ArrayXXf::Zero(c.nBands, nOutputFrames);

        auto cUpscale = upscale.getCoefficients();
        cUpscale.resize(nFilterbanks);
        for (auto i = 0; i < nFilterbanks; i++)
        {
            cUpscale[i].factorHorizontal = positivePow2(nFilterbanks - 1 - i);
            cUpscale[i].factorVertical = positivePow2(i);
            cUpscale[i].leftBoundaryExcluded = true;
        }
        upscale.setCoefficients(cUpscale);
    }

    FilterbankSetAnalysisWOLA filterbankSet;
    VectorAlgo<Upscale2DLinear> upscale;
    DEFINE_MEMBER_ALGORITHMS(filterbankSet, upscale)

  private:
    FilterbankSetAnalysisWOLA::Coefficients initializeFilterbankSet(const Coefficients &c)
    {
        FilterbankSetAnalysisWOLA::Coefficients cFilterbank;
        cFilterbank.bufferSize = c.bufferSize;
        cFilterbank.nBands = c.nBands;
        cFilterbank.nFilterbanks = nFilterbanks;
        cFilterbank.filterbankType =
            (c.algorithmType == Coefficients::ADAPTIVE_WOLA_8) ? FilterbankSetAnalysisWOLA::Coefficients::WOLA : FilterbankSetAnalysisWOLA::Coefficients::HANN;
        return cFilterbank;
    }

    void inline processAlgorithm(Input input, Output output)
    {
        filterbankSet.process(input, filterbankOut);

        spectrogramRaw[0].col(0) = spectrogramRaw[0].col(1); // copy prevous frame
        spectrogramRaw[0].col(1) = filterbankOut[0].abs2();
        upscale[0].process(spectrogramRaw[0], output);
        for (auto iFB = 1; iFB < static_cast<int>(filterbankOut.size()); iFB++)
        {
            spectrogramRaw[iFB].leftCols(1 << iFB) = spectrogramRaw[iFB].rightCols(1 << iFB); // copy prevous frames
            spectrogramRaw[iFB].rightCols(filterbankOut[iFB].cols()) = filterbankOut[iFB].abs2();
            upscale[iFB].process(spectrogramRaw[iFB].leftCols(filterbankOut[iFB].cols() + 1), spectrogramUpscaled);
            output = output.min(spectrogramUpscaled);
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = 0;
        for (auto i = 0; i < static_cast<int>(filterbankOut.size()); i++)
        {
            size += filterbankOut[i].getDynamicMemorySize();
            size += spectrogramRaw[i].getDynamicMemorySize();
        }
        size += spectrogramUpscaled.getDynamicMemorySize();
        return size;
    }

    static constexpr int nFilterbanks = 4;                               // 4 filterbanks that each halves the frame size resulting in 2^(4-1) = 8 output frames
    static constexpr int nOutputFrames = positivePow2(nFilterbanks - 1); // 8 output frames corresponding to the 4 filterbanks
    std::vector<Eigen::ArrayXXcf> filterbankOut;
    std::vector<Eigen::ArrayXXf> spectrogramRaw;
    Eigen::ArrayXXf spectrogramUpscaled;

    friend BaseAlgorithm;
};
