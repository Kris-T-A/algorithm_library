#pragma once
#include "algorithm_library/critical_bands.h"
#include "framework/framework.h"

// Common CriticalBandsBark class that following classes inherit from and specialize
class CriticalBandsBark
{
  public:
    CriticalBandsBark(CriticalBandsConfiguration::Coefficients c = CriticalBandsConfiguration::Coefficients())
    {
        // calculate number of critical bands
        nBandsCritical = CriticalBandsConfiguration::getNCriticalBands(c.sampleRate);

        // convert Frequency corners to FFT indices
        auto nFFT = (c.nBands - 1) * 2; // number of FFT points
        Eigen::ArrayXi frequencyCornersIndex = (CriticalBandsConfiguration::getCornerFrequencies(c.sampleRate) * nFFT / c.sampleRate).round().cast<int>();
        indexStart = frequencyCornersIndex.head(nBandsCritical);
        nBandsSum = (frequencyCornersIndex.tail(nBandsCritical) - indexStart).cwiseMax(1);
        indexEnd = indexStart(nBandsCritical - 1) + nBandsSum(nBandsCritical - 1);
    }

    inline void inverse(I::Real2D xPower, O::Real2D yPower, const int nBands)
    {
        assert(xPower.rows() == nBandsCritical);

        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            yPower.block(0, channel, indexStart(0), 1).setConstant(xPower(0, channel));
            for (auto i = 0; i < nBandsCritical; i++)
            {
                yPower.block(indexStart(i), channel, nBandsSum(i), 1).setConstant(xPower(i, channel));
            }
            yPower.block(indexEnd, channel, nBands - indexEnd, 1).setConstant(xPower(nBandsCritical - 1, channel));
        }
    }

  protected:
    size_t getDynamicSizeVariables() const
    {
        size_t size = indexStart.getDynamicMemorySize();
        size += nBandsSum.getDynamicMemorySize();
        return size;
    }

    int nBandsCritical, indexEnd;
    Eigen::ArrayXi indexStart;
    Eigen::ArrayXi nBandsSum;
};

// --------------------------------------------------- CriticalBandsBarkSum ----------------------------------------------------------------

// Sum FFT-bin powers within each Bark critical band independently for every channel.
class CriticalBandsBarkSum : public CriticalBandsBark, public AlgorithmImplementation<CriticalBandsSumConfiguration, CriticalBandsBarkSum>
{
  public:
    CriticalBandsBarkSum(Coefficients c = Coefficients()) : CriticalBandsBark(c), BaseAlgorithm{c} {}

    inline void inverse(I::Real2D xPower, O::Real2D yPower) { CriticalBandsBark::inverse(xPower, yPower, C.nBands); }

  private:
    inline void processAlgorithm(Input xPower, Output yPower)
    {
        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            for (auto i = 0; i < nBandsCritical; i++)
            {
                // sum in a critical band
                yPower(i, channel) = xPower.block(indexStart(i), channel, nBandsSum(i), 1).sum();
            }
        }
    }
    size_t getDynamicSizeVariables() const final { return CriticalBandsBark::getDynamicSizeVariables(); }

    friend BaseAlgorithm;
};

// --------------------------------------------------- CriticalBandsBarkMax ----------------------------------------------------------------

// Take the maximum FFT-bin power within each Bark critical band for every channel.
class CriticalBandsBarkMax : public CriticalBandsBark, public AlgorithmImplementation<CriticalBandsMaxConfiguration, CriticalBandsBarkMax>
{
  public:
    CriticalBandsBarkMax(Coefficients c = Coefficients()) : CriticalBandsBark(c), BaseAlgorithm{c} {}

    inline void inverse(I::Real2D xPower, O::Real2D yPower) { CriticalBandsBark::inverse(xPower, yPower, C.nBands); }

  private:
    inline void processAlgorithm(Input xPower, Output yPower)
    {
        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            for (auto i = 0; i < nBandsCritical; i++)
            {
                // find max in a critical band
                yPower(i, channel) = xPower.block(indexStart(i), channel, nBandsSum(i), 1).maxCoeff();
            }
        }
    }
    size_t getDynamicSizeVariables() const final { return CriticalBandsBark::getDynamicSizeVariables(); }

    friend BaseAlgorithm;
};

// --------------------------------------------------- CriticalBandsBarkMean ----------------------------------------------------------------

// Average FFT-bin powers within each Bark critical band independently for every channel.
class CriticalBandsBarkMean : public CriticalBandsBark, public AlgorithmImplementation<CriticalBandsMeanConfiguration, CriticalBandsBarkMean>
{
  public:
    CriticalBandsBarkMean(Coefficients c = Coefficients()) : CriticalBandsBark(c), BaseAlgorithm{c} {}

    inline void inverse(I::Real2D xPower, O::Real2D yPower) { CriticalBandsBark::inverse(xPower, yPower, C.nBands); }

  private:
    inline void processAlgorithm(Input xPower, Output yPower)
    {
        for (auto channel = 0; channel < xPower.cols(); channel++)
        {
            for (auto i = 0; i < nBandsCritical; i++)
            {
                // mean in a critical band
                yPower(i, channel) = xPower.block(indexStart(i), channel, nBandsSum(i), 1).mean();
            }
        }
    }

    size_t getDynamicSizeVariables() const final { return CriticalBandsBark::getDynamicSizeVariables(); }

    friend BaseAlgorithm;
};