#pragma once
#include "algorithm_library/interface/interface.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// author: Kristian Timm Andersen

// Convert a gain spectrogram with nBands rows and 2^(timeOversampling - 1) frames into
// timeOversampling resolutions. Level i has 2^i frames and (nBands - 1) / 2^i + 1 bands:
// take the minimum gain over each group of 2^i frequency bins and 2^(timeOversampling - 1 - i)
// consecutive frames. Keep the Nyquist bin separate and reduce it only across time.
// Taking the minimum preserves the strongest attenuation within each group.
struct DecimateGainConfiguration
{
    using Input = I::Real2D;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int nBands = 2049;         // number of frequency bands in the gain spectrogram
        int timeOversampling = 4; // number of resolutions; input has 2^(timeOversampling - 1) frames
        DEFINE_TUNABLE_COEFFICIENTS(nBands, timeOversampling)
    };

    struct Parameters
    { DEFINE_NO_TUNABLE_PARAMETERS };

    static Eigen::ArrayXXf initInput(const Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(c.nBands, positivePow2(c.timeOversampling - 1)).abs(); // gain between 0 and 1
    }

    static std::vector<Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXf> output(c.timeOversampling);
        for (auto i = 0; i < c.timeOversampling; i++)
        {
            int nFrames = positivePow2(i);
            int nBands = (c.nBands - 1) / nFrames + 1;
            output[i] = Eigen::ArrayXXf::Zero(nBands, nFrames);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c)
    {
        return input.allFinite() && (input.rows() == c.nBands) && (input.cols() == positivePow2(c.timeOversampling - 1)) && (input >= 0.f).all() &&
               (input <= 1.f).all();
    }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != c.timeOversampling) { return false; }
        for (auto i = 0; i < c.timeOversampling; i++)
        {
            int nFrames = positivePow2(i);
            int nBands = (c.nBands - 1) / nFrames + 1;
            if ((output[i].rows() != nBands) || (output[i].cols() != nFrames) || !output[i].allFinite() || (output[i] < 0.f).any() || (output[i] > 1.f).any())
            {
                return false;
            }
        };
        return true;
    }
};

class DecimateGain : public AlgorithmImplementation<DecimateGainConfiguration, DecimateGain>
{
  public:
    DecimateGain(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        assert(c.timeOversampling > 0 && c.timeOversampling < 31);
        assert(c.nBands > 1 && (c.nBands - 1) % positivePow2(c.timeOversampling - 1) == 0);
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        const int inputFrames = positivePow2(C.timeOversampling - 1);
        for (int level = 0; level < C.timeOversampling; ++level)
        {
            const int frequencyFactor = positivePow2(level);
            const int timeFactor = inputFrames / frequencyFactor;
            const int nBands = (C.nBands - 1) / frequencyFactor;
            for (int frame = 0; frame < frequencyFactor; ++frame)
            {
                for (int band = 0; band < nBands; ++band)
                {
                    output[level](band, frame) = input.block(band * frequencyFactor, frame * timeFactor, frequencyFactor, timeFactor).minCoeff();
                }
                // Nyquist remains a separate band at every resolution.
                output[level](nBands, frame) = input.block(C.nBands - 1, frame * timeFactor, 1, timeFactor).minCoeff();
            }
        }
    }

    friend BaseAlgorithm;
};
