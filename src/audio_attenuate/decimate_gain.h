#pragma once
#include "algorithm_library/interface/interface.h"
#include "framework/framework.h"
#include "utilities/fastonebigheader.h"

// author: Kristian Timm Andersen

struct DecimateGainConfiguration
{
    using Input = I::Real2D;
    using Output = O::VectorReal2D;

    struct Coefficients
    {
        int nBands = 2049; // number of frequency bands in the gain spectrogram
        DEFINE_TUNABLE_COEFFICIENTS(nBands)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(c.nBands, 8).abs(); // gain between 0 and 1
    }

    static std::vector<Eigen::ArrayXXf> initOutput(Input input, const Coefficients &c)
    {
        std::vector<Eigen::ArrayXXf> output(4);
        for (auto i = 0; i < 4; i++)
        {
            int nFrames = positivePow2(i);
            int nBands = (c.nBands - 1) / nFrames + 1;
            output[i] = Eigen::ArrayXXf::Zero(nBands, nFrames);
        }
        return output;
    }

    static bool validInput(Input input, const Coefficients &c)
    {
        return input.allFinite() && (input.rows() == c.nBands) && (input.cols() <= 8) && (input >= 0.f).all() && (input <= 1.f).all();
    }

    static bool validOutput(Output output, const Coefficients &c)
    {
        if (static_cast<int>(output.size()) != 4) { return false; }
        for (auto i = 0; i < 4; i++)
        {
            int nFrames = positivePow2(i);
            int nBands = (c.nBands - 1) / nFrames + 1;
            if ((output[i].rows() != nBands) || (output[i].cols() != nFrames) || (output[i] < 0.f).any() || (output[i] > 1.f).any()) { return false; }
        };
        return true;
    }
};

class DecimateGain : public AlgorithmImplementation<DecimateGainConfiguration, DecimateGain>
{
  public:
    DecimateGain(const Coefficients &c = Coefficients()) : BaseAlgorithm{c}
    {
        nFreq4 = (c.nBands - 1) / 4;
        memoryBuffer = Eigen::ArrayXXf::Zero(nFreq4, 8);
    }

  private:
    void processAlgorithm(Input input, Output output)
    {
        Eigen::Map<Eigen::ArrayXXf> memoryBuffer2(memoryBuffer.data(), C.nBands - 1, 2);
        memoryBuffer2.col(0) = input.block(0, 0, C.nBands - 1, 4).rowwise().minCoeff();
        memoryBuffer2.col(1) = input.block(0, 4, C.nBands - 1, 4).rowwise().minCoeff();
        output[0].col(0).head(C.nBands - 1) = memoryBuffer2.rowwise().minCoeff();

        // min along column length 2
        for (auto j = 0; j < 2; j++)
        {
            for (auto i = 0; i < (C.nBands - 1) / 2; i++)
            {
                output[1](i, j) = std::min(memoryBuffer2(2 * i, j), memoryBuffer2(2 * i + 1, j));
            }
        }

        // min along column length 4
        for (auto j = 0; j < 8; j++)
        {
            for (auto i = 0; i < nFreq4; i++)
            {
                memoryBuffer(i, j) = std::min(std::min(input(4 * i, j), input(4 * i + 1, j)), std::min(input(4 * i + 2, j), input(4 * i + 3, j)));
            }
        }

        for (auto i = 0; i < 4; i++)
        {
            output[2].col(i).head(nFreq4) = memoryBuffer.block(0, 2 * i, nFreq4, 2).rowwise().minCoeff();
        }

        // min along column length 2
        for (auto j = 0; j < 8; j++)
        {
            for (auto i = 0; i < nFreq4 / 2; i++)
            {
                output[3](i, j) = std::min(memoryBuffer(2 * i, j), memoryBuffer(2 * i + 1, j));
            }
        }

        output[3].row(nFreq4 / 2) = input.row(C.nBands - 1);
        for (auto i = 0; i < 4; i++)
        {
            output[2](nFreq4, i) = std::min(output[3](nFreq4 / 2, 2 * i), output[3](nFreq4 / 2, 2 * i + 1));
        }
        for (auto i = 0; i < 2; i++)
        {
            output[1](nFreq4 * 2, i) = std::min(output[2](nFreq4, 2 * i), output[2](nFreq4, 2 * i + 1));
        }
        output[0](C.nBands - 1, 0) = std::min(output[1](nFreq4 * 2, 0), output[1](nFreq4 * 2, 1));
    }

    size_t getDynamicSizeVariables() const final { return memoryBuffer.getDynamicMemorySize(); }

    Eigen::ArrayXXf memoryBuffer;
    int nFreq4;

    friend BaseAlgorithm;
};