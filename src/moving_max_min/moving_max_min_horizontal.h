#pragma once
#include "filter_min_max/filter_min_max_lemire.h"
#include "framework/framework.h"

struct MovingMaxMinHorizontalConfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    struct Coefficients
    {
        int filterLength = 5;
        int nChannels = 100;
        DEFINE_TUNABLE_COEFFICIENTS(filterLength, nChannels);
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(c.nChannels, 1000); } // arbitrary number of inputs

    static Eigen::ArrayXXf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXf::Zero(c.nChannels, input.cols()); }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nChannels) && (input.cols() > 0) && input.allFinite(); }

    static bool validOutput(Output output, const Coefficients &c) { return (output.rows() == c.nChannels) && (output.cols() > 0) && output.allFinite(); }
};

class MovingMaxMinHorizontal : public AlgorithmImplementation<MovingMaxMinHorizontalConfiguration, MovingMaxMinHorizontal>
{
  public:
    MovingMaxMinHorizontal(const Coefficients &c = Coefficients())
        : BaseAlgorithm{c},                                                         //
          streamingMax({.filterLength = c.filterLength, .nChannels = c.nChannels}), //
          streamingMin({.filterLength = c.filterLength, .nChannels = c.nChannels})
    {
        streamingMax.resetInitialValue(-std::numeric_limits<float>::infinity());
        streamingMin.resetInitialValue(std::numeric_limits<float>::infinity());
        maxOut.resize(1, c.nChannels);
    }

    StreamingMaxLemire streamingMax;
    StreamingMinLemire streamingMin;
    DEFINE_MEMBER_ALGORITHMS(streamingMax, streamingMin)

  private:
    void processAlgorithm(Input input, Output output)
    {
        for (auto sample = 0; sample < input.cols(); sample++)
        {
            streamingMax.process(Eigen::Map<const Eigen::ArrayXXf>(input.col(sample).data(), 1, C.nChannels), maxOut);
            streamingMin.process(maxOut, Eigen::Map<Eigen::ArrayXXf>(output.col(sample).data(), 1, C.nChannels));
        }
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = maxOut.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf maxOut;
    friend BaseAlgorithm;
};