#pragma once
#include "algorithm_library/interpolation.h"
#include "framework/framework.h"

// 4 point, 3rd order Hermite interpolation based on :
// http ://yehar.com/blog/wp-content/uploads/2009/08/deip.pdf
//
// Input is a size 4 array and FractionalDelay is
// in the interval[0, 1[ between Input[1] and Input[2].
//
// author: Kristian Timm Andersen

// ----------------------------- Cubic interpolation of one sample -----------------------------------------

class InterpolationCubicSample : public AlgorithmImplementation<InterpolationSampleConfiguration, InterpolationCubicSample>
{
  public:
    InterpolationCubicSample(Coefficients c = Coefficients()) : BaseAlgorithm{c} {}

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        const auto c0 = input.samples(1);
        const auto c1 = .5f * (input.samples(2) - input.samples(0));
        const auto c2 = input.samples(0) - 2.5f * input.samples(1) + 2.f * input.samples(2) - .5f * input.samples(3);
        const auto c3 = .5f * (input.samples(3) - input.samples(0)) + 1.5f * (input.samples(1) - input.samples(2));
        output = ((c3 * input.fractionalDelay + c2) * input.fractionalDelay + c1) * input.fractionalDelay + c0;
        assert(input.fractionalDelay >= 0.f);
        assert(input.fractionalDelay <= 1.f); // ASSERT: fractionalDelay must be between 0 and 1
    }

    friend BaseAlgorithm;
};

// ----------------------------- Cubic interpolation of array ------------------------------------------------------

// Evaluate four-point cubic Hermite interpolation at each requested fractional
// index of an input sample array.
class InterpolationCubic : public AlgorithmImplementation<InterpolationConfiguration, InterpolationCubic>
{
  public:
    InterpolationCubic(Coefficients c = Coefficients()) : BaseAlgorithm{c} {}

    InterpolationCubicSample interpolation; // this class uses default constructor
    DEFINE_MEMBER_ALGORITHMS(interpolation)

  private:
    inline void processAlgorithm(Input x, Output y)
    {
        for (auto i = 0; i < x.fractionalIndices.size(); i++)
        {
            auto index = static_cast<int>(x.fractionalIndices(i));  // truncate index to interpolate
            float fractionalIndex = x.fractionalIndices(i) - index; // fractional part of index (between 0 and 1)
            assert(index > 0);
            assert(index < x.samples.size() - 1);                                      // ASSERT: indices must be between 1.0 and N-2.0
            Eigen::Map<const Eigen::Array4f> samples(x.samples.data() + index - 1, 4); // map 4 samples starting from one sample before current index
            interpolation.process({samples, fractionalIndex}, y(i));
        }
    }

    friend BaseAlgorithm;
};

// ----------------------------- Cubic interpolation of array with constant fractionalDelay. This is faster ------------------------------------------------------

class InterpolationCubicConstant : public AlgorithmImplementation<InterpolationConstantConfiguration, InterpolationCubicConstant>
{
  public:
    InterpolationCubicConstant(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        const auto d2 = C.fractionalDelay * C.fractionalDelay;
        const auto dm1 = C.fractionalDelay - 1.f;
        const auto dm12 = dm1 * dm1;
        a0 = -dm12 * C.fractionalDelay * .5f;
        a1 = dm12 * (1.f + 2.f * C.fractionalDelay) - d2 * dm1 * .5f;
        a2 = dm12 * C.fractionalDelay * .5f + d2 * (3.f - 2.f * C.fractionalDelay);
        a3 = d2 * dm1 * .5f;
        assert(C.fractionalDelay <= 1);
        assert(C.fractionalDelay >= 0); // ASSERT: fractionalDelay must be between 0 and 1
    }

  private:
    inline void processAlgorithm(Input x, Output y)
    {
        const int N = static_cast<int>(x.size()) - 3; // number of output samples
        assert(N > 0);                                // ASSERT: must be at least 4 input samples to give 1 output sample
        y = a0 * x.head(N) + a1 * x.segment(1, N) + a2 * x.segment(2, N) + a3 * x.tail(N);
    }

    float a0, a1, a2, a3;

    friend BaseAlgorithm;
};
