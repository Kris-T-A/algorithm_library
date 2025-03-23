#pragma once
#include "algorithm_library/delay.h"
#include "framework/framework.h"

// Circular buffer.
//
// author: Kristian Timm Andersen

class CircularBuffer : public AlgorithmImplementation<DelayConfiguration, CircularBuffer>
{
  public:
    CircularBuffer(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        buffer.resize(c.delayLength, c.nChannels);
        reset();
    }

    inline void push(Input input)
    {
        for (auto sample = 0; sample < input.rows(); sample++)
        {
            buffer.row(index) = input.row(sample);
            increment();
        }
    }

    inline void pop(Output output)
    {
        for (auto sample = 0; sample < output.rows(); sample++)
        {
            decrement();
            output.row(sample) = buffer.row(index);
        }
    }

    inline Eigen::ArrayXf get(const int indexGet) const
    {
        auto newIndex = index - indexGet - 1;
        if (newIndex < 0) { newIndex += C.delayLength; }
        return buffer.row(newIndex);
    }

    inline Eigen::ArrayXf get(const float indexGet) const
    {
        auto newIndex = index - indexGet - 1;
        if (newIndex < 0.f) { newIndex += C.delayLength; }
        auto intIndex = static_cast<int>(newIndex);
        const auto remainder = newIndex - intIndex;
        Eigen::ArrayXf value = buffer.row(intIndex).transpose();
        intIndex++;
        if (intIndex == C.delayLength) { intIndex = 0; }
        value += remainder * (buffer.row(intIndex).transpose() - value);
        return value;
    }

  private:
    // push input values into buffer and write output values from buffer
    inline void processAlgorithm(Input input, Output output)
    {
        if (input.rows() < C.delayLength)
        {
            const int bRows = std::min(static_cast<int>(input.rows()), C.delayLength - index);
            const int tRows = std::max(0, static_cast<int>(input.rows()) - C.delayLength + index);
            output.topRows(bRows) = buffer.middleRows(index, bRows);
            output.bottomRows(tRows) = buffer.topRows(tRows);
            buffer.middleRows(index, bRows) = input.topRows(bRows);
            buffer.topRows(tRows) = input.bottomRows(tRows);
            index = tRows > 0 ? tRows : index + bRows;
        }
        else
        {
            const int bRows = C.delayLength - index;
            const int tRows = static_cast<int>(input.rows()) - C.delayLength;
            output.topRows(bRows) = buffer.bottomRows(bRows);
            output.middleRows(bRows, index) = buffer.topRows(index);
            output.bottomRows(tRows) = input.topRows(tRows);
            buffer = input.bottomRows(C.delayLength);
            index = 0;
        }
    }

    inline void increment()
    {
        index++;
        if (index >= C.delayLength) { index = 0; }
    }
    inline void increment(const int increment)
    {
        index += increment;
        while (index >= C.delayLength)
        {
            index -= C.delayLength;
        }
    }
    inline void decrement()
    {
        index--;
        if (index < 0) { index = C.delayLength - 1; }
    }
    inline void decrement(const int decrement)
    {
        index -= decrement;
        while (index < 0)
        {
            index += C.delayLength;
        }
    }

    void resetVariables() final
    {
        buffer.setZero();
        index = 0;
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = buffer.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXXf buffer;
    int index;

    friend BaseAlgorithm;
};
