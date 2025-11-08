#pragma once
#include "algorithm_library/convert_rgba.h"
#include "framework/framework.h"

// Convert a matrix of normalized values (0-1) to RGBA colors
//
// author: Kristian Timm Andersen

class ConvertRGBAOcean : public AlgorithmImplementation<ConvertRGBAConfiguration, ConvertRGBAOcean>
{
  public:
    ConvertRGBAOcean(Coefficients c = {.colorScale = Coefficients::OCEAN}) : BaseAlgorithm{c} { assert(c.colorScale == Coefficients::OCEAN); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto col = 0; col < input.cols(); col++)
        {
            for (auto row = 0, rowOutput = 0; row < input.rows(); row++, rowOutput += 4)
            {
                scale(input(row, col), &output(rowOutput, col));
            }
        }
    }

    void inline scale(float value, uint8_t *output) const
    {
        if (value < 0.125f) // Deep blue to blue (0.0 - 0.125)
        {
            value *= 8.f;
            output[0] = static_cast<uint8_t>(value * 15.f + 24.f);  // R
            output[1] = static_cast<uint8_t>(value * 55.f + 15.f);  // G
            output[2] = static_cast<uint8_t>(value * 56.f + 122.f); // B
        }
        else if (value < 0.375f) // Blue to teal (0.125 - 0.375)
        {
            value = (value - 0.125f) * 4.f;
            output[0] = static_cast<uint8_t>(value * 21.f + 39.f);   // R
            output[1] = static_cast<uint8_t>(value * 92.f + 70.f);   // G
            output[2] = static_cast<uint8_t>(-value * 56.f + 122.f); // B
        }
        else if (value < 0.625f) // Teal to Green (0.375 - 0.625)
        {
            value = (value - 0.375f) * 4.f;
            output[0] = static_cast<uint8_t>(value * 100.f + 60.f);  // R
            output[1] = static_cast<uint8_t>(value * 92.f + 70.f);   // G
            output[2] = static_cast<uint8_t>(-value * 42.f + 178.f); // B
        }
        else if (value < 0.875f) // Green to Yellow (0.625 - 0.875)
        {
            value = (value - 0.625f) * 4.f;
            output[0] = static_cast<uint8_t>(value * 70.f + 160.f); // R
            output[1] = static_cast<uint8_t>(value * 36.f + 191.f); // G
            output[2] = static_cast<uint8_t>(-value * 8.f + 28.f);  // B
        }
        else // Yellow to bright Yellow (0.875 - 1.0)
        {
            value = (value - 0.875f) * 8.f;
            output[0] = static_cast<uint8_t>(value * 25.f + 230.f); // R
            output[1] = static_cast<uint8_t>(value * 28.f + 227.f); // G
            output[2] = static_cast<uint8_t>(value * 20.f + 20.f);  // B
        }
        output[3] = C.alpha; // A
    }

    int rowOutput;

    friend BaseAlgorithm;
};

class ConvertRGBAParula : public AlgorithmImplementation<ConvertRGBAConfiguration, ConvertRGBAParula>
{
  public:
    ConvertRGBAParula(Coefficients c = {.colorScale = Coefficients::PARULA}) : BaseAlgorithm{c} { assert(c.colorScale == Coefficients::PARULA); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto col = 0; col < input.cols(); col++)
        {
            for (auto row = 0, rowOutput = 0; row < input.rows(); row++, rowOutput += 4)
            {
                scale(input(row, col), &output(rowOutput, col));
            }
        }
    }

    void inline scale(float value, uint8_t *output) const
    {
        if (value < 0.2f) // Deep blue/purple
        {
            float t = value * 5.0f;
            output[0] = static_cast<uint8_t>(53 + t * (15 - 53));    // R
            output[1] = static_cast<uint8_t>(42 + t * (92 - 42));    // G
            output[2] = static_cast<uint8_t>(135 + t * (221 - 135)); // B
        }
        else if (value < 0.4f) // Blue to teal
        {
            float t = (value - 0.2f) * 5.0f;
            output[0] = static_cast<uint8_t>(15 + t * (18 - 15));    // R
            output[1] = static_cast<uint8_t>(92 + t * (125 - 92));   // G
            output[2] = static_cast<uint8_t>(221 + t * (216 - 221)); // B
        }
        else if (value < 0.6f) // Teal to green
        {
            float t = (value - 0.4f) * 5.0f;
            output[0] = static_cast<uint8_t>(18 + t * (7 - 18));     // R
            output[1] = static_cast<uint8_t>(125 + t * (156 - 125)); // G
            output[2] = static_cast<uint8_t>(216 + t * (165 - 216)); // B
        }
        else if (value < 0.8f) // Green to yellow-green
        {
            float t = (value - 0.6f) * 5.0f;
            output[0] = static_cast<uint8_t>(7 + t * (21 - 7));      // R
            output[1] = static_cast<uint8_t>(156 + t * (177 - 156)); // G
            output[2] = static_cast<uint8_t>(165 + t * (131 - 165)); // B
        }
        else // Yellow-green to yellow
        {
            float t = (value - 0.8f) * 5.0f;
            output[0] = static_cast<uint8_t>(21 + t * (251 - 21));   // R
            output[1] = static_cast<uint8_t>(177 + t * (250 - 177)); // G
            output[2] = static_cast<uint8_t>(131 + t * (50 - 131));  // B
        }
        output[3] = C.alpha; // A
    }

    int rowOutput;

    friend BaseAlgorithm;
};

class ConvertRGBAViridis : public AlgorithmImplementation<ConvertRGBAConfiguration, ConvertRGBAViridis>
{
  public:
    ConvertRGBAViridis(Coefficients c = {.colorScale = Coefficients::VIRIDIS}) : BaseAlgorithm{c} { assert(c.colorScale == Coefficients::VIRIDIS); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto col = 0; col < input.cols(); col++)
        {
            for (auto row = 0, rowOutput = 0; row < input.rows(); row++, rowOutput += 4)
            {
                scale(input(row, col), &output(rowOutput, col));
            }
        }
    }

    void inline scale(float value, uint8_t *output) const
    {
        if (value < 0.25f) // Dark purple to blue (0.0 - 0.25)
        {
            float t = value * 4.0f;
            output[0] = static_cast<uint8_t>(68 + t * (59 - 68));  // R
            output[1] = static_cast<uint8_t>(1 + t * (82 - 1));    // G
            output[2] = static_cast<uint8_t>(84 + t * (139 - 84)); // B
        }
        else if (value < 0.5f) // Blue to teal (0.25 - 0.5)
        {
            float t = (value - 0.25f) * 4.0f;
            output[0] = static_cast<uint8_t>(59 + t * (33 - 59));    // R
            output[1] = static_cast<uint8_t>(82 + t * (145 - 82));   // G
            output[2] = static_cast<uint8_t>(139 + t * (140 - 139)); // B
        }
        else if (value < 0.75f) // Teal to green (0.5 - 0.75)
        {
            float t = (value - 0.5f) * 4.0f;
            output[0] = static_cast<uint8_t>(33 + t * (94 - 33));    // R
            output[1] = static_cast<uint8_t>(145 + t * (201 - 145)); // G
            output[2] = static_cast<uint8_t>(140 + t * (98 - 140));  // B
        }
        else // Green to yellow (0.75 - 1.0)
        {
            float t = (value - 0.75f) * 4.0f;
            output[0] = static_cast<uint8_t>(94 + t * (253 - 94));   // R
            output[1] = static_cast<uint8_t>(201 + t * (231 - 201)); // G
            output[2] = static_cast<uint8_t>(98 + t * (37 - 98));    // B
        }
        output[3] = C.alpha; // A
    }

    int rowOutput;

    friend BaseAlgorithm;
};

class ConvertRGBAPlasma : public AlgorithmImplementation<ConvertRGBAConfiguration, ConvertRGBAPlasma>
{
  public:
    ConvertRGBAPlasma(Coefficients c = {.colorScale = Coefficients::PLASMA}) : BaseAlgorithm{c} { assert(c.colorScale == Coefficients::PLASMA); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto col = 0; col < input.cols(); col++)
        {
            for (auto row = 0, rowOutput = 0; row < input.rows(); row++, rowOutput += 4)
            {
                scale(input(row, col), &output(rowOutput, col));
            }
        }
    }

    void inline scale(float value, uint8_t *output) const
    {
        if (value < 0.25f) // Dark purple to magenta (0.0 - 0.25)
        {
            float t = value * 4.0f;
            output[0] = static_cast<uint8_t>(13 + t * (126 - 13));   // R
            output[1] = static_cast<uint8_t>(8 + t * (3 - 8));       // G
            output[2] = static_cast<uint8_t>(135 + t * (168 - 135)); // B
        }
        else if (value < 0.5f) // Magenta to red-pink (0.25 - 0.5)
        {
            float t = (value - 0.25f) * 4.0f;
            output[0] = static_cast<uint8_t>(126 + t * (203 - 126)); // R
            output[1] = static_cast<uint8_t>(3 + t * (70 - 3));      // G
            output[2] = static_cast<uint8_t>(168 + t * (121 - 168)); // B
        }
        else if (value < 0.75f) // Red-pink to orange (0.5 - 0.75)
        {
            float t = (value - 0.5f) * 4.0f;
            output[0] = static_cast<uint8_t>(203 + t * (248 - 203)); // R
            output[1] = static_cast<uint8_t>(70 + t * (149 - 70));   // G
            output[2] = static_cast<uint8_t>(121 + t * (64 - 121));  // B
        }
        else // Orange to yellow (0.75 - 1.0)
        {
            float t = (value - 0.75f) * 4.0f;
            output[0] = static_cast<uint8_t>(248 + t * (240 - 248)); // R
            output[1] = static_cast<uint8_t>(149 + t * (249 - 149)); // G
            output[2] = static_cast<uint8_t>(64 + t * (33 - 64));    // B
        }
        output[3] = C.alpha; // A
    }

    int rowOutput;

    friend BaseAlgorithm;
};

class ConvertRGBAMagma : public AlgorithmImplementation<ConvertRGBAConfiguration, ConvertRGBAMagma>
{
  public:
    ConvertRGBAMagma(Coefficients c = {.colorScale = Coefficients::MAGMA}) : BaseAlgorithm{c} { assert(c.colorScale == Coefficients::MAGMA); }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto col = 0; col < input.cols(); col++)
        {
            for (auto row = 0, rowOutput = 0; row < input.rows(); row++, rowOutput += 4)
            {
                scale(input(row, col), &output(rowOutput, col));
            }
        }
    }

    void inline scale(float value, uint8_t *output) const
    {
        if (value < 0.25f) // Black to dark purple (0.0 - 0.25)
        {
            float t = value * 4.0f;
            output[0] = static_cast<uint8_t>(0 + t * (87 - 0));  // R
            output[1] = static_cast<uint8_t>(0 + t * (15 - 0));  // G
            output[2] = static_cast<uint8_t>(4 + t * (109 - 4)); // B
        }
        else if (value < 0.5f) // Dark purple to pink (0.25 - 0.5)
        {
            float t = (value - 0.25f) * 4.0f;
            output[0] = static_cast<uint8_t>(87 + t * (180 - 87));   // R
            output[1] = static_cast<uint8_t>(15 + t * (54 - 15));    // G
            output[2] = static_cast<uint8_t>(109 + t * (122 - 109)); // B
        }
        else if (value < 0.75f) // Pink to orange (0.5 - 0.75)
        {
            float t = (value - 0.5f) * 4.0f;
            output[0] = static_cast<uint8_t>(180 + t * (251 - 180)); // R
            output[1] = static_cast<uint8_t>(54 + t * (136 - 54));   // G
            output[2] = static_cast<uint8_t>(122 + t * (97 - 122));  // B
        }
        else // Orange to white/pale yellow (0.75 - 1.0)
        {
            float t = (value - 0.75f) * 4.0f;
            output[0] = static_cast<uint8_t>(251 + t * (252 - 251)); // R
            output[1] = static_cast<uint8_t>(136 + t * (253 - 136)); // G
            output[2] = static_cast<uint8_t>(97 + t * (191 - 97));   // B
        }
        output[3] = C.alpha; // A
    }

    int rowOutput;

    friend BaseAlgorithm;
};
