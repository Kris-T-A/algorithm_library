#pragma once
#include "interface/interface.h"
#include <iostream>
// Convert a matrix of normalized values (0-1) to RGBA colors using the Parula colormap
//
// author: Kristian Timm Andersen

struct ConvertRGBAConfiguration
{

    using Input = I::Real2D;
    using Output = O::U8Int2D;

    struct Coefficients
    {
        uint8_t alpha = 255;
        enum ColorScales { OCEAN, PARULA, VIRIDIS, MAGMA, PLASMA };
        ColorScales colorScale = PARULA;
        DEFINE_TUNABLE_ENUM(ColorScales, {{OCEAN, "Ocean"}, {PARULA, "Parula"}, {VIRIDIS, "Viridis"}, {MAGMA, "Magma"}, {PLASMA, "Plasma"}})
        DEFINE_TUNABLE_COEFFICIENTS(alpha, colorScale)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static Eigen::ArrayXXf initInput(const Coefficients &c) { return Eigen::ArrayXXf::Random(10, 12).abs2(); } // matrix of normalized values (0-1)

    static Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> initOutput(Input input, const Coefficients &c)
    {
        Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic> output(input.rows() * 4, input.cols());
        output.setZero();
        for (auto iCol = 0; iCol < input.cols(); iCol++)
        {
            Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<4>>(&output(3, iCol), input.rows()).setConstant(c.alpha);
        }

        return output;
    }

    static bool validInput(Input input, const Coefficients &c) { return (input >= 0.f).all() && (input <= 1.f).all(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        int nRows4 = output.rows() / 4;
        bool flag = nRows4 * 4 == output.rows(); // must be multiple of 4
        for (int iCol = 0; iCol < output.cols(); iCol++) // every 4th value must be equal to alpha
        {
            flag &= (Eigen::Map<Eigen::Array<uint8_t, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<4>>(&output(3,iCol), nRows4) == c.alpha)
                        .all();
        }
        return flag;
    }
};

class ConvertRGBA : public Algorithm<ConvertRGBAConfiguration>
{
  public:
    ConvertRGBA() = default;
    ConvertRGBA(const Coefficients &c);
};