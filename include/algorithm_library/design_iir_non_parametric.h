#pragma once
#include "interface/interface.h"

// Design minimum phase IIR filter from an array of frequency and gain pairs. The IIR filter is factorized in a cascade of 2nd order sections.
//
// author: Kristian Timm Andersen

struct DesignIIRNonParametricConfiguration
{
    struct Input
    {
        I::Real frequencies;
        I::Real gaindB;
    };

    // sos is structured as:
    // [b0 bb0 ... ]
    // [b1 bb1 ... ]
    // [b2 bb2 ... ]
    // [1  1   ... ]
    // [a1 aa1 ... ]
    // [a2 aa2 ... ]
    struct Output
    {
        O::Real6X sos;
        O::Float gain;
    };

    struct Coefficients
    {
        int nBands = 257; // nBands should be significantly higher than the filter order: nOrder
        int nGains = 5;   // number of frequency/gain pairs. High orders are not numerically stable when calculating roots (about >64)
        float sampleRate = 16000;
        DEFINE_TUNABLE_COEFFICIENTS(nBands, nGains, sampleRate)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXf, Eigen::ArrayXf> initInput(const Coefficients &c) // frequency and gain pairs
    {
        Eigen::ArrayXf frequencies = Eigen::ArrayXf::LinSpaced(c.nGains, std::log(0.005f * c.sampleRate), std::log(0.25f * c.sampleRate)).exp(); // logarithmic spacing
        Eigen::ArrayXf gaindB = Eigen::ArrayXf::Random(c.nGains) * 20;                                                                           // random gain
        return std::make_tuple(frequencies, gaindB);
    }

    static std::tuple<Eigen::Array<float, 6, Eigen::Dynamic>, float> initOutput(Input input, const Coefficients &c) // second-order sections and gain
    {
        Eigen::ArrayXXf sos = Eigen::ArrayXXf::Zero(6, c.nGains);
        sos.row(0).setOnes();
        sos.row(3).setOnes();
        return std::make_tuple(sos, 1.f);
    }

    static bool validInput(Input input, const Coefficients &c)
    {
        bool flag = (input.frequencies.rows() == c.nGains) && (input.frequencies.rows() >= 2) &&
                    (input.frequencies > 0.f).all(); // number of frequency/gain pairs must be at least 2 to allow interpolation
        flag &= (input.gaindB.rows() == c.nGains) && input.gaindB.allFinite();
        return flag;
    }

    static bool validOutput(Output output, const Coefficients &c)
    {
        bool flag = (output.sos.cols() == c.nGains) && (output.sos.rows() == 6) && output.sos.allFinite();
        flag &= (output.sos.row(3) == 1.f).all();
        flag &= output.gain >= 0;
        return flag;
    }
};

// Interpolate frequency/gain control points and design a minimum-phase IIR filter
// represented by a gain and second-order sections.
class DesignIIRNonParametric : public Algorithm<DesignIIRNonParametricConfiguration>
{
  public:
    DesignIIRNonParametric() = default;
    DesignIIRNonParametric(const Coefficients &c);
};
