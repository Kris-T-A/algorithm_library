#pragma once
#include "interface/interface.h"

// Design minimum phase IIR filter from a magnitude spectrum. The IIR filter is factorized in a cascade of 2nd order sections.
//
// author: Kristian Timm Andersen

struct DesignIIRMinPhaseConfiguration
{
    using Input = I::Real;

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
        float sampleRate = 16000;
        int nOrder = 8; // filter order. High orders are not numerically stable when calculating roots (about >32)
        enum WeightTypes { LINEAR, MELSCALE };
        WeightTypes weightType = MELSCALE;
        DEFINE_TUNABLE_ENUM(WeightTypes, {{LINEAR, "Linear"}, {MELSCALE, "MelScale"}})
        DEFINE_TUNABLE_COEFFICIENTS(nBands, sampleRate, nOrder, weightType)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static int getNSos(int nOrder) { return static_cast<int>(std::ceil(static_cast<float>(nOrder) / 2)); } // get number of second-order sections from filter order

    static Eigen::ArrayXf initInput(const Coefficients &c) { return Eigen::ArrayXf::Random(c.nBands).abs2(); } // magnitude spectrum

    static std::tuple<Eigen::Array<float, 6, Eigen::Dynamic>, float> initOutput(Input input, const Coefficients &c) // second-order sections and gain
    {
        Eigen::ArrayXXf sos = Eigen::ArrayXXf::Zero(6, getNSos(c.nOrder));
        sos.row(0).setOnes();
        sos.row(3).setOnes();
        return std::make_tuple(sos, 1.f);
    }

    static bool validInput(Input input, const Coefficients &c) { return (input.rows() == c.nBands) && (input >= 0).all(); }

    static bool validOutput(Output output, const Coefficients &c)
    {
        bool flag = (output.sos.cols() == getNSos(c.nOrder)) && (output.sos.rows() == 6) && output.sos.allFinite();
        flag &= (output.sos.row(3) == 1.f).all();
        flag &= output.gain >= 0;
        return flag;
    }
};

// Fit a minimum-phase IIR filter to a magnitude spectrum and return its gain
// and cascade of second-order sections.
class DesignIIRMinPhase : public Algorithm<DesignIIRMinPhaseConfiguration>
{
  public:
    DesignIIRMinPhase() = default;
    DesignIIRMinPhase(const Coefficients &c);

    int getNSos() const { return Configuration::getNSos(getCoefficients().nOrder); }
};
