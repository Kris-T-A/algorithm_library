#include "design_iir_min_phase/design_iir_min_phase_tf2sos.h"
#include "iir_filter/iir_filter_2nd_order.h" // used for calculating impulse response
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(DesignIIRMinPhase, Interface)
{
    bool testMallocFlag = false;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<DesignIIRMinPhaseTF2SOS>(testMallocFlag));
}

TEST(DesignIIRMinPhase, CheckCalculation)
{
    DesignIIRMinPhaseTF2SOS::Coefficients c;
    c.nOrder = 5;
    c.nBands = 17;
    c.weightType = c.LINEAR;
    DesignIIRMinPhaseTF2SOS filterDesigner(c);

    // magnitude spectrum for testing
    ArrayXf magnitudeSpectrum(c.nBands);
    magnitudeSpectrum << 1.4560f, 2.2489f, 0.9559f, 5.0838f, 1.2221f, 2.3314f, 1.9187f, 1.5807f, 1.9052f, 1.8399f, 6.1568f, 1.3541f, 0.5104f, 1.9502f, 1.5755f, 4.1232f,
        5.1997f;

    // process impulse through reference filter
    ArrayXXf sosRef(6, filterDesigner.getNSos());
    sosRef << 1.f, 1.f, 1.f, 1.04831f, -1.34835f, 0.345899f, 0.842527f, 0.751896f, 0.f, 1.f, 1.f, 1.f, 0.759416f, -1.50516f, 0.850303f, 0.870153f, 0.884276f, 0.f;
    float gainRef = 1.89722f;
    int nSamples = 64;
    ArrayXf impulse(nSamples);
    impulse.setZero();
    impulse(0) = 1.f; // impulse
    IIRFilterCascaded iirFilter({.nChannels = 1, .nSos = filterDesigner.getNSos()});
    ArrayXf outputRef = iirFilter.initOutput(impulse);
    iirFilter.setFilter(sosRef, gainRef);
    iirFilter.process(impulse, outputRef);

    // process impulse through designed filter and compare with reference
    ArrayXXf sos(6, filterDesigner.getNSos());
    float gain;
    filterDesigner.process(magnitudeSpectrum, {sos, gain});
    iirFilter.reset();
    iirFilter.setFilter(sos, gain);
    ArrayXf output = iirFilter.initOutput(impulse);
    iirFilter.process(impulse, output);

    // calculate errors
    float errorImpulse = (output - outputRef).abs2().sum() / outputRef.abs2().sum();
    fmt::print("Impulse response relative error: {}\n", errorImpulse);
    EXPECT_LT(errorImpulse, 1e-9f);
}