#include "design_iir_min_phase/design_iir_min_phase_tf2sos.h"
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

    ArrayXf input(c.nBands);
    input << 1.4560f, 2.2489f, 0.9559f, 5.0838f, 1.2221f, 2.3314f, 1.9187f, 1.5807f, 1.9052f, 1.8399f, 6.1568f, 1.3541f, 0.5104f, 1.9502f, 1.5755f, 4.1232f, 5.1997f;
    ArrayXXf sos(6, filterDesigner.getNSos());
    float gain;
    filterDesigner.process(input, {sos, gain});
    ArrayXXf sosRef(6, filterDesigner.getNSos());
    sosRef << 1.f, 1.f, 1.f, 1.04831f, -1.34835f, 0.345899f, 0.842527f, 0.751896f, 0.f, 1.f, 1.f, 1.f, 0.759416f, -1.50516f, 0.850303f, 0.870153f, 0.884276f, 0.f;
    float errorSOS = (sos - sosRef).abs2().sum() / sosRef.abs2().sum();
    fmt::print("SOS relative error: {}\n", errorSOS);
    float errorGain = std::fabs(gain - 1.89722f);
    errorGain = errorGain * errorGain / (1.89722f * 1.89722f);
    fmt::print("gain relative error: {}\n", errorGain);

    EXPECT_LT(errorSOS, 1e-10f);
    EXPECT_LT(errorGain, 1e-10f);
}