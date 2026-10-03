#include "spectrogram_adaptive_set/spectrogram_adaptive_set_min.h"
#include "unit_test.h"
#include "gtest/gtest.h"

TEST(SpectrogramAdaptiveSet, InterfaceMin)
{
    bool dummy = true;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramAdaptiveSetMin>(dummy));
}
