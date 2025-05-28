#include "spectrogram/spectrogram_nonlinear.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(SpectrogramNonlinear, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramNonlinear>()); }
