#include "spectrogram_set/spectrogram_set_wola.h"
#include "spectrogram_set/spectrogram_set_zeropad.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(SpectrogramSet, Interface) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramSetWOLA>()); }

TEST(SpectrogramSet, InterfaceZeropad) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<SpectrogramSetZeropad>()); }