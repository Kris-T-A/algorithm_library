#include "convert_rgba/convert_rgba_variations.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(ConvertRGBA, InterfaceOcean) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<ConvertRGBAOcean>()); }

TEST(ConvertRGBA, InterfaceParula) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<ConvertRGBAParula>()); }

TEST(ConvertRGBA, InterfaceViridis) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<ConvertRGBAViridis>()); }

TEST(ConvertRGBA, InterfaceMagma) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<ConvertRGBAMagma>()); }

TEST(ConvertRGBA, InterfacePlasma) { EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<ConvertRGBAPlasma>()); }