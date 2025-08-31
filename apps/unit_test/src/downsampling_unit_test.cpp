#include "downsampling/downsampling_2nd_dim.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(Downsampling, Interface2ndDim) 
{ 
    bool dummy = true;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<DownSampling2ndDim>(dummy)); 
}
