#include "pyplot_cpp/pyplot_cpp.h"
#include "fmt/ranges.h"
#include "gtest/gtest.h"

using namespace Pyplotcpp;
using namespace Eigen;

// --------------------------------------------- TEST FIXTURE ---------------------------------------------

class PlotTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        clear(); // clear figure before each test
    }
    void TearDown() override
    {
        clear(); // clear figure after each test
    }
};

// --------------------------------------------- TEST CASES ---------------------------------------------

// Note that first test will always take a longer time to run due to the initialization of the Python interpreter.

TEST_F(PlotTest, Plot) 
{ 
    ArrayXXf x(10, 10);
    x.setRandom();
    plot(x);
}

TEST_F(PlotTest, Plot2Inputs)
{
    ArrayXXf x(10,3);
    x.setRandom();
    ArrayXXf y(10, 3);
    y.setRandom();
    plot(x, y);
}

TEST_F(PlotTest, Imagesc)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    imagesc(x);
}

TEST_F(PlotTest, ImagescWithScaling)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    imagesc(x, {0.0f, 1.0f});
}


TEST_F(PlotTest, PlotSettings)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    imagesc(x);

    title("Test Title");
    xlim(0.0f, 1.0f);
    ylim(0.0f, 1.0f);
    xlabel("X Axis");
    ylabel("Y Axis");
}

TEST_F(PlotTest, Colorbar)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    Figure* fig = imagesc(x);
    
    colorbar(fig);
}

TEST_F(PlotTest, Save)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    imagesc(x);

    save("test_plot.png");
}

TEST_F(PlotTest, FigureAndClose)
{
    ArrayXXf x(10, 10);
    x.setRandom();
    imagesc(x);

    figure(3);
    x.setOnes();
    imagesc(x);
    figure(1);
    close();
}