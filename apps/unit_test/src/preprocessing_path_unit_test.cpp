#include "preprocessing_path/beamformer_path.h"
#include "unit_test.h"
#include "gtest/gtest.h"

using namespace Eigen;

// --------------------------------------------- TEST CASES ---------------------------------------------

TEST(PreprocessingPath, Interface)
{
    bool testMallocFlag = false;
    EXPECT_TRUE(InterfaceTests::algorithmInterfaceTest<BeamformerPath>(testMallocFlag));
}

TEST(PreprocessingPath, PublicInterface)
{
    EXPECT_TRUE(InterfaceTests::algorithmBufferInterfaceTest<PreprocessingPath>());
}

// test that bufferMode is correctly changed to asynchronous mode
TEST(PreprocessingPath, SetAsynchronousMode)
{
    PreprocessingPath prePath;
    prePath.setBufferMode(BufferMode::ASYNCHRONOUS_BUFFER);
    auto mode = prePath.getBufferMode();
    EXPECT_TRUE(mode == BufferMode::ASYNCHRONOUS_BUFFER);
}

// test that coefficients are saved when changing buffer mode
TEST(PreprocessingPath, CoefficientsSetBufferMode)
{
    PreprocessingPath prePath;

    auto c = prePath.getCoefficients();
    c.bufferSize = c.bufferSize * 2;
    c.nChannels = c.nChannels * 2;
    prePath.setCoefficients(c);

    prePath.setBufferMode(BufferMode::ASYNCHRONOUS_BUFFER);

    auto cNew = prePath.getCoefficients();
    auto mode = prePath.getBufferMode();

    EXPECT_TRUE(c.bufferSize == cNew.bufferSize);
    EXPECT_TRUE(c.nChannels == cNew.nChannels);
    EXPECT_TRUE(mode == BufferMode::ASYNCHRONOUS_BUFFER);
}

// test that anysize init/valid functions can be called
TEST(PreprocessingPath, AnySize)
{
    PreprocessingPath prePath;
    auto c = prePath.getCoefficients();

    // set buferSizeInput to be smaller than bufferSize
    int bufferSizeInput = static_cast<int>(c.bufferSize * .7f);

    prePath.setBufferMode(BufferMode::SYNCHRONOUS_BUFFER);
    auto input = prePath.initInputAnySize(bufferSizeInput);
    auto output = prePath.initOutputAnySize(input);
    prePath.processAnySize(input, output);
    EXPECT_TRUE(prePath.validInputAnySize(input));
    EXPECT_TRUE(prePath.validOutputAnySize(output, static_cast<int>(input.rows())));
    fmt::print("Synchronous processing with small bufferSize succesful.\n");

    prePath.setBufferMode(BufferMode::ASYNCHRONOUS_BUFFER);
    output = prePath.initOutputAnySize(input);
    prePath.processAnySize(input, output);
    EXPECT_TRUE(prePath.validInputAnySize(input));
    EXPECT_TRUE(prePath.validOutputAnySize(output, static_cast<int>(input.rows())));
    fmt::print("Asynchronous processing with small bufferSize succesful.\n");

    // set bufferSizeInput to be larger than bufferSize
    bufferSizeInput = static_cast<int>(c.bufferSize * 5.7f);

    prePath.setBufferMode(BufferMode::SYNCHRONOUS_BUFFER);
    input = prePath.initInputAnySize(bufferSizeInput);
    output = prePath.initOutputAnySize(input);
    prePath.processAnySize(input, output);
    EXPECT_TRUE(prePath.validInputAnySize(input));
    EXPECT_TRUE(prePath.validOutputAnySize(output, static_cast<int>(input.rows())));
    fmt::print("Synchronous processing with large bufferSize succesful.\n");

    prePath.setBufferMode(BufferMode::ASYNCHRONOUS_BUFFER);
    output = prePath.initOutputAnySize(input);
    prePath.processAnySize(input, output);
    EXPECT_TRUE(prePath.validInputAnySize(input));
    EXPECT_TRUE(prePath.validOutputAnySize(output, static_cast<int>(input.rows())));
    fmt::print("Asynchronous processing with large bufferSize succesful.\n");
}