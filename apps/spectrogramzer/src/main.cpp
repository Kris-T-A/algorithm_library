#include "AudioFile.h"
#include <algorithm_library/fft.h>
#include <cxxopts.hpp>
#include <iostream>
#include "spectrogram_process.h"

using namespace Eigen;
using namespace Pyplotcpp;

// set constants
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

int main(int argc, char **argv)
{
    cxxopts::Options options(*argv, "An app for creating spectrograms of audio files.");

    std::string inputName;
    std::string outputName;
    float hopSizeMilliseconds;
    float spectrumSizeMilliseconds;

    options.add_options()("h,help", "Show help")("v,version", "Print the current version number")("i,input", "Name of input file",
                                                                                                  cxxopts::value(inputName)->default_value("input.wav"))(
        "o,output", "Name of output file", cxxopts::value(outputName)->default_value("output.png"))("t,hop_size", "Size of each time hop between spectrums in milliseconds",
                                                                                                    cxxopts::value(hopSizeMilliseconds)->default_value("10.0"))(
        "s,spectrum_size", "Size of each frame used for calculating the spectrum in milliseconds", cxxopts::value(spectrumSizeMilliseconds)->default_value("80.0"));

    options.allow_unrecognised_options();

    auto result = options.parse(argc, argv);

    if (result["help"].as<bool>())
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result["version"].as<bool>())
    {
        std::cout << "version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
        return 0;
    }

    AudioFile<float> audioFileInput;
    audioFileInput.shouldLogErrorsToConsole(false);
    audioFileInput.load(inputName);

    if (audioFileInput.getLengthInSeconds() == 0)
    {
        std::cerr << "Error: Audio file is empty or could not be loaded." << std::endl;
        return 1;
    }

    std::cout << "Input file summary:\n";
    audioFileInput.printSummary();
    std::cout << "\n";

    std::cout << "Processing summary:\n";
    int bufferSize = static_cast<int>(hopSizeMilliseconds / 1000.f * audioFileInput.getSampleRate());
    std::cout << "Buffer size: " << bufferSize << "\n";
    float fftSize = spectrumSizeMilliseconds / 1000.f * audioFileInput.getSampleRate();
    fftSize = FFTConfiguration::getValidFFTSize(fftSize);
    std::cout << "FFT size: " << fftSize << "\n";
    int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
    std::cout << "Number of frequency bins: " << nBands << "\n";
    int nFrames = audioFileInput.getNumSamplesPerChannel() / bufferSize;
    std::cout << "Number of frames: " << nFrames << "\n";

    int nFolds = 1; // no overlap
    int nonlinearity = 0; // no nonlinearity


    std::cout << "Processing audio file...\n";
    spectrogramProcess(&audioFileInput.samples[0][0], outputName, bufferSize, nBands, nFolds, nonlinearity, nFrames);

    std::cout << "DONE!\n" << std::endl;

    return 0;
}