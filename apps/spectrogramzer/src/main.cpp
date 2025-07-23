#include "AudioFile.h"
#include <algorithm_library/fft.h>
#include <cxxopts.hpp>
#include <iostream>
#include "spectrogram_process.h"
#include "spectrogram_adaptive_zeropad_process.h"
#include <chrono>

using namespace Eigen;
using namespace Pyplotcpp;

// set constants
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// split a string by a delimiter
// This function is used to split the output file name into file name and extension
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    
    while (end != std::string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    
    tokens.push_back(str.substr(start));
    return tokens;
}

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

    float sampleRate = audioFileInput.getSampleRate();

    auto outputSplit = split(outputName, '.');

    std::cout << "Processing summary:\n";
    int bufferSize = static_cast<int>(hopSizeMilliseconds / 1000.f * sampleRate);
    std::cout << "Buffer size: " << bufferSize << "\n";
    float fftSize = spectrumSizeMilliseconds / 1000.f * sampleRate;
    fftSize = FFTConfiguration::getValidFFTSize(fftSize);
    std::cout << "FFT size: " << fftSize << "\n";
    int nBands = FFTConfiguration::convertFFTSizeToNBands(fftSize);
    std::cout << "Number of frequency bins: " << nBands << "\n";
    int nFrames = audioFileInput.getNumSamplesPerChannel() / bufferSize;
    std::cout << "Number of frames: " << nFrames << "\n";

    int nFolds = 1; // no overlap
    int nonlinearity = 0; // no nonlinearity


    std::cout << "Processing audio file...\n";

    std::cout << "Processing spectrogram...\n";
    std::string spectrogramOutputName = outputSplit[0] + "_spectrogram." + outputSplit[1];
    auto start = std::chrono::high_resolution_clock::now();
    spectrogramProcess(&audioFileInput.samples[0][0], sampleRate, spectrogramOutputName, bufferSize, nBands, nFolds, nonlinearity, nFrames);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << duration.count() / 1000000.f << " seconds" << std::endl;

    std::cout << "Processing spectrogram adaptive zeropad...\n";
    spectrogramOutputName = outputSplit[0] + "_spectrogram_adaptive_zeropad." + outputSplit[1];

    start = std::chrono::high_resolution_clock::now();
    spectrogramAdaptiveZeropadProcess(&audioFileInput.samples[0][0], sampleRate, spectrogramOutputName, bufferSize, nBands, nFolds, nonlinearity, nFrames);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Execution time: " << duration.count() / 1000000.f << " seconds" << std::endl;

    std::cout << "DONE!\n" << std::endl;

    return 0;
}