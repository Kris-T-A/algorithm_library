#pragma once
#include "algorithm_library/noise_reduction.h"
#include "framework/framework.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_run_options_config_keys.h"
#include "onnxruntime_session_options_config_keys.h"
#include <iostream>

// ONNXModel use Row-major layout internally, so the input shape must be specified in such a way, i.e. a 2 x 4 x 64 tensor contains 2*4*64 = 512 elements where each 64 length
// vector is placed contiguously in memory.
//
// ONNXModel only supports float data type as input/output.
class ONNXModel
{
  public:
    ONNXModel(const std::string &mPath, const std::vector<std::string> &inNames, const std::vector<std::vector<int64_t>> &inShapes, const std::vector<std::string> &outNames,
              const std::vector<std::vector<int64_t>> &outShapes)
        : modelPath(mPath.begin(), mPath.end()), // .begin() and .end() is necessary since we don't know the type of mPath (see comment where modelPath is defined)
          inputNames(inNames), inputShapes(inShapes), outputNames(outNames), outputShapes(outShapes)
    {
        Ort::MemoryInfo memInfo{Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)};

        // create vector of char pointers to input/output names and create vectors of input/output tensors
        nInputs = static_cast<int>(std::min(inputNames.size(), inputShapes.size()));
        inputNamesChar.resize(nInputs);
        inputData.resize(nInputs);
        for (auto i = 0; i < nInputs; i++)
        {
            inputNamesChar[i] = inputNames[i].c_str();
            pushTensor(inputTensors, memInfo, inputData[i], inputShapes[i], i);
        }

        nOutputs = static_cast<int>(std::min(outputNames.size(), outputShapes.size()));
        outputNamesChar.resize(nInputs);
        outputData.resize(nOutputs);
        for (auto i = 0; i < nOutputs; i++)
        {
            outputNamesChar[i] = outputNames[i].c_str();
            pushTensor(outputTensors, memInfo, outputData[i], outputShapes[i], i);
        }

        size_t maxMemory = 0;           // 0 = default
        int arenaExtendStrategy = -1;   // -1 = default
        int initialChunkSizeBytes = -1; // -1 = default
        int maxDeadBytesPerChunk = -1;  // -1 = default
        Ort::ArenaCfg arenaCfg = Ort::ArenaCfg(maxMemory, arenaExtendStrategy, initialChunkSizeBytes, maxDeadBytesPerChunk);

        Ort::ThreadingOptions threadOptions;
        threadOptions.SetGlobalInterOpNumThreads(2);
        env = std::make_unique<Ort::Env>(threadOptions, ORT_LOGGING_LEVEL_WARNING, "global_environment");

        // env.CreateAndRegisterAllocator(memInfo, arenaCfg); // TODO: enable this again. It fails because you can not register the same allocator several times (in
        // different algorithms)
        env.get()->DisableTelemetryEvents();

        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOptions.DisablePerSessionThreads();
        sessionOptions.DisableProfiling();
        sessionOptions.DisableMemPattern();
        sessionOptions.DisableCpuMemArena();
        sessionOptions.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");

        session = std::make_unique<Ort::Session>(*env.get(), modelPath.c_str(), sessionOptions);

        // inspired from: https://github.com/microsoft/onnxruntime/issues/11627
        // runOption.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "cpu:0;gpu:0");
    }

    // POTENTIAL TODO: an alternative to copying the input/output vectors is to create new tensors each run.
    void process(std::vector<Eigen::Ref<Eigen::ArrayXXf>> input, std::vector<Eigen::Ref<Eigen::ArrayXXf>> output)
    {
        for (auto i = 0; i < nInputs; i++)
        {
            inputData[i] = Eigen::Map<Eigen::ArrayXf>(input[i].data(), inputData[i].size());
        }
        run();
        for (auto i = 0; i < nOutputs; i++)
        {
            output[i] = Eigen::Map<Eigen::ArrayXXf>(outputData[i].data(), output[i].rows(), output[i].cols());
        }
    }

  private:
    void run() { session.get()->Run(runOption, inputNamesChar.data(), inputTensors.data(), nInputs, outputNamesChar.data(), outputTensors.data(), nOutputs); }

    void pushTensor(std::vector<Ort::Value> &tensorVector, const Ort::MemoryInfo &memInfo, Eigen::ArrayXf &array, const std::vector<int64_t> &shape, int i)
    {
        size_t nDims = shape.size();
        size_t nElements = 1;
        for (const auto &e : shape)
        {
            nElements *= static_cast<size_t>(e);
        }
        array = Eigen::ArrayXf::Zero(nElements);
        tensorVector.emplace_back(Ort::Value::CreateTensor<float>(memInfo, array.data(), nElements, shape.data(), nDims));
    }

    int nInputs;
    int nOutputs;
    std::basic_string<ORTCHAR_T> modelPath; // ORTCHAR_T is defined in onnxruntime_c_api.h and is wchar_t on Windows, and char_t on Linux
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::Env> env;
    Ort::RunOptions runOption{nullptr};
    std::vector<Eigen::ArrayXf> inputData;
    std::vector<Eigen::ArrayXf> outputData;
    std::vector<std::string> inputNames;
    std::vector<const char *> inputNamesChar;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<Ort::Value> inputTensors;
    std::vector<std::string> outputNames;
    std::vector<const char *> outputNamesChar;
    std::vector<std::vector<int64_t>> outputShapes;
    std::vector<Ort::Value> outputTensors;
};

// Feed spectral power, phase, and persistent recurrent state into an ONNX model
// and apply its predicted gains to the complex input spectrum.
class NoiseReductionML : public AlgorithmImplementation<NoiseReductionConfiguration, NoiseReductionML>
{
  public:
    NoiseReductionML(const Coefficients &c = {.nBands = 257, .nChannels = 1, .algorithmType = Coefficients::ML})
        : BaseAlgorithm{c}, onnxModel("pp2model.onnx", {"magnitude", "phase", "time state", "gru1 state", "gru2 state"},
                                      {{1, 1, C.nBands}, {1, 1, C.nBands}, {1, 5, 2, C.nBands}, {1, 64, 32}, {1, 64, 32}},
                                      {"gain", "time state out", "gru1 state out", "gru2 state out"}, {{1, C.nBands}, {1, 5, 2, C.nBands}, {1, 64, 32}, {1, 64, 32}})
    {
        magnitude = Eigen::ArrayXf::Zero(C.nBands);
        phase = Eigen::ArrayXf::Zero(C.nBands);
        gain = Eigen::ArrayXf::Zero(C.nBands);
        timeBuffer = Eigen::ArrayXXf::Zero(10, C.nBands);
        gru1Buffer = Eigen::ArrayXXf::Zero(64, 32);
        gru2Buffer = Eigen::ArrayXXf::Zero(64, 32);
    }

  private:
    Eigen::ArrayXf magnitude;
    Eigen::ArrayXf phase;
    Eigen::ArrayXf gain;
    Eigen::ArrayXXf timeBuffer;
    Eigen::ArrayXXf gru1Buffer;
    Eigen::ArrayXXf gru2Buffer;

    ONNXModel onnxModel;

    void processAlgorithm(Input xFreq, Output yFreq)
    {
        magnitude = xFreq.abs2();
        phase = xFreq.arg();
        onnxModel.process({magnitude, phase, timeBuffer, gru1Buffer, gru2Buffer}, {gain, timeBuffer, gru1Buffer, gru2Buffer});
        yFreq = gain * xFreq;
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = magnitude.getDynamicMemorySize();
        size += phase.getDynamicMemorySize();
        size += gain.getDynamicMemorySize();
        size += timeBuffer.getDynamicMemorySize();
        size += gru1Buffer.getDynamicMemorySize();
        size += gru2Buffer.getDynamicMemorySize();
        return size;
    }

    void resetVariables() final
    {
        magnitude.setZero();
        phase.setZero();
        gain.setZero();
        timeBuffer.setZero();
        gru1Buffer.setZero();
        gru2Buffer.setZero();
    }

    bool isCoefficientsValid() const final
    {
        bool flag = C.algorithmType == C.ML;
        flag &= C.nBands == 257;
        flag &= C.nChannels == 1;
        return flag;
    }

    friend BaseAlgorithm;
};
