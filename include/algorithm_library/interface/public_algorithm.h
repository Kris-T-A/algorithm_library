#pragma once
#include "input_output.h"
#include "macros_json.h"

template <typename Tconfiguration>
struct TSetup
{
    using Coefficients = typename Tconfiguration::Coefficients;
    using Parameters = typename Tconfiguration::Parameters;

    TSetup() = default;
    TSetup(const Coefficients &c, const Parameters &p)
    {
        coefficients = c;
        parameters = p;
    }
    Coefficients coefficients;
    Parameters parameters;
    DEFINE_TUNABLE_SETUP(coefficients, parameters)
};

template <typename Tconfiguration>
class Algorithm
{
  public:
    using Configuration = Tconfiguration;
    using Input = const typename Configuration::Input &; // force inputs to be const reference
    using Output = typename Configuration::Output;
    using Coefficients = typename Configuration::Coefficients;
    using Parameters = typename Configuration::Parameters;
    using Setup = TSetup<Configuration>;

    Algorithm() : Algorithm(Coefficients()) {} // default constructor
    Algorithm(const Coefficients &c) { setImplementation(c); }

    void process(Input input, Output output) { pimpl->process(input, output); }
    Coefficients getCoefficients() const { return pimpl->getCoefficients(); }
    Parameters getParameters() const { return pimpl->getParameters(); }
    Setup getSetup() const { return pimpl->getSetup(); }
    void reset() { pimpl->reset(); }

    void setImplementation(const Coefficients &c); // define this in derived source file
    void setCoefficients(const Coefficients &c)
    {
        auto p = getParameters();
        setImplementation(c);
        setParameters(p);
    }
    void setParameters(const Parameters &p) { pimpl->setParameters(p); }
    void setSetup(const Setup &s)
    {
        setImplementation(s.coefficients);
        setParameters(s.parameters);
    }

    nlohmann::json getDebugJson() const { return pimpl->getDebugJson(); }
    void setDebugJson(const nlohmann::json &s) { pimpl->setDebugJson(s); }

    auto validInput(Input input) const { return pimpl->validInput(input); }
    auto validOutput(Output output) const { return pimpl->validOutput(output); }
    auto initInput() const { return pimpl->initInput(); }
    auto initOutput(Input input) const { return pimpl->initOutput(input); }

    bool isConfigurationValid() const { return pimpl->isConfigurationValid(); } // if this returns false, then behaviour of algorithm is undefined

    static constexpr size_t ALGORITHM_VERSION_MAJOR = 1; // version changes in ABI

    struct BaseImplementation // Base of implementation. Allows to derive different implementations from this struct.
    {
        BaseImplementation() = default;
        virtual ~BaseImplementation() = default;
        virtual void process(Input input, Output output) = 0;
        virtual Coefficients getCoefficients() const = 0;
        virtual Parameters getParameters() const = 0;
        virtual Setup getSetup() const = 0;
        virtual void setCoefficients(const Coefficients &c) = 0;
        virtual void setParameters(const Parameters &p) = 0;
        virtual void setSetup(const Setup &s) = 0;
        virtual void reset() = 0;
        virtual nlohmann::json getDebugJson() const = 0;
        virtual void setDebugJson(const nlohmann::json &s) = 0;
        virtual bool isConfigurationValid() const = 0;
        virtual bool validInput(typename Configuration::Input input) const = 0;
        virtual bool validOutput(typename Configuration::Output output) const = 0;
        virtual decltype(Configuration::initInput(std::declval<Coefficients>())) initInput() const = 0;
        virtual decltype(Configuration::initOutput(std::declval<Input>(), std::declval<Coefficients>())) initOutput(typename Configuration::Input input) const = 0;
    };

  protected:
    ~Algorithm() = default;

    std::unique_ptr<BaseImplementation> pimpl; // PIMPL. Define in derived source file
};

enum BufferMode { SYNCHRONOUS_BUFFER, ASYNCHRONOUS_BUFFER };

template <typename Tconfiguration>
struct ConfigurationBuffer : public Tconfiguration
{
    using Input = I::Real2D;
    using Output = O::Real2D;

    static Eigen::ArrayXXf initInput(const typename Tconfiguration::Coefficients &c) { return Eigen::ArrayXXf::Random(c.bufferSize, c.nChannels); } // time samples

    static Eigen::ArrayXXf initOutput(Input input, const typename Tconfiguration::Coefficients &c)
    {
        return Eigen::ArrayXXf::Zero(c.bufferSize, Tconfiguration::getNChannelsOut(c));
    } // time samples

    static bool validInput(Input input, const typename Tconfiguration::Coefficients &c)
    {
        return (input.rows() == c.bufferSize) && (input.cols() == c.nChannels) && input.allFinite();
    }

    static bool validOutput(Output output, const typename Tconfiguration::Coefficients &c)
    {
        return (output.rows() == c.bufferSize) && (output.cols() == Tconfiguration::getNChannelsOut(c)) && output.allFinite();
    }

    // return size of output buffer given the input bufferSize and bufferMode
    static int getOutputAnySizeBufferSize(int bufferAnySize, const typename Tconfiguration::Coefficients &c, BufferMode bufferMode)
    {
        if (bufferMode == BufferMode::SYNCHRONOUS_BUFFER) { return static_cast<int>(std::ceil(static_cast<float>(bufferAnySize) / c.bufferSize) * c.bufferSize); }
        return bufferAnySize;
    }

    static Eigen::ArrayXXf initInputAnySize(int bufferAnySize, const typename Tconfiguration::Coefficients &c)
    {
        return Eigen::ArrayXXf::Random(bufferAnySize, c.nChannels);
    } // time samples

    static Eigen::ArrayXXf initOutputAnySize(Input input, const typename Tconfiguration::Coefficients &c, BufferMode bufferMode)
    {
        return Eigen::ArrayXXf::Zero(getOutputAnySizeBufferSize(static_cast<int>(input.rows()), c, bufferMode), Tconfiguration::getNChannelsOut(c));
    } // time samples

    static bool validInputAnySize(Input input, const typename Tconfiguration::Coefficients &c)
    {
        return (input.rows() > 0) && (input.cols() == c.nChannels) && input.allFinite();
    }

    static bool validOutputAnySize(Output output, int bufferAnySize, const typename Tconfiguration::Coefficients &c, BufferMode bufferMode)
    {
        return (output.rows() == getOutputAnySizeBufferSize(bufferAnySize, c, bufferMode)) && (output.cols() == Tconfiguration::getNChannelsOut(c)) && output.allFinite();
    }
};
// AlgorithmBuffer<Tconfiguration> is a class that derives from Algorithm<Tconfiguration>
// It is for algorithms that have a dynamic size input/output array and allows to change
// the mode that the algorithm is using to process the input using processAnySize:
//
//	SYNCHRONOUS_BUFFER - zeropad the input to be a multiple of Configuration.bufferSize and process the entire input by successive calls to process
//	ASYNCHRONOUS_BUFFER - Create an internal buffer and fill it with input values in a for-loop. Every time the buffer is full, call process and output the result. This
// results in an additional delay equal to bufferSize

template <typename Tconfiguration>
class AlgorithmBuffer : public Algorithm<Tconfiguration>
{
  public:
    using Base = Algorithm<Tconfiguration>;
    using Configuration = typename Base::Configuration;
    using Input = typename Base::Input;
    using Output = typename Base::Output;
    using Coefficients = typename Base::Coefficients;
    using Parameters = typename Base::Parameters;
    using Setup = typename Base::Setup;

    // check if conditions for AlgorithmBuffer<Tconfiguration> are fulfilled
    static_assert(std::is_same<int, decltype(Coefficients::bufferSize)>::value,
                  "Coefficients must have integer variable bufferSize");                                                 // Coefficients has integer member variable bufferSize
    static_assert(Eigen::Dynamic == I::getType<Input>::type::RowsAtCompileTime, "Input number of rows must be dynamic"); // input rows size is Dynamic
    static_assert(Eigen::Dynamic == O::getType<Output>::type::RowsAtCompileTime, "Output number of rows must be dynamic"); // output rows size is Dynamic

    AlgorithmBuffer() : AlgorithmBuffer(Coefficients()) {}
    AlgorithmBuffer(const Coefficients &c) : Algorithm<Tconfiguration>(c) {}

    struct BufferBaseImplementation : public Base::BaseImplementation
    {
        BufferBaseImplementation() = default;
        virtual ~BufferBaseImplementation() = default;
        virtual BufferMode getBufferMode() const = 0;
        virtual int getBufferSize() const = 0;
        virtual int getDelaySamples() const = 0;
        virtual void processAnySize(Input input, Output output) = 0;
    };

    BufferMode getBufferMode() const { return static_cast<BufferBaseImplementation *>(Base::pimpl.get())->getBufferMode(); }
    int getBufferSize() const { return static_cast<BufferBaseImplementation *>(Base::pimpl.get())->getBufferSize(); }
    int getDelaySamples() const { return static_cast<BufferBaseImplementation *>(Base::pimpl.get())->getDelaySamples(); }
    void processAnySize(Input input, Output output) { static_cast<BufferBaseImplementation *>(Base::pimpl.get())->processAnySize(input, output); }
    void setBufferMode(BufferMode bufferMode); // define in source file

    Eigen::ArrayXXf initInputAnySize(int bufferAnySize) { return Configuration::initInputAnySize(bufferAnySize, Base::getCoefficients()); }
    Eigen::ArrayXXf initOutputAnySize(Input input) { return Configuration::initOutputAnySize(input, Base::getCoefficients(), getBufferMode()); }
    bool validInputAnySize(Input input) { return Configuration::validInputAnySize(input, Base::getCoefficients()); }
    bool validOutputAnySize(Output output, int bufferAnySize) { return Configuration::validOutputAnySize(output, bufferAnySize, Base::getCoefficients(), getBufferMode()); }

  protected:
    ~AlgorithmBuffer() = default;
};