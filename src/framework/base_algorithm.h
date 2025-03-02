#pragma once
#include "algorithm_library/interface/input_output.h"
#include "algorithm_library/interface/macros_json.h"
#include "algorithm_library/interface/public_algorithm.h"

template <typename Talgo, typename Tconfiguration>
struct Implementation : public Algorithm<Tconfiguration>::BaseImplementation
{
    using Base = typename Algorithm<Tconfiguration>::BaseImplementation;

    Implementation() : algo{} {}
    Implementation(const typename Tconfiguration::Coefficients &c) : algo{c} {}
    Talgo algo;
    void process(typename Algorithm<Tconfiguration>::Input input, typename Algorithm<Tconfiguration>::Output output) override { algo.process(input, output); }
    typename Algorithm<Tconfiguration>::Coefficients getCoefficients() const final { return algo.getCoefficients(); }
    typename Algorithm<Tconfiguration>::Parameters getParameters() const final { return algo.getParameters(); }
    typename Algorithm<Tconfiguration>::Setup getSetup() const final { return algo.getSetup(); }
    void setCoefficients(const typename Algorithm<Tconfiguration>::Coefficients &c) final { algo.setCoefficients(c); }
    void setParameters(const typename Algorithm<Tconfiguration>::Parameters &p) final { algo.setParameters(p); }
    void setSetup(const typename Algorithm<Tconfiguration>::Setup &s) final { algo.setSetup(s); }
    void reset() override { algo.reset(); }
    nlohmann::json getDebugJson() const final
    {
        nlohmann::json j = algo.getSetupTree();
        return j;
    }
    void setDebugJson(const nlohmann::json &s) final { algo.setSetupTree(s); }
    bool isConfigurationValid() const final { return algo.isConfigurationValid(); }

    bool validInput(typename Tconfiguration::Input input) const final { return algo.validInput(input); }
    bool validOutput(typename Tconfiguration::Output output) const final { return algo.validOutput(output); }
    decltype(Tconfiguration::initInput(std::declval<typename Tconfiguration::Coefficients>())) initInput() const final { return algo.initInput(); }
    decltype(Tconfiguration::initOutput(std::declval<typename Tconfiguration::Input>(), std::declval<typename Tconfiguration::Coefficients>())) initOutput(
        typename Tconfiguration::Input input) const final
    {
        return algo.initOutput(input);
    }
};

template <typename Talgo, typename Tconfiguration>
struct BufferImplementation : public AlgorithmBuffer<Tconfiguration>::BufferBaseImplementation
{
    BufferImplementation() : BufferImplementation(Algorithm<Tconfiguration>::Coefficients()) {}
    BufferImplementation(const typename Tconfiguration::Coefficients &c) : algo{c}
    {
        bufferIn = algo.initInput();
        bufferOut = algo.initOutput(bufferIn);
        bufferIn.setZero();
        bufferOut.setZero();
    }
    template <typename TcoefficientsTree>
    BufferImplementation(const TcoefficientsTree &cTree)
    {
        algo.setCoefficientsTree(cTree);
        bufferIn = algo.initInput();
        bufferOut = algo.initOutput(bufferIn);
        bufferIn.setZero();
        bufferOut.setZero();
    }

    Talgo algo;

    void process(typename Talgo::Input input, typename Talgo::Output output) final { algo.process(input, output); }
    typename Talgo::Coefficients getCoefficients() const final { return algo.getCoefficients(); }
    typename Talgo::Parameters getParameters() const final { return algo.getParameters(); }
    typename Talgo::Setup getSetup() const final { return algo.getSetup(); }
    void setCoefficients(const typename Talgo::Coefficients &c) final { algo.setCoefficients(c); }
    void setParameters(const typename Talgo::Parameters &p) final { algo.setParameters(p); }
    void setSetup(const typename Talgo::Setup &s) final { algo.setSetup(s); }
    void reset() override
    {
        bufferIn.setZero();
        bufferOut.setZero();
        algo.reset();
    }
    nlohmann::json getDebugJson() const final
    {
        nlohmann::json j = algo.getSetupTree();
        return j;
    }
    void setDebugJson(const nlohmann::json &s) final { algo.setSetupTree(s); }
    bool isConfigurationValid() const final { return algo.isConfigurationValid(); }

    bool validInput(typename Tconfiguration::Input input) const final { return algo.validInput(input); }
    bool validOutput(typename Tconfiguration::Output output) const final { return algo.validOutput(output); }
    decltype(Tconfiguration::initInput(std::declval<typename Tconfiguration::Coefficients>())) initInput() const final { return algo.initInput(); }
    decltype(Tconfiguration::initOutput(std::declval<typename Tconfiguration::Input>(), std::declval<typename Tconfiguration::Coefficients>())) initOutput(
        typename Tconfiguration::Input input) const final
    {
        return algo.initOutput(input);
    }

    BufferMode getBufferMode() const override { return BufferMode::SYNCHRONOUS_BUFFER; }
    int getBufferSize() const final { return algo.getCoefficients().bufferSize; }
    int getDelaySamples() const override { return algo.getDelaySamples(); }

    // zeropad the input to be a multiple of Configuration.bufferSize and process the entire input by successive calls to process
    void processAnySize(typename Talgo::Input input, typename Talgo::Output output) override
    {
        int i = 0;
        const int bufferSize = getBufferSize();
        // Process as many full buffers as possible
        for (; i <= (input.rows() - bufferSize); i += bufferSize)
        {
            algo.process(input.middleRows(i, bufferSize), output.middleRows(i, bufferSize));
        }
        // if we have been given a size that is not an integer multiple of bufferSize, zeropad and process.
        const int remainingSamples = static_cast<int>(input.rows()) - i;
        if (remainingSamples > 0)
        {
            bufferIn.topRows(remainingSamples) = input.middleRows(i, remainingSamples);
            bufferIn.bottomRows(bufferSize - remainingSamples).setZero();
            algo.process(bufferIn, bufferOut);
            output.bottomRows(remainingSamples) = bufferOut.topRows(remainingSamples);
        }
    }

    typename I::getType<typename Talgo::Input>::type bufferIn;
    typename O::getType<typename Talgo::Output>::type bufferOut;
};

template <typename Talgo, typename Tconfiguration>
struct AsynchronousBufferImplementation : public BufferImplementation<Talgo, Tconfiguration>
{
    using Base = BufferImplementation<Talgo, Tconfiguration>;

    AsynchronousBufferImplementation() : AsynchronousBufferImplementation(Algorithm<Tconfiguration>::Coefficients()) {}
    template <typename TcoefficientsTree>
    AsynchronousBufferImplementation(const TcoefficientsTree &c) : Base{c} // this constructor is used both for the constructor with coefficients and coefficients tree
    {
        index = 0;
    }

    // Create an internal buffer and fill it with input values in a for-loop. Every time the buffer is full, call process and output the result. This results in an additional
    // delay equal to bufferSize
    void processAnySize(typename Algorithm<Tconfiguration>::Input input, typename Algorithm<Tconfiguration>::Output output) final
    {
        int const bufferSize = Base::getBufferSize();
        for (auto i = 0; i < input.rows(); i++)
        {
            Base::bufferIn.row(index) = input.row(i);
            output.row(i) = Base::bufferOut.row(index);
            index++;
            if (index == bufferSize)
            {
                Base::algo.process(Base::bufferIn, Base::bufferOut);
                index = 0;
            }
        }
    }

    void reset() final
    {
        index = 0;
        Base::reset();
    }

    BufferMode getBufferMode() const final { return BufferMode::ASYNCHRONOUS_BUFFER; }

    // when processing asynchronously an extra buffer is added to the delay in the process() method
    int getDelaySamples() const final { return Base::getDelaySamples() + Base::getBufferSize(); }

    int index;
};

template <typename Tconfiguration, typename Talgo>
class AlgorithmImplementation
{
  public:
    using Configuration = Tconfiguration;
    using Input = const typename Configuration::Input &; // force inputs to be const reference
    using Output = typename Configuration::Output;
    using BaseAlgorithm = AlgorithmImplementation;
    using Coefficients = typename Configuration::Coefficients;
    using Parameters = typename Configuration::Parameters;
    using Setup = TSetup<Configuration>;

    static_assert(std::is_trivially_copyable<Coefficients>::value, "Coefficients data type must be trivially copyable.");
    static_assert(std::is_trivially_copyable<Parameters>::value, "Parameters data type must be trivially copyable.");

    AlgorithmImplementation() = default;
    AlgorithmImplementation(const Coefficients &c) : C(c) {}
    AlgorithmImplementation(const Setup &s) : C(s.coefficients), P(s.parameters) {}
    virtual ~AlgorithmImplementation() = default;

    size_t getStaticSize() const { return sizeof(Talgo); }

    size_t getDynamicSize() const { return getDynamicSizeVariables() + getDynamicSizeAlgorithms(); }

    // Processing method. This is where the core of the algorithm is calculated.
    // When profiling using MSVC compiler it was found that CRTP is faster than virtual methods.
    // However, using GCC it was found that virtual methods are as fast as CRTP (maybe because the virtual methods in header files can be inlined?).
    inline void process(Input input, Output output) { static_cast<Talgo &>(*this).processAlgorithm(input, output); }

    // templated process functions that allows to call process with tuples
    template <typename... TupleTypes>
    void process(const std::tuple<TupleTypes...> &input, Output output)
    {
        processImpl(input, std::make_index_sequence<sizeof...(TupleTypes)>{}, output);
    }

    template <typename... TupleTypes>
    void process(Input input, std::tuple<TupleTypes...> &output)
    {
        processImpl(input, output, std::make_index_sequence<sizeof...(TupleTypes)>{});
    }

    template <typename... TupleTypesInput, typename... TupleTypesOutput>
    void process(const std::tuple<TupleTypesInput...> &input, std::tuple<TupleTypesOutput...> &output)
    {
        processImpl(input, std::make_index_sequence<sizeof...(TupleTypesInput)>{}, output, std::make_index_sequence<sizeof...(TupleTypesOutput)>{});
    }

    void reset()
    {
        resetVariables();
        resetAlgorithms();
    }

    static constexpr size_t ALGORITHM_VERSION_MINOR = 1; // version changes in implementation

    Coefficients getCoefficients() const { return C; }
    Parameters getParameters() const { return P; }
    Setup getSetup() const { return Setup{getCoefficients(), getParameters()}; }

    // use SFINAE to call default constructor if Coefficients is empty
    template <typename T = Coefficients>
    typename std::enable_if<std::is_empty<T>::value>::type setCoefficients(const Coefficients &c)
    {
        auto p = getParameters();
        static_cast<Talgo &>(*this) = Talgo();
        setParameters(p);
    }

    template <typename T = Coefficients>
    typename std::enable_if<!std::is_empty<T>::value>::type setCoefficients(const Coefficients &c)
    {
        auto p = getParameters();
        static_cast<Talgo &>(*this) = Talgo(c);
        setParameters(p);
    }

    void setParameters(const Parameters &p)
    {
        P = p;
        static_cast<Talgo &>(*this).onParametersChanged();
    }

    void setSetup(const Setup &setup)
    {
        setCoefficients(setup.coefficients);
        setParameters(setup.parameters);
    }

    // return type is deduced compile-time so can't be virtual
    auto getCoefficientsTree() const { return static_cast<Talgo const &>(*this).getCoefficientsTreeImpl(); }
    auto getParametersTree() const { return static_cast<Talgo const &>(*this).getParametersTreeImpl(); }
    auto getSetupTree() const { return static_cast<Talgo const &>(*this).getSetupTreeImpl(); }

    template <typename Tcoefficients>
    void setCoefficientsTree(const Tcoefficients &c)
    {
        static_cast<Talgo &>(*this).setCoefficientsTreeImpl(c);
    }

    template <typename Tparameters>
    void setParametersTree(const Tparameters &p)
    {
        static_cast<Talgo &>(*this).setParametersTreeImpl(p);
    }

    template <typename Tsetup>
    void setSetupTree(const Tsetup &s)
    {
        static_cast<Talgo &>(*this).setSetupTreeImpl(s);
    }

    auto initInput() const { return Configuration::initInput(C); }
    auto initOutput(Input input) const { return Configuration::initOutput(input, C); }
    auto initDefaultOutput() const { return Configuration::initOutput(initInput(), C); }
    auto validInput(Input input) const { return Configuration::validInput(input, C); }
    auto validOutput(Output output) const { return Configuration::validOutput(output, C); }

    // if this returns false, then behaviour of algorithm is undefined
    virtual bool isConfigurationValid() const
    {
        bool flag = isCoefficientsValid();
        flag &= isParametersValid();
        flag &= isAlgorithmsValid();
        return flag;
    }

    // template functions to allow to call initOutput, validInput, validOutput with tuples
    template <typename... TupleTypes>
    auto initOutput(const std::tuple<TupleTypes...> &input) const
    {
        return initOutputImpl(input, std::make_index_sequence<sizeof...(TupleTypes)>{});
    }

    template <typename... TupleTypes>
    auto validInput(const std::tuple<TupleTypes...> &input) const
    {
        return validInputImpl(input, std::make_index_sequence<sizeof...(TupleTypes)>{});
    }

    template <typename... TupleTypes>
    auto validOutput(std::tuple<TupleTypes...> &output) const
    {
        return validOutputImpl(output, std::make_index_sequence<sizeof...(TupleTypes)>{});
    }

  protected:
    // these functions will be overridden if defined in derived Talgo
    virtual size_t getDynamicSizeVariables() const { return 0; }
    virtual size_t getDynamicSizeAlgorithms() const { return 0; }
    virtual void resetVariables() {}
    virtual void resetAlgorithms() {}
    virtual bool isCoefficientsValid() const { return true; }
    virtual bool isParametersValid() const { return true; }
    virtual bool isAlgorithmsValid() const { return true; }
    void onParametersChanged() {} // If more advanced functionality is needed, then write your own setters but remember to call the setters from this function.

    // these functions will be hidden if macro DEFINE_STATIC_MEMBER_ALGORITHMS(...) or DEFINE_SIMPLE_MEMBER_ALGORITHMS(...) is declared in derived Talgo
    Coefficients getCoefficientsTreeImpl() const { return getCoefficients(); }
    void setCoefficientsTreeImpl(const Coefficients &c) { setCoefficients(c); }
    Parameters getParametersTreeImpl() const { return getParameters(); }
    void setParametersTreeImpl(const Parameters &p) { setParameters(p); }
    Setup getSetupTreeImpl() const { return getSetup(); }
    void setSetupTreeImpl(const Setup &s) { setSetup(s); }

    Coefficients C;
    Parameters P;

  private:
    // template implementations that allow to call methods with tuples
    template <typename TupleType, std::size_t... Is>
    void processImpl(const TupleType &input, std::index_sequence<Is...>, Output output)
    {
        process({std::get<Is>(input)...}, output);
    }

    template <typename TupleType, std::size_t... Is>
    void processImpl(Input input, TupleType &output, std::index_sequence<Is...>)
    {
        process(input, {std::get<Is>(output)...});
    }

    template <typename TupleTypeInput, std::size_t... IsInput, typename TupleTypeOutput, std::size_t... IsOutput>
    void processImpl(const TupleTypeInput &input, std::index_sequence<IsInput...>, TupleTypeOutput &output, std::index_sequence<IsOutput...>)
    {
        process({std::get<IsInput>(input)...}, {std::get<IsOutput>(output)...});
    }

    template <typename TupleType, std::size_t... Is>
    auto initOutputImpl(const TupleType &input, std::index_sequence<Is...>) const
    {
        return initOutput({std::get<Is>(input)...});
    }

    template <typename TupleType, std::size_t... Is>
    auto validInputImpl(const TupleType &input, std::index_sequence<Is...>) const
    {
        return validInput({std::get<Is>(input)...});
    }

    template <typename TupleType, std::size_t... Is>
    auto validOutputImpl(TupleType &output, std::index_sequence<Is...>) const
    {
        return validOutput({std::get<Is>(output)...});
    }
};
