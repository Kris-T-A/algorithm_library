#pragma once
#define EIGEN_DENSEBASE_PLUGIN                                                                                                                                                \
    "algorithm_library/interface/get_dynamic_memory_size.h" //  member function added to Eigen DenseBase class to get dynamic memory size of array and matrices
#define EIGEN_MPL2_ONLY                                     // don't allow LGPL licensed code from Eigen
#include <Eigen/Dense>                                      // Eigen Library.

// author: Kristian Timm Andersen, 2019

// Eigen inputs should be const Eigen::Ref<const T>& types:
namespace I
{
// define In type
template <typename T>
using In = const Eigen::Ref<const T> &;

template <typename T>
using InArray = In<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using InArray2D = In<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

using Real = In<Eigen::ArrayXf>;
using Complex = In<Eigen::ArrayXcf>;
using Bool = In<Eigen::Array<bool, Eigen::Dynamic, 1>>;
using Real2D = In<Eigen::ArrayXXf>;
using Complex2D = In<Eigen::ArrayXXcf>;
using Bool2D = In<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>;
using Boolean = const bool &;
using Int = const int &;
using Float = const float &;
using Void = void *;

using VectorComplex2D = const std::vector<Eigen::ArrayXXcf> &;

using Real4X = In<Eigen::Array4Xf>;
using Real4 = In<Eigen::Array4f>;
using RealX2 = In<Eigen::ArrayX2f>;
using Real6X = In<Eigen::Array<float, 6, Eigen::Dynamic>>;

using Real32 = In<Eigen::Array<float, 3, 2>>;

struct RealComplex
{
    Real R;
    Complex C;
};
struct RealReal
{
    Real R1;
    Real R2;
};
struct ComplexComplex
{
    Complex C1;
    Complex C2;
};

// extract type using partial template specialization: https://stackoverflow.com/questions/301203/extract-c-template-parameters
template <typename T>
struct getType
{
    typedef T type;
};

template <typename T>
struct getType<In<T>>
{
    typedef T type;
};

template <typename T>
struct getType<Eigen::Array<T, Eigen::Dynamic, 1>>
{
    typedef T type;
};

template <typename T>
struct getType<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>
{
    typedef T type;
};

template <typename>
struct getScalarType;

template <typename T>
struct getScalarType<In<Eigen::Array<T, Eigen::Dynamic, 1>>>
{
    typedef T type;
};

template <typename T>
struct getScalarType<In<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>>
{
    typedef T type;
};
}; // namespace I

// Eigen outputs should be Eigen::Ref<T> types:
namespace O
{
// define Out type
template <typename T>
using Out = Eigen::Ref<T>;

template <typename T>
using OutArray = Out<Eigen::Array<T, Eigen::Dynamic, 1>>;

template <typename T>
using OutArray2D = Out<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;

using Real = Out<Eigen::ArrayXf>;
using Complex = Out<Eigen::ArrayXcf>;
using Bool = Out<Eigen::Array<bool, Eigen::Dynamic, 1>>;
using Real2D = Out<Eigen::ArrayXXf>;
using Complex2D = Out<Eigen::ArrayXXcf>;
using Bool2D = Out<Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>>;
using U8Int2D = Out<Eigen::Array<uint8_t, Eigen::Dynamic, Eigen::Dynamic>>;

using VectorReal2D = std::vector<Eigen::ArrayXXf> &;
using VectorComplex2D = std::vector<Eigen::ArrayXXcf> &;

using RealX6 = Out<Eigen::Array<float, Eigen::Dynamic, 6>>;
using Real6X = Out<Eigen::Array<float, 6, Eigen::Dynamic>>;

struct RealComplex
{
    Real R;
    Complex C;
};
struct RealReal
{
    Real R1;
    Real R2;
};
struct ComplexComplex
{
    Complex C1;
    Complex C2;
};

using Boolean = bool &;
using Float = float &;
using Void = void *;

using Real2 = Out<Eigen::Array<float, 2, 1>>;

// extract type using partial template specialization: https://stackoverflow.com/questions/301203/extract-c-template-parameters
template <typename>
struct getType;

template <typename T>
struct getType<Out<T>>
{
    typedef T type;
};

template <typename T>
struct getType<Eigen::Array<T, Eigen::Dynamic, 1>>
{
    typedef T type;
};

template <typename T>
struct getType<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>
{
    typedef T type;
};

template <typename>
struct getScalarType;

template <typename T>
struct getScalarType<Out<Eigen::Array<T, Eigen::Dynamic, 1>>>
{
    typedef T type;
};

template <typename T>
struct getScalarType<Out<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>>
{
    typedef T type;
};
}; // namespace O
