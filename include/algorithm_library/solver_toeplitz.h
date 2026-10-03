#pragma once
#include "interface/interface.h"

// Toeplitz Solver that can solve multiple right hand sides at once.
//
// Input:
// aToeplitz - 1D complex array that corresponds to the first row in lefthand toeplitz matrix
// BRighthand - 2D complex array where each column is a right hand side vector
//
// author: Kristian Timm Andersen

// ----------------------------- Cubic interpolation ------------------------------------------------------

struct SolverToeplitzConfiguration
{
    struct Input
    {
        I::Complex aToeplitz;
        I::Complex2D BRighthand;
    };

    using Output = O::Complex2D;

    struct Coefficients
    {
        int nRHS = 8; // number of right-hand sides to solve (columns in BRighthand)
        DEFINE_TUNABLE_COEFFICIENTS(nRHS)
    };

    struct Parameters
    {
        DEFINE_NO_TUNABLE_PARAMETERS
    };

    static std::tuple<Eigen::ArrayXcf, Eigen::ArrayXXcf> initInput(const Coefficients &c)
    {
        int dimension = 8;                                              // dimension of toeplitz matrix is arbitrary
        Eigen::ArrayXcf aToeplitz = Eigen::ArrayXcf::Random(dimension); // first row in A toeplitz matrix: A X = B
        aToeplitz(0) = 1.f;
        Eigen::ArrayXXcf BRightHand = Eigen::ArrayXXcf::Random(dimension, c.nRHS); // B matrix: A X = B
        return std::make_tuple(aToeplitz, BRightHand);
    }

    static Eigen::ArrayXXcf initOutput(Input input, const Coefficients &c) { return Eigen::ArrayXXcf::Zero(input.BRighthand.rows(), c.nRHS); } // X matrix: A X = B

    static bool validInput(Input input, const Coefficients &c)
    {
        bool flag = (input.aToeplitz.size() > 0) && input.aToeplitz.allFinite();
        flag &= (input.BRighthand.rows() == input.aToeplitz.size()) && (input.BRighthand.cols() == c.nRHS) && input.BRighthand.allFinite();
        return flag;
    }

    static bool validOutput(Output output, const Coefficients &c)
    {
        return (output.rows() > 0) && (output.cols() == c.nRHS) && output.allFinite();
        ;
    }
};

// Solve a complex Toeplitz system for multiple right-hand sides using
// the matrix first row rather than storing the full matrix.
class SolverToeplitz : public Algorithm<SolverToeplitzConfiguration>
{
  public:
    SolverToeplitz() = default;
    SolverToeplitz(const Coefficients &c);
};
