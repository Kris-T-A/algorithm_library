#pragma once
#include "algorithm_library/normal3d.h"
#include "framework/framework.h"

// Calculate the 3-dimensional normal vector of a plane for each point in the plane.
// The input is a 2-dimensional matrix containing the z-value of each point in a x/y grid
// The output is a 2-dimensional matrix, where the 1st output dimension is 3 times the size of the 1st input dimension to hold the interleaved 3-dimensional normal vector.
//
// This method calculates the normal vector by the difference between the neighbour points.
// Along the 1st dimension, the boundary conditions are assumed to be symmetric so the normal is 0 at the edge.
// Along the 2nd dimension, the input is assumed to be streamed, so there is a 1 sample delay in the output.
//
// author: Kristian Timm Andersen

// ----------------------------- Cubic interpolation of one sample -----------------------------------------

// Compute surface-normal components from neighboring height differences,
// normalize each vector, and stream the result with one column of delay.
class Normal3dDiff : public AlgorithmImplementation<Normal3dConfiguration, Normal3dDiff>
{
  public:
    Normal3dDiff(Coefficients c = Coefficients()) : BaseAlgorithm{c}
    {
        valuesOld0.resize(c.nValuesX);
        valuesOld1.resize(c.nValuesX);
        normals.resize(c.nValuesX, 3);
        normalsTransposed.resize(3, c.nValuesX);
        length.resize(c.nValuesX);
        den1 = 1.f / (2.f * P.distance1);
        den2 = 1.f / (2.f * P.distance2);
        resetVariables();
    }

  private:
    inline void processAlgorithm(Input input, Output output)
    {
        for (auto frame = 0; frame < input.cols(); frame++)
        {
            normals.col(0) << 0.f, den1 * (input.col(frame).tail(C.nValuesX - 2) - input.col(frame).head(C.nValuesX - 2)), 0.f; // 1st dimension
            normals.col(1) = den2 * (input.col(frame) - valuesOld0);                                                            // 2nd dimension
            normals.col(2) = Eigen::ArrayXf::Ones(C.nValuesX);                                                                  // Z direction
            length = (normals.col(0).abs2() + normals.col(1).abs2() + Eigen::ArrayXf::Ones(C.nValuesX)).sqrt();                 // calculate length
            normals.col(0) /= length; // normalize by length (profiled to be much faster than using .replicate())
            normals.col(1) /= length; // normalize by length (profiled to be much faster than using .replicate())
            normals.col(2) /= length; // normalize by length (profiled to be much faster than using .replicate())

            normalsTransposed = normals.transpose(); // saving temp is profiled to be faster than writing directly to output
            output.col(frame) = Eigen::Map<Eigen::ArrayXf>(normalsTransposed.data(), 3 * C.nValuesX, 1);

            valuesOld0 = valuesOld1;
            valuesOld1 = input.col(frame);
        }
    }

    void resetVariables() final
    {
        valuesOld0.setZero();
        valuesOld1.setZero();
    }

    size_t getDynamicSizeVariables() const final
    {
        size_t size = valuesOld0.getDynamicMemorySize();
        size += valuesOld1.getDynamicMemorySize();
        size += normals.getDynamicMemorySize();
        size += length.getDynamicMemorySize();
        size += normalsTransposed.getDynamicMemorySize();
        return size;
    }

    Eigen::ArrayXf valuesOld0;
    Eigen::ArrayXf valuesOld1;
    Eigen::ArrayXXf normals;
    Eigen::ArrayXf length;
    Eigen::ArrayXXf normalsTransposed;
    float den1, den2;

    friend BaseAlgorithm;
};
