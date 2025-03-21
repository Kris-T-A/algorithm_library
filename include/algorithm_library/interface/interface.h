#pragma once
#define EIGEN_DENSEBASE_PLUGIN                                                                                                                                                \
    "algorithm_library/interface/get_dynamic_memory_size.h" //  member function added to Eigen DenseBase class to get dynamic memory size of array and matrices
#define EIGEN_MPL2_ONLY                                     // don't allow LGPL licensed code from Eigen
#include "algorithm_library/interface/constrained_type.h"   // ConstrainedType class
#include "algorithm_library/interface/helper_functions.h"
#include "algorithm_library/interface/input_output.h"     // Input/Output structs.
#include "algorithm_library/interface/macros_json.h"      // macros for coefficients and parameters
#include "algorithm_library/interface/public_algorithm.h" // Base Algorithm class
#include <Eigen/Dense>                                    // Eigen Library.
#include <Eigen/IterativeLinearSolvers>


// This is the main interface file that includes the necessary files for the public interface
//
// author: Kristian Timm Andersen, 2019