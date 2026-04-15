cmake_minimum_required(VERSION 3.16)

project(
  PythonAlgorithmLibrary
  VERSION 0.0.1
  DESCRIPTION "Python bindings for algorithm library"
  LANGUAGES CXX
)

# ---- Include guards ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "builds are not allowed in this folder. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# ---- Add dependencies via CPM ----

# set folder for 3rd party libraries
set(ENV{CPM_SOURCE_CACHE} ${PROJECT_SOURCE_DIR}/../../extern)
include(../../cmake/CPM.cmake)

# add algorithm_library
cpmaddpackage(NAME AlgorithmLibrary SOURCE_DIR ${PROJECT_SOURCE_DIR}/../.. OPTIONS ALGORITHM_LIBRARY_POSITION_INDEPENDENT_CODE # build using -fPIC flag
)

# add pybind11 library
cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG origin/master)

# add pybind11_json library
cpmaddpackage(NAME pybind11_json GITHUB_REPOSITORY pybind/pybind11_json GIT_TAG origin/master)

# add pfr non-boost library
cpmaddpackage(NAME pfr GITHUB_REPOSITORY apolukhin/pfr_non_boost GIT_TAG origin/master)

find_package(
  Python
  COMPONENTS Interpreter Development
  REQUIRED
)

# This line configures CMake to perform an optimized release build by default unless another build type is specified. Without this addition, binding code may run slowly and
# produce large binaries.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE
  )
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# NOTE: it seems this line fails to create build files if build files have already been created. If this gives problems, try to delete build folder and run cmake again
pybind11_add_module(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cpp")

# We enforce standards conformance on MSVC both compiler and  preprocessor: https://learn.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview?view=msvc-160
# The preproccessor is needed for macros to be interpreted correctly. Older versions of MSVC might not support these flags. The solution should be to update MSVC (should we
# print error?)
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive-; /Zc:preprocessor>")

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/../../include)

# With MSVC, enable natvis files to improve debugging experience and increase warning level
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /W4 -wd4127 -wd4100 -wd4459") # suppress warnings from 3rd party libraries - "C4100": unreferenced formal parameter,
                                                                                 # "C4127": conditional expression is constant, "C4459": Suppress 'declaration of 'var' hides
                                                                                 # global declaration'
else() # on GCC increase warning level
# removed -Wextra flag since it resulted in a "maybe-uninitialized" error from pybind11 in release mode using gcc 13.3.0
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-unused-parameter -ffast-math") # "no-unused-parameter": suppress unused parameter warning from 3rd party
                                                                                                    # libraries
endif()

# All users of this library will need at least C++17
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

target_link_libraries(${PROJECT_NAME} PUBLIC AlgorithmLibrary pybind11_json pfr)

add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}/python_script/
)
