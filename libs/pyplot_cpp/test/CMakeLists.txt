cmake_minimum_required(VERSION 3.16)

# set project name and version
project(
  PyPlotCPP_Test
  VERSION 0.0.1
  DESCRIPTION "Tests of PyPlot_CPP library"
  LANGUAGES CXX
)

# ---- Include guards ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "builds are not allowed in this folder. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# ---- Add dependencies via CPM ----

# set folder for 3rd party libraries
set(ENV{CPM_SOURCE_CACHE} ${PROJECT_SOURCE_DIR}/../../../extern)
include(../../../cmake/CPM.cmake)

# add Google test
cpmaddpackage(
  NAME
  googletest
  GITHUB_REPOSITORY
  google/googletest
  GIT_TAG
  release-1.12.1
  VERSION
  1.12.1
  OPTIONS
  "INSTALL_GTEST OFF"
  "gtest_force_shared_crt"
)

# add format library
cpmaddpackage(
  NAME
  fmt
  GIT_TAG
  9.0.0
  GITHUB_REPOSITORY
  fmtlib/fmt
  OPTIONS
  "FMT_INSTALL YES" # create an installable target
)

# add algorithm_library
cpmaddpackage(NAME PyPlotCPP SOURCE_DIR ${PROJECT_SOURCE_DIR}/..)

file(GLOB_RECURSE SOURCE_LIST_TEST_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/test.cpp")

add_executable(${PROJECT_NAME} ${SOURCE_LIST_TEST_CPP})

# We need to include this path in our search path
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../include ${PROJECT_SOURCE_DIR}/../src)

# The project depends on the following libraries
target_link_libraries(${PROJECT_NAME} PRIVATE PyPlotCPP gtest gtest_main fmt::fmt)

# All users of this library will need at least C++14
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
