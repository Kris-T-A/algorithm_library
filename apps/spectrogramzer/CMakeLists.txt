cmake_minimum_required(VERSION 3.16)

# set project name and version
project(
  Spectrogramzer
  VERSION 0.0.1
  DESCRIPTION "Application for creating spectrograms from audio files"
  LANGUAGES CXX
)

# ---- Include guards ----
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(FATAL_ERROR "builds are not allowed in this folder. Please make a new directory (called a build directory) and run CMake from there.")
endif()

# set folder for 3rd party libraries
set(ENV{CPM_SOURCE_CACHE} ${PROJECT_SOURCE_DIR}/../../extern)
include(../../cmake/CPM.cmake)

# add cxxopts library
cpmaddpackage(
  NAME
  cxxopts
  GITHUB_REPOSITORY
  jarro2783/cxxopts
  GIT_TAG
  v3.1.1
  OPTIONS
  "CXXOPTS_BUILD_EXAMPLES OFF"
  "CXXOPTS_BUILD_TESTS OFF"
  "CXXOPTS_ENABLE_INSTALL OFF"
)

# add audio file library
cpmaddpackage(
  NAME
  AudioFile
  GITHUB_REPOSITORY
  adamstark/AudioFile
  GIT_TAG
  1.1.3
  OPTIONS
  "BUILD_TESTS OFF"
  "BUILD_EXAMPLES OFF"
)

# add PyPlotCPP
cpmaddpackage(NAME PyPlotCPP SOURCE_DIR ${PROJECT_SOURCE_DIR}/../../libs/pyplot_cpp)

# add algorithm_library
cpmaddpackage(NAME AlgorithmLibrary SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)

add_executable(${PROJECT_NAME} src/main.cpp)

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Werror -Wextra -Wno-unused-parameter -fno-strict-aliasing) # "no-unused-parameter": suppress unused parameter warning, -fno-strict-aliasing is necessary in release mode as otherwise AudioFile library reports a warning (which is treated as errors)

# We need to include this path in our search path
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../include ${PROJECT_SOURCE_DIR}/../../src)

# The project depends on the following libraries
target_link_libraries(${PROJECT_NAME} PRIVATE AlgorithmLibrary cxxopts::cxxopts AudioFile PyPlotCPP)
