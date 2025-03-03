cmake_minimum_required(VERSION 3.16)

# set project name and version
project(
  DSP_Benchmark
  VERSION 0.0.1
  DESCRIPTION "Benchmark of DSP algorithms"
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

# add Google benchmark
cpmaddpackage(
  NAME
  benchmark
  GITHUB_REPOSITORY
  google/benchmark
  VERSION
  1.7.1
  OPTIONS
  "BENCHMARK_ENABLE_TESTING Off"
)

# add algorithm_library
cpmaddpackage(NAME AlgorithmLibrary SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)

# add onnx runtime
cpmaddpackage( NAME ONNXRuntime URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz DOWNLOAD_ONLY YES )
add_library(onnxruntime SHARED IMPORTED)

if (MSVC) # TODO: MSVC properties are not set correctly
  set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB ${ONNXRuntime_SOURCE_DIR}/runtimes/win-x86/native/onnxruntime.lib)
  set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_SOURCE_DIR}/build/native/include)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRuntime_SOURCE_DIR}/runtimes/win-x86/native/onnxruntime.dll)
else()
  set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_SOURCE_DIR}/include)
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRuntime_SOURCE_DIR}/lib/libonnxruntime.so)
endif()

add_executable(${PROJECT_NAME} src/benchmark_algorithms.cpp)

if(NOT MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE "$<$<CONFIG:Release>:-ffast-math>")
endif()

# We need to include this path in our search path
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../include ${PROJECT_SOURCE_DIR}/../../src)

# The project depends on the following libraries
target_link_libraries(${PROJECT_NAME} PRIVATE AlgorithmLibrary benchmark::benchmark onnxruntime)

# All users of this library will need at least C++14
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_14)

# copy pp2model.onnx model to target dir
add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/../../libs/onnx/pp2model.onnx" $<TARGET_FILE_DIR:${PROJECT_NAME}> 
)

# copy onnxruntime to target dir
if (MSVC)
add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${AlgorithmLibrary_BINARY_DIR}/$<CONFIG>/onnxruntime.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}>  
)
else()
# get onnx file paths from AlgorithmLibrary target and copy files to target dir
get_target_property(ONNXRUNTIME_FILES AlgorithmLibrary ONNXRUNTIME_FILES)
add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_FILES} $<TARGET_FILE_DIR:${PROJECT_NAME}> 
)
endif()
