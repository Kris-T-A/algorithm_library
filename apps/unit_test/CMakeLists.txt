cmake_minimum_required(VERSION 3.16)

# set project name and version
project(
  DSP_Test
  VERSION 0.0.1
  DESCRIPTION "Tests of DSP algorithms"
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
cpmaddpackage(NAME AlgorithmLibrary SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)

file(GLOB_RECURSE SOURCE_LIST_TEST_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCE_LIST_TEST_CPP})

# We need to include this path in our search path
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../include ${PROJECT_SOURCE_DIR}/../../src)

# The project depends on the following libraries
target_link_libraries(${PROJECT_NAME} PRIVATE AlgorithmLibrary gtest gtest_main fmt::fmt)

# C++17 is required, not just preferred: nlohmann_json's from_json for std::vector of macro-generated
# MemberCoefficients structs (DEFINE_MEMBER_ALGORITHMS over a VectorAlgo<...> whose element also has
# member algorithms - e.g. SpectrogramSetWOLA -> VectorAlgo<SpectrogramNonlinear> -> VectorAlgo<Filterbank>)
# relies on copy elision to skip move-construct-then-destruct on intermediate temporaries. Under C++14
# at -O0 those moves and matching destructors actually run, and the implicit move constructor of the
# generated struct leaves the source's nested vector header in a half-cleared state, so the destructor
# segfaults walking _M_start..nullptr. C++17's mandatory copy elision skips that dance entirely.
# Repro: drop to cxx_std_14, clean rebuild, run SpectrogramAdaptive.InterfaceEnvelope or SpectrogramSet.Interface.
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# define this to detect dynamic memory allocation in debug mode
target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_RUNTIME_NO_MALLOC)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})

# copy model.onnx model to target dir
add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/../../libs/onnx/model.onnx" $<TARGET_FILE_DIR:${PROJECT_NAME}> 
)

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