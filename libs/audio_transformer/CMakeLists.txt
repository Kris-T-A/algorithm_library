cmake_minimum_required(VERSION 3.14)

# Check Emscripten environment early
if(NOT DEFINED ENV{EMSDK})
    message(FATAL_ERROR "EMSDK environment variable is not set. Please source emsdk_env.sh")
endif()

# Toolchain file must be set before project()
if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{EMSDK}/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"
        CACHE STRING "Emscripten toolchain file" FORCE)
endif()

project(audio_transformer)

# set folder for 3rd party libraries
set(ENV{CPM_SOURCE_CACHE} ${PROJECT_SOURCE_DIR}/../../extern)
include(${PROJECT_SOURCE_DIR}/../../cmake/CPM.cmake)

# add Eigen library (CMakeLists.txt in this project is not suitable for including as subdirectory, and it's a header-only project so only download it)
cpmaddpackage(NAME Eigen GITLAB_REPOSITORY libeigen/eigen GIT_TAG master DOWNLOAD_ONLY YES)
add_library(Eigen INTERFACE IMPORTED)
target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})

# add nlohmann json library
cpmaddpackage(
  NAME
  nlohmann_json
  VERSION
  3.11.3
  GITHUB_REPOSITORY
  nlohmann/json 
  OPTIONS
  "JSON_BuildTests OFF"
  "JSON_Install OFF"
)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure we're using Emscripten compiler
if (NOT DEFINED EMSCRIPTEN)
    message(FATAL_ERROR "This CMake file requires Emscripten. Please compile with: 
    mkdir build && cd build
    emcmake cmake ..
    emmake make")
endif()

## Set the target to WebAssembly
set(CMAKE_EXECUTABLE_SUFFIX ".js")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
endif()

# Add the source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "../../src/filterbank/filterbank.cpp" "../../src/utilities/pffft.c" "../../src/fft/fft.cpp" "../../src/perceptual_spectral_analysis/perceptual_spectral_analysis.cpp" "../../src/convert_rgba/convert_rgba.cpp")

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/../../include)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/../../src)
target_include_directories(${PROJECT_NAME} PRIVATE $ENV{EMSDK}/upstream/emscripten/cache/sysroot/include)

target_compile_options(${PROJECT_NAME} PRIVATE  
-msimd128 -msse4.2 -ffast-math -Wno-nan-infinity-disabled )

# The project depends on the following libraries
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen nlohmann_json::nlohmann_json)

# Set Emscripten link flags - Compatible with traditional loading
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS "-s WASM=1 \
                -s EXPORTED_RUNTIME_METHODS=['cwrap','HEAPF32','HEAP32'] \
                -s EXPORTED_FUNCTIONS=['_audio_attenuate','_audio_spectral_analysis','_create_audio_spectral_analysis','_audio_spectral_analysis_stateful','_destroy_audio_spectral_analysis','_get_n_frequency_bands','_convert_rgba','_scale_and_convert_rgba','_malloc','_free'] \
                -s NO_EXIT_RUNTIME=1 \
                -s INITIAL_MEMORY=67108864 \
                -s MAXIMUM_MEMORY=536870912 \
                -s ALLOW_MEMORY_GROWTH=1 \
                -s MODULARIZE=0 \
                --closure 1"
)

# Debug build with assertions
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS_DEBUG "-s ASSERTIONS=1"
)

# Release build without assertions
set_target_properties(${PROJECT_NAME} PROPERTIES
    LINK_FLAGS_RELEASE "-s ASSERTIONS=0"
)