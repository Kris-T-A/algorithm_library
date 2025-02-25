cmake_minimum_required(VERSION 3.16)

# set project name and version
project(
  AlgorithmLibrary
  VERSION 0.0.1
  DESCRIPTION "Library of DSP algorithms"
  LANGUAGES CXX C
)

# ---- Add dependencies via CPM ----

# set folder for 3rd party libraries
set(ENV{CPM_SOURCE_CACHE} ${PROJECT_SOURCE_DIR}/extern)
include(cmake/CPM.cmake)

# add nlohmann json library
cpmaddpackage(
  NAME
  nlohmann_json
  VERSION
  3.11.3
  GITHUB_REPOSITORY
  nlohmann/json # ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  OPTIONS
  "JSON_BuildTests OFF"
  "JSON_Install ON"
)

# add Eigen library (CMakeLists.txt in this project is not suitable for including as subdirectory, and it's a header-only project so only download it)
cpmaddpackage(NAME Eigen URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip DOWNLOAD_ONLY YES)
add_library(Eigen INTERFACE IMPORTED)
target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})

# add clang and cmake formatting (requires clang-format and cmake-format is installed)
cpmaddpackage(
  NAME
  Format.cmake
  VERSION
  1.7.3
  GITHUB_REPOSITORY
  TheLartians/Format.cmake
  OPTIONS
  # set to yes skip cmake formatting
  "FORMAT_SKIP_CMAKE NO"
  # set to yes skip clang formatting
  "FORMAT_SKIP_CLANG NO"
  # path to exclude (optional, supports regular expressions)
  "CMAKE_FORMAT_EXCLUDE cmake/CPM.cmake"
  # extra arguments for cmake_format (optional)
  "CMAKE_FORMAT_EXTRA_ARGS -c .cmake-format"
)

add_library(onnxruntime SHARED IMPORTED)
if (MSVC) # TODO: MSVC properties are not set correctly
  cpmaddpackage( NAME ONNXRuntime URL https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/Microsoft.ML.OnnxRuntime.DirectML.1.19.2.nupkg  DOWNLOAD_ONLY YES )
  if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
        message(STATUS "64 bit")
        set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB ${ONNXRuntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.lib)
    else()
        message(STATUS "32 bit")
        set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB ${ONNXRuntime_SOURCE_DIR}/runtimes/win-x86/native/onnxruntime.lib)
    endif()
  set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_SOURCE_DIR}/build/native/include)
else()
  cpmaddpackage( NAME ONNXRuntime URL https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz DOWNLOAD_ONLY YES )
  set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${ONNXRuntime_SOURCE_DIR}/lib/libonnxruntime.so)
  set_target_properties(onnxruntime PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_SOURCE_DIR}/include)
endif()

# If we want the library to be SHARED, aka a DLL, then we have to tell cmake to export symbols for the applications that are using the DLL. Therefore, uncomment this line if
# add_library is SHARED set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

file(GLOB_RECURSE HEADER_LIST_ALGORITHMS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE SOURCE_LIST_CPP CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SOURCE_LIST_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE SOURCE_LIST_UTILITIES_C CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/utilities/*.c")
file(GLOB_RECURSE SOURCE_LIST_UTILITIES_HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/utilities/*.h")

# add SHARED if a DLL is needed (see CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS comment above about exporting symbols)
if(ALGORITHM_LIBRARY_SHARED_LIBRARY)
  add_library(${PROJECT_NAME} SHARED ${SOURCE_LIST_CPP} ${SOURCE_LIST_HEADERS} ${SOURCE_LIST_UTILITIES_C} ${SOURCE_LIST_UTILITIES_HEADERS} ${HEADER_LIST_ALGORITHMS})

  # Strip shared binary for release builds
  add_custom_command(
    TARGET "${PROJECT_NAME}"
    POST_BUILD
    COMMAND $<$<CONFIG:release>:${CMAKE_STRIP}> ARGS --strip-all $<TARGET_FILE:${PROJECT_NAME}>
  )
else()
  add_library(${PROJECT_NAME} ${SOURCE_LIST_CPP} ${SOURCE_LIST_HEADERS} ${SOURCE_LIST_UTILITIES_C} ${SOURCE_LIST_UTILITIES_HEADERS} ${HEADER_LIST_ALGORITHMS})
endif()

# We enforce standards conformance on MSVC both compiler and  preprocessor: https://learn.microsoft.com/en-us/cpp/preprocessor/preprocessor-experimental-overview?view=msvc-160
# The preproccessor is needed for macros to be interpreted correctly. Older versions of MSVC might not support these flags. The solution should be to update MSVC (should we
# print error?)
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive-; /Zc:preprocessor>")

# target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-DNDEBUG -O2 -ffast-math -fno-exceptions -s -ffunction-sections -fdata-sections -flto>)

# With MSVC, enforce C++20 and enable natvis files to improve debugging experience and increase warning level
if(MSVC)
  target_sources(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${Eigen_SOURCE_DIR}/debug/msvc/eigen.natvis>)
  # suppress warnings from 3rd party libraries - "C4100": unreferenced formal parameter, "C4127": conditional expression is constant, "C4459": Suppress 'declaration of 'var'
  # hides global declaration'
  target_compile_options(${PROJECT_NAME} PUBLIC /fp:fast /W4 -wd4127 -wd4100 -wd4459 /std:c++20) # we enforce C++20 for MSVC since it is needed for designated initializers lists. GCC and Clang already implements this in C++14 so there it is not needed.

else() # on GCC/CLANG increase warning level
  target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Werror -Wextra -Wno-unused-parameter) # "no-unused-parameter": suppress unused parameter warning
  target_compile_options(${PROJECT_NAME} PRIVATE -ffast-math)
endif()

# We need to include this path in our search path and users of this library also need it
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

# The project depends on the following libraries. They are PUBLIC since they are used in the headers so anyone using this library needs them as well
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen nlohmann_json::nlohmann_json onnxruntime)
#target_link_libraries(${PROJECT_NAME} PRIVATE onnxruntime)

# All users of this library will need at least C++14
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

# Let's ensure -std=c++xx instead of -std=g++xx
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CXX_EXTENSIONS OFF)

# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# IDEs should put the files in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Public Header Files"
  FILES ${HEADER_LIST_ALGORITHMS}
)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/src"
  PREFIX "Internal Files"
  FILES ${SOURCE_LIST_CPP} ${SOURCE_LIST_HEADERS} ${SOURCE_LIST_UTILITIES_C} ${SOURCE_LIST_UTILITIES_HEADERS}
)

# copy onnxruntime to target directory
if (MSVC)
    # if 32 bit: https://discourse.cmake.org/t/how-to-distinguish-windows-architecture/5104
    if(NOT "${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
        add_custom_command(
            TARGET ${PROJECT_NAME} 
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${ONNXRuntime_SOURCE_DIR}/runtimes/win-x86/native/onnxruntime.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}> 
        )
    else() # 64 bit
        add_custom_command(
            TARGET ${PROJECT_NAME} 
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy "${ONNXRuntime_SOURCE_DIR}/runtimes/win-x64/native/onnxruntime.dll" $<TARGET_FILE_DIR:${PROJECT_NAME}> 
        )
    endif()
else()
file(GLOB ONNXRUNTIME_FILES CONFIGURE_DEPENDS "${ONNXRuntime_SOURCE_DIR}/lib/libonnxruntime*")
add_custom_command(
  TARGET ${PROJECT_NAME} 
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_FILES} $<TARGET_FILE_DIR:${PROJECT_NAME}> 
)
# save the paths of onnx files for the projects so it can be imported by other projects
set_target_properties(${PROJECT_NAME} PROPERTIES ONNXRUNTIME_FILES "${ONNXRUNTIME_FILES}")
endif()