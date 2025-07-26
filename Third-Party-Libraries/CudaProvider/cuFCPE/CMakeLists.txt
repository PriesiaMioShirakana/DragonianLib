cmake_minimum_required(VERSION 3.28)
project(cuFCPE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
string(REGEX MATCH "^[0-9]+\\.[0-9]+" PYBIND11_PYTHON_VERSION "${Python3_VERSION}")

find_program(CONDA_EXECUTABLE NAMES conda conda.exe PATHS 
    "$ENV{CONDA_PREFIX}"
    "$ENV{CONDA_PREFIX}/bin"
    "$ENV{CONDA_PREFIX}/Scripts"
    "$ENV{USERPROFILE}/anaconda3"
    "$ENV{USERPROFILE}/anaconda3/bin"
    "$ENV{USERPROFILE}/anaconda3/Scripts"
    "C:/anaconda3"
    "C:/anaconda3/bin"
    "C:/anaconda3/Scripts"
    "/opt/anaconda3/bin"
    "/usr/local/anaconda3/bin"
    DOC "Path to conda executable")

if(CONDA_EXECUTABLE)
    get_filename_component(ANACONDA_PATH "${CONDA_EXECUTABLE}" DIRECTORY)
    get_filename_component(ANACONDA_PATH "${ANACONDA_PATH}" DIRECTORY)
    message(STATUS "ANACONDA_PATH: ${ANACONDA_PATH}")
else()
    message(WARNING "Could not find conda installation, trying system Python")
    set(ANACONDA_PATH "")
endif()

message(STATUS "PYBIND11_PYTHON_VERSION: ${PYBIND11_PYTHON_VERSION}")

if(WIN32 AND ANACONDA_PATH)
    set(pybind11_DIR "${ANACONDA_PATH}/Lib/site-packages/pybind11/share/cmake/pybind11")
else()
    set(pybind11_DIR "${ANACONDA_PATH}/lib/python${PYBIND11_PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11")
endif()

if(EXISTS "${pybind11_DIR}/pybind11Config.cmake")
    message(STATUS "Found pybind11Config.cmake at: ${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
else()
    message(STATUS "pybind11Config.cmake not found, trying alternative method")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    
    if(pybind11_RESULT EQUAL 0 AND EXISTS "${pybind11_DIR}")
        message(STATUS "Found pybind11 via Python at: ${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "Could not find pybind11. Please install it with: pip install pybind11")
    endif()
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES all)

file(GLOB_RECURSE CUFCPE_SOURCES
	"../Provider/*.cpp" "../Provider/*.cu"  "../Provider/*.c"
	"../Provider/*.h"   "../Provider/*.hpp" "../Provider/*.cuh"
	"../Layers/*.cpp"   "../Layers/*.cu"    "../Layers/*.c"
	"../Layers/*.h"     "../Layers/*.hpp"   "../Layers/*.cuh")

pybind11_add_module(
    cuFCPEPythonImp
    # C++ source files
    cuFCPE.cpp
    cuFCPEPythonImp.cpp
    ${CUFCPE_SOURCES}
    # CUDA source files
    FCPE.cu
)

# release: 使用O3
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "/O2")
else()
    # For GCC/Clang, use -O3
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
endif()

set_target_properties(cuFCPEPythonImp PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(cuFCPEPythonImp PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

target_include_directories(cuFCPEPythonImp PRIVATE 
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/../
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(cuFCPEPythonImp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_definitions(cuFCPEPythonImp PRIVATE 
    VERSION_INFO="1.0.0"
)

if(WIN32)
    set_target_properties(cuFCPEPythonImp PROPERTIES
        SUFFIX ".pyd"
    )
endif()