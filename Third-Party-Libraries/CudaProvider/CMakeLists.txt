cmake_minimum_required(VERSION 3.28)
project(CudaProvider LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE DLIB_CUDA_SOURCES
	"Provider/*.cpp" "Provider/*.cu" "Provider/*.c"
	"Provider/*.h" "Provider/*.hpp" "Provider/*.cuh"
	"Layers/*.cpp" "Layers/*.cu" "Layers/*.c"
	"Layers/*.h" "Layers/*.hpp" "Layers/*.cuh")

add_library(CudaProvider STATIC ${DLIB_CUDA_SOURCES})

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES all)

message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} /O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

set_target_properties(CudaProvider PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    # CUDA_RESOLVE_DEVICE_SYMBOLS ON 
)

target_link_libraries(CudaProvider PRIVATE 
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
)

target_include_directories(CudaProvider PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(CudaProvider PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_definitions(CudaProvider PRIVATE 
    VERSION_INFO="1.0.0"
)

#Set library output directory

add_subdirectory(cuFCPE)