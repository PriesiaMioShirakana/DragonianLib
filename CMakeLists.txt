cmake_minimum_required (VERSION 3.13)

#Version
set(DRAGONIANLIB_VERSION_MAJOR 1)
set(DRAGONIANLIB_VERSION_MINOR 4)
set(DRAGONIANLIB_VERSION_PATCH 0)

#Global Visual Studio Macros
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DUNICODE)
add_definitions(-D_UNICODE)
add_definitions(-D_SILENCE_ALL_CXX23_DEPRECATION_WARNINGS)
add_definitions(-DNOMINMAX)

#DragonianLib Build Options
option(DRAGONIANLIB_DML									"dml backend"									ON)
option(DRAGONIANLIB_CUDA								"cuda backend"									ON)
option(DRAGONIANLIB_ROCM								"rocm backend"									OFF)
option(DRAGONIANLIB_VULKAN								"vk backend"									OFF)

option(DRAGONIANLIB_ORT_SUPER_RESOLUTION				"build ort super resolution libs"				ON)
option(DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION		"build ort singing voice conversion libs"		ON)
option(DRAGONIANLIB_ORT_TEXT_TO_SPEECH					"build ort text to speech libs"					ON)
option(DRAGONIANLIB_ORT_MUSIC_TRANSCRIPTION			"build ort music transcription libs"			ON)
option(DRAGONIANLIB_ORT_DEMIX							"build ort demix libs"							ON)
option(DRAGONIANLIB_ORT_F0_EXTRACTOR					"build ort f0 extractor libs"					ON)
option(DRAGONIANLIB_ORT_CONTEXT							"build ort context extractor libs"				ON)
option(DRAGONIANLIB_ORT_G2P								"build ort g2p libs"							ON)
option(DRAGONIANLIB_ORT_UNIT_ENCODER					"build ort unit extractor libs"					ON)
option(DRAGONIANLIB_ORT_VOCODER							"build ort vocoder libs"						ON)

option(DRAGONIANLIB_TRT_SUPER_RESOLUTION				"build trt super resolution libs"				OFF)
option(DRAGONIANLIB_TRT_SINGING_VOICE_CONVERSION		"build trt singing voice conversion libs"		OFF)
option(DRAGONIANLIB_TRT_TEXT_TO_SPEECH					"build trt text to speech libs"					OFF)
option(DRAGONIANLIB_TRT_MUSIC_TRANSCRIPTION			"build trt music transcription libs"			OFF)

option(DRAGONIANLIB_NCNN_SUPER_RESOLUTION				"build ncnn super resolution libs"				ON)
option(DRAGONIANLIB_NCNN_SINGING_VOICE_CONVERSION		"build ncnn singing voice conversion libs"		ON)
option(DRAGONIANLIB_NCNN_TEXT_TO_SPEECH				"build ncnn text to speech libs"				ON)
option(DRAGONIANLIB_NCNN_MUSIC_TRANSCRIPTION			"build ncnn music transcription libs"			ON)

option(DRAGONIANLIB_STATIC_FFMPEG						"use static ffmpeg"								ON)
option(DRAGONIANLIB_SHARED_LIBS							"build shared libs"								OFF)
option(DRAGONIANLIB_BUILD_DEMO							"build demo"									ON)

set(DRAGONIANLIB_ONNXRUNTIME OFF)
if(DRAGONIANLIB_ORT_SUPER_RESOLUTION OR 
	DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION OR 
	DRAGONIANLIB_ORT_TEXT_TO_SPEECH OR 
	DRAGONIANLIB_ORT_MUSIC_TRANSCRIPTION OR
	DRAGONIANLIB_ORT_DEMIX OR 
	DRAGONIANLIB_ORT_F0_EXTRACTOR OR 
	DRAGONIANLIB_ORT_CONTEXT OR 
	DRAGONIANLIB_ORT_G2P OR
	DRAGONIANLIB_ORT_UNIT_ENCODER OR 
	DRAGONIANLIB_ORT_VOCODER)
	set(DRAGONIANLIB_ONNXRUNTIME ON)
endif()

if (DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION OR
	DRAGONIANLIB_ORT_TEXT_TO_SPEECH)
	set(DRAGONIANLIB_ORT_UNIT_ENCODER ON)
	set(DRAGONIANLIB_ORT_VOCODER ON)
endif()

if (DRAGONIANLIB_ORT_TEXT_TO_SPEECH)
	set(DRAGONIANLIB_ORT_G2P ON)
	set(DRAGONIANLIB_ORT_CONTEXT ON)
endif()

if (DRAGONIANLIB_ORT_SINGING_VOICE_CONVERSION)
	set(DRAGONIANLIB_ORT_F0_EXTRACTOR ON)
endif()

set (DRAGONIANLIB_TENSORRT OFF)
if(DRAGONIANLIB_TRT_SUPER_RESOLUTION OR
	DRAGONIANLIB_TRT_SINGING_VOICE_CONVERSION OR
	DRAGONIANLIB_TRT_TEXT_TO_SPEECH OR
	DRAGONIANLIB_TRT_MUSIC_TRANSCRIPTION)
	set(DRAGONIANLIB_TENSORRT ON)
endif()

set (DRAGONIANLIB_NCNN OFF)
if(DRAGONIANLIB_NCNN_SUPER_RESOLUTION OR
	DRAGONIANLIB_NCNN_SINGING_VOICE_CONVERSION OR
	DRAGONIANLIB_NCNN_TEXT_TO_SPEECH OR
	DRAGONIANLIB_NCNN_MUSIC_TRANSCRIPTION)
	set(DRAGONIANLIB_NCNN ON)
endif()	

if(DRAGONIANLIB_TENSORRT)
	set(DRAGONIANLIB_CUDA ON)
endif()

if(DRAGONIANLIB_NCNN)
	set(DRAGONIANLIB_VULKAN ON)
endif()

#DragonianLib Configs
add_definitions(-DDRAGONIANLIB_ALLOC_ALIG=32)
add_definitions(-DDRAGONIANLIB_ALIG_DIM_SHAPE=8)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_FRONT=8)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_BACK=32)
add_definitions(-DDRAGONIANLIB_EMPTY_CAPACITY=16)
add_definitions(-DDRAGONIANLIB_PADDING_COUNT=64000)
add_definitions(-DDRAGONIANLIB_CONT_THRESHOLD_MIN_SIZE=8192)
add_definitions(-DDRAGONIANLIB_NAME_MAX_SIZE=1024)

#Global Config
set(BUILD_TESTING OFF)
set(BUILD_BENCHMARKS OFF)

#Project Name
set(DRAGONIANLIB_LIB DragonianLib)

#Languages
set(DRAGONIANLIB_LANGUAGES CXX ASM_MASM)

#Execusion Providers
if(DRAGONIANLIB_CUDA)
	add_definitions(-DDRAGONIANLIB_ENABLECUDA)
	list(APPEND DRAGONIANLIB_LANGUAGES CUDA)
	set(CMAKE_CUDA_ARCHITECTURES all)
endif()
if (DRAGONIANLIB_ROCM)
	add_definitions(-DDRAGONIANLIB_ENABLEROCM)
	#list(APPEND DRAGONIANLIB_LANGUAGES HIP)
endif ()
if (DRAGONIANLIB_DML)
	add_definitions(-DDRAGONIANLIB_ENABLEDML)
endif ()

#Project
project (
	${DRAGONIANLIB_LIB}
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	LANGUAGES ${DRAGONIANLIB_LANGUAGES}
)

add_definitions(-D__DRAGONIANLIB_SOURCE_DIRECTORY="${PROJECT_SOURCE_DIR}")

#Set FFMPEG Libraries And Include Directories
if(NOT DEFINED FFMPEG_LIBRARIES)
	set(FFMPEG_LIBRARIES)
	if(DRAGONIANLIB_STATIC_FFMPEG)
		list(
			APPEND FFMPEG_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavcodec.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavdevice.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavfilter.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavformat.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libavutil.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libswresample.a
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/lib/libswscale.a
		)
		if(WIN32)
			list(
				APPEND FFMPEG_LIBRARIES
				dxva2.lib
				evr.lib
				mf.lib
				mfplat.lib
				mfplay.lib
				mfreadwrite.lib
				mfuuid.lib
				bcrypt.lib
				ws2_32.lib
				Secur32.lib
				Strmiids.lib
			)
		endif()
	else()
		list(
			APPEND FFMPEG_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avcodec.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avformat.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/avutil.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/swresample.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/bin/swscale.lib
		)
	endif()
endif()
if(NOT DEFINED FFMPEG_INCLUDE_DIRS)
	set(FFMPEG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ffmpeg/output/include)
endif()

#Find OpenBLAS Libraries And Include Directories
find_package(OpenBLAS)
if(NOT OpenBLAS_FOUND)
	if(NOT DEFINED OpenBLAS_LIBRARIES)
		set(OpenBLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/lib/openblas.lib)
	endif()
	if(NOT DEFINED OpenBLAS_INCLUDE_DIRS)
		set(OpenBLAS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/OpenBLAS/out/install/x64-Release/include/openblas)
	endif()
endif()

#Set CUDA Libraries And Include Directories
if(DRAGONIANLIB_CUDA)
	find_package(CUDAToolkit REQUIRED)
	set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
	set(CUDA_LIBRARIES 
		CUDA::cudart
		CUDA::cublas
		CUDA::cufft)
endif()

#Find LibreMidi Libraries And Include Directories
find_package(libremidi)
if(NOT libremidi_FOUND)
	if(NOT DEFINED LIBREMIDI_LIBRARIES)
		if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
			set(LIBREMIDI_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/out/install/x64-Debug/lib/libremidi.lib)
		else()
			set(LIBREMIDI_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/out/install/x64-Release/lib/libremidi.lib)
		endif()
	endif()
	if(NOT DEFINED LIBREMIDI_INCLUDE_DIRS)
		set(LIBREMIDI_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/libmidi/out/install/x64-Release/include)
	endif()
endif()

#Find FFTW Libraries And Include Directories
find_package(FFTW3)
if(NOT FFTW3_FOUND)
	if(NOT DEFINED FFTW3_LIBRARIES)
		set(FFTW3_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/fftw/out/install/x64-Release/lib/fftw3.lib)
	endif()
	if(NOT DEFINED FFTW3_INCLUDE_DIRS)
		set(FFTW3_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/fftw/out/install/x64-Release/include)
	endif()
endif()

#Find World Libraries And Include Directories
find_package(world)
if(NOT world_FOUND)
	if(NOT DEFINED WORLD_LIBRARIES)
		set(WORLD_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/out/install/x64-Release/lib/world.lib)
	endif()
	if(NOT DEFINED WORLD_INCLUDE_DIRS)
		set(WORLD_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/World/out/install/x64-Release/include)
	endif()
endif()

#Find Faiss Libraries And Include Directories
find_package(faiss)
if(NOT faiss_FOUND)
	if(NOT DEFINED FAISS_LIBRARIES)
		if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
			set(FAISS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Debug/lib/faiss_avx2.lib)
		else()
			set(FAISS_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Release/lib/faiss_avx2.lib)
		endif()
	endif()
	if(NOT DEFINED FAISS_INCLUDE_DIRS)
		if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
			set(FAISS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Debug/include)
		else()
			set(FAISS_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/faiss/out/install/x64-Release/include)
		endif()
	endif()
endif()

find_package(yyjson)
if(NOT yyjson_FOUND)
	if(NOT DEFINED YYJSON_LIBRARIES)
		set(YYJSON_LIBRARIES ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/yyjson/out/install/x64-Release/lib/yyjson.lib)
	endif()
	if(NOT DEFINED YYJSON_INCLUDE_DIRS)
		set(YYJSON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/yyjson/out/install/x64-Release/include)
	endif()
endif()

if(DRAGONIANLIB_NCNN)
	find_package(ncnn)
	if(NOT ncnn_FOUND)
		if(NOT DEFINED NCNN_LIBRARIES)
			if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
				set(
					NCNN_LIBRARIES 
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/GenericCodeGend.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/glslangd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/glslang-default-resource-limitsd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/MachineIndependentd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/ncnnd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/OSDependentd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/OGLCompilerd.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Debug/lib/SPIRVd.lib
				)
			else()
				set(
					NCNN_LIBRARIES 
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/GenericCodeGen.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/glslang.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/glslang-default-resource-limits.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/MachineIndependent.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/ncnn.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/OSDependent.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/OGLCompiler.lib
					${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/lib/SPIRV.lib
				)
			endif()
		endif()
		if(NOT DEFINED NCNN_INCLUDE_DIRS)
			set(NCNN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ncnn/out/install/x64-Release/include)
		endif()
	endif()
endif()

#Set OnnxRuntime Libraries And Include Directories
if(DRAGONIANLIB_ONNXRUNTIME)
	if(NOT DEFINED ONNXRUNTIME_LIBRARIES)
		set(
			ONNXRUNTIME_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/RelWithDebInfo/RelWithDebInfo/onnxruntime_providers_shared.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/RelWithDebInfo/RelWithDebInfo/onnxruntime_providers_dml.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/RelWithDebInfo/RelWithDebInfo/onnxruntime_providers_cuda.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/build/Windows/RelWithDebInfo/RelWithDebInfo/onnxruntime.lib
		)
	endif()
	if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIRS)
		set(
			ONNXRUNTIME_INCLUDE_DIRS
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/ortmain/include/onnxruntime/core/session
		)
	endif()
endif()

#Set TensorRT Libraries And Include Directories
if(DRAGONIANLIB_TENSORRT)
	if(NOT DEFINED TENSORRT_LIBRARIES)
		set(
			TENSORRT_LIBRARIES
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_dispatch_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_lean_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_plugin_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvinfer_vc_plugin_10.lib
			${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/lib/nvonnxparser_10.lib
		)
	endif()
	if(NOT DEFINED TENSORRT_INCLUDE_DIRS)
		set(TENSORRT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/Third-Party-Libraries/TensorRT/include)
	endif()
endif()

message(">--------------------------------------DragonianLib--------------------------------------<")
message("")

#Output Directories
if(DRAGONIANLIB_SHARED_LIBS)
	set(DRAGONIANLIB_TYPE_DIR "Shared")
else()
	set(DRAGONIANLIB_TYPE_DIR "Static")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Archive)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/Runtime)
set(LIBRARY_OUTPUT_PATH             ${PROJECT_SOURCE_DIR}/OutPuts/${DRAGONIANLIB_TYPE_DIR}/${CMAKE_BUILD_TYPE}/OLibrary)

if(MSVC)
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
	endif()
endif()

set(DRAGONIANLIB_LIBRARIES_SRC)
file(GLOB_RECURSE DRAGONIANLIB_LIBRARIES_SRC 
	${PROJECT_SOURCE_DIR}/Libraries/*.cpp
	${PROJECT_SOURCE_DIR}/Libraries/*.h
	${PROJECT_SOURCE_DIR}/Libraries/*.c
	${PROJECT_SOURCE_DIR}/Libraries/*.hpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.hpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.h
	${PROJECT_SOURCE_DIR}/TensorLib/*.cpp
	${PROJECT_SOURCE_DIR}/TensorLib/*.c
	${PROJECT_SOURCE_DIR}/TensorLib/*.ixx
	#${PROJECT_SOURCE_DIR}/Libraries/*.asm
)

#Source code
if(DRAGONIANLIB_SHARED_LIBS)
	add_definitions(-DDRAGONIANLIB_EXPORT)
	set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()
add_library(${DRAGONIANLIB_LIB} STATIC ${DRAGONIANLIB_LIBRARIES_SRC})

message("Project Name: "${DRAGONIANLIB_LIB})

#Build Type
if(MSVC)
	set(DRAGONIANLIB_LIB_ARGUMENT_RELEASE
		/GS- /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /bigobj /std:c++latest)
	set(DRAGONIANLIB_LIB_ARGUMENT_RELEASE_DEB
		/Gy  /O2 /Ot /Oi /Ob3 /Zc:inline /arch:AVX2 /experimental:module /openmp:experimental /Zi /bigobj /std:c++latest)
	set(DRAGONIANLIB_LIB_ARGUMENT_DEBUG
		/Zi /Gy /experimental:module /openmp:experimental /bigobj /std:c++latest)
else()
	set(DRAGONIANLIB_LIB_ARGUMENT_RELEASE
		-O3 -std=c++23)
	set(DRAGONIANLIB_LIB_ARGUMENT_RELEASE_DEB
		-O3 -std=c++23)
	set(DRAGONIANLIB_LIB_ARGUMENT_DEBUG
		-g -O0 -std=c++23)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
	message("Build Type: \"Release\"")
	target_compile_options(${DRAGONIANLIB_LIB} PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE})
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	message("Build Type: \"RelWithDebInfo\"")
	target_compile_options(${DRAGONIANLIB_LIB} PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_RELEASE_DEB})
else()
	message("Build Type: \"Debug\"")
	add_definitions(-DDRAGONIANLIB_DEBUG)
	add_definitions(-DDRAGONIANLIB_DEBUG_MACRO)
	target_compile_options(${DRAGONIANLIB_LIB} PRIVATE ${DRAGONIANLIB_LIB_ARGUMENT_DEBUG})
endif()

#Link Libraries
if(DEFINED FFMPEG_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FFMPEG_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FFMPEG_INCLUDE_DIRS})
else()
	message(FATAL_ERROR "FFMPEG Libraries Not Found")
endif()

if(WIN32)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC winmm.lib)
endif()

if(DEFINED LIBREMIDI_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${LIBREMIDI_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${LIBREMIDI_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC libremidi)
	get_target_property(LIBREMIDI_INCLUDE_DIRS libremidi INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT LIBREMIDI_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for libremidi")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${LIBREMIDI_INCLUDE_DIRS})
endif()

if(DEFINED FFTW3_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FFTW3_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FFTW3_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC fftw3)
	get_target_property(FFTW3_INCLUDE_DIRS fftw3 INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT FFTW3_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for fftw3")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FFTW3_INCLUDE_DIRS})
endif()

if(DEFINED WORLD_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${WORLD_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${WORLD_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC world)
	get_target_property(WORLD_INCLUDE_DIRS world INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT WORLD_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for world")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${WORLD_INCLUDE_DIRS})
endif()

if(DEFINED FAISS_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${FAISS_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FAISS_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC faiss_avx2)
	get_target_property(FAISS_INCLUDE_DIRS faiss_avx2 INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT FAISS_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for faiss_avx2")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${FAISS_INCLUDE_DIRS})
endif()

if(DEFINED OpenBLAS_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${OpenBLAS_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC OpenBLAS::OpenBLAS)
	get_target_property(OpenBLAS_INCLUDE_DIRS OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT OpenBLAS_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for OpenBLAS")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${OpenBLAS_INCLUDE_DIRS})
endif()

if(DEFINED YYJSON_LIBRARIES)
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${YYJSON_LIBRARIES})
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${YYJSON_INCLUDE_DIRS})
else()
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC yyjson::yyjson)
	get_target_property(YYJSON_INCLUDE_DIRS yyjson::yyjson INTERFACE_INCLUDE_DIRECTORIES)
	if(NOT YYJSON_INCLUDE_DIRS)
		message(FATAL_ERROR "Failed to get INTERFACE_INCLUDE_DIRECTORIES for yyjson")
	endif()
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${YYJSON_INCLUDE_DIRS})
endif()

if(DRAGONIANLIB_CUDA)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${CUDA_INCLUDE_DIRS})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${CUDA_LIBRARIES})
	set_target_properties(${DRAGONIANLIB_LIB} PROPERTIES
		CUDA_STANDARD 20
		CUDA_STANDARD_REQUIRED ON
		CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
		CUDA_SEPARABLE_COMPILATION OFF
	)
endif()

#project include
set(DRAGONIANLIB_BASE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR})
target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${PROJECT_SOURCE_DIR})

if(DRAGONIANLIB_ONNXRUNTIME)
	message("")
	message("OnnxRuntime Libraries: Enabled")
	add_definitions(-DDRAGONIANLIB_ONNXRT_LIB)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${ONNXRUNTIME_LIBRARIES})
	add_subdirectory(OnnxLibrary)
	set(DRAGONIANLIB_ORT_LIB ${DRAGONIANLIB_LIB}OrtLib)
endif()

if(DRAGONIANLIB_TENSORRT)
	message("")
	message("TensorRT Libraries: Enabled")
	add_definitions(-DDRAGONIANLIB_TENSORRT_LIB)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${TENSORRT_INCLUDE_DIRS})
	target_link_libraries(${DRAGONIANLIB_LIB} PUBLIC ${TENSORRT_LIBRARIES})
	add_subdirectory(TensorRT)
	set(DRAGONIANLIB_TRT_LIB ${DRAGONIANLIB_LIB}TRTLib)
endif()

if(DRAGONIANLIB_NCNN)
	message("")
	message("NCNN Libraries: Enabled")
	add_definitions(-DDRAGONIANLIB_NCNN_LIB)
	target_include_directories(${DRAGONIANLIB_LIB} PRIVATE ${NCNN_INCLUDE_DIRS})
	target_link_directories(${DRAGONIANLIB_LIB} PRIVATE ${NCNN_LIBRARIES})
	add_subdirectory(NCNNLibrary)
	set(DRAGONIANLIB_NCNN_LIB ${DRAGONIANLIB_LIB}NCNNLib)
endif()

message("")

#Natvis File
set(DragonianLibNatvisFile ${PROJECT_SOURCE_DIR}/NatvisFile.natvis)
if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
else()
	add_link_options(
		/NATVIS:${DragonianLibNatvisFile}
	)
endif()

set(DRAGONIANLIB_LIBRARIES ${DRAGONIANLIB_LIB})
set(DRAGONIANLIB_INCLUDE_DIRS ${PROJECT_SOURCE_DIR})

set(DRAGONIANLIB_ONNX_LIBRARIES ${DRAGONIANLIB_ORT_LIB})
set(DRAGONIANLIB_ONNX_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIRS})

set(DRAGONIANLIB_TENSORRT_LIBRARIES ${DRAGONIANLIB_TRT_LIB})
set(DRAGONIANLIB_TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIRS})

set(DRAGONIANLIB_NCNN_LIBRARIES ${DRAGONIANLIB_NCNN_LIB})
set(DRAGONIANLIB_NCNN_INCLUDE_DIRS ${NCNN_INCLUDE_DIRS})

#Install Target
install(
	TARGETS ${DRAGONIANLIB_LIB}
	EXPORT ${DRAGONIANLIB_LIB}Targets
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin
	INCLUDES DESTINATION include
)

#Install Header File
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/Libraries/
	DESTINATION include/DragonianLib/Libraries
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/OnnxLibrary/
	DESTINATION include/DragonianLib/OnnxLibrary
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/TensorRT/
	DESTINATION include/DragonianLib/TensorRT
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/TensorLib/
	DESTINATION include/DragonianLib/TensorLib
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/NCNNLibrary/
	DESTINATION include/DragonianLib/NCNNLibrary
	FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
	PATTERN "*/" EXCLUDE
)

#Export Subprojects
install(
	EXPORT ${DRAGONIANLIB_LIB}Targets
	FILE ${DRAGONIANLIB_LIB}Targets.cmake
	NAMESPACE ${DRAGONIANLIB_LIB}::
	DESTINATION lib/cmake/${DRAGONIANLIB_LIB}
)

if(DRAGONIANLIB_ONNXRUNTIME)
	install(
		EXPORT ${DRAGONIANLIB_ORT_LIB}Targets
		FILE ${DRAGONIANLIB_ORT_LIB}Targets.cmake
		NAMESPACE ${DRAGONIANLIB_LIB}::
		DESTINATION lib/cmake/${DRAGONIANLIB_ORT_LIB}
	)
endif()

if(DRAGONIANLIB_TENSORRT)
	install(
		EXPORT ${DRAGONIANLIB_TRT_LIB}Targets
		FILE ${DRAGONIANLIB_TRT_LIB}Targets.cmake
		NAMESPACE ${DRAGONIANLIB_LIB}::
		DESTINATION lib/cmake/${DRAGONIANLIB_TRT_LIB}
	)
endif()

if(DRAGONIANLIB_NCNN)
	install(
		EXPORT ${DRAGONIANLIB_NCNN_LIB}Targets
		FILE ${DRAGONIANLIB_NCNN_LIB}Targets.cmake
		NAMESPACE ${DRAGONIANLIB_LIB}::
		DESTINATION lib/cmake/${DRAGONIANLIB_NCNN_LIB}
	)
endif()

#Config Files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}ConfigVersion.cmake
	VERSION ${DRAGONIANLIB_VERSION_MAJOR}.${DRAGONIANLIB_VERSION_MINOR}.${DRAGONIANLIB_VERSION_PATCH}
	COMPATIBILITY AnyNewerVersion
)
configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}Config.cmake
	COPYONLY
)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${DRAGONIANLIB_LIB}ConfigVersion.cmake
	DESTINATION lib/cmake/${DRAGONIANLIB_LIB}
)

if(DRAGONIANLIB_ONNXRUNTIME)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/OnnxLibrary/${DRAGONIANLIB_ORT_LIB}Config.cmake
		${CMAKE_CURRENT_BINARY_DIR}/OnnxLibrary/${DRAGONIANLIB_ORT_LIB}ConfigVersion.cmake
		DESTINATION lib/cmake/${DRAGONIANLIB_ORT_LIB}
	)
endif()

if(DRAGONIANLIB_TENSORRT)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/TensorRT/${DRAGONIANLIB_TRT_LIB}Config.cmake
		${CMAKE_CURRENT_BINARY_DIR}/TensorRT/${DRAGONIANLIB_TRT_LIB}ConfigVersion.cmake
		DESTINATION lib/cmake/${DRAGONIANLIB_TRT_LIB}
	)
endif()

if(DRAGONIANLIB_NCNN)
	install(FILES
		${CMAKE_CURRENT_BINARY_DIR}/NCNNLibrary/${DRAGONIANLIB_NCNN_LIB}Config.cmake
		${CMAKE_CURRENT_BINARY_DIR}/NCNNLibrary/${DRAGONIANLIB_NCNN_LIB}ConfigVersion.cmake
		DESTINATION lib/cmake/${DRAGONIANLIB_NCNN_LIB}
	)
endif()

message(">--------------------------------------DragonianLib--------------------------------------<")
message("")

#Build Demo
if (DRAGONIANLIB_BUILD_DEMO)
	if(DRAGONIANLIB_ONNXRUNTIME)
		set(DRAGONIANLIBORTDEMO ON)
	endif()
	add_subdirectory(Demo)
endif()

