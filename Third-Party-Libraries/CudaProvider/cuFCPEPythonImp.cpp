// ReSharper disable CppClangTidyClangDiagnosticCastFunctionTypeStrict
// ReSharper disable CppClangTidyClangDiagnosticMissingFieldInitializers
#include <codecvt>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

#include "cuFCPE.h"
#include "cuFCPEPythonImp.h"
#include "fcpe.h"
#include "npy.h"

using namespace DragonianLib::CudaModules::FCPE;

DragonianLib::PythonImplement::Processor::cuFCPEProcessor::cuFCPEProcessor(
    ModelConfig modelConfig,
    PreProcessConfig preProcessConfig
) : modelConfig(modelConfig), preProcessConfig(preProcessConfig)
{
    this->model = new Model(
        modelConfig.inputChannels,
        modelConfig.outputDims,
        modelConfig.hiddenDims,
        modelConfig.numLayers,
        modelConfig.numHeads,
        modelConfig.f0Max,
        modelConfig.f0Min,
        modelConfig.useFaNorm,
        modelConfig.convOnly,
        modelConfig.useHarmonicEmb
    );

    this->melKernel = new DragonianLib::CudaModules::MelKernel(
        preProcessConfig.samplingRate,
        preProcessConfig.melBins,
        preProcessConfig.fftLength,
        preProcessConfig.windowSize,
        preProcessConfig.hopSize,
        preProcessConfig.freqMin,
        preProcessConfig.freqMax,
        preProcessConfig.clipVal
    );

	this->model->LoadFromFile(modelConfig.modelPath);

    auto handle = DragonianLib::CudaModules::createHandle();
    auto stream = DragonianLib::CudaProvider::createCudaStream();
    this->stream = stream;

    setHandleStream(handle, stream);
    this->cacheTensors.res.Handle = handle;
    this->cacheTensors.res.Resize(this->batchSize, this->audioLength * preProcessConfig.samplingRate);
}

DragonianLib::PythonImplement::Processor::cuFCPEProcessor::~cuFCPEProcessor() {
    if (this->model != nullptr)
		delete this->model;
    if (this->melKernel != nullptr)
		delete this->melKernel;
    // TAP: Steam一释放就Segment Fault了, 可能object销毁之后Stream已经被释放了，但是防止意外的内存泄漏，此处注释一下
    // if (this->stream != nullptr) {
    //     delete this->stream;
    //     this->stream = nullptr;
    // }
}

template<typename ARRAY_TYPE>
py::array_t<ARRAY_TYPE> DragonianLib::PythonImplement::Processor::cuFCPEProcessor::excute(py::array_t<ARRAY_TYPE> inputData) {
	// inputData size [batchSize, audioLength * samplingRate]
    // 只支持float32类型，其他类型报错
    if(std::is_same<ARRAY_TYPE, float>::value == false) {
        throw std::runtime_error("Input data must be of type float32.");
	}
    if (inputData.ndim() != 2) {
        throw std::runtime_error("Input data must be a 2D array.");
    }

    auto inputShape = inputData.shape();
    if(inputShape[0] != this->batchSize || inputShape[1] != this->audioLength * this->preProcessConfig.samplingRate) {
        this->cacheTensors.res.Resize(
            static_cast<unsigned>(inputShape[0]),
            static_cast<unsigned>(inputShape[1])
        );
	}

    DragonianLib::CudaProvider::cpy2Device(
        this->cacheTensors.res.Data,
        (const DragonianLib::CudaModules::moduleValueType*)inputData.data(),
        inputData.size(),
        this->stream
    );

    auto timeBegin = std::chrono::high_resolution_clock::now();
    this->melKernel->Forward(this->cacheTensors.res, this->cacheTensors.input, this->cacheTensors.output);
    this->model->Infer(this->cacheTensors);
    auto output = this->cacheTensors.output.Cpy2Host(this->stream);
    //DragonianLib::CudaProvider::asyncCudaStream(this->cacheTensors.res.Handle->Stream);

    return py::array_t<ARRAY_TYPE>(
        {this->cacheTensors.output.N, this->cacheTensors.output.C, this->cacheTensors.output.H, this->cacheTensors.output.W},
        output.data()
	);
}

// creat cuFCPEProcessor from python
DragonianLib::PythonImplement::Processor::cuFCPEProcessor create_cuFCPEProcessor(
    DragonianLib::PythonImplement::Processor::ModelConfig modelConfig,
    DragonianLib::PythonImplement::Processor::PreProcessConfig preProcessConfig
) {
    return DragonianLib::PythonImplement::Processor::cuFCPEProcessor(modelConfig, preProcessConfig);
}

// Register the cuFCPEProcessor class and its methods in Python
PYBIND11_MODULE(cuFCPEPythonImp, m) {
    m.doc() = "cuFCPE for Python implement"; // optional module docstring
    py::class_<DragonianLib::PythonImplement::Processor::cuFCPEProcessor>(m, "cuFCPEProcessor")
        .def(py::init<DragonianLib::PythonImplement::Processor::ModelConfig, DragonianLib::PythonImplement::Processor::PreProcessConfig>())
        .def("execute", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::excute<float>)
        .def_readonly("modelConfig", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::modelConfig)
        .def_readonly("preProcessConfig", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::preProcessConfig);
    py::class_<DragonianLib::PythonImplement::Processor::ModelConfig>(m, "ModelConfig")
        .def(py::init<std::string, unsigned, unsigned, unsigned, unsigned, unsigned, DragonianLib::CudaModules::moduleValueType, DragonianLib::CudaModules::moduleValueType, bool, bool, bool>())
        .def_property_readonly("modelPath", [](const DragonianLib::PythonImplement::Processor::ModelConfig& config) {
                return config.modelPath.string();
        })
        .def_readonly("inputChannels", &DragonianLib::PythonImplement::Processor::ModelConfig::inputChannels)
        .def_readonly("outputDims", &DragonianLib::PythonImplement::Processor::ModelConfig::outputDims)
        .def_readonly("hiddenDims", &DragonianLib::PythonImplement::Processor::ModelConfig::hiddenDims)
        .def_readonly("numLayers", &DragonianLib::PythonImplement::Processor::ModelConfig::numLayers)
        .def_readonly("numHeads", &DragonianLib::PythonImplement::Processor::ModelConfig::numHeads)
        .def_readonly("f0Max", &DragonianLib::PythonImplement::Processor::ModelConfig::f0Max)
        .def_readonly("f0Min", &DragonianLib::PythonImplement::Processor::ModelConfig::f0Min)
        .def_readonly("useFaNorm", &DragonianLib::PythonImplement::Processor::ModelConfig::useFaNorm)
        .def_readonly("convOnly", &DragonianLib::PythonImplement::Processor::ModelConfig::convOnly)
		.def_readonly("useHarmonicEmb", &DragonianLib::PythonImplement::Processor::ModelConfig::useHarmonicEmb);
    py::class_<DragonianLib::PythonImplement::Processor::PreProcessConfig>(m, "PreProcessConfig")
        .def(py::init<unsigned, unsigned, unsigned, unsigned, unsigned, DragonianLib::CudaModules::moduleValueType, DragonianLib::CudaModules::moduleValueType, DragonianLib::CudaModules::moduleValueType>())
        .def_readonly("samplingRate", &DragonianLib::PythonImplement::Processor::PreProcessConfig::samplingRate)
        .def_readonly("melBins", &DragonianLib::PythonImplement::Processor::PreProcessConfig::melBins)
        .def_readonly("fftLength", &DragonianLib::PythonImplement::Processor::PreProcessConfig::fftLength)
        .def_readonly("windowSize", &DragonianLib::PythonImplement::Processor::PreProcessConfig::windowSize)
        .def_readonly("hopSize", &DragonianLib::PythonImplement::Processor::PreProcessConfig::hopSize)
        .def_readonly("freqMin", &DragonianLib::PythonImplement::Processor::PreProcessConfig::freqMin)
        .def_readonly("freqMax", &DragonianLib::PythonImplement::Processor::PreProcessConfig::freqMax)
		.def_readonly("clipVal", &DragonianLib::PythonImplement::Processor::PreProcessConfig::clipVal);
	m.def("create_cuFCPEProcessor", &create_cuFCPEProcessor, "Create a cuFCPEProcessor instance");
}