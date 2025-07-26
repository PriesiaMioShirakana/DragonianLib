// ReSharper disable CppClangTidyClangDiagnosticCastFunctionTypeStrict
// ReSharper disable CppClangTidyClangDiagnosticMissingFieldInitializers
#include <codecvt>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

#include "cuFCPEPythonImp.h"
#include "Layers/ThirdParty/Numpy.h"

using namespace DragonianLib::CudaModules::FCPE;

class semaphoreGuard
{
public:
    semaphoreGuard(std::binary_semaphore& sem) : sem(sem) { sem.acquire(); }
    ~semaphoreGuard() { sem.release(); }
private:
    std::binary_semaphore& sem;
};

DragonianLib::PythonImplement::Processor::cuFCPEProcessor::cuFCPEProcessor(
    ModelConfig modelConf,
    std::optional<PreProcessConfig> preProcessConf
) : modelConfig(std::move(modelConf))
{
    
    this->model = std::make_shared<Model>(
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

    if (std::optional<PreProcessConfig> melConf = std::move(preProcessConf))
    {
        preProcessConfig = std::move(*melConf);
        this->melKernel = std::make_shared<CudaModules::MelKernel>(
            preProcessConfig.samplingRate,
            preProcessConfig.melBins,
            preProcessConfig.fftLength,
            preProcessConfig.windowSize,
            preProcessConfig.hopSize,
            preProcessConfig.freqMin,
            preProcessConfig.freqMax,
            preProcessConfig.clipVal
        );
    }

	this->model->LoadFromFile(modelConfig.modelPath);
}

py::array_t<float> DragonianLib::PythonImplement::Processor::executionContextWithAudio::execute(
    const py::array_t<float>& audio,
    CudaModules::moduleValueType threshold,
    const std::string& decoder,
    uint64_t streamPtr
)
{
	// inputData size [batchSize, audioLength * samplingRate]
    semaphoreGuard Guard(sigVal);

    if (audio.ndim() != 2)
        throw std::runtime_error("Input data must be a 2D array.");

    const auto stream = stream_t(streamPtr);

    if (!this->cacheTensors.res.Handle)
        this->cacheTensors.res.Handle = CudaModules::createHandle();
    if (!this->cacheTensors.res.Handle)
        throw std::runtime_error("Failed to create CUDA handle");
    if (const auto st = getHandleStream(this->cacheTensors.res.Handle); st != stream)
    {
        if (st) CudaProvider::asyncCudaStream(st);
        setHandleStream(this->cacheTensors.res.Handle, stream);
    }

    const auto inputShape = audio.shape();
    this->cacheTensors.res.Resize(
        static_cast<unsigned>(inputShape[0]),
        static_cast<unsigned>(inputShape[1])
    );

    if (const auto ret = CudaProvider::cpy2Device(
        this->cacheTensors.res.Data,
        audio.data(),
        audio.size(),
        stream
    )) throw std::runtime_error(CudaProvider::getCudaError(ret));

    if (const auto ret = this->melKernel->Forward(
        this->cacheTensors.res,
        this->cacheTensors.input,
        this->cacheTensors.output
    )) throw std::runtime_error(getErrorString(ret));
    if (const auto ret = this->model->Infer(
        this->cacheTensors,
        threshold,
        decoder == "local_argmax" ? Model::DECODER_LOCAL_ARGMAX : Model::DECODER_ARGMAX
    )) throw std::runtime_error(getErrorString(ret));

    py::array_t<float> output(
        {
            this->cacheTensors.output.N,
            this->cacheTensors.output.C,
            this->cacheTensors.output.H,
            this->cacheTensors.output.W
        }
    );

    CudaProvider::cpy2Host(output.mutable_data(), this->cacheTensors.output.Data, output.size(), stream);

    return output;
}

uint64_t DragonianLib::PythonImplement::Processor::executionContextWithAudio::executeGpuBuffer(
    uint64_t deviceInputBuffer,
    py::array::ShapeContainer shape,
    CudaModules::moduleValueType threshold,
    const std::string& decoder,
    uint64_t streamPtr
)
{
    semaphoreGuard Guard(sigVal);

    if (shape->size() != 2)
        throw std::runtime_error("Input data must be a 2D array.");

    const auto stream = stream_t(streamPtr);

    if (!this->cacheTensors.res.Handle)
        this->cacheTensors.res.Handle = CudaModules::createHandle();
    if (!this->cacheTensors.res.Handle)
        throw std::runtime_error("Failed to create CUDA handle");
    if (const auto st = getHandleStream(this->cacheTensors.res.Handle); st != stream)
    {
        if (st) CudaProvider::asyncCudaStream(st);
        setHandleStream(this->cacheTensors.res.Handle, stream);
    }

    const auto inputShape = shape->data();
    this->cacheTensors.res.Resize(
        static_cast<unsigned>(inputShape[0]),
        static_cast<unsigned>(inputShape[1])
    );

    if (const auto ret = CudaProvider::cpy2Device(
        this->cacheTensors.res.Data,
        (const CudaModules::moduleValueType*)deviceInputBuffer,
        this->cacheTensors.res.BufferSize,
        stream
    )) throw std::runtime_error(CudaProvider::getCudaError(ret));

    if (const auto ret = this->melKernel->Forward(
        this->cacheTensors.res,
        this->cacheTensors.input,
        this->cacheTensors.output
    )) throw std::runtime_error(getErrorString(ret));
    if (const auto ret = this->model->Infer(
        this->cacheTensors,
        threshold,
        decoder == "local_argmax" ? Model::DECODER_LOCAL_ARGMAX : Model::DECODER_ARGMAX
    )) throw std::runtime_error(getErrorString(ret));

    return uint64_t(this->cacheTensors.output.Data);
}

py::array_t<float> DragonianLib::PythonImplement::Processor::executionContextWithMel::execute(
    const py::array_t<float>& mel,
    CudaModules::moduleValueType threshold,
    const std::string& decoder,
    uint64_t streamPtr
)
{
    // inputData size [batchSize, audioLength * samplingRate]
    semaphoreGuard Guard(sigVal);

    if (mel.ndim() != 3)
        throw std::runtime_error("Input data must be a 3D array.");

    const auto stream = stream_t(streamPtr);

    if (!this->cacheTensors.input.Handle)
        this->cacheTensors.input.Handle = CudaModules::createHandle();
    if (!this->cacheTensors.input.Handle)
        throw std::runtime_error("Failed to create CUDA handle");
    if (const auto st = getHandleStream(this->cacheTensors.input.Handle); st != stream)
    {
        if (st) CudaProvider::asyncCudaStream(st);
        setHandleStream(this->cacheTensors.input.Handle, stream);
    }

    const auto inputShape = mel.shape();
    this->cacheTensors.input.Resize(
        static_cast<unsigned>(inputShape[0]),
        static_cast<unsigned>(inputShape[1]),
        static_cast<unsigned>(inputShape[2])
    );

    if (const auto ret = CudaProvider::cpy2Device(
        this->cacheTensors.input.Data,
        mel.data(),
        mel.size(),
        stream
    )) throw std::runtime_error(CudaProvider::getCudaError(ret));

    if (const auto ret = this->model->Infer(
        this->cacheTensors,
        threshold,
        decoder == "local_argmax" ? Model::DECODER_LOCAL_ARGMAX : Model::DECODER_ARGMAX
    )) throw std::runtime_error(getErrorString(ret));

    py::array_t<float> output(
        {
            this->cacheTensors.output.N,
            this->cacheTensors.output.C,
            this->cacheTensors.output.H,
            this->cacheTensors.output.W
        }
    );

    CudaProvider::cpy2Host(output.mutable_data(), this->cacheTensors.output.Data, output.size(), stream);

    return output;
}

DragonianLib::PythonImplement::Processor::executionContextWithAudio DragonianLib::PythonImplement::Processor::cuFCPEProcessor::createAudioExecutionContext() const
{
	return executionContextWithAudio(
        this->model,
        this->melKernel
	);
}

DragonianLib::PythonImplement::Processor::executionContextWithMel DragonianLib::PythonImplement::Processor::cuFCPEProcessor::createMelExecutionContext() const
{
    return executionContextWithMel(
        this->model
    );
}


uint64_t DragonianLib::PythonImplement::Processor::executionContextWithMel::executeGpuBuffer(
    uint64_t deviceInputBuffer,
    py::array::ShapeContainer shape,
    CudaModules::moduleValueType threshold,
    const std::string& decoder,
    uint64_t streamPtr
)
{
    // inputData size [batchSize, audioLength * samplingRate]
    semaphoreGuard Guard(sigVal);

    if (shape->size() != 3)
        throw std::runtime_error("Input data must be a 3D array.");

    const auto stream = stream_t(streamPtr);

    if (!this->cacheTensors.input.Handle)
        this->cacheTensors.input.Handle = CudaModules::createHandle();
    if (!this->cacheTensors.input.Handle)
        throw std::runtime_error("Failed to create CUDA handle");
    if (const auto st = getHandleStream(this->cacheTensors.input.Handle); st != stream)
    {
        if (st) CudaProvider::asyncCudaStream(st);
        setHandleStream(this->cacheTensors.input.Handle, stream);
    }

    const auto inputShape = shape->data();
    this->cacheTensors.input.Resize(
        static_cast<unsigned>(inputShape[0]),
        static_cast<unsigned>(inputShape[1]),
        static_cast<unsigned>(inputShape[2])
    );

    if (const auto ret = CudaProvider::cpy2Device(
        this->cacheTensors.input.Data,
        (const CudaModules::moduleValueType*)deviceInputBuffer,
        this->cacheTensors.input.BufferSize,
        stream
    )) throw std::runtime_error(CudaProvider::getCudaError(ret));

    if (const auto ret = this->model->Infer(
        this->cacheTensors,
        threshold,
        decoder == "local_argmax" ? Model::DECODER_LOCAL_ARGMAX : Model::DECODER_ARGMAX
    )) throw std::runtime_error(getErrorString(ret));

    return uint64_t(this->cacheTensors.output.Data);
}

// Register the cuFCPEProcessor class and its methods in Python
PYBIND11_MODULE(cuFCPEPythonImp, m) {
    m.doc() = "cuFCPE for Python implement"; // optional module docstring

    py::class_<DragonianLib::PythonImplement::Processor::executionContextWithAudio>(m, "audioExecutionContext")
        .def("execute", &DragonianLib::PythonImplement::Processor::executionContextWithAudio::execute)
        .def("executeGpuBuffer", &DragonianLib::PythonImplement::Processor::executionContextWithAudio::executeGpuBuffer);

    py::class_<DragonianLib::PythonImplement::Processor::executionContextWithMel>(m, "melExecutionContext")
        .def("execute", &DragonianLib::PythonImplement::Processor::executionContextWithMel::execute)
        .def("executeGpuBuffer", &DragonianLib::PythonImplement::Processor::executionContextWithMel::executeGpuBuffer);

    py::class_<DragonianLib::PythonImplement::Processor::cuFCPEProcessor>(m, "cuFCPEProcessor")
        .def(py::init<DragonianLib::PythonImplement::Processor::ModelConfig, DragonianLib::PythonImplement::Processor::PreProcessConfig>())
        .def("createAudioExecutionContext", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::createAudioExecutionContext)
        .def("createMelExecutionContext", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::createMelExecutionContext)
        .def_readonly("modelConfig", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::modelConfig)
        .def_readonly("preProcessConfig", &DragonianLib::PythonImplement::Processor::cuFCPEProcessor::preProcessConfig);

    py::class_<DragonianLib::PythonImplement::Processor::ModelConfig>(m, "ModelConfig")
        .def(py::init<const std::string&, unsigned, unsigned, unsigned, unsigned, unsigned, DragonianLib::CudaModules::moduleValueType, DragonianLib::CudaModules::moduleValueType, bool, bool, bool>())
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

}