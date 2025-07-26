#pragma once
#include <memory>
#include <semaphore>
#include <pybind11/numpy.h>

#include "fcpe.h"
#include "Provider/Kernel.h"

namespace py = pybind11;

namespace DragonianLib
{
    namespace PythonImplement
    {
        namespace Processor
        {
            struct ModelConfig
            {
                unsigned inputChannels;
                unsigned outputDims;
                unsigned hiddenDims;
                unsigned numLayers;
                unsigned numHeads;
                DragonianLib::CudaModules::moduleValueType f0Max;
                DragonianLib::CudaModules::moduleValueType f0Min;
                bool useFaNorm;
                bool convOnly;
                bool useHarmonicEmb;
                std::filesystem::path modelPath;

                ModelConfig(
                    const std::string& modelPath,
                    unsigned inputChannels = 128,
                    unsigned outputDims = 360,
                    unsigned hiddenDims = 512,
                    unsigned numLayers = 6,
                    unsigned numHeads = 8,
                    DragonianLib::CudaModules::moduleValueType f0Max = 1975.5f,
                    DragonianLib::CudaModules::moduleValueType f0Min = 32.70f,
                    bool useFaNorm = false,
                    bool convOnly = true,
                    bool useHarmonicEmb = false
                ) : inputChannels(inputChannels), outputDims(outputDims), hiddenDims(hiddenDims),
                    numLayers(numLayers), numHeads(numHeads), f0Max(f0Max), f0Min(f0Min),
                    useFaNorm(useFaNorm), convOnly(convOnly), useHarmonicEmb(useHarmonicEmb)
                {
                    this->modelPath = std::filesystem::path(modelPath);
                }
            };

            struct PreProcessConfig
            {
                unsigned samplingRate;
                unsigned fftLength;
                unsigned windowSize;
                unsigned hopSize;
                unsigned melBins;
                DragonianLib::CudaModules::moduleValueType freqMin;
                DragonianLib::CudaModules::moduleValueType freqMax;
                DragonianLib::CudaModules::moduleValueType clipVal;

                PreProcessConfig(
                    unsigned samplingRate = 16000,
                    unsigned fftLength = 1024,
                    unsigned windowSize = 1024,
                    unsigned hopSize = 160,
                    unsigned melBins = 128,
                    DragonianLib::CudaModules::moduleValueType freqMin = 0.f,
                    DragonianLib::CudaModules::moduleValueType freqMax = 8000.f,
                    DragonianLib::CudaModules::moduleValueType clipVal = 1e-5f
                ) : samplingRate(samplingRate), fftLength(fftLength), windowSize(windowSize),
                    hopSize(hopSize), melBins(melBins), freqMin(freqMin), freqMax(freqMax), clipVal(clipVal)
                {

                }
            };

            class executionContextWithAudio  // NOLINT(cppcoreguidelines-special-member-functions)
            {
            public:
                friend class cuFCPEProcessor;

                py::array_t<float> execute(
                    const py::array_t<float>& audio,
                    CudaModules::moduleValueType threshold,
                    const std::string& decoder,
                    uint64_t streamPtr
                );
                uint64_t executeGpuBuffer(
                    uint64_t deviceInputBuffer,
                    py::array::ShapeContainer shape,
                    CudaModules::moduleValueType threshold,
                    const std::string& decoder,
                    uint64_t streamPtr
                );
            protected:
                executionContextWithAudio(
                    std::shared_ptr<CudaModules::FCPE::Model> model,
                    std::shared_ptr<CudaModules::MelKernel> melKernel
                ) : model(std::move(model)), melKernel(std::move(melKernel))
                {

                }
                CudaModules::FCPE::Model::CacheTensors cacheTensors;
                std::shared_ptr<CudaModules::FCPE::Model> model = nullptr;
                std::shared_ptr<CudaModules::MelKernel> melKernel = nullptr;
                std::binary_semaphore sigVal{ 1 };
            };

            class executionContextWithMel  // NOLINT(cppcoreguidelines-special-member-functions)
            {
            public:
                friend class cuFCPEProcessor;

                py::array_t<float> execute(
                    const py::array_t<float>& mel,
                    CudaModules::moduleValueType threshold,
                    const std::string& decoder,
                    uint64_t streamPtr
                );
                uint64_t executeGpuBuffer(
                    uint64_t deviceInputBuffer,
                    py::array::ShapeContainer shape,
                    CudaModules::moduleValueType threshold,
                    const std::string& decoder,
                    uint64_t streamPtr
                );
            protected:
                executionContextWithMel(
                    std::shared_ptr<CudaModules::FCPE::Model> model
                ) : model(std::move(model))
                {

                }
                CudaModules::FCPE::Model::CacheTensors cacheTensors;
                std::shared_ptr<CudaModules::FCPE::Model> model = nullptr;
                std::binary_semaphore sigVal{ 1 };
            };

            class cuFCPEProcessor  // NOLINT(cppcoreguidelines-special-member-functions)
            {
            public:
                std::shared_ptr<CudaModules::FCPE::Model> model = nullptr;
                std::shared_ptr<CudaModules::MelKernel> melKernel = nullptr;
                ModelConfig modelConfig;
                PreProcessConfig preProcessConfig;

                cuFCPEProcessor(
                    ModelConfig modelConf,
                    std::optional<PreProcessConfig> preProcessConf
                );

                executionContextWithAudio createAudioExecutionContext() const;
                executionContextWithMel createMelExecutionContext() const;
            };
        }
    }
}