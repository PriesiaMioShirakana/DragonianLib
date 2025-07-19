#pragma once
#include "fcpe.h"
#include <pybind11/numpy.h>
#include "kernel.h"

using namespace DragonianLib::CudaModules::FCPE;
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
                    std::string modelPath,
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
                    useFaNorm(useFaNorm), convOnly(convOnly), useHarmonicEmb(useHarmonicEmb){ 
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
                    hopSize(hopSize), melBins(melBins), freqMin(freqMin), freqMax(freqMax), clipVal(clipVal) {
                }

            };

            class cuFCPEProcessor
            {
                public:
                    Model* model = nullptr;
					DragonianLib::CudaModules::MelKernel* melKernel = nullptr;
					ModelConfig modelConfig;
					PreProcessConfig preProcessConfig;
                    Model::CacheTensors cacheTensors;
                    stream_t stream = nullptr;

					unsigned audioLength = 10; // Default audio length in seconds
                    unsigned batchSize = 1; // Default batch size

                    cuFCPEProcessor(
                        ModelConfig modelConfig,
                        PreProcessConfig preProcessConfig
                    );
                    ~cuFCPEProcessor();

                    template<typename ARRAY_TYPE>
                    py::array_t<ARRAY_TYPE> excute(py::array_t<ARRAY_TYPE> inputData);
            };
        }
    }
}