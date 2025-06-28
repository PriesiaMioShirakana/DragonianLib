#include <string>
#include <device_launch_parameters.h>

#include "fcpe.h"
#include "cuda_runtime.h"
#include "npy.h"

namespace DragonianLib
{
    namespace CudaProvider
    {
        extern thread_local std::string __LastError;
    }

	namespace CudaModules
	{
		namespace FCPE
		{
            template <unsigned blockSizeX>
            static __global__ void layerReduceSumCentKernel(
                const moduleValueType* iFeat,
                const moduleValueType* iCentTable,
                moduleValueType* oSumCent,
                moduleValueType* oSum,
                unsigned sampleCount,
                unsigned featureSize
            )
            {
                const unsigned ty = threadIdx.y;
                const unsigned tx = threadIdx.x;
                const unsigned featureIdx = blockIdx.x * blockDim.x + tx;
                const unsigned batchIdx = blockIdx.y * blockDim.y + ty;
                const unsigned sharedIdx = ty * blockDim.x + tx;

                extern __shared__ moduleValueType sharedReduceSumData[];
                auto sharedReduceSumCentData = sharedReduceSumData + (ptrdiff_t)blockDim.y * blockDim.x;
                auto sharedReduceCentData = sharedReduceSumData + 2ll * blockDim.y * blockDim.x;

                sharedReduceSumData[sharedIdx] = 0.f;
                sharedReduceSumCentData[sharedIdx] = 0.f;

                if (featureIdx >= featureSize || batchIdx >= sampleCount)
                    return;

                if (ty == 0)
                    sharedReduceCentData[tx] = iCentTable[featureIdx];

                __syncthreads();

                moduleValueType cFeat = iFeat[batchIdx * featureSize + featureIdx];
                sharedReduceSumData[sharedIdx] = cFeat;
                sharedReduceSumCentData[sharedIdx] = cFeat * sharedReduceCentData[tx];

                __syncthreads();

                if constexpr (blockSizeX >= 1024)
                {
                    if (tx < 512)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 512];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 512];
                    }
                    __syncthreads();
                }
                if constexpr (blockSizeX >= 512)
                {
                    if (tx < 256)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 256];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 256];
                    }
                    __syncthreads();
                }
                if constexpr (blockSizeX >= 256)
                {
                    if (tx < 128)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 128];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 128];
                    }
                    __syncthreads();
                }
                if constexpr (blockSizeX >= 128)
                {
                    if (tx < 64)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 64];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 64];
                    }
                    __syncthreads();
                }
                if (tx < 32)
                {
                    if constexpr (blockSizeX >= 64)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 32];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 32];
                        __syncthreads();
                    }
                    if constexpr (blockSizeX >= 32)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 16];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 16];
                        __syncthreads();
                    }
                    if constexpr (blockSizeX >= 16)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 8];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 8];
                        __syncthreads();
                    }
                    if constexpr (blockSizeX >= 8)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 4];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 4];
                        __syncthreads();
                    }
                    if constexpr (blockSizeX >= 4)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 2];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 2];
                        __syncthreads();
                    }
                    if constexpr (blockSizeX >= 2)
                    {
                        sharedReduceSumData[sharedIdx] += sharedReduceSumData[sharedIdx + 1];
                        sharedReduceSumCentData[sharedIdx] += sharedReduceSumCentData[sharedIdx + 1];
                        __syncthreads();
                    }
                    if (tx == 0)
                    {
                        atomicAdd(oSum + batchIdx, sharedReduceSumData[(ptrdiff_t)ty * blockDim.x]);
                        atomicAdd(oSumCent + batchIdx, sharedReduceSumCentData[(ptrdiff_t)ty * blockDim.x]);
                        if (blockIdx.x == gridDim.x - 1)
	                        *(oSumCent + batchIdx) /= *(oSum + batchIdx);
                    }
                }
            }

            static __global__ void layerReduceSumCentIdxKernel(
                const moduleValueType* iFeat,
                const moduleValueType* iCentTable,
                const int* iIndex,
                moduleValueType* oSumCent,
                unsigned sampleCount,
                unsigned featureSize
            )
            {
                constexpr unsigned iFeatureSize = 9;
                const unsigned batchIdx = blockIdx.x * blockDim.x + threadIdx.x;

                if (batchIdx >= sampleCount)
                    return;

                auto cFeat = iFeat + (ptrdiff_t)batchIdx * featureSize;
                auto cIndex = iIndex + (ptrdiff_t)batchIdx * iFeatureSize;

                moduleValueType sum = 0.f, sumCent = 0.f;

				#pragma unroll
                for (unsigned i = 0; i < iFeatureSize; ++i)
                {
                    int idx = cIndex[i];
                    sum += cFeat[idx];
                    sumCent += cFeat[idx] * iCentTable[idx];
                }

                oSumCent[batchIdx] = sumCent / sum;
            }

            template <unsigned outputIdx = false>
            static __global__ void layerReduceMaxKernel(
                const moduleValueType* iFeat,
                moduleValueType* oMax,
                unsigned sampleCount,
                int featureSize,
                int* oMaxIdx = nullptr
            )
            {
                const unsigned batchIdx = blockIdx.x * blockDim.x + threadIdx.x;

                if (batchIdx >= sampleCount)
                    return;

                const moduleValueType* featData = iFeat + (ptrdiff_t)batchIdx * featureSize;

                moduleValueType maxCount = -1e+10f;
                int maxIdx = 0;
                for (int i = 0; i < featureSize; ++i)
                {
                    auto cond = featData[i] > maxCount;
                    maxCount = cond ? featData[i] : maxCount;
                    if constexpr (outputIdx)
                        maxIdx = cond ? i : maxIdx;
                }

                oMax[batchIdx] = maxCount;
                if constexpr (outputIdx)
                {
                    auto oIdx = oMaxIdx + batchIdx * 9ll;
					#pragma unroll
                    for (int i = 0; i < 9; ++i)
                    {
                        auto mIdx = i + maxIdx - 4;
                        mIdx = mIdx < 0             ? 0                 : mIdx;
                        mIdx = mIdx >= featureSize  ? featureSize - 1   : mIdx;
                        oIdx[i] = mIdx;
                    }
                }
            }

            static __global__ void layerMaskKernel(
                const moduleValueType* iMask,
                const moduleValueType* iFeat,
                moduleValueType* oFeat,
                unsigned sampleCount,
                moduleValueType thr
            )
            {
                const unsigned batchIdx = blockIdx.x * blockDim.x + threadIdx.x;

                if (batchIdx >= sampleCount)
                    return;

                oFeat[batchIdx] = iMask[batchIdx] > thr ? iFeat[batchIdx] : -INFINITY;
            }

            static __global__ void layerCent2F0(
				moduleValueType* oFeat,
                unsigned sampleCount
            )
            {
                const unsigned batchIdx = blockIdx.x * blockDim.x + threadIdx.x;
                if (batchIdx >= sampleCount)
                    return;
                oFeat[batchIdx] = 10.f * powf(2.f, oFeat[batchIdx] / 1200.f);
            }

            ConformerConvModule::ConformerConvModule(
                Module* parent, const std::string& name,
                unsigned dimModel, unsigned expandFactor, unsigned kernelSize
            ) : Module(parent, name)
            {
                auto inner_dim = dimModel * expandFactor;

                net_0 = std::make_shared<LayerNorm1D>(
                    this,
					"net.0",
                    dimModel
                );
                net_2 = std::make_shared<Conv1D>(
                    this,
                    "net.2",
                    dimModel,
                    inner_dim * 2,
                    1
                );
                net_4_conv = std::make_shared<Conv1D>(
                    this,
                    "net.4.conv",
                    inner_dim,
                    inner_dim,
                    kernelSize,
                    1,
                    kernelSize / 2,
					1,
                    inner_dim
                );
                net_6 = std::make_shared<Conv1D>(
                    this,
                    "net.6",
                    inner_dim,
                    dimModel,
                    1
                );
            }

            layerStatus_t ConformerConvModule::Forward(
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& mean,
                Tensor<moduleValueType>& var,
                Tensor<moduleValueType>& cache,
                Tensor<moduleValueType>& col
            ) const noexcept try
            {
                if (auto Ret = net_0->Forward(output, mean, var)) return Ret;

                if (auto Ret = net_1.Forward(output, cache)) return Ret;

                if (auto Ret = net_2->Forward(cache, output, col)) return Ret;

                if (auto Ret = net_3.Forward(output, cache)) return Ret;

                if (auto Ret = net_4_conv->Forward(cache, output, col)) return Ret;

                if (auto Ret = net_5.Forward(output)) return Ret;

                if (auto Ret = net_6->Forward(output, cache, col)) return Ret;

                return net_7.Forward(cache, output);
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            CFNEncoderLayer::CFNEncoderLayer(
                Module* parent, const std::string& name,
                unsigned dimModel, unsigned numHeads,
                bool useNorm, bool convOnly
            ) : Module(parent, name)
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                conformer = std::make_shared<ConformerConvModule>(
                    this,
                    "conformer",
                    dimModel
                );
                norm = std::make_shared<LayerNorm1D>(
                    this,
                    "norm",
                    dimModel
                );
            }

            layerStatus_t CFNEncoderLayer::Forward(
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& mean,
                Tensor<moduleValueType>& var,
                Tensor<moduleValueType>& res,
                Tensor<moduleValueType>& cache,
                Tensor<moduleValueType>& col
            ) const noexcept try
            {
                res.Copy(output);

                if (auto Ret = conformer->Forward(
                    output, mean, var, cache, col
                )) return Ret;

                return AddTensor(output, res);
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            ConformerNaiveEncoder::ConformerNaiveEncoder(
                Module* parent, const std::string& name,
                unsigned numLayers, unsigned numHeads, unsigned dimModel,
                bool useNorm, bool convOnly
            ) : Module(parent, name)
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                for (unsigned i = 0; i < numLayers; ++i)
                    encoder_layers.emplace_back(
                        std::make_shared<CFNEncoderLayer>(
                            this,
                            "encoder_layers." + std::to_string(i),
                            dimModel,
                            numHeads,
                            useNorm,
                            convOnly
                        )
                    );
            }

            layerStatus_t ConformerNaiveEncoder::Forward(
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& mean,
                Tensor<moduleValueType>& var,
                Tensor<moduleValueType>& res,
                Tensor<moduleValueType>& cache,
                Tensor<moduleValueType>& col
            ) const noexcept try
            {
                for (const auto& layer : encoder_layers)
                    if (auto Ret = layer->Forward(
                        output,
                        mean,
                        var,
                        res,
                        cache,
                        col
                    )) return Ret;
                return LAYER_STATUS_SUCCESS;
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

			Model::Model(
                unsigned inputChannels, unsigned outputDims, unsigned hiddenDims,
                unsigned numLayers, unsigned numHeads,
                moduleValueType f0Max, moduleValueType f0Min,
                bool useFaNorm, bool convOnly,
                bool useHarmonicEmb
            ) : Module(nullptr, "")
            {
                if (!convOnly)
                    throw std::overflow_error("not impl yet!");

                if (useHarmonicEmb)
                    throw std::overflow_error("not impl yet!");

                input_stack_0 = std::make_shared<Conv1D>(
                    this,
                    "input_stack.0",
                    inputChannels,
                    hiddenDims,
                    3,
                    1,
                    1
                );
                input_stack_1 = std::make_shared<GroupNorm1D>(
                    this,
                    "input_stack.1",
                    4,
                    hiddenDims
                );
                input_stack_3 = std::make_shared<Conv1D>(
                    this,
                    "input_stack.3",
                    hiddenDims,
                    hiddenDims,
                    3,
                    1,
                    1
                );

                net = std::make_shared<ConformerNaiveEncoder>(
                    this,
					"net",
                    numLayers,
                    numHeads,
                    hiddenDims,
                    useFaNorm,
                    convOnly
                );

                norm = std::make_shared<LayerNorm1D>(
                    this,
                    "norm",
                    hiddenDims
                );

                output_proj = std::make_shared<Linear>(
                    this,
                    "output_proj",
                    hiddenDims,
                    outputDims
                );

                cent_table = std::make_shared<Parameter>(
                    this,
					"cent_table",
                    Tensor<moduleValueType>(),
                    false
                );

                gaussian_blurred_cent_mask = std::make_shared<Parameter>(
                    this,
                    "gaussian_blurred_cent_mask",
                    Tensor<moduleValueType>(),
                    false
                );
            }

            layerStatus_t Model::Forward(
                CacheTensors& caches
            ) const noexcept try
            {
                if (auto Ret = input_stack_0->Forward(
                    caches.input,
                    caches.res,
                    caches.col
                )) return Ret;

                if (auto Ret = input_stack_1->Forward(
                    caches.res,
                    caches.mean,
                    caches.var
                )) return Ret;

                if (auto Ret = input_stack_2.Forward(
                    caches.res
                )) return Ret;

                if (auto Ret = input_stack_3->Forward(
                    caches.res,
                    caches.output,
                    caches.col
                )) return Ret;

                if (auto Ret = Transpose::Forward(
                    caches.output,
                    caches.input
                )) return Ret;

                if (auto Ret = net->Forward(
                    caches.input,
                    caches.mean,
                    caches.var,
                    caches.res,
                    caches.output,
                    caches.col
                )) return Ret;

                if (auto Ret = norm->Forward(
                    caches.input,
                    caches.mean,
                    caches.var
                )) return Ret;

                if (auto Ret = output_proj->Forward(
                    caches.input,
                    caches.output
                )) return Ret;

                return SigmoidTensor(caches.output);
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            layerStatus_t Model::Latent2Cents(
                CacheTensors& caches,
                moduleValueType threshold
            ) const noexcept try
            {
                const auto sampleCount = caches.output.N * caches.output.C * caches.output.H;
                const auto featureSize = caches.output.W;

                if (featureSize != cent_table->GetTensor().W)
                    return LAYER_STATUS_SIZE_MISMATCH;

                auto Stream = (cudaStream_t)getHandleStream(caches.output.Handle);

                caches.mean.Resize(caches.output.N * caches.output.C, caches.output.H);
                if (caches.mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.mean.Handle));
                caches.mean.Handle = caches.output.Handle;

                caches.var.Resize(caches.output.N * caches.output.C, caches.output.H);
                if (caches.var.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.var.Handle));
                caches.var.Handle = caches.output.Handle;

                caches.col.Resize(caches.output.N * caches.output.C, caches.output.H);
                if (caches.col.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.col.Handle));
                caches.col.Handle = caches.output.Handle;

                cudaMemsetAsync(caches.mean.Data, 0, sizeof(moduleValueType)* sampleCount, Stream);
                cudaMemsetAsync(caches.var.Data, 0, sizeof(moduleValueType)* sampleCount, Stream);

                dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSize(
                    (featureSize + blockSize.x - 1) / blockSize.x,
                    (sampleCount + blockSize.y - 1) / blockSize.y
                );
                const auto sharedMemSize = (2ull * blockSize.x * blockSize.y + blockSize.x) * sizeof(moduleValueType);

                layerReduceSumCentKernel<DRAGONIANLIB_CUDA_BLOCK_SIZE / 32>
            	<<<gridSize, blockSize, sharedMemSize, Stream>>>(
                    caches.output.Data,
                    cent_table->GetTensor().Data,
                    caches.mean.Data,
                    caches.var.Data,
                    sampleCount,
                    featureSize
                    );

                dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
                dim3 gridLength((sampleCount + blockLength.x - 1) / blockLength.x);
                layerReduceMaxKernel<<<gridLength, blockLength, 0, Stream>>>(
                    caches.output.Data,
                    caches.col.Data,
                    sampleCount,
                    featureSize
                    );

                caches.output.Resize(caches.output.N * caches.output.C, caches.output.H);

                layerMaskKernel<<<gridLength, blockLength, 0, Stream>>>(
                    caches.col.Data,
                    caches.mean.Data,
                    caches.output.Data,
                    sampleCount,
                    threshold
                    );

                return LAYER_STATUS_SUCCESS;
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            layerStatus_t Model::Latent2CentsLocal(
                CacheTensors& caches,
                moduleValueType threshold
            ) const noexcept try
            {
                const auto sampleCount = caches.output.N * caches.output.C * caches.output.H;
                const auto featureSize = caches.output.W;

                if (featureSize != cent_table->GetTensor().W)
                    return LAYER_STATUS_SIZE_MISMATCH;

                auto Stream = (cudaStream_t)getHandleStream(caches.output.Handle);

                caches.mean.Resize(caches.output.N* caches.output.C, caches.output.H);
                if (caches.mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.mean.Handle));
                caches.mean.Handle = caches.output.Handle;

                caches.col.Resize(caches.output.N * caches.output.C, caches.output.H);
                if (caches.col.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.col.Handle));
                caches.col.Handle = caches.output.Handle;

                caches.res.Resize(caches.output.N * caches.output.C, caches.output.H, 9);
                if (caches.res.Handle) CudaProvider::asyncCudaStream(getHandleStream(caches.res.Handle));
                caches.res.Handle = caches.output.Handle;

                cudaMemsetAsync(caches.mean.Data, 0, sizeof(moduleValueType)* sampleCount, Stream);
                cudaMemsetAsync(caches.var.Data, 0, sizeof(moduleValueType)* sampleCount, Stream);

                dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
                dim3 gridLength((sampleCount + blockLength.x - 1) / blockLength.x);
                layerReduceMaxKernel<true><<<gridLength, blockLength, 0, Stream>>>(
                    caches.output.Data,
                    caches.col.Data,
                    sampleCount,
                    featureSize,
                    (int*)caches.res.Data
                    );

                layerReduceSumCentIdxKernel<<<gridLength, blockLength, 0, Stream>>>(
                    caches.output.Data,
                    cent_table->GetTensor().Data,
                    (int*)caches.res.Data,
                    caches.mean.Data,
                    sampleCount,
                    featureSize
                    );

                caches.output.Resize(caches.output.N * caches.output.C, caches.output.H);

                layerMaskKernel<<<gridLength, blockLength, 0, Stream>>>(
                    caches.col.Data,
                    caches.mean.Data,
                    caches.output.Data,
                    sampleCount,
                    threshold
                    );

                return LAYER_STATUS_SUCCESS;
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            layerStatus_t Model::Infer(
                CacheTensors& caches,
                moduleValueType threshold,
                DECODER decoder
            ) const noexcept try
            {
                if (auto Ret = Forward(caches))
                    return Ret;
				
                if (decoder == DECODER_ARGMAX)
                {
	                if (auto Ret = Latent2Cents(caches, threshold))
	                	return Ret;
                }
                else
                {
                    if (auto Ret = Latent2CentsLocal(caches, threshold))
                        return Ret;
                }

                const auto sampleCount = caches.output.N * caches.output.C * caches.output.H * caches.output.W;
                auto Stream = (cudaStream_t)getHandleStream(caches.output.Handle);
                dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
                dim3 gridLength((sampleCount + blockLength.x - 1) / blockLength.x);
                layerCent2F0<<<gridLength, blockLength, 0, Stream>>>(caches.output.Data, sampleCount);
                return LAYER_STATUS_SUCCESS;
            }
            catch (std::exception& except)
            {
                CudaProvider::__LastError = except.what();
                return LAYER_STATUS_FATAL_ERROR;
            }

            void Model::LoadFromFile(const std::wstring& view)
            {
                auto Dict = Util::LoadNumpyFileToDict(view);
                Module::LoadModel(Dict);
            }
		}
	}
}
