#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_runtime.h"
#include "kernel.h"

namespace DragonianLib
{
	namespace CudaProvider
	{
		thread_local std::string __LastError = "Empty";  // NOLINT(misc-use-internal-linkage)

		void* cudaAllocate(size_t size)
		{
			void* block = nullptr;
			size_t memFree, memTotal;
			cudaMemGetInfo(&memFree, &memTotal);
			if (size > memFree)
			{
				fprintf(stderr, "Cuda memory allocation failed: %zu bytes requested, but only %zu bytes available.\n", size, memFree);
				throw std::runtime_error("Cuda Out of Memory!");
			}
			if (const auto err = ::cudaMalloc(&block, size))
				fprintf(stderr, "%s\n", cudaGetErrorString(err));
			return block;
		}

		int cudaFree(void* block) noexcept
		{
			return ::cudaFree(block);
		}

		int host2Device(void* dst, const void* src, size_t size, stream_t stream) noexcept
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
		}

		int device2Host(void* dst, const void* src, size_t size, stream_t stream) noexcept
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
		}

		int device2Device(void* dst, const void* src, size_t size, stream_t stream) noexcept
		{
			return cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
		}

		const std::string& getLastErrorString() noexcept
		{
			return __LastError;
		}

		stream_t createCudaStream() noexcept
		{
			cudaStream_t Ret;
			if (const auto err = cudaStreamCreate(&Ret))
				fprintf(stderr, "%s\n", cudaGetErrorString(err));
			return stream_t(Ret);
		}

		int destroyCudaStream(stream_t stream) noexcept
		{
			return cudaStreamDestroy((cudaStream_t)stream);
		}

		int asyncCudaStream(stream_t stream) noexcept
		{
			return cudaStreamSynchronize((cudaStream_t)stream);
		}

		const char* getCudaError(int errorId) noexcept
		{
			return cudaGetErrorString(static_cast<cudaError_t>(errorId));
		}

		int getLastError() noexcept
		{
			return cudaGetLastError();
		}
	}
}
