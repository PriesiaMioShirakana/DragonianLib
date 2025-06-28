#pragma once
#include <stdexcept>

struct __half;
struct __nv_bfloat16;
struct float2;
struct double2;
typedef float2 cuFloatComplex;
typedef double2 cuDoubleComplex;
typedef struct __Stream* stream_t;

namespace DragonianLib
{
	namespace CudaProvider
	{
		void* cudaAllocate(size_t size);
		int cudaFree(void* block);
		int host2Device(void* dst, const void* src, size_t size, stream_t stream);
		int device2Host(void* dst, const void* src, size_t size, stream_t stream);
		int device2Device(void* dst, const void* src, size_t size, stream_t stream);
		stream_t createCudaStream();
		int destoryCudaStream(stream_t stream);
		int asyncCudaStream(stream_t stream);
		const char* getCudaError(int errorId);
		int getLastError();
		const std::string& getLastErrorString();
		template <typename T>
		T* cudaAlloc(size_t size)
		{
			if (auto block = (T*)cudaAllocate(sizeof(T) * size))
				return block;
			throw std::runtime_error(getCudaError(getLastError()));
		}
		template <typename T>
		int cpy2Device(T* dst, const T* src, size_t size, stream_t stream)
		{
			return host2Device(dst, src, size * sizeof(T), stream);
		}
		template <typename T>
		int cpy2Host(T* dst, const T* src, size_t size, stream_t stream)
		{
			return device2Host(dst, src, size * sizeof(T), stream);
		}
		template <typename T>
		int cpyData(T* dst, const T* src, size_t size, stream_t stream)
		{
			return device2Device(dst, src, size * sizeof(T), stream);
		}

		template <typename T>
		class CudaArray
		{
		public:
			CudaArray(size_t N)
			{
				_MyData = cudaAlloc<T>(N);
				if (!_MyData)
					throw std::runtime_error("Failed to allocate device memory for CudaArray.");
				_MySize = N;
			}

			~CudaArray()
			{
				Release();
			}

			CudaArray(const CudaArray&) = delete;
			CudaArray& operator=(const CudaArray&) = delete;
			CudaArray(CudaArray&& other) noexcept = delete;
			CudaArray& operator=(CudaArray&& other) noexcept = delete;

			T* Data() const
			{
				return _MyData;
			}
			size_t Size() const
			{
				return _MySize;
			}

		protected:
			T* _MyData = nullptr;
			size_t _MySize = 0; // Size in bytes

		private:
			void Release()
			{
				if (_MyData)
				{
					if (auto Ret = cudaFree(_MyData))
					{
						fprintf(stderr, "Error freeing CudaArray memory: %s\n", getCudaError(Ret));
						abort();
					}
					_MyData = nullptr;
				}
			}
		};

		class CudaDimensions
		{
		public:
			CudaDimensions(size_t N, const int64_t* Dims, stream_t Stream)
			{
				_MyData = cudaAlloc<uint32_t>(N << 1);
				auto New = new uint32_t[N];
				for (size_t i = 0; i < N; ++i)
				{
					if (Dims[i] < 0 || Dims[i] > UINT32_MAX)
					{
						delete[] New;
						throw std::out_of_range("Dimension value out of range for CudaDimensions.");
					}
					New[i] = static_cast<uint32_t>(Dims[i]);
				}
				std::reverse(New, New + N);
				if (auto Ret = host2Device(_MyData, New, N * sizeof(uint32_t), Stream))
				{
					delete[] New;
					throw std::runtime_error(getCudaError(Ret));
				}
				delete[] New;
			}

			~CudaDimensions()
			{
				if (_MyData)
				{
					if (auto Ret = cudaFree(_MyData))
					{
						fprintf(stderr, "Error freeing CudaDimensions memory: %s\n", getCudaError(Ret));
						abort();
					}
					_MyData = nullptr;
				}
			}

			const uint32_t* Data() const
			{
				return _MyData;
			}

			CudaDimensions(const CudaDimensions&) = delete;
			CudaDimensions& operator=(const CudaDimensions&) = delete;
			CudaDimensions(CudaDimensions&& other) noexcept = delete;
			CudaDimensions& operator=(CudaDimensions&& other) noexcept = delete;

		private:
			uint32_t* _MyData; // Pointer to device memory for dimensions
		};
	}
}