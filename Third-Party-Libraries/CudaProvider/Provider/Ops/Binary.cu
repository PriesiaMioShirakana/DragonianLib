#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "binary.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuComplex.h"

namespace DragonianLib
{
	namespace CudaProvider
	{
		namespace Binary
		{
			template<typename T>
			struct CudaAdd
			{
				__device__ constexpr T operator()(const T& a, const T& b) const
				{
					if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>)
						return __hadd(a, b);
					else if constexpr (std::is_same_v<T, cuFloatComplex> || std::is_same_v<T, cuDoubleComplex>)
						return cuCadd(a, b);
					else
						return a + b;
				}
			};

			template<typename T1, typename T2, typename T3, typename O>
			__global__ static void BinaryKernelImpl(
				T1* Dest, const T2* Left, const T3* Right,
				uint32_t N, const unsigned* Shape, const O& Op,
				const unsigned* StrideDest, const unsigned* StrideLeft, const unsigned* StrideRight,
				size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;

				uint32_t linearIdxLeft = 0, linearIdxRight = 0, linearIdxDest = 0;
				for (uint32_t i = 0; i < N; ++i)
				{
					const auto multiDimIndex = idx % Shape[i];
					linearIdxLeft += multiDimIndex * StrideLeft[i];
					linearIdxRight += multiDimIndex * StrideRight[i];
					linearIdxDest += multiDimIndex * StrideDest[i];
					idx /= Shape[i];
				}

				Dest[linearIdxDest] = Op(Left[linearIdxLeft], Right[linearIdxRight]);
			}

			template<bool L, typename T1, typename T2, typename T3, typename O>
			__global__ static void BinaryKernelImpl(
				T1* Dest, const T2* Left, const T3& Right,
				uint32_t N, const unsigned* Shape, const O& Op,
				const unsigned* StrideDest, const unsigned* StrideLeft,
				size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;
				
				uint32_t linearIdxLeft = 0, linearIdxDest = 0;
				for (uint32_t i = 0; i < N; ++i)
				{
					const auto multiDimIndex = idx % Shape[i];
					linearIdxLeft += multiDimIndex * StrideLeft[i];
					linearIdxDest += multiDimIndex * StrideDest[i];
					idx /= Shape[i];
				}

				if constexpr (L)
					Dest[linearIdxDest] = Op(Right, Left[linearIdxLeft]);
				else
					Dest[linearIdxDest] = Op(Left[linearIdxLeft], Right);
			}

			template<typename T1, typename T2, typename O>
			__global__ static void BinaryInplaceKernelImpl(
				T1* Dest, const T2* Right,
				uint32_t N, const unsigned* Shape, const O& Op,
				const unsigned* StrideDest, const unsigned* StrideRight,
				size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;

				uint32_t linearIdxRight = 0, linearIdxDest = 0;
				for (uint32_t i = 0; i < N; ++i)
				{
					const auto multiDimIndex = idx % Shape[i];
					linearIdxRight += multiDimIndex * StrideRight[i];
					linearIdxDest += multiDimIndex * StrideDest[i];
					idx /= Shape[i];
				}
				
				Dest[linearIdxDest] = Op(Dest[linearIdxDest], Right[linearIdxRight]);
			}

			template<bool L, typename T1, typename T2, typename O>
			__global__ static void BinaryInplaceKernelImpl(
				T1* Dest, const T2& Right,
				uint32_t N, const unsigned* Shape, const O& Op,
				const unsigned* StrideDest,
				size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;

				uint32_t linearIdxDest = 0;
				for (uint32_t i = 0; i < N; ++i)
				{
					const auto multiDimIndex = idx % Shape[i];
					linearIdxDest += multiDimIndex * StrideDest[i];
					idx /= Shape[i];
				}

				if constexpr (L)
					Dest[linearIdxDest] = Op(Right, Dest[linearIdxDest]);
				else
					Dest[linearIdxDest] = Op(Dest[linearIdxDest], Right);
			}

			template<typename T1, typename T2, typename T3, typename O>
			__global__ static void BinaryKernelImpl(
				T1* Dest, const T2* Left, const T3* Right, const O& Op, size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;
				Dest[idx] = Op(Left[idx], Right[idx]);
			}

			template<bool L, typename T1, typename T2, typename T3, typename O>
			__global__ static void BinaryKernelImpl(
				T1* Dest, const T2* Left, const T3& Right, const O& Op, size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;
				if constexpr (L)
					Dest[idx] = Op(Right, Left[idx]);
				else
					Dest[idx] = Op(Left[idx], Right);
			}

			template<typename T1, typename T2, typename O>
			__global__ static void BinaryInplaceKernelImpl(
				T1* Dest, const T2* Right, const O& Op, size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;
				Dest[idx] = Op(Dest[idx], Right[idx]);
			}

			template<bool L, typename T1, typename T2, typename O>
			__global__ static void BinaryInplaceKernelImpl(
				T1* Dest, const T2& Right, const O& Op, size_t ElementCount
			)
			{
				uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
				if (idx >= ElementCount)
					return;
				if constexpr (L)
					Dest[idx] = Op(Right, Dest[idx]);
				else
					Dest[idx] = Op(Dest[idx], Right);
			}

			void ImplAdd(
				float* Dest, const float* Left, const float* Right,
				size_t Rank, const unsigned* Shape,
				const unsigned* StrideDest, const unsigned* StrideLeft, const unsigned* StrideRight,
				size_t ElementCount, bool Cont, stream_t CudaStream
			)
			{
				dim3 block(256);
				dim3 grid(unsigned((ElementCount + block.x - 1) / size_t(block.x)));
				if (!Cont)  // NOLINT(bugprone-branch-clone)
				{
					if (Left == nullptr)
						BinaryInplaceKernelImpl<<<grid, block, 0, (cudaStream_t)CudaStream>>>(
							Dest, Right,
							static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
							StrideDest, StrideRight,
							ElementCount
							);
					else
						BinaryKernelImpl<<<grid, block, 0, (cudaStream_t)CudaStream>>>(
							Dest, Left, Right,
							static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
							StrideDest, StrideLeft, StrideRight,
							ElementCount
							);
				}
				else
				{
					if (Left == nullptr)
						BinaryInplaceKernelImpl<<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Right, CudaAdd<float>(), ElementCount);
					else
						BinaryKernelImpl<<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Left, Right, CudaAdd<float>(), ElementCount);
				}
			}

			void ImplAddScalar(
				float* Dest, const float* Left, const float Right,
				size_t Rank, const unsigned* Shape,
				const unsigned* StrideDest, const unsigned* StrideLeft,
				size_t ElementCount, bool Reverse, bool Cont, stream_t CudaStream
			)
			{
				dim3 block(256);
				dim3 grid(unsigned((ElementCount + block.x - 1) / size_t(block.x)));
				if (!Cont)  // NOLINT(bugprone-branch-clone)
				{
					if (Reverse)  // NOLINT(bugprone-branch-clone)
					{
						if (Left == nullptr)
							BinaryInplaceKernelImpl<true><<<grid, block, 0, (cudaStream_t)CudaStream>>>(
								Dest, Right,
								static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
								StrideDest,
								ElementCount
								);
						else
							BinaryKernelImpl<true><<<grid, block, 0, (cudaStream_t)CudaStream>>>(
								Dest, Left, Right,
								static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
								StrideDest, StrideLeft,
								ElementCount
								);
					}
					else
					{
						if (Left == nullptr)
							BinaryInplaceKernelImpl<false><<<grid, block, 0, (cudaStream_t)CudaStream>>>(
								Dest, Right,
								static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
								StrideDest,
								ElementCount
								);
						else
							BinaryKernelImpl<false><<<grid, block, 0, (cudaStream_t)CudaStream>>>(
								Dest, Left, Right,
								static_cast<uint32_t>(Rank), Shape, CudaAdd<float>(),
								StrideDest, StrideLeft,
								ElementCount
								);
					}
				}
				else
				{
					if (Reverse)  // NOLINT(bugprone-branch-clone)
					{
						if (Left == nullptr)
							BinaryInplaceKernelImpl<true><<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Right, CudaAdd<float>(), ElementCount);
						else
							BinaryKernelImpl<true><<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Left, Right, CudaAdd<float>(), ElementCount);
					}
					else
					{
						if (Left == nullptr)
							BinaryInplaceKernelImpl<false><<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Right, CudaAdd<float>(), ElementCount);
						else
							BinaryKernelImpl<false><<<grid, block, 0, (cudaStream_t)CudaStream>>>(Dest, Left, Right, CudaAdd<float>(), ElementCount);
					}
				}
			}
		}
	}
}
