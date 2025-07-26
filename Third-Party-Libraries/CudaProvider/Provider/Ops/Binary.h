#pragma once
#include "Provider/Kernel.h"

#define _DRAGONIANLIB_BINARY_DCL(Fun, Type) \
__declspec(dllexport) void Impl##Fun##( \
	Type##* Dest, const Type##* Left, const Type##* Right, \
	size_t Rank, const unsigned* Shape, \
	const unsigned* StrideDest, const unsigned* StrideLeft, const unsigned* StrideRight, \
	size_t ElementCount, bool Cont, stream_t CudaStream \
)

#define _DRAGONIANLIB_BINARY_DCL_Scalar(Fun, Type) \
__declspec(dllexport) void Impl##Fun##Scalar( \
	Type##* Dest, const Type##* Left, const Type Right, \
	size_t Rank, const unsigned* Shape, \
	const unsigned* StrideDest, const unsigned* StrideLeft, \
	size_t ElementCount, bool Reverse, bool Cont, stream_t CudaStream \
)

#define _DRAGONIANLIB_BINARY_DCL_EXPORTS(Fun) \
	_DRAGONIANLIB_BINARY_DCL(Fun, int8_t); \
	_DRAGONIANLIB_BINARY_DCL(Fun, int16_t); \
	_DRAGONIANLIB_BINARY_DCL(Fun, int32_t); \
	_DRAGONIANLIB_BINARY_DCL(Fun, int64_t); \
	_DRAGONIANLIB_BINARY_DCL(Fun, float); \
	_DRAGONIANLIB_BINARY_DCL(Fun, double); \
	_DRAGONIANLIB_BINARY_DCL(Fun, __half); \
	_DRAGONIANLIB_BINARY_DCL(Fun, __nv_bfloat16); \
	_DRAGONIANLIB_BINARY_DCL(Fun, cuFloatComplex); \
	_DRAGONIANLIB_BINARY_DCL(Fun, cuDoubleComplex); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, int8_t); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, int16_t); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, int32_t); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, int64_t); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, float); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, double); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, __half); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, __nv_bfloat16); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, cuFloatComplex); \
	_DRAGONIANLIB_BINARY_DCL_Scalar(Fun, cuDoubleComplex)

namespace DragonianLib
{
	namespace CudaProvider
	{
		namespace Binary
		{
			_DRAGONIANLIB_BINARY_DCL_EXPORTS(Add);
			_DRAGONIANLIB_BINARY_DCL_EXPORTS(Sub);
			_DRAGONIANLIB_BINARY_DCL_EXPORTS(Mul);
			_DRAGONIANLIB_BINARY_DCL_EXPORTS(Div);
		}
	}
}