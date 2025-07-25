/**
 * @file Simd.h
 * @author NaruseMioShirakana
 * @email shirakanamio@foxmail.com
 * @copyright Copyright (C) 2022-2025 NaruseMioShirakana (shirakanamio@foxmail.com)
 * @license GNU Affero General Public License v3.0
 * @attentions
 *  - This file is part of DragonianLib.
 *  - DragonianLib is free software: you can redistribute it and/or modify it under the terms of the
 *  - GNU Affero General Public License as published by the Free Software Foundation, either version 3
 *  - of the License, or any later version.
 *
 *  - DragonianLib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  - without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  - See the GNU Affero General Public License for more details.
 *
 *  - You should have received a copy of the GNU Affero General Public License along with Foobar.
 *  - If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
 * @brief SIMD wrapper for cpu operators
 * @changes
 *  > 2025/3/22 NaruseMioShirakana Refactored <
 */

#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/OperatorBase.h"
#include <immintrin.h>
#define _D_Dragonian_Lib_Simd_Not_Mask(_Type) (::DragonianLib::Operators::Vectorized<_Type>((_Type)(0)))
#define _D_Dragonian_Lib_Simd_Complement_Mask(_Type) (::DragonianLib::Operators::Vectorized<_Type>((_Type)(-1)))

#define _D_Dragonian_Lib_Simd_Int8_Fp(_Function) do {\
auto BLower = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(*this, 0)); \
auto BHigher = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(*this, 1)); \
auto LowerLower = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(BLower, 0)))))); \
auto LowerHigher = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(BLower, 1)))))); \
auto HigherLower = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(BHigher, 0)))))); \
auto HigherHigher = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(BHigher, 1)))))); \
auto CLower = _mm256_cvtepi16_epi8(_mm256_insertf128_si256(_mm256_castsi128_si256(LowerLower), LowerHigher, 1)); \
auto CHigher = _mm256_cvtepi16_epi8(_mm256_insertf128_si256(_mm256_castsi128_si256(HigherLower), HigherHigher, 1)); \
return _mm256_insertf128_si256(_mm256_castsi128_si256(CLower), CHigher, 1);} while(false)

#define _D_Dragonian_Lib_Simd_Int16_Fp(_Function) do {\
auto Lower = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(*this, 0)))))); \
auto Higher = _mm256_cvtepi32_epi16(_mm256_cvtps_epi32(_mm256_##_Function##_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(*this, 1)))))); \
return _mm256_insertf128_si256(_mm256_castsi128_si256(Lower), Higher, 1); \
} while(false)

_D_Dragonian_Lib_Operator_Space_Begin

/**
 * @class Vectorized
 * @brief Warpper for SIMD
 * @tparam Type Type of the data
 */
template<typename Type>
class Vectorized
{
public:
	static constexpr size_t Alignment = sizeof(__m256); // AVX-256 requires 32-byte alignment
	static constexpr size_t Stride = IsAvx256SupportedValue<Type> ? sizeof(__m256) / sizeof(Type) : 8;
	static constexpr size_t size() { return Stride; }
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline ~Vectorized() = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(const Vectorized&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(Vectorized&&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized& operator=(const Vectorized&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized& operator=(Vectorized&&) = default;
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(const Type* _ValPtr)
	{
		LoadPointer(_ValPtr);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(__m256i _Val) : _YMMX(_Val) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(__m256 _Val) : _YMMX(_mm256_castps_si256(_Val)) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(__m256d _Val) : _YMMX(_mm256_castpd_si256(_Val)) {}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized(Type _Val)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			_YMMX = _mm256_castps_si256(_mm256_set1_ps(_Val));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			_YMMX = _mm256_castpd_si256(_mm256_set1_pd(_Val));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_YMMX = _mm256_castps_si256(
				_mm256_set_ps(
					_Val.imag(), _Val.real(),
					_Val.imag(), _Val.real(),
					_Val.imag(), _Val.real(),
					_Val.imag(), _Val.real()
				)
			);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_YMMX = _mm256_castpd_si256(
				_mm256_set_pd(
					_Val.imag(), _Val.real(),
					_Val.imag(), _Val.real()
				)
			);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_YMMX = _mm256_set1_epi8(_Val);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_YMMX = _mm256_set1_epi16(_Val);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			_YMMX = _mm256_set1_epi32(_Val);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			_YMMX = _mm256_set1_epi64x(_Val);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void LoadPointer(const Type* _ValPtr)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			_YMMX = _mm256_castps_si256(_mm256_load_ps(reinterpret_cast<float const*>(_ValPtr)));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			_YMMX = _mm256_castpd_si256(_mm256_load_pd(reinterpret_cast<double const*>(_ValPtr)));
		else
			_YMMX = _mm256_load_si256((const __m256i*)_ValPtr);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline void Store(Type* _Dest) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			_mm256_store_ps(reinterpret_cast<float*>(_Dest), *this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			_mm256_store_pd(reinterpret_cast<double*>(_Dest), *this);
		else
			_mm256_store_si256((__m256i*)_Dest, _YMMX);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline void StoreBool(void* _Dest) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			bool Dest[8];
			auto Mask = _mm256_movemask_ps(_mm256_castsi256_ps(*this));
			for (int i = 0; i < 8; ++i)
				Dest[i] = Mask & (1 << i);
			((bool*)_Dest)[0] = Dest[0] && Dest[1];
			((bool*)_Dest)[1] = Dest[2] && Dest[3];
			((bool*)_Dest)[2] = Dest[4] && Dest[5];
			((bool*)_Dest)[3] = Dest[6] && Dest[7];
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			bool Dest[4];
			auto Mask = _mm256_movemask_pd(_mm256_castsi256_pd(*this));
			for (int i = 0; i < 4; ++i)
				Dest[i] = Mask & (1 << i);
			((bool*)_Dest)[0] = Dest[0] && Dest[1];
			((bool*)_Dest)[1] = Dest[2] && Dest[3];
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			const auto Mask = _mm256_movemask_ps(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 8; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			const auto Mask = _mm256_movemask_pd(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 4; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
		{
			const auto Mask = _mm256_movemask_epi8(*this);
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 32; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi16_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 16; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi32_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 8; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
		{
			const auto Mask = _mm256_movemask_epi8(_mm256_castsi128_si256(_mm256_cvtepi64_epi8(*this)));
			auto Dest = (bool*)_Dest;
			for (int i = 0; i < 4; ++i)
				Dest[i] = Mask & (1 << i);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline operator __m256i() const
	{
		static_assert(TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean> || TypeTraits::IsAnyOfValue<Type, Int16, UInt16> || TypeTraits::IsAnyOfValue<Type, Int32, UInt32> || TypeTraits::IsAnyOfValue<Type, Int64, UInt64>);
		return _YMMX;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline operator __m256() const
	{
		static_assert(TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>);
		return _mm256_castsi256_ps(_YMMX);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline operator __m256d() const
	{
		static_assert(TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>);
		return _mm256_castsi256_pd(_YMMX);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline operator __mmask32() const
	{
		return _mm256_movemask_epi8(_YMMX);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline operator __mmask16() const
	{
		return _mm256_movemask_epi8(_YMMX) >> 16;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline operator __mmask8() const
	{
		return _mm256_movemask_epi8(_YMMX) >> 24;
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator<(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cmp_ps(*this, _Right, _CMP_LT_OQ);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cmp_pd(*this, _Right, _CMP_LT_OQ);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_cmpgt_epi8(_Right, *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_cmpgt_epi16(_Right, *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cmpgt_epi32(_Right, *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cmpgt_epi64(_Right, *this);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator<=(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cmp_ps(*this, _Right, _CMP_LE_OQ);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cmp_pd(*this, _Right, _CMP_LE_OQ);
		else
			return *this < _Right || *this == _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator>(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cmp_ps(*this, _Right, _CMP_GT_OQ);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cmp_pd(*this, _Right, _CMP_GT_OQ);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_cmpgt_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_cmpgt_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cmpgt_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cmpgt_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator>=(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cmp_ps(*this, _Right, _CMP_GE_OQ);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cmp_pd(*this, _Right, _CMP_GE_OQ);
		else
			return *this > _Right || *this == _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator==(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			const auto Diff = _mm256_sub_ps(*this, _Right);
			const auto Abs = _mm256_and_ps(Diff, _mm256_set1_ps(-0.0f));
			return _mm256_cmp_ps(Abs, _mm256_set1_ps(FLT_EPSILON), _CMP_LE_OQ);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			const auto Diff = _mm256_sub_pd(*this, _Right);
			const auto Abs = _mm256_and_pd(Diff, _mm256_set1_pd(-0.0));
			return _mm256_cmp_pd(Abs, _mm256_set1_pd(DBL_EPSILON), _CMP_LE_OQ);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_cmpeq_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_cmpeq_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cmpeq_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cmpeq_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator!=(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			const auto Diff = _mm256_sub_ps(*this, _Right);
			const auto Abs = _mm256_and_ps(Diff, _mm256_set1_ps(-0.0f));
			return _mm256_cmp_ps(Abs, _mm256_set1_ps(FLT_EPSILON), _CMP_GT_OQ);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			const auto Diff = _mm256_sub_pd(*this, _Right);
			const auto Abs = _mm256_and_pd(Diff, _mm256_set1_pd(-0.0));
			return _mm256_cmp_pd(Abs, _mm256_set1_pd(DBL_EPSILON), _CMP_GT_OQ);
		}
		else
			return _mm256_xor_si256(*this, _Right);
	}

	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator+(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_add_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_add_pd(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_add_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_add_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_add_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_add_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator+=(const Vectorized& _Right)
	{
		return *this = *this + _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator-(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_sub_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_sub_pd(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_sub_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_sub_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_sub_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_sub_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator-=(const Vectorized& _Right)
	{
		return *this = *this - _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator*(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_mul_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			__m256 c_real = _mm256_shuffle_ps(_Right, _Right, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 d_imag = _mm256_shuffle_ps(_Right, _Right, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 real_part = _mm256_fmsub_ps(a_real, c_real, _mm256_mul_ps(b_imag, d_imag));
			__m256 imag_part = _mm256_fmadd_ps(b_imag, c_real, _mm256_mul_ps(a_real, d_imag));

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_mul_pd(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);
			__m256d c_real = _mm256_movedup_pd(_Right);
			__m256d d_imag = _mm256_permute_pd(_Right, 0b0101);

			__m256d real_part = _mm256_fmsub_pd(a_real, c_real, _mm256_mul_pd(b_imag, d_imag));
			__m256d imag_part = _mm256_fmadd_pd(b_imag, c_real, _mm256_mul_pd(a_real, d_imag));

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean> || TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_mullo_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_mullo_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_mullo_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator*=(const Vectorized& _Right)
	{
		return *this = *this * _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator/(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_div_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			__m256 c_real = _mm256_shuffle_ps(_Right, _Right, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 d_imag = _mm256_shuffle_ps(_Right, _Right, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 real_part = _mm256_fmadd_ps(a_real, c_real, _mm256_mul_ps(b_imag, d_imag));
			__m256 imag_part = _mm256_fmsub_ps(b_imag, c_real, _mm256_mul_ps(a_real, d_imag));

			__m256 denominator = _mm256_fmadd_ps(c_real, c_real, _mm256_mul_ps(d_imag, d_imag));

			return _mm256_div_ps(_mm256_blend_ps(real_part, imag_part, 0b10101010), denominator);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_div_pd(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);
			__m256d c_real = _mm256_movedup_pd(_Right);
			__m256d d_imag = _mm256_permute_pd(_Right, 0b0101);

			__m256d real_part = _mm256_fmadd_pd(a_real, c_real, _mm256_mul_pd(b_imag, d_imag));
			__m256d imag_part = _mm256_fmsub_pd(b_imag, c_real, _mm256_mul_pd(a_real, d_imag));

			__m256d denominator = _mm256_fmadd_pd(c_real, c_real, _mm256_mul_pd(d_imag, d_imag));

			return _mm256_div_pd(_mm256_blend_pd(real_part, imag_part, 0b1010), denominator);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_div_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_div_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_div_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_div_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator%(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_rem_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_rem_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_rem_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_rem_epi64(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_fmod_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_fmod_pd(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator%=(const Vectorized& _Right)
	{
		return *this = *this % _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator/=(const Vectorized& _Right)
	{
		return *this = *this / _Right;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator&(const Vectorized& _Right) const
	{
		return _mm256_and_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator|(const Vectorized& _Right) const
	{
		return _mm256_or_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator^(const Vectorized& _Right) const
	{
		return _mm256_xor_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator<<(const int _Count) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_slli_epi32(*this, _Count);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_slli_epi64(*this, _Count);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Not_Implemented_Error;
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_slli_epi16(*this, _Count);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator>>(const int _Count) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_srli_epi32(*this, _Count);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_srli_epi64(*this, _Count);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Not_Implemented_Error;
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_srli_epi16(*this, _Count);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator&&(const Vectorized& _Right) const
	{
		return _mm256_and_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator||(const Vectorized& _Right) const
	{
		return _mm256_or_si256(*this, _Right);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator!() const
	{
		if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_cmpeq_epi8(*this, _mm256_setzero_ps());
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_cmpeq_epi16(*this, _mm256_setzero_ps());
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cmpeq_epi32(*this, _mm256_setzero_ps());
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cmpeq_epi64(*this, _mm256_setzero_ps());
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cmp_ps(*this, _mm256_setzero_ps(), _CMP_EQ_OQ);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cmp_pd(*this, _mm256_setzero_pd(), _CMP_EQ_OQ);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized operator~() const
	{
		return _mm256_xor_si256(*this, _mm256_set1_epi64x(-1));
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Complement() const
	{
		if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_sub_epi8(_mm256_set1_epi8(UINT8_MAX), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_sub_epi16(_mm256_set1_epi16(UINT16_MAX), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32, Float32>)
			return _mm256_sub_epi32(_mm256_set1_epi32(UINT32_MAX), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64, Float64>)
			return _mm256_sub_epi64(_mm256_set1_epi64x(UINT64_MAX), *this);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Negative() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_xor_ps(*this, _mm256_set1_ps(-0.0f));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_xor_pd(*this, _mm256_set1_pd(-0.0));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_sub_epi8(_mm256_setzero_si256(), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_sub_epi16(_mm256_setzero_si256(), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_sub_epi32(_mm256_setzero_si256(), *this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_sub_epi64(_mm256_setzero_si256(), *this);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractAngle() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			return _mm256_atan2_ps(b_imag, a_real);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			return _mm256_atan2_pd(b_imag, a_real);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractMagnitude() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			return _mm256_sqrt_ps(_mm256_fmadd_ps(a_real, a_real, _mm256_mul_ps(b_imag, b_imag)));
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			return _mm256_sqrt_pd(_mm256_fmadd_pd(a_real, a_real, _mm256_mul_pd(b_imag, b_imag)));
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractReal() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 magnitude = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 angle = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			return _mm256_mul_ps(magnitude, _mm256_cos_ps(angle));
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d magnitude = _mm256_movedup_pd(*this);
			__m256d angle = _mm256_permute_pd(*this, 0b0101);
			return _mm256_mul_pd(magnitude, _mm256_cos_pd(angle));
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractImag() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 magnitude = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 angle = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			return _mm256_mul_ps(magnitude, _mm256_sin_ps(angle));
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d magnitude = _mm256_movedup_pd(*this);
			__m256d angle = _mm256_permute_pd(*this, 0b0101);
			return _mm256_mul_pd(magnitude, _mm256_sin_pd(angle));
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	static _D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractAngle(const Vectorized& Real, const Vectorized& Imag)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_atan2_ps(Imag, Real);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_atan2_pd(Imag, Real);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	static _D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractMagnitude(const Vectorized& Real, const Vectorized& Imag)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_sqrt_ps(_mm256_fmadd_ps(Real, Real, _mm256_mul_ps(Imag, Imag)));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_sqrt_pd(_mm256_fmadd_pd(Real, Real, _mm256_mul_pd(Imag, Imag)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	static _D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractReal(const Vectorized& Magnitude, const Vectorized& Angle)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_mul_ps(Magnitude, _mm256_cos_ps(Angle));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_mul_pd(Magnitude, _mm256_cos_pd(Angle));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	static _D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ExtractImag(const Vectorized& Magnitude, const Vectorized& Angle)
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_mul_ps(Magnitude, _mm256_sin_ps(Angle));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_mul_pd(Magnitude, _mm256_sin_pd(Angle));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Pow(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_pow_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_pow_pd(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 pow = _mm256_shuffle_ps(_Right, _Right, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			auto magnitude = _mm256_sqrt_ps(_mm256_fmadd_ps(a_real, a_real, _mm256_mul_ps(b_imag, b_imag)));
			auto angle = _mm256_atan2_ps(b_imag, a_real);

			magnitude = _mm256_pow_ps(magnitude, pow);
			angle = _mm256_mul_ps(angle, pow);

			__m256 real_part = _mm256_mul_ps(magnitude, _mm256_cos_ps(angle));
			__m256 imag_part = _mm256_mul_ps(magnitude, _mm256_sin_ps(angle));

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d pow = _mm256_movedup_pd(_Right);
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			auto magnitude = _mm256_sqrt_pd(_mm256_fmadd_pd(a_real, a_real, _mm256_mul_pd(b_imag, b_imag)));
			auto angle = _mm256_atan2_pd(b_imag, a_real);

			magnitude = _mm256_pow_pd(magnitude, pow);
			angle = _mm256_mul_pd(angle, pow);

			__m256d real_part = _mm256_mul_pd(magnitude, _mm256_cos_pd(angle));
			__m256d imag_part = _mm256_mul_pd(magnitude, _mm256_sin_pd(angle));

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Not_Implemented_Error;
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Not_Implemented_Error;
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_pow_ps(_mm256_cvtepi32_ps(*this), _mm256_cvtepi32_ps(_Right)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_pow_pd(_mm256_cvtepi64_pd(*this), _mm256_cvtepi64_pd(_Right)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Sqrt() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_sqrt_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_sqrt_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			auto magnitude = _mm256_sqrt_ps(_mm256_fmadd_ps(a_real, a_real, _mm256_mul_ps(b_imag, b_imag)));
			auto angle = _mm256_atan2_ps(b_imag, a_real);

			magnitude = _mm256_sqrt_ps(magnitude);
			angle = _mm256_mul_ps(angle, _mm256_set1_ps(0.5f));

			__m256 real_part = _mm256_mul_ps(magnitude, _mm256_cos_ps(angle));
			__m256 imag_part = _mm256_mul_ps(magnitude, _mm256_sin_ps(angle));

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			auto magnitude = _mm256_sqrt_pd(_mm256_fmadd_pd(a_real, a_real, _mm256_mul_pd(b_imag, b_imag)));
			auto angle = _mm256_atan2_pd(b_imag, a_real);

			magnitude = _mm256_sqrt_pd(magnitude);
			angle = _mm256_mul_pd(angle, _mm256_set1_pd(0.5));

			__m256d real_part = _mm256_mul_pd(magnitude, _mm256_cos_pd(angle));
			__m256d imag_part = _mm256_mul_pd(magnitude, _mm256_sin_pd(angle));

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(sqrt);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(sqrt);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_sqrt_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_sqrt_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized RSqrt() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_rsqrt_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_rsqrt_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(*this)))));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			auto magnitude = _mm256_sqrt_ps(_mm256_fmadd_ps(a_real, a_real, _mm256_mul_ps(b_imag, b_imag)));
			auto angle = _mm256_atan2_ps(b_imag, a_real);

			magnitude = _mm256_rsqrt_ps(magnitude);
			angle = _mm256_mul_ps(angle, _mm256_set1_ps(-0.5f));

			__m256 real_part = _mm256_mul_ps(magnitude, _mm256_cos_ps(angle));
			__m256 imag_part = _mm256_mul_ps(magnitude, _mm256_sin_ps(angle));

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			auto magnitude = _mm256_sqrt_pd(_mm256_fmadd_pd(a_real, a_real, _mm256_mul_pd(b_imag, b_imag)));
			auto angle = _mm256_atan2_pd(b_imag, a_real);

			magnitude = _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_rsqrt_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(magnitude)))));
			angle = _mm256_mul_pd(angle, _mm256_set1_pd(-0.5));

			__m256d real_part = _mm256_mul_pd(magnitude, _mm256_cos_pd(angle));
			__m256d imag_part = _mm256_mul_pd(magnitude, _mm256_sin_pd(angle));

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else
			return _mm256_setzero_si256();
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Reciprocal() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_rcp_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cvtps_pd(_mm256_castps256_ps128(_mm256_rcp_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(*this)))));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			__m256 c_real = _mm256_set1_ps(1.f);
			__m256 d_imag = _mm256_setzero_ps();

			__m256 real_part = _mm256_fmadd_ps(a_real, c_real, _mm256_mul_ps(b_imag, d_imag));
			__m256 imag_part = _mm256_fmsub_ps(b_imag, c_real, _mm256_mul_ps(a_real, d_imag));

			__m256 denominator = _mm256_fmadd_ps(c_real, c_real, _mm256_mul_ps(d_imag, d_imag));

			return _mm256_div_ps(_mm256_blend_ps(real_part, imag_part, 0b10101010), denominator);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);
			__m256d c_real = _mm256_set1_pd(1.0);
			__m256d d_imag = _mm256_setzero_pd();

			__m256d real_part = _mm256_fmadd_pd(a_real, c_real, _mm256_mul_pd(b_imag, d_imag));
			__m256d imag_part = _mm256_fmsub_pd(b_imag, c_real, _mm256_mul_pd(a_real, d_imag));

			__m256d denominator = _mm256_fmadd_pd(c_real, c_real, _mm256_mul_pd(d_imag, d_imag));

			return _mm256_div_pd(_mm256_blend_pd(real_part, imag_part, 0b1010), denominator);
		}
		else
			return _mm256_setzero_si256();
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Abs() const
	{
		if constexpr (std::is_unsigned_v<Type>)
			return *this;
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			auto magnitude = _mm256_sqrt_ps(_mm256_fmadd_ps(real, real, _mm256_mul_ps(imag, imag)));
			return _mm256_blend_ps(magnitude, _mm256_setzero_ps(), 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d real = _mm256_movedup_pd(*this);
			__m256d imag = _mm256_permute_pd(*this, 0b0101);
			auto magnitude = _mm256_sqrt_pd(_mm256_fmadd_pd(real, real, _mm256_mul_pd(imag, imag)));
			return _mm256_blend_pd(magnitude, _mm256_setzero_pd(), 0b1010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_and_ps(*this, _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MAX)));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_and_pd(*this, _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_abs_epi8(*this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_abs_epi16(*this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_abs_epi32(*this);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_abs_epi64(*this);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;

	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Sin() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_sin_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_sin_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 s_real = _mm256_sin_ps(real);
			__m256 ch_imag = _mm256_cosh_ps(imag);

			__m256 real_part = _mm256_mul_ps(s_real, ch_imag);

			__m256 c_real = _mm256_cos_ps(real);
			__m256 sh_imag = _mm256_sinh_ps(imag);

			__m256 imag_part = _mm256_mul_ps(c_real, sh_imag);

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d real = _mm256_movedup_pd(*this);
			__m256d imag = _mm256_permute_pd(*this, 0b0101);

			__m256d s_real = _mm256_sin_pd(real);
			__m256d ch_imag = _mm256_cosh_pd(imag);

			__m256d real_part = _mm256_mul_pd(s_real, ch_imag);

			__m256d c_real = _mm256_cos_pd(real);
			__m256d sh_imag = _mm256_sinh_pd(imag);

			__m256d imag_part = _mm256_mul_pd(c_real, sh_imag);

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(sin);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(sin);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_sin_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_sin_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Cos() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cos_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cos_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 c_real = _mm256_cos_ps(real);
			__m256 ch_imag = _mm256_cosh_ps(imag);

			__m256 real_part = _mm256_mul_ps(c_real, ch_imag);

			__m256 s_real = _mm256_sin_ps(real);
			__m256 sh_imag = _mm256_sinh_ps(imag);

			__m256 imag_part = _mm256_mul_ps(s_real, sh_imag);

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d real = _mm256_movedup_pd(*this);
			__m256d imag = _mm256_permute_pd(*this, 0b0101);

			__m256d c_real = _mm256_cos_pd(real);
			__m256d ch_imag = _mm256_cosh_pd(imag);

			__m256d real_part = _mm256_mul_pd(c_real, ch_imag);

			__m256d s_real = _mm256_sin_pd(real);
			__m256d sh_imag = _mm256_sinh_pd(imag);

			__m256d imag_part = _mm256_mul_pd(s_real, sh_imag);

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(cos);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(cos);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_cos_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_cos_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Tan() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_tan_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_tan_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 _2a = _mm256_add_ps(a, a);
			__m256 _2b = _mm256_add_ps(b, b);

			__m256 sin_2a = _mm256_sin_ps(_2a);
			__m256 cos_2a = _mm256_cos_ps(_2a);
			__m256 sinh_2b = _mm256_sinh_ps(_2b);
			__m256 cosh_2b = _mm256_cosh_ps(_2b);

			__m256 denominator = _mm256_add_ps(cos_2a, cosh_2b);

			return _mm256_div_ps(_mm256_blend_ps(sin_2a, sinh_2b, 0b10101010), denominator);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a = _mm256_movedup_pd(*this);
			__m256d b = _mm256_permute_pd(*this, 0b0101);

			__m256d _2a = _mm256_add_pd(a, a);
			__m256d _2b = _mm256_add_pd(b, b);

			__m256d sin_2a = _mm256_sin_pd(_2a);
			__m256d cos_2a = _mm256_cos_pd(_2a);
			__m256d sinh_2b = _mm256_sinh_pd(_2b);
			__m256d cosh_2b = _mm256_cosh_pd(_2b);

			__m256d denominator = _mm256_add_pd(cos_2a, cosh_2b);

			return _mm256_div_pd(_mm256_blend_pd(sin_2a, sinh_2b, 0b1010), denominator);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(tan);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(tan);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_tan_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_tan_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ASin() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_asin_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_asin_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(asin);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(asin);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_asin_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_asin_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ACos() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_acos_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_acos_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(acos);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(acos);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_acos_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_acos_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ATan() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_atan_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_atan_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(atan);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(atan);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_atan_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_atan_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ATan2() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 a_real = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b_imag = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 magnitude = _mm256_sqrt_ps(_mm256_fmadd_ps(a_real, a_real, _mm256_mul_ps(b_imag, b_imag)));
			__m256 angle = _mm256_atan2_ps(b_imag, a_real);

			return _mm256_blend_ps(magnitude, angle, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d a_real = _mm256_movedup_pd(*this);
			__m256d b_imag = _mm256_permute_pd(*this, 0b0101);

			__m256d magnitude = _mm256_sqrt_pd(_mm256_fmadd_pd(a_real, a_real, _mm256_mul_pd(b_imag, b_imag)));
			__m256d angle = _mm256_atan2_pd(b_imag, a_real);

			return _mm256_blend_pd(magnitude, angle, 0b1010);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Tan2() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32> || TypeTraits::IsSameTypeValue<Type, Float32>)
		{
			__m256 magnitude = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 angle = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 real = _mm256_cos_ps(angle);
			__m256 imag = _mm256_sin_ps(angle);

			return _mm256_mul_ps(magnitude, _mm256_blend_ps(real, imag, 0b10101010));
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64> || TypeTraits::IsSameTypeValue<Type, Float64>)
		{
			__m256d magnitude = _mm256_movedup_pd(*this);
			__m256d angle = _mm256_permute_pd(*this, 0b0101);

			__m256d real = _mm256_cos_pd(angle);
			__m256d imag = _mm256_sin_pd(angle);

			return _mm256_mul_pd(magnitude, _mm256_blend_pd(real, imag, 0b1010));
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Sinh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_sinh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_sinh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 sinh_a = _mm256_sinh_ps(a);
			__m256 cos_b = _mm256_cos_ps(b);

			__m256 cosh_a = _mm256_cosh_ps(a);
			__m256 sin_b = _mm256_sin_ps(b);

			__m256 real_part = _mm256_mul_ps(sinh_a, cos_b);
			__m256 imag_part = _mm256_mul_ps(cosh_a, sin_b);

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a = _mm256_movedup_pd(*this);
			__m256d b = _mm256_permute_pd(*this, 0b0101);

			__m256d sinh_a = _mm256_sinh_pd(a);
			__m256d cos_b = _mm256_cos_pd(b);

			__m256d cosh_a = _mm256_cosh_pd(a);
			__m256d sin_b = _mm256_sin_pd(b);

			__m256d real_part = _mm256_mul_pd(sinh_a, cos_b);
			__m256d imag_part = _mm256_mul_pd(cosh_a, sin_b);

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(sinh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(sinh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_sinh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_sinh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Cosh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_cosh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_cosh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			__m256 a = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			__m256 b = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));

			__m256 cosh_a = _mm256_cosh_ps(a);
			__m256 cos_b = _mm256_cos_ps(b);

			__m256 sinh_a = _mm256_sinh_ps(a);
			__m256 sin_b = _mm256_sin_ps(b);

			__m256 real_part = _mm256_mul_ps(cosh_a, cos_b);
			__m256 imag_part = _mm256_mul_ps(sinh_a, sin_b);

			return _mm256_blend_ps(real_part, imag_part, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			__m256d a = _mm256_movedup_pd(*this);
			__m256d b = _mm256_permute_pd(*this, 0b0101);

			__m256d cosh_a = _mm256_cosh_pd(a);
			__m256d cos_b = _mm256_cos_pd(b);

			__m256d sinh_a = _mm256_sinh_pd(a);
			__m256d sin_b = _mm256_sin_pd(b);

			__m256d real_part = _mm256_mul_pd(cosh_a, cos_b);
			__m256d imag_part = _mm256_mul_pd(sinh_a, sin_b);

			return _mm256_blend_pd(real_part, imag_part, 0b1010);
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(cosh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(cosh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_cosh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_cosh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Tanh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_tanh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_tanh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(tanh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(tanh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_tanh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_tanh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ASinh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_asinh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_asinh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(asinh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(asinh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_asinh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_asinh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ACosh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_acosh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_acosh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(acosh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(acosh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_acosh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_acosh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized ATanh() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_atanh_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_atanh_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(atanh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(atanh);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_atanh_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_atanh_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Log() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_log_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_log_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(log);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(log);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_log_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_log_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Log2() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_log2_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_log2_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(log2);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(log2);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_log2_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_log2_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Log10() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_log10_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_log10_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(log10);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(log10);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_log10_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_log10_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Exp() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_exp_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_exp_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(exp);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(exp);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_exp_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_exp_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Exp2() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_exp2_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_exp2_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(exp2);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(exp2);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_exp2_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_exp2_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Exp10() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32>)
			return _mm256_exp10_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64>)
			return _mm256_exp10_pd(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			_D_Dragonian_Lib_Not_Implemented_Error; //TODO
		}
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			_D_Dragonian_Lib_Simd_Int8_Fp(exp10);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			_D_Dragonian_Lib_Simd_Int16_Fp(exp10);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_cvtps_epi32(_mm256_exp10_ps(_mm256_cvtepi32_ps(*this)));
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_cvtpd_epi64(_mm256_exp10_pd(_mm256_cvtepi64_pd(*this)));
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Ceil() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_ceil_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_ceil_pd(*this);
		else
			return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Floor() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_floor_ps(*this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_floor_pd(*this);
		else
			return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Round() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_round_ps(*this, _MM_FROUND_TO_NEAREST_INT);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_round_pd(*this, _MM_FROUND_TO_NEAREST_INT);
		else
			return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Trunc() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_round_ps(*this, _MM_FROUND_TO_ZERO);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_round_pd(*this, _MM_FROUND_TO_ZERO);
		else
			return *this;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Frac() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_sub_ps(*this, _mm256_floor_ps(*this));
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_sub_pd(*this, _mm256_floor_pd(*this));
		else
			return _mm256_setzero_si256();
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Min(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_min_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_min_pd(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_min_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_min_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_min_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_min_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Max(const Vectorized& _Right) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_max_ps(*this, _Right);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_max_pd(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int8, UInt8, Boolean>)
			return _mm256_max_epi8(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int16, UInt16>)
			return _mm256_max_epi16(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int32, UInt32>)
			return _mm256_max_epi32(*this, _Right);
		else if constexpr (TypeTraits::IsAnyOfValue<Type, Int64, UInt64>)
			return _mm256_max_epi64(*this, _Right);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Lerp(const Vectorized& _Right, const Vectorized& _Alpha) const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Float32> || TypeTraits::IsSameTypeValue<Type, Complex32>)
			return _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(_Right, *this), _Alpha), *this);
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Float64> || TypeTraits::IsSameTypeValue<Type, Complex64>)
			return _mm256_add_pd(_mm256_mul_pd(_mm256_sub_pd(_Right, *this), _Alpha), *this);
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Clamp(const Vectorized& _Min, const Vectorized& _Max) const
	{
		return Max(_Min).Min(_Max);
	}
	_D_Dragonian_Lib_Constexpr_Force_Inline Vectorized Polar() const
	{
		if constexpr (TypeTraits::IsSameTypeValue<Type, Complex32>)
		{
			auto magnitude = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(2, 2, 0, 0));
			auto angle = _mm256_shuffle_ps(*this, *this, _MM_SHUFFLE(3, 3, 1, 1));
			auto real = _mm256_mul_ps(magnitude, _mm256_cos_ps(angle));
			auto imag = _mm256_mul_ps(magnitude, _mm256_sin_ps(angle));
			return _mm256_blend_ps(real, imag, 0b10101010);
		}
		else if constexpr (TypeTraits::IsSameTypeValue<Type, Complex64>)
		{
			auto magnitude = _mm256_movedup_pd(*this);
			auto angle = _mm256_permute_pd(*this, 0b0101);
			auto real = _mm256_mul_pd(magnitude, _mm256_cos_pd(angle));
			auto imag = _mm256_mul_pd(magnitude, _mm256_sin_pd(angle));
			return _mm256_blend_pd(real, imag, 0b1010);
		}
		else
			_D_Dragonian_Lib_Not_Implemented_Error;
	}

private:
	__m256i _YMMX;

};

_D_Dragonian_Lib_Force_Inline void DragonianLibMemcpy256_8(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	const __m256i m1 = _mm256_load_si256(_Src + 1);
	const __m256i m2 = _mm256_load_si256(_Src + 2);
	const __m256i m3 = _mm256_load_si256(_Src + 3);
	const __m256i m4 = _mm256_load_si256(_Src + 4);
	const __m256i m5 = _mm256_load_si256(_Src + 5);
	const __m256i m6 = _mm256_load_si256(_Src + 6);
	const __m256i m7 = _mm256_load_si256(_Src + 7);
	_mm256_store_si256(_Dst + 0, m0);
	_mm256_store_si256(_Dst + 1, m1);
	_mm256_store_si256(_Dst + 2, m2);
	_mm256_store_si256(_Dst + 3, m3);
	_mm256_store_si256(_Dst + 4, m4);
	_mm256_store_si256(_Dst + 5, m5);
	_mm256_store_si256(_Dst + 6, m6);
	_mm256_store_si256(_Dst + 7, m7);
}
_D_Dragonian_Lib_Force_Inline void DragonianLibMemcpy256_4(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	const __m256i m1 = _mm256_load_si256(_Src + 1);
	const __m256i m2 = _mm256_load_si256(_Src + 2);
	const __m256i m3 = _mm256_load_si256(_Src + 3);
	_mm256_store_si256(_Dst + 0, m0);
	_mm256_store_si256(_Dst + 1, m1);
	_mm256_store_si256(_Dst + 2, m2);
	_mm256_store_si256(_Dst + 3, m3);
}
_D_Dragonian_Lib_Force_Inline void DragonianLibMemcpy256_2(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	const __m256i m1 = _mm256_load_si256(_Src + 1);
	_mm256_store_si256(_Dst + 0, m0);
	_mm256_store_si256(_Dst + 1, m1);
}
_D_Dragonian_Lib_Force_Inline void DragonianLibMemcpy256_1(__m256i* __restrict _Dst, const __m256i* __restrict _Src) {
	const __m256i m0 = _mm256_load_si256(_Src + 0);
	_mm256_store_si256(_Dst + 0, m0);
}

_D_Dragonian_Lib_Force_Inline void DragonianLibMemCpy(
	void* const __restrict _Dst, const void* const __restrict _Src, size_t _Size
)
{
	__m256i* __restrict _Dst_Ptr = (__m256i*)_Dst;
	const __m256i* __restrict _Src_Ptr = (const __m256i*)_Src;
	while (true)
	{
		if (!(_Size >> 8))
			break;
		DragonianLibMemcpy256_8(_Dst_Ptr, _Src_Ptr);
		_Dst_Ptr += 8;
		_Src_Ptr += 8;
		_Size -= alignof(__m256i) * 8;
	}
	if (_Size >> 7)
	{
		DragonianLibMemcpy256_4(_Dst_Ptr, _Src_Ptr);
		_Dst_Ptr += 4;
		_Src_Ptr += 4;
		_Size -= alignof(__m256i) * 4;
	}
	if (_Size >> 6)
	{
		DragonianLibMemcpy256_2(_Dst_Ptr, _Src_Ptr);
		_Dst_Ptr += 2;
		_Src_Ptr += 2;
		_Size -= alignof(__m256i) * 2;
	}
	if (_Size >> 5)
	{
		DragonianLibMemcpy256_1(_Dst_Ptr, _Src_Ptr);
		++_Dst_Ptr;
		++_Src_Ptr;
		_Size -= alignof(__m256i);
	}
	if (_Size)
		memcpy(_Dst_Ptr, _Src_Ptr, _Size);
}

namespace SimdTypeTraits
{
	template <typename Type>
	constexpr bool IsVectorized = false;
	template <typename Type>
	constexpr bool IsVectorized<Vectorized<Type>> = true;
	template <typename Type>
	constexpr bool IsVectorizedValue = IsVectorized<RemoveARPCVType<Type>>;
	template <typename _Type>
	concept IsSimdVector = IsVectorizedValue<_Type>;

	template <typename _Type, typename _FunctionType, _FunctionType _Function>
	class IsAvxEnabled
	{
	public:
		_D_Dragonian_Lib_Constexpr_Force_Inline static bool Get()
		{
			static IsAvxEnabled CheckInstance;
			return Value;
		}
	private:
		static inline bool Value;
		_D_Dragonian_Lib_Constexpr_Force_Inline IsAvxEnabled()
		{
			if constexpr (TypeTraits::IsAvx256SupportedValue<_Type>)
			{
				if constexpr (requires(Vectorized<_Type>&_a, Vectorized<_Type>&_b) { _Function(_a, _b); })
				{
					try
					{
						_Function(Vectorized<_Type>(_Type(1)), Vectorized<_Type>(_Type(1)));
						Value = true;
						return;
					}
					catch (std::exception& _Except)
					{
						_D_Dragonian_Lib_Namespace GetDefaultLogger()->LogWarn(UTF8ToWideString(_Except.what()) + L" Some operator is not Avx256 implemented! It will fall back to scalar mode! ");
					}
				}
				else if constexpr (requires(Vectorized<_Type>&_a) { _Function(_a); })
				{
					try
					{
						_Function(Vectorized<_Type>(_Type(1)));
						Value = true;
						return;
					}
					catch (std::exception& _Except)
					{
						_D_Dragonian_Lib_Namespace GetDefaultLogger()->LogWarn(UTF8ToWideString(_Except.what()) + L" Some operator is not Avx256 implemented! It will fall back to scalar mode! ");
					}
				}
			}
			Value = false;
		}
	};
}

template <
	Int64 Throughput, typename Type,
	typename _FunctionTypePre, typename _FunctionTypeMid, typename _FunctionTypeEnd,
	typename _FunctionTypePreVec, typename _FunctionTypeMidVec
> Type ReduceFunction(
	const Type* _Src, SizeType _Size, const Type& _InitValue,
	_FunctionTypePre _PreFunction,
	_FunctionTypePreVec _PreFunctionVec,
	_FunctionTypeMid _MidFunction,
	_FunctionTypeMidVec _MidFunctionVec,
	_FunctionTypeEnd _EndFunction
) requires (IsInvocableValue<_FunctionTypeMid, Type, Type>)
{
	using VectorType = Vectorized<Type>;

	constexpr Int64 Stride = VectorType::Stride;
	constexpr Int64 LoopStride = Throughput * Stride;

	auto _MyResultValue = _InitValue;
	if constexpr (TypeTraits::IsAvx256SupportedValue<Type> && IsInvocableValue<_FunctionTypeMidVec, VectorType, VectorType>)
	{
		if (_Size >= LoopStride)
		{
			while (size_t(_Src) % sizeof(__m256) && _Size > 0)
			{
				auto _Value = *_Src++;
				if constexpr (IsInvocableValue<_FunctionTypePre, decltype(_Value)>)
					_Value = _PreFunction(_Value);
				_MyResultValue = _MidFunction(_MyResultValue, _Value);
				--_Size;
			}

			if (_Size >= LoopStride)
			{
				VectorType VectorizedValue[Throughput]; VectorType VectorizedSource[Throughput];
				for (Int64 i = 0; i < Throughput; ++i)
					VectorizedValue[i] = VectorType(_InitValue);
				while (_Size >= LoopStride)
				{
					for (Int64 i = 0; i < Throughput; ++i)
						VectorizedSource[i] = VectorType(_Src + i * Stride);
					if constexpr (IsInvocableValue<_FunctionTypePreVec, VectorType>)
						for (Int64 i = 0; i < Throughput; ++i)
							VectorizedSource[i] = _PreFunctionVec(VectorizedSource[i]);
					for (Int64 i = 0; i < Throughput; ++i)
						VectorizedValue[i] = _MidFunctionVec(VectorizedValue[i], VectorizedSource[i]);
					_Size -= LoopStride;
					_Src += LoopStride;
				}
				for (Int64 i = 1; i < Throughput; ++i)
					VectorizedValue[0] = _MidFunctionVec(VectorizedValue[0], VectorizedValue[i]);
				Type ResultVec[Stride];
				VectorizedValue[0].Store(ResultVec);
				for (Int64 i = 0; i < Stride; ++i)
					_MyResultValue = _MidFunction(_MyResultValue, ResultVec[i]);
			}
		}
	}

	while (_Size >= LoopStride)
	{
		for (Int64 i = 0; i < LoopStride; ++i)
		{
			auto _Value = _Src[i];
			if constexpr (IsInvocableValue<_FunctionTypePre, decltype(_Value)>)
				_Value = _PreFunction(_Value);
			_MyResultValue = _MidFunction(_MyResultValue, _Value);
		}
		_Src += LoopStride;
		_Size -= LoopStride;
	}

	while (_Size > 0)
	{
		auto _Value = *_Src++;
		if constexpr (IsInvocableValue<_FunctionTypePre, decltype(_Value)>)
			_Value = _PreFunction(_Value);
		_MyResultValue = _MidFunction(_MyResultValue, _Value);
		--_Size;
	}
	if constexpr (IsInvocableValue<_FunctionTypeEnd, decltype(_MyResultValue)>)
		return _EndFunction(_MyResultValue);
	else
		return _MyResultValue;
}

template <
	typename _RetType, typename _InputType, typename _ParameterType,
	typename _FunctionType, TypeDef::OperatorType _OType, bool IsCompare, Int64 OpThroughput
> void VectorizedFunction(
	_RetType* _Dest,
	SizeType _DestSize,
	const _InputType* _Src1,
	const _InputType* _Src2,
	_ParameterType _Value,
	const _FunctionType& _Function
)
{
	using VectorType = Vectorized<_InputType>;

	constexpr Int64 Stride = VectorType::Stride;
	constexpr Int64 LoopStride = OpThroughput * Stride;

	if constexpr (TypeTraits::IsAvx256SupportedValue<_InputType>)
	{
		if (SimdTypeTraits::IsAvxEnabled<_InputType, decltype(_Function), _Function>::Get())
		{
			while (size_t(_Dest) % sizeof(__m256) && _DestSize > 0)
			{
				if constexpr (_OType == TypeDef::UnaryOperatorType)
					*_Dest++ = (_RetType)_Function(*_Src1++);
				else if constexpr (_OType == TypeDef::BinaryOperatorType)
					*_Dest++ = (_RetType)_Function(*_Src1++, *_Src2++);
				else if constexpr (_OType == TypeDef::ConstantOperatorType)
					*_Dest++ = (_RetType)_Function(*_Src1++, _Value);
				else if constexpr (_OType == TypeDef::ReversedConstantOperatorType)
					*_Dest++ = (_RetType)_Function(_Value, *_Src1++);
				--_DestSize;
			}

			Vectorized<_InputType> VectorizedValue[OpThroughput * 2];
			if constexpr (_OType == TypeDef::ConstantOperatorType)
				VectorizedValue[OpThroughput] = Vectorized<_InputType>(_Value);

			while (_DestSize >= LoopStride)
			{
				for (Int64 i = 0; i < OpThroughput; ++i)
					VectorizedValue[i] = Vectorized<_InputType>(_Src1 + i * Stride);

				if constexpr (_OType == TypeDef::UnaryOperatorType)
					for (Int64 i = 0; i < OpThroughput; ++i)
						VectorizedValue[i] = _Function(VectorizedValue[i]);
				else if constexpr (_OType == TypeDef::BinaryOperatorType)
				{
					for (Int64 i = 0; i < OpThroughput; ++i)
						VectorizedValue[i + OpThroughput] = Vectorized<_InputType>(_Src2 + i * Stride);
					for (Int64 i = 0; i < OpThroughput; ++i)
						VectorizedValue[i] = _Function(VectorizedValue[i], VectorizedValue[i + OpThroughput]);
				}
				else if constexpr (_OType == TypeDef::ConstantOperatorType)
					for (Int64 i = 0; i < OpThroughput; ++i)
						VectorizedValue[i] = _Function(VectorizedValue[i], VectorizedValue[OpThroughput]);
				else if constexpr (_OType == TypeDef::ReversedConstantOperatorType)
					for (Int64 i = 0; i < OpThroughput; ++i)
						VectorizedValue[i] = _Function(VectorizedValue[OpThroughput], VectorizedValue[i]);

				for (Int64 i = 0; i < OpThroughput; ++i)
					if constexpr (IsCompare)
						VectorizedValue[i].StoreBool(_Dest + i * Stride);
					else
						VectorizedValue[i].Store(_Dest + i * Stride);

				_Dest += LoopStride; _Src1 += LoopStride; _DestSize -= LoopStride;
				if constexpr (_OType == TypeDef::BinaryOperatorType) _Src2 += LoopStride;
			}
		}
	}
	
	while (_DestSize >= LoopStride)
	{
		for (Int64 i = 0; i < LoopStride; ++i)
		{
			if constexpr (_OType == TypeDef::UnaryOperatorType)
				_Dest[i] = (_RetType)_Function(_Src1[i]);
			else if constexpr (_OType == TypeDef::BinaryOperatorType)
				_Dest[i] = (_RetType)_Function(_Src1[i], _Src2[i]);
			else if constexpr (_OType == TypeDef::ConstantOperatorType)
				_Dest[i] = (_RetType)_Function(_Src1[i], _Value);
			else if constexpr (_OType == TypeDef::ReversedConstantOperatorType)
				_Dest[i] = (_RetType)_Function(_Value, _Src1[i]);
		}
		_Dest += LoopStride;
		_Src1 += LoopStride;
		if constexpr (_OType == TypeDef::BinaryOperatorType)
			_Src2 += LoopStride;
		_DestSize -= LoopStride;
	}

	while (_DestSize > 0)
	{
		if constexpr (_OType == TypeDef::UnaryOperatorType)
			*_Dest++ = (_RetType)_Function(*_Src1++);
		else if constexpr (_OType == TypeDef::BinaryOperatorType)
			*_Dest++ = (_RetType)_Function(*_Src1++, *_Src2++);
		else if constexpr (_OType == TypeDef::ConstantOperatorType)
			*_Dest++ = (_RetType)_Function(*_Src1++, _Value);
		else if constexpr (_OType == TypeDef::ReversedConstantOperatorType)
			*_Dest++ = (_RetType)_Function(_Value, *_Src1++);
		--_DestSize;
	}
}

_D_Dragonian_Lib_Operator_Space_End