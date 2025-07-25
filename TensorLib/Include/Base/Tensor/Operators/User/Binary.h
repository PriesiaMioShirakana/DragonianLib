#pragma once
#include "TensorLib/Include/Base/Tensor/Operators/OperatorBase.h"

#define _D_Dragonian_Lib_Operator_Binary_Traits(_Function) \
namespace BinaryOperators \
{ \
    namespace _Function##Binary \
    { \
        template <class _ValueType> \
        concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) \
        { \
            { _D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l) } -> \
                TypeTraits::NotType<decltype(std::nullopt)>; \
        }; \
        template <class _ValueType> \
        concept HasVectorOperatorValue = requires(Vectorized<_ValueType> & __r, Vectorized<_ValueType> & __l) \
        { \
            { _D_Dragonian_Lib_Namespace Operators::BinaryOperators::_Function(__r, __l) } -> \
                TypeTraits::NotType<decltype(std::nullopt)>; \
        }; \
    } \
}

#define _D_Dragonian_Lib_Operator_Binary_Bool_Traits(_Function) \
namespace ComparisonOperators \
{ \
    namespace _Function##Binary \
    { \
        template <class _ValueType> \
        concept HasOperatorValue = requires(_ValueType & __r, _ValueType & __l) \
        { \
            { _D_Dragonian_Lib_Namespace Operators::ComparisonOperators::_Function(__r, __l) } -> \
                TypeTraits::NotType<decltype(std::nullopt)>; \
        }; \
        template <class _ValueType> \
        concept HasVectorOperatorValue = requires(Vectorized<_ValueType> & __r, Vectorized<_ValueType> & __l) \
        { \
            { _D_Dragonian_Lib_Namespace Operators::ComparisonOperators::_Function(__r, __l) } -> \
                TypeTraits::NotType<decltype(std::nullopt)>; \
        }; \
    } \
}

_D_Dragonian_Lib_Operator_Space_Begin

namespace BinaryOperators
{
	template <typename _Type>
	concept HasClampMaxOp = requires(const _Type & _Val, const _Type & _M) { { _Val > _M }; };
	template <typename _Type>
	concept HasClampMinOp = requires(const _Type & _Val, const _Type & _M) { { _Val < _M }; };
	template <typename _Type>
	concept HasClampOp = HasClampMaxOp<_Type> && HasClampMinOp<_Type>;

	template <typename _Type>
	constexpr _Type ClampMax(const _Type& Value, const _Type& Max)
		requires(HasClampMaxOp<_Type>)
	{
		return Value > Max ? Max : Value;
	}

	template <typename _Type>
	constexpr _Type ClampMin(const _Type& Value, const _Type& Min)
		requires(HasClampMinOp<_Type>)
	{
		return Value < Min ? Min : Value;
	}

	template <typename _Type>
	constexpr _Type Clamp(const _Type& Value, const _Type& Min, const _Type& Max)
		requires(HasClampOp<_Type>)
	{
		return Value < Min ? Min : Value > Max ? Max : Value;
	}
}

namespace BinaryOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

    constexpr class AddOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A + _B }; })
                return _Left + _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Add(_B) }; })
                return _Left.Add(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.add(_B) }; })
                return _Left.add(_Right);
            else
                return std::nullopt;
        }
    } Add;

    constexpr class SubOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A - _B }; })
                return _Left - _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Sub(_B) }; })
                return _Left.Sub(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.sub(_B) }; })
                return _Left.sub(_Right);
            else
                return std::nullopt;
        }
    } Sub;

    constexpr class MulOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A* _B }; })
                return _Left * _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Mul(_B) }; })
                return _Left.Mul(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.mul(_B) }; })
                return _Left.mul(_Right);
            else
                return std::nullopt;
        }
    } Mul;

    constexpr class DivOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A / _B }; })
                return _Left / _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Div(_B) }; })
                return _Left.Div(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.div(_B) }; })
                return _Left.div(_Right);
            else
                return std::nullopt;
        }
    } Div;

    constexpr class ModOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (IsFloatingPointValue<_Type>)
                return std::fmod(_Left, _Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A% _B }; })
                return _Left % _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Mod(_B) }; })
                return _Left.Mod(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.mod(_B) }; })
                return _Left.mod(_Right);
            else
                return std::nullopt;
        }
    } Mod;

    constexpr class AndOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A&& _B }; })
                return _Left && _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.And(_B) }; })
                return _Left.And(_Right);
            else
                return std::nullopt;
        }
    } And;

    constexpr class OrOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A || _B }; })
                return _Left || _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Or(_B) }; })
                return _Left.Or(_Right);
            else
                return std::nullopt;
        }
    } Or;

    constexpr class BitwiseAndOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(_Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A& _B }; })
                return _Left & _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.BitwiseAnd(_B) }; })
                return _Left.BitwiseAnd(_Right);
            else
                return std::nullopt;
        }
    } BitwiseAnd;

    constexpr class BitwiseOrOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(_Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A | _B }; })
                return _Left | _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.BitwiseOr(_B) }; })
                return _Left.BitwiseOr(_Right);
            else
                return std::nullopt;
        }
    } BitwiseOr;

    constexpr class XorOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A^ _B }; })
                return _Left ^ _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.Xor(_B) }; })
                return _Left.Xor(_Right);
            else
                return std::nullopt;
        }
    } Xor;

    constexpr class LShiftOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A << _B }; })
                return _Left << _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.LShift(_B) }; })
                return _Left.LShift(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.lshift(_B) }; })
                return _Left.lshift(_Right);
            else
                return std::nullopt;
        }
    } LShift;

    constexpr class RShiftOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A >> _B }; })
                return _Left >> _Right;
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.RShift(_B) }; })
                return _Left.RShift(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.rshift(_B) }; })
                return _Left.rshift(_Right);
            else
                return std::nullopt;
        }
    } RShift;

    constexpr class PowOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A.Pow(_B) }; })
                return _Left.Pow(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.pow(_B) }; })
                return _Left.pow(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { std::pow(_A, _B) }; })
                return std::pow(_Left, _Right);
            else
                return std::nullopt;
        }
    } Pow;

    constexpr class MaxOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A.Max(_B) }; })
                return _Left.Max(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.max(_B) }; })
                return _Left.max(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { std::max(_A, _B) }; })
                return std::max(_Left, _Right);
            else
                return std::nullopt;
        }
    } Max;

    constexpr class MinOp
    {
    public:
        template <typename _Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const _Type& _Left, const _Type& _Right)
        {
            if constexpr (requires(_Type & _A, _Type & _B) { { _A.Min(_B) }; })
                return _Left.Min(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { _A.min(_B) }; })
                return _Left.min(_Right);
            else if constexpr (requires(_Type & _A, _Type & _B) { { std::min(_A, _B) }; })
                return std::min(_Left, _Right);
            else
                return std::nullopt;
        }
    } Min;
}

_D_Dragonian_Lib_Operator_Binary_Traits(Add);
_D_Dragonian_Lib_Operator_Binary_Traits(Sub);
_D_Dragonian_Lib_Operator_Binary_Traits(Mul);
_D_Dragonian_Lib_Operator_Binary_Traits(Div);
_D_Dragonian_Lib_Operator_Binary_Traits(Mod);
_D_Dragonian_Lib_Operator_Binary_Traits(And);
_D_Dragonian_Lib_Operator_Binary_Traits(Or);
_D_Dragonian_Lib_Operator_Binary_Traits(Xor);
_D_Dragonian_Lib_Operator_Binary_Traits(BitwiseOr);
_D_Dragonian_Lib_Operator_Binary_Traits(BitwiseAnd);
_D_Dragonian_Lib_Operator_Binary_Traits(LShift);
_D_Dragonian_Lib_Operator_Binary_Traits(RShift);
_D_Dragonian_Lib_Operator_Binary_Traits(Pow);
_D_Dragonian_Lib_Operator_Binary_Traits(Max);
_D_Dragonian_Lib_Operator_Binary_Traits(Min);

namespace ComparisonOperators
{
	using namespace DragonianLib::Operators::SimdTypeTraits;

    constexpr class EqualOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (IsAnyOfValue<Type, float, double>)
                return std::fabs(A - B) <= std::numeric_limits<Type>::epsilon();
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left == _Right }; })
                return A == B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Equal(_Right) }; })
                return A.Equal(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.equal(_Right) }; })
                return A.equal(B);
            else
                return std::nullopt;
        }
    } Equal;

    constexpr class NotEqualOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (IsAnyOfValue<Type, float, double>)
                return std::fabs(A - B) > std::numeric_limits<Type>::epsilon();
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left != _Right }; })
                return A != B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.NotEqual(_Right) }; })
                return A.NotEqual(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.not_equal(_Right) }; })
                return A.not_equal(B);
            else
                return std::nullopt;
        }
    } NotEqual;

    constexpr class GreaterOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (requires(Type & _Left, Type & _Right) { { _Left > _Right }; })
                return A > B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Greater(_Right) }; })
                return A.Greater(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.greater(_Right) }; })
                return A.greater(B);
            else
                return std::nullopt;
        }
    } Greater;

    constexpr class GreaterEqualOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (requires(Type & _Left, Type & _Right) { { _Left >= _Right }; })
                return A >= B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.GreaterEqual(_Right) }; })
                return A.GreaterEqual(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.greater_equal(_Right) }; })
                return A.greater_equal(B);
            else
                return std::nullopt;
        }
    } GreaterEqual;

    constexpr class LessOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (requires(Type & _Left, Type & _Right) { { _Left < _Right }; })
                return A < B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.Less(_Right) }; })
                return A.Less(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.less(_Right) }; })
                return A.less(B);
            else
                return std::nullopt;
        }
    } Less;

    constexpr class LessEqualOp
    {
    public:
        template <typename Type>
        static _D_Dragonian_Lib_Constexpr_Force_Inline decltype(auto) operator()(const Type& A, const Type& B)
        {
            if constexpr (requires(Type & _Left, Type & _Right) { { _Left <= _Right }; })
                return A <= B;
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.LessEqual(_Right) }; })
                return A.LessEqual(B);
            else if constexpr (requires(Type & _Left, Type & _Right) { { _Left.less_equal(_Right) }; })
                return A.less_equal(B);
            else
                return std::nullopt;
        }
    } LessEqual;
}

_D_Dragonian_Lib_Operator_Binary_Bool_Traits(Equal);
_D_Dragonian_Lib_Operator_Binary_Bool_Traits(NotEqual);
_D_Dragonian_Lib_Operator_Binary_Bool_Traits(Greater);
_D_Dragonian_Lib_Operator_Binary_Bool_Traits(GreaterEqual);
_D_Dragonian_Lib_Operator_Binary_Bool_Traits(Less);
_D_Dragonian_Lib_Operator_Binary_Bool_Traits(LessEqual);



_D_Dragonian_Lib_Operator_Space_End

#undef _D_Dragonian_Lib_Operator_Binary_Traits
#undef _D_Dragonian_Lib_Operator_Binary_Bool_Traits
