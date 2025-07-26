#pragma once
#include <memory>
#include <unordered_map>

#include "Provider/Kernel.h"

#ifndef DRAGONIANLIB_CUDA_EP_BENCHMARK
#define DRAGONIANLIB_CUDA_EP_BENCHMARK 0
#endif // DRAGONIANLIB_CUDA_EP_BENCHMARK

#ifndef DRAGONIANLIB_CUDA_BLOCK_SIZE
#define DRAGONIANLIB_CUDA_BLOCK_SIZE 1024
#endif // DRAGONIANLIB_CUDA_BLOCK_SIZE

#if DRAGONIANLIB_CUDA_BLOCK_SIZE % 32 != 0
#error "DRAGONIANLIB_CUDA_BLOCK_SIZE must be a multiple of 32"
#endif // DRAGONIANLIB_CUDA_BLOCK_SIZE % 32 != 0

namespace DragonianLib
{
    namespace CudaModules
    {
        typedef struct __Handle* handle_t;
        typedef float moduleValueType;

        enum layerStatus_t : int8_t {
            LAYER_STATUS_FATAL_ERROR = -2,
            LAYER_STATUS_SIZE_MISMATCH = -1,
            LAYER_STATUS_SUCCESS = 0,
            LAYER_STATUS_NOT_INITIALIZED = 1,
            LAYER_STATUS_ALLOC_FAILED = 3,
            LAYER_STATUS_INVALID_VALUE = 7,
            LAYER_STATUS_ARCH_MISMATCH = 8,
            LAYER_STATUS_MAPPING_ERROR = 11,
            LAYER_STATUS_EXECUTION_FAILED = 13,
            LAYER_STATUS_INTERNAL_ERROR = 14,
            LAYER_STATUS_NOT_SUPPORTED = 15,
            LAYER_STATUS_LICENSE_ERROR = 16
        };

        handle_t createHandle();
        layerStatus_t destoryHandle(handle_t handle);
        const char* getErrorString(layerStatus_t errorId);
        layerStatus_t setHandleStream(handle_t handle, stream_t stream);
        stream_t getHandleStream(handle_t handle);

        template <typename T>
        class Tensor
        {
        public:
            Tensor() : Data(nullptr), N(0), C(0), H(0), W(0) {}
            Tensor(T* _Data, unsigned _N, unsigned _C, unsigned _H, unsigned _W) : Data(_Data), N(_N), C(_C), H(_H), W(_W), Dim(4), BufferSize(_N* _C* _H* _W) {}
            Tensor(T* _Data, unsigned _N, unsigned _H, unsigned _W) : Data(_Data), N(_N), C(1), H(_H), W(_W), Dim(3), BufferSize(_N* _H* _W) {}
            Tensor(T* _Data, unsigned _H, unsigned _W) : Data(_Data), N(1), C(1), H(_H), W(_W), Dim(2), BufferSize(_H* _W) {}
            Tensor(T* _Data, unsigned _W) : Data(_Data), N(1), C(1), H(1), W(_W), Dim(1), BufferSize(_W) {}

            Tensor(unsigned _N, unsigned _C, unsigned _H, unsigned _W) : Data(CudaProvider::cudaAlloc<T>(_W* _H* _C* _N)), N(_N), C(_C), H(_H), W(_W), Dim(4), BufferSize(_N* _C* _H* _W), Own(true) {}
            Tensor(unsigned _N, unsigned _H, unsigned _W) : Data(CudaProvider::cudaAlloc<T>(_W* _H* _N)), N(_N), C(1), H(_H), W(_W), Dim(3), BufferSize(_N* _H* _W), Own(true) {}
            Tensor(unsigned _H, unsigned _W) : Data(CudaProvider::cudaAlloc<T>(_W* _H)), N(1), C(1), H(_H), W(_W), Dim(2), BufferSize(_H* _W), Own(true) {}
            Tensor(unsigned _W) : Data(CudaProvider::cudaAlloc<T>(_W)), N(1), C(1), H(1), W(_W), Dim(1), BufferSize(_W), Own(true) {}
            ~Tensor()
            {
                Release();
            }

            Tensor(const Tensor&) = delete;
            Tensor& operator=(const Tensor&) = delete;
            Tensor(Tensor&& _Right) noexcept
            {
                Data = _Right.Data;
                N = _Right.N; C = _Right.C; H = _Right.H; W = _Right.W;
                Dim = _Right.Dim; BufferSize = _Right.BufferSize; Handle = _Right.Handle;
                Own = _Right.Own;

                _Right.Data = nullptr;
                _Right.N = _Right.C = _Right.H = _Right.W = _Right.Dim = _Right.BufferSize = 0;
                _Right.Handle = nullptr;
            }
            Tensor& operator=(Tensor&& _Right) noexcept
            {
                if (this != &_Right)
                {
                    Release();
                    Data = _Right.Data;
                    N = _Right.N; C = _Right.C; H = _Right.H; W = _Right.W;
                    Dim = _Right.Dim; BufferSize = _Right.BufferSize; Handle = _Right.Handle;
                    Own = _Right.Own;

                    _Right.Data = nullptr;
                    _Right.N = _Right.C = _Right.H = _Right.W = _Right.Dim = _Right.BufferSize = 0;
                    _Right.Handle = nullptr;
                }
                return *this;
            }

            void Resize(unsigned _N, unsigned _C, unsigned _H, unsigned _W)
            {
                if (Dim == 4 && N == _N && C == _C && H == _H && W == _W) return;

                const auto NewSize = _W * _H * _C * _N;
                
                if (NewSize > BufferSize)
                {
                    Release();
                    Data = CudaProvider::cudaAlloc<T>(NewSize);
                    BufferSize = NewSize;
                    Own = true;
                }

                N = _N; C = _C; H = _H; W = _W; Dim = 4;
            }
            void Resize(unsigned _N, unsigned _H, unsigned _W)
            {
                if (Dim == 3 && N == _N && H == _H && W == _W) return;

                const auto NewSize = _W * _H * _N;
                
                if (NewSize > BufferSize)
                {
                    Release();
                    Data = CudaProvider::cudaAlloc<T>(NewSize);
                    BufferSize = NewSize;
                    Own = true;
                }

                N = _N; C = 1; H = _H; W = _W; Dim = 3;
            }
            void Resize(unsigned _H, unsigned _W)
            {
                if (Dim == 2 && H == _H && W == _W) return;

                const auto NewSize = _W * _H;
                
                if (NewSize > BufferSize)
                {
                    Release();
                    Data = CudaProvider::cudaAlloc<T>(NewSize);
                    BufferSize = NewSize;
                    Own = true;
                }

                N = 1; C = 1; H = _H; W = _W; Dim = 2;
            }
            void Resize(unsigned _W)
            {
                if (Dim == 1 && W == _W) return;

                const auto NewSize = _W;
                
                if (NewSize > BufferSize)
                {
                    Release();
                    Data = CudaProvider::cudaAlloc<T>(NewSize);
                    BufferSize = NewSize;
                    Own = true;
                }

                N = 1; C = 1; H = 1; W = _W; Dim = 1;
            }
            void Resize(const Tensor& _Ref)
            {
                switch (_Ref.Dim)
                {
                case 4:
                    Resize(_Ref.N, _Ref.C, _Ref.H, _Ref.W);
                    break;
                case 3:
                    Resize(_Ref.N, _Ref.H, _Ref.W);
                    break;
                case 2:
                    Resize(_Ref.H, _Ref.W);
                    break;
                case 1:
                    Resize(_Ref.W);
                    break;
                default:
                    break;
                }
            }
            auto SizeInBytes() const
            {
                return sizeof(T) * N * C * W * H;
            }

            std::vector<T> Cpy2Host(stream_t _Stream = nullptr) const
            {
                auto size = N * C * H * W;
                std::vector<T> data(size);
                CudaProvider::cpy2Host(data.data(), Data, size, _Stream);
                return data;
            }

            template <typename Type>
            std::vector<Type> Cpy2Host(stream_t _Stream = nullptr) const
            {
                static_assert(sizeof(Type) == sizeof(T), "error");
                auto size = N * C * H * W;
                std::vector<Type> data(size);
                CudaProvider::cpy2Host(data.data(), (const Type*)Data, size, _Stream);
                return data;
            }

            Tensor Clone(stream_t _Stream = nullptr) const
            {
                if (!_Stream) _Stream = getHandleStream(Handle);

                Tensor Ret; Ret.Resize(*this);
                Ret.Handle = Handle;

                auto size = N * C * H * W;
                if (auto code = CudaProvider::cpyData(Ret.Data, Data, size, _Stream))
                    throw std::runtime_error(CudaProvider::getCudaError(code));
                
                return Ret;
            }

            Tensor& Copy(const Tensor& _Left, stream_t _Stream = nullptr)
            {
                if (Handle) CudaProvider::asyncCudaStream(getHandleStream(Handle));
                if (!_Stream) _Stream = getHandleStream(_Left.Handle);

                Resize(_Left);
                Handle = _Left.Handle;

                auto size = N * C * H * W;
                if (auto code = CudaProvider::cpyData(Data, _Left.Data, size, _Stream))
                    throw std::runtime_error(CudaProvider::getCudaError(code));

                return *this;
            }

            T* Data;
            unsigned N, C, H, W;
            unsigned Dim = 0;
            unsigned BufferSize = 0;
            handle_t Handle = nullptr;

        private:
            void Release()
            {
                if (Handle) CudaProvider::asyncCudaStream(getHandleStream(Handle));
                if (Own && Data)
                    if (auto Ret = CudaProvider::cudaFree(Data))
                    {
                        fprintf(stderr, "%s\n", CudaProvider::getCudaError(Ret));
                        abort();
                    }
                Data = nullptr;
                N = C = H = W = Dim = BufferSize = 0;
            }
            bool Own = false;
        };

        class Module : std::enable_shared_from_this<Module>
        {
        public:
            using DictType = std::unordered_map<std::string, Tensor<moduleValueType>>;
            Module() = delete;
            
            virtual ~Module() = default;
            virtual void SetName(const std::string& name) { Name = name; }
            virtual const std::string& GetName() const { return Name; }
            virtual void LoadModel(DictType& dict);

            Module(const Module&) = delete;
            Module(Module&&) = default;
            Module& operator=(const Module&) = delete;
            Module& operator=(Module&&) = default;

            Module(
                Module* parent,
                const std::string& name
            );

            std::string Name;
            std::vector<Module*> Children;
        };

        class MelKernel
        {
        public:
            MelKernel(
                unsigned samplingRate = 16000,
                unsigned melBins = 128,
                unsigned fftSize = 1024,
                unsigned windowSize = 1024,
                unsigned hopSize = 160,
                moduleValueType freqMin = 0.f,
                moduleValueType freqMax = 8000.f,
                moduleValueType clipVal = 1e-5f
            ) : m_samplingRate(samplingRate), m_melBins(melBins), m_fftSize(fftSize), m_windowSize(windowSize),
                m_hopSize(hopSize), m_freqMin(freqMin), m_freqMax(freqMax), m_clipVal(clipVal)
            {
                Init();
            }

            static layerStatus_t Stft(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& cacheWindows,
                unsigned fftSize = 1024,
                unsigned windowSize = 1024,
                unsigned hopSize = 160,
                const moduleValueType* window = nullptr,
                bool padding = true,
                bool center = false
            ) noexcept;

            layerStatus_t Forward(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& midCache,
                bool padding = true,
                bool center = false
            ) const noexcept;

        private:
            unsigned m_samplingRate = 16000;
            unsigned m_melBins = 128;
            unsigned m_fftSize = 1024;
            unsigned m_specBins = 513;
            unsigned m_windowSize = 1024;
            unsigned m_hopSize = 160;
            moduleValueType m_freqMin = 0.f;
            moduleValueType m_freqMax = 8000.f;
            moduleValueType m_clipVal = 1e-5f;
            Tensor<moduleValueType> m_melBasis;
            Tensor<moduleValueType> m_window;

            void Init();
        };

        class Parameter : public Module
        {
        public:
            Parameter() = delete;
            const Tensor<moduleValueType>& GetTensor() const { return TensorData; }
            Tensor<moduleValueType>& SetTensor() { return TensorData; }
            void LoadModel(DictType& dict) override;

            Parameter(
                Module* parent,
                const std::string& name,
                Tensor<moduleValueType>&& tensor,
                bool strict = true
            ) : Module(parent, name), TensorData(std::move(tensor)), Strict(strict)
            {

            }

        private:
			Tensor<moduleValueType> TensorData;
			bool Strict = true;
        };

        class Conv1D : public Module
        {
        public:
            Conv1D() = delete;

            layerStatus_t Forward(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& col
            ) const noexcept;

            Conv1D(
                Module* parent,
                const std::string& name,
                unsigned inputChannels,
                unsigned outputChannels,
                unsigned kernelSize,
                unsigned stride = 1,
                unsigned padding = 0,
                unsigned dilation = 1,
                unsigned groups = 1,
                bool bias = true
            );

        private:
            std::shared_ptr<Parameter> Weight;
            std::shared_ptr<Parameter> Bias;
            unsigned KernelSize = 1;
            unsigned Stride = 1;
            unsigned Padding = 0;
            unsigned Dilation = 1;
            unsigned Groups = 1;
            unsigned InputChannels = 1;
            unsigned OutputChannels = 1;
            bool BiasEnabled = true;
            bool IsContiguous = false;
            bool IsDwConv = false;
            bool UseGemm = false;
        };

        class LeakyReLU
        {
        public:
            LeakyReLU(moduleValueType negativeSlope = 1e-2f) : NegativeSlope(negativeSlope) {}

            layerStatus_t Forward(
                Tensor<moduleValueType>& output
            ) const noexcept;
        private:
            moduleValueType NegativeSlope = 1e-2f;  // NOLINT(clang-diagnostic-unused-private-field)
        };

        class GroupNorm1D : public Module
        {
        public:
            GroupNorm1D() = delete;

            layerStatus_t Forward(
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& mean,
                Tensor<moduleValueType>& var
            ) const noexcept;

            GroupNorm1D(
                Module* parent,
                const std::string& name,
                unsigned numGroups,
                unsigned numChannels,
                moduleValueType eps = 1e-5f,
                bool affine = true
            );

        private:
            std::shared_ptr<Parameter> Weight;
            std::shared_ptr<Parameter> Bias;
            unsigned NumGroups = 1;
            unsigned NumChannels = 1;
            moduleValueType Epsilon = 1e-5f;  // NOLINT(clang-diagnostic-unused-private-field)
            bool AffineEnabled = true;
        };

        class LayerNorm1D : public Module
        {
        public:
            LayerNorm1D() = delete;
            
            layerStatus_t Forward(
                Tensor<moduleValueType>& output,
                Tensor<moduleValueType>& mean,
                Tensor<moduleValueType>& var
			) const noexcept;

            LayerNorm1D(
                Module* parent,
                const std::string& name,
                unsigned numChannels,
                moduleValueType eps = 1e-5f,
                bool affine = true,
                bool bias = true
            );

        private:
            std::shared_ptr<Parameter> Weight;
            std::shared_ptr<Parameter> Bias;
            unsigned NumChannels = 1;
            moduleValueType Epsilon = 1e-5f;  // NOLINT(clang-diagnostic-unused-private-field)
            bool BiasEnabled = true;
            bool AffineEnabled = true;
        };

        class Linear : public Module
        {
        public:
            Linear() = delete;

            layerStatus_t Forward(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output
			) const noexcept;

            Linear(
                Module* parent,
                const std::string& name,
                unsigned inFeatureDim,
                unsigned outFeatureDim,
                bool bias = true
            );

        private:
            std::shared_ptr<Parameter> Weight;
            std::shared_ptr<Parameter> Bias;
            unsigned InFeatureDim = 1;
            unsigned OutFeatureDim = 1;
			bool BiasEnabled = true;
        };

        layerStatus_t AddTensor(
            Tensor<moduleValueType>& output,
            const Tensor<moduleValueType>& input
        ) noexcept;

        layerStatus_t SigmoidTensor(
            Tensor<moduleValueType>& output
        ) noexcept;

        class Transpose
        {
        public:
            Transpose() = default;

            static layerStatus_t Forward(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output
            ) noexcept;
        };

        class GLU
        {
        public:
            GLU() = default;

            static layerStatus_t Forward(
                const Tensor<moduleValueType>& input,
                Tensor<moduleValueType>& output
            ) noexcept;
        };

        class SiLU
        {
        public:
            SiLU() = default;

            static layerStatus_t Forward(
                Tensor<moduleValueType>& output
            ) noexcept;
        };
    }
}
