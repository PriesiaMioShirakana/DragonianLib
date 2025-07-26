#include <chrono>
#include <device_launch_parameters.h>

#include "base.h"
#include "cublas_v2.h"
#include "cufftw.h"

namespace DragonianLib
{
    namespace CudaProvider
    {
        extern thread_local std::string __LastError;
    }

    namespace CudaModules
    {

        class Timer  // NOLINT(cppcoreguidelines-special-member-functions)
        {
        public:
            Timer(std::string name, const handle_t* handle = nullptr) : Handle(handle), Name(std::move(name)), Start(std::chrono::high_resolution_clock::now()) {}
            ~Timer()
            {
                if (Handle && *Handle) CudaProvider::asyncCudaStream(getHandleStream(*Handle));
                const auto end = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - Start).count();
                printf("%s: %lld us\n", Name.c_str(), duration);
            }

        private:
            const handle_t* Handle;
            std::string Name;
            std::chrono::high_resolution_clock::time_point Start;
        };

        handle_t createHandle()
        {
            cublasHandle_t Handle;
            if (const auto Ret = cublasCreate(&Handle))
                fprintf(stderr, "%s\n", cublasGetStatusString(Ret));
            cublasSetMathMode(Handle, CUBLAS_TF32_TENSOR_OP_MATH);
            return handle_t(Handle);
        }

        layerStatus_t destoryHandle(handle_t handle)
        {
            return static_cast<layerStatus_t>(cublasDestroy(cublasHandle_t(handle)));
        }

        const char* getErrorString(layerStatus_t errorId)
        {
            if (errorId == LAYER_STATUS_SIZE_MISMATCH)
                return "Input size mismatch!";
            return cublasGetStatusString(static_cast<cublasStatus_t>(errorId));
        }

        layerStatus_t setHandleStream(handle_t handle, stream_t stream)
        {
            return static_cast<layerStatus_t>(cublasSetStream((cublasHandle_t)handle, (cudaStream_t)stream));
        }

        stream_t getHandleStream(handle_t handle)
        {
            cudaStream_t stream;
            if (const auto Ret = cublasGetStream((cublasHandle_t)handle, &stream))
                fprintf(stderr, "%s\n", cublasGetStatusString(Ret));
            return stream_t(stream);
        }

        static std::vector<moduleValueType> HannWindow(
            size_t WindowSize,
            bool Periodic = false
        )
        {
            std::vector<moduleValueType> Window(WindowSize);
            const size_t Denominator = Periodic ? WindowSize : WindowSize - 1;
            const auto Step = moduleValueType(2) * 3.1415926535f / static_cast<moduleValueType>(Denominator);  // NOLINT(modernize-use-std-numbers)
            for (size_t i = 0; i < WindowSize; i++)
                Window[i] = static_cast<moduleValueType>(0.5) * (static_cast<moduleValueType>(1) - cos(Step * static_cast<moduleValueType>(i)));
            return Window;
        }

        static double HZ2Mel(const double frequency)
        {
            constexpr auto f_min = 0.0;
            constexpr auto f_sp = 200.0 / 3;
            auto mel = (frequency - f_min) / f_sp;
            constexpr auto min_log_hz = 1000.0;
            constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;
            const auto logstep = log(6.4) / 27.0;
            if (frequency >= min_log_hz)
                mel = min_log_mel + log(frequency / min_log_hz) / logstep;
            return mel;
        }

        static double Mel2HZ(const double mel)
        {
            constexpr auto f_min = 0.0;
            constexpr auto f_sp = 200.0 / 3;
            auto freqs = f_min + f_sp * mel;
            constexpr auto min_log_hz = 1000.0;
            constexpr auto min_log_mel = (min_log_hz - f_min) / f_sp;
            const auto logstep = log(6.4) / 27.0;
            if (mel >= min_log_mel)
                freqs = min_log_hz * exp(logstep * (mel - min_log_mel));
            return freqs;
        }

        static std::vector<moduleValueType> Arange(moduleValueType begin, moduleValueType end, moduleValueType step)
        {
            auto size = size_t((end - begin) / step);
            std::vector<moduleValueType> ret;
            ret.reserve(size);
            while (size--)
            {
                ret.emplace_back(begin);
                begin += step;
            }
            return ret;
        }

        static std::vector<moduleValueType> Linspace(moduleValueType _Begin,moduleValueType _End,size_t _Count,bool _EndPoint = false)
        {
            if (_EndPoint)
            {
                const auto Step = (_End - _Begin) / moduleValueType(_Count - 1);
                return Arange(_Begin, _End + (Step * 1.01f), Step);
            }
            const auto Step = (_End - _Begin) / moduleValueType(_Count);
            return Arange(_Begin, _End + Step * 0.01f, Step);
        }

        static std::vector<moduleValueType> Diff(const std::vector<moduleValueType>& input)
        {
            if (input.size() < 2)
                return {};
            std::vector<moduleValueType> differences;
            differences.reserve(input.size() - 1);
            for (size_t i = 1; i < input.size(); ++i)
                differences.emplace_back(input[i] - input[i - 1]);
            return differences;
        }

        static std::vector<std::vector<moduleValueType>> Outer(const std::vector<moduleValueType>& a, const std::vector<moduleValueType>& b)
        {
            std::vector<std::vector<moduleValueType>> result(a.size(), std::vector<moduleValueType>(b.size()));
            for (size_t i = 0; i < a.size(); ++i)
                for (size_t j = 0; j < b.size(); ++j)
                    result[i][j] = a[i] - b[j];
            return result;
        }

        static std::vector<moduleValueType> operator-(const std::vector<moduleValueType>& a)
        {
            std::vector<moduleValueType> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = -a[i];
            return result;
        }

        static std::vector<moduleValueType> operator/(const std::vector<moduleValueType>& a, moduleValueType b)
        {
            std::vector<moduleValueType> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = a[i] / b;
            return result;
        }

        static std::vector<moduleValueType> Max(const std::vector<moduleValueType>& a, moduleValueType b)
        {
            std::vector<moduleValueType> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = std::max(a[i], b);
            return result;
        }

        static std::vector<moduleValueType> Min(const std::vector<moduleValueType>& a, const std::vector<moduleValueType>& b)
        {
            if (a.size() != b.size())
                throw std::invalid_argument("Vectors a and b must have the same size.");

            std::vector<moduleValueType> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = std::min(a[i], b[i]);
            return result;
        }

        static std::vector<moduleValueType> operator-(const std::vector<moduleValueType>& a, const std::vector<moduleValueType>& b)
        {
            if (a.size() != b.size())
                throw std::invalid_argument("Vectors a and b must have the same size.");

            std::vector<moduleValueType> result(a.size());
            for (size_t i = 0; i < a.size(); ++i)
                result[i] = a[i] - b[i];
            return result;
        }

        static std::vector<std::vector<moduleValueType>>& operator*=(std::vector<std::vector<moduleValueType>>& input, moduleValueType val)
        {
            for (auto& i : input)
                for (auto& j : i)
                    j *= val;
            return input;
        }

        static std::vector<std::vector<moduleValueType>>& operator/=(std::vector<std::vector<moduleValueType>>& input, const std::vector<moduleValueType>& val)
        {
            if (input.size() == val.size())
                for (size_t i = 0; i < input.size(); ++i)
                    for (size_t j = 0; j < input[0].size(); ++j)
                        input[i][j] /= val[i];
            return input;
        }

        void MelKernel::Init()
        {
            if (m_fftSize < m_windowSize)
                throw std::runtime_error("m_fftSize could not less than m_windowSize");

            m_window.Resize(m_windowSize);
            const auto Window = HannWindow(m_windowSize);
            CudaProvider::cpy2Device(m_window.Data, Window.data(), Window.size(), nullptr);

            const double MEL_MIN = HZ2Mel(static_cast<double>(m_freqMin));
            const double MEL_MAX = HZ2Mel(static_cast<double>(m_freqMax));
            m_specBins = m_fftSize / 2 + 1;

            auto Weight = std::vector<std::vector<moduleValueType>>(m_melBins, std::vector<moduleValueType>(m_specBins));
        	/*Tensor<Float32, 2, Device::CPU>::Empty(
                Dimensions{ MEL_BINS, FFT_BINS }
            );*/
            const auto DSR = 1.f / moduleValueType(m_samplingRate);
            const auto VAl = 1.f / (DSR * moduleValueType(m_fftSize));
            const auto N = moduleValueType(m_specBins);
            auto FFT_FREQS = Arange(
                0.f, N, 1.f
            );
            auto MEL_F = Linspace(
                moduleValueType(MEL_MIN), moduleValueType(MEL_MAX), m_melBins + 2, true
            );

            for (auto& POINT : MEL_F)
                POINT = (moduleValueType)Mel2HZ((double)POINT);
            for (auto& POINT : FFT_FREQS)
                POINT *= VAl;

            const auto F_DIFF = Diff(MEL_F);
            const auto RAMPS = Outer(MEL_F, FFT_FREQS);

            for (unsigned i = 0; i < m_melBins; ++i)
            {
                auto UPPER = RAMPS[i + 2] / F_DIFF[i + 1];
                auto LOWER = -RAMPS[i] / F_DIFF[i];
                Weight[i] = Max(Min(LOWER, UPPER), 0.f);
            }
            const auto ENORM = std::vector<moduleValueType>{ MEL_F.begin() + 2, MEL_F.end() } -
                std::vector<moduleValueType>{ MEL_F.begin(), MEL_F.begin() + m_melBins };
            (Weight *= 2) /= ENORM;
            m_melBasis.Resize(m_melBins, m_specBins);
            for (size_t i = 0; i < m_melBins; ++i)
                CudaProvider::cpy2Device(
                    m_melBasis.Data + i * m_specBins,
                    Weight[i].data(),
                    Weight[i].size(),
                    nullptr
                );
        }

        static __global__ void appendWindow(
            moduleValueType* outWindows,
            const moduleValueType* inSignal,
            const moduleValueType* inWindow,
            unsigned frameCount,
            unsigned fftSize,
            unsigned signalLength,
            unsigned windowSize,
            unsigned hopSize,
            unsigned paddingCount
        )
        {
			//[batchSize, frameCount, fftSize]
			//[batchSize, signalLength]
            const unsigned batchIdx = blockIdx.z;
            const unsigned frameIdx = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned sigIdx = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned off = (fftSize - windowSize) / 2;

            if (frameIdx >= frameCount || sigIdx >= fftSize)
                return;

            const int srcBegIdx = int(frameIdx * hopSize) - int(paddingCount) / 2;
			const unsigned curRemain = signalLength - srcBegIdx - 1;

			//inSignal[batchSize, :]
			const auto srcData = inSignal + (ptrdiff_t)batchIdx * signalLength + srcBegIdx;
			//outWindows[batchIdx, frameIdx, :]
			const auto dstData = outWindows + (ptrdiff_t)(batchIdx * frameCount + frameIdx) * fftSize;

            if (sigIdx < off || sigIdx > curRemain || int(sigIdx) + srcBegIdx < 0)
            	dstData[sigIdx] = 0.f;
            else
                dstData[sigIdx] = inWindow[sigIdx - off] * srcData[sigIdx];
        }

        layerStatus_t MelKernel::Stft(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output,
            Tensor<moduleValueType>& cacheWindows,
            unsigned fftSize,
			unsigned windowSize,
			unsigned hopSize,
			const moduleValueType* window,
            bool padding,
            bool center
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
            auto __BENCH_TM_BEG = Timer("Stft", &output.Handle);
#endif

            if (!window)
            {
				CudaProvider::__LastError = "Window is not set!";
	            return LAYER_STATUS_FATAL_ERROR;
            }

			const unsigned m_specBins = fftSize / 2 + 1;
            const auto Stream = (cudaStream_t)getHandleStream(input.Handle);

            auto paddingCount = padding ? windowSize - hopSize : 0;
            paddingCount = center ? windowSize : paddingCount;
			printf("Padding count: %u\n", paddingCount);
            const auto signalBatch = input.H * input.C * input.N; const auto signalLength = input.W;
            const auto frames = (signalLength - windowSize + paddingCount) / hopSize + 1;
            //const auto correctSize = frames * hopSize + (fftSize - hopSize);

            cacheWindows.Resize(signalBatch, frames, fftSize);
            if (cacheWindows.Handle) CudaProvider::asyncCudaStream(getHandleStream(cacheWindows.Handle));
            cacheWindows.Handle = input.Handle;

            dim3 blockSize(
                DRAGONIANLIB_CUDA_BLOCK_SIZE / 32,
                32
            );
            dim3 gridSize(
                (fftSize + blockSize.x - 1) / blockSize.x,
                (frames + blockSize.y - 1) / blockSize.y,
                signalBatch
            );

            appendWindow<<<gridSize, blockSize, 0, Stream>>>(
                cacheWindows.Data,
                input.Data,
                window,
                frames,
                fftSize,
                signalLength,
                windowSize,
                hopSize,
                paddingCount
            );

            output.Resize(signalBatch, frames, m_specBins, 2);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            cufftHandle handle;
            int n[] = { static_cast<int>(fftSize) };
            int inembed[] = { static_cast<int>(signalBatch * frames), static_cast<int>(fftSize) };
            int oembed[] = { static_cast<int>(signalBatch * frames), static_cast<int>(m_specBins) };
            if (cufftPlanMany(
                &handle, 1, n,
                inembed, 1, static_cast<int>(fftSize),
                oembed, 1, static_cast<int>(m_specBins),
                CUFFT_R2C, static_cast<int>(signalBatch * frames)) ||
                cufftSetStream(handle, Stream) ||
                cufftExecR2C(handle, cacheWindows.Data, (cuFloatComplex*)output.Data) ||
                cudaStreamSynchronize(Stream))
            {
                if (handle) cufftDestroy(handle);
                return LAYER_STATUS_FATAL_ERROR;
            }
            cufftDestroy(handle);
            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __global__ void implPowerSpec(
            moduleValueType* oPowerSpec,
            const moduleValueType* iSpec,
            unsigned frameLength,
            unsigned specBins
        )
        {
			const unsigned batchIdx = blockIdx.z;
			const unsigned binIdx = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned frameIdx = blockIdx.x * blockDim.x + threadIdx.x;

            if (frameIdx >= frameLength || binIdx >= specBins)
				return;

            //batchSize, specBins, frameLength
            const unsigned outIdx = (batchIdx * specBins + binIdx) * frameLength + frameIdx;
            //batchSize, frameLength, specBins, 2
            const unsigned inIdx = ((batchIdx * frameLength + frameIdx) * specBins + binIdx) * 2;

			const moduleValueType real = iSpec[inIdx], imag = iSpec[inIdx + 1];
            oPowerSpec[outIdx] = sqrtf(real * real + imag * imag + 1e-9f);
        }

        static __global__ void dynamicRangeCompression(
            moduleValueType* output,
            unsigned size,
            moduleValueType clipVal,
            moduleValueType C
        )
        {
            const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size)
                return;
            auto& val = output[idx];
            val = logf((val < clipVal ? clipVal : val) * C);
        }

        layerStatus_t MelKernel::Forward(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output,
            Tensor<moduleValueType>& midCache,
            bool padding,
            bool center
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
            auto __BENCH_TM_BEG = Timer("MelSpec", &output.Handle);
#endif

            if (const auto Ret = Stft(
                input,
                output,
                midCache,
				m_fftSize,
				m_windowSize,
				m_hopSize,
				m_window.Data,
                padding,
                center
            )) return Ret;

            const unsigned batchSize = output.N;
            const unsigned frameLength = output.C;
            const unsigned specBins = output.H;
            const unsigned melBins = m_melBins;
            if (specBins != m_specBins)
                return LAYER_STATUS_SIZE_MISMATCH;

            midCache.Resize(batchSize, specBins, frameLength); //power spec
            if (midCache.Handle) CudaProvider::asyncCudaStream(getHandleStream(midCache.Handle));
            midCache.Handle = input.Handle;

            const auto Stream = (cudaStream_t)getHandleStream(output.Handle);
            dim3 blockSize(
                DRAGONIANLIB_CUDA_BLOCK_SIZE / 32,
                32
            );
            dim3 gridSize(
                (frameLength + blockSize.x - 1) / blockSize.x,
                (specBins + blockSize.y - 1) / blockSize.y,
                batchSize
            );
            implPowerSpec<<<gridSize, blockSize, 0, Stream>>>(
                midCache.Data,
                output.Data,
                frameLength,
                specBins
                );
            cudaStreamSynchronize(Stream);
            //[batchSize, specBins, frameLength] * [1, melBins, specBins] -> [batchSize, melBins, frameLength]
            output.Resize(batchSize, melBins, frameLength);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
			output.Handle = input.Handle;

            static constexpr moduleValueType Alpha = 1.0f;
            static constexpr moduleValueType Beta = 0.0f;

            if (auto Ret = cublasSgemmStridedBatched(
                cublasHandle_t(output.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                (int)frameLength, (int)melBins, (int)specBins,
                &Alpha,
                midCache.Data, (int)frameLength, (ptrdiff_t)specBins * frameLength,
                m_melBasis.Data, (int)specBins, 0,
                &Beta,
                output.Data, (int)frameLength, (ptrdiff_t)melBins * frameLength,
                (int)batchSize
            )) return static_cast<layerStatus_t>(Ret);

            const auto size = output.N * output.C * output.W * output.H;
            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);
            dynamicRangeCompression<<<gridLength, blockLength, 0, Stream>>>(
                output.Data,
                size,
                1e-5f,
                1.f
                );
            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        Module::Module(Module* parent, const std::string& name)
        {
            if (parent)
            {
                if (parent->Name.empty())
                    Name = name;
                else
                    Name = parent->Name + '.' + name;
                parent->Children.emplace_back(this);
            }
            else
                Name = name;
        }

        void Module::LoadModel(DictType& dict)
        {
            for (const auto layer : Children)
                layer->LoadModel(dict);
        }

        void Parameter::LoadModel(DictType& dict)
        {
	        const auto weight = dict.find(Name);
            if (weight != dict.end())
            {
                if (Strict)
                {
                    if (weight->second.N != TensorData.N ||
                        weight->second.C != TensorData.C ||
                        weight->second.H != TensorData.H ||
                        weight->second.W != TensorData.W)
                        throw std::runtime_error("Parameter " + Name + " shape mismatch in model dictionary.");
                }
                TensorData = std::move(weight->second);
            }
            else
                throw std::runtime_error("Parameter " + Name + " not found in model dictionary.");
        }

        static __global__ void broadcastAddKernel(moduleValueType* A, const moduleValueType* bias, unsigned YL, unsigned XL)
    	{
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ moduleValueType broadcastAddSharedBias[];

            if (threadIdx.y == 0 && x < XL)
                broadcastAddSharedBias[threadIdx.x] = bias[x];

            __syncthreads();

            if (y < YL && x < XL)
                A[y * XL + x] += broadcastAddSharedBias[threadIdx.x];
        }

        template <unsigned blockSizeX>
		static __global__ void layerReduceMeanKernel(
			const moduleValueType* iFeat,
			moduleValueType* oMean,
            unsigned sampleCount,
			unsigned featureSize
		)
		{
			//shape: [sampleCount, featureSize] -> [[gridDim.y, blockDim.y], [gridDim.x, blockDim.x]]
			//idx = sampleIdx * featureSize + featureIdx
			const unsigned ty = threadIdx.y;
			const unsigned tx = threadIdx.x;
			const unsigned featureIdx = blockIdx.x * blockDim.x + tx;
            const unsigned batchIdx = blockIdx.y * blockDim.y + ty;
            const unsigned sharedIdx = ty * blockDim.x + tx;

			extern __shared__ moduleValueType sharedReduceMeanData[];

            //[4, 256]
            //sharedReduceMeanData[blockDim.y][blockDim.x]    ty * blockDim.x + tx = i
			sharedReduceMeanData[sharedIdx] = 0.f;
			if (featureIdx >= featureSize || batchIdx >= sampleCount)
				return;

			sharedReduceMeanData[sharedIdx] = iFeat[batchIdx * featureSize + featureIdx];

			__syncthreads();

			if constexpr (blockSizeX >= 1024)
			{
				if (tx < 512) sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 512];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 512)
			{
				if (tx < 256) sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 256];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 256)
			{
				if (tx < 128) sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 128];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 128)
			{
				if (tx < 64) sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 64];
				__syncthreads();
			}
			if (tx < 32)
			{
				if constexpr (blockSizeX >= 64)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 32];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 32)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 16];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 16)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 8];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 8)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 4];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 4)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 2];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 2)
				{
					sharedReduceMeanData[sharedIdx] += sharedReduceMeanData[sharedIdx + 1];
					__syncthreads();
				}
				if (tx == 0) atomicAdd(oMean + batchIdx, sharedReduceMeanData[(ptrdiff_t)ty * blockDim.x] / moduleValueType(featureSize));
			}
		}

		template <unsigned blockSizeX>
		static __global__ void layerReduceVarKernel(
			const moduleValueType* iFeat,
			const moduleValueType* iMean,
			moduleValueType* oVar,
            unsigned sampleCount,
			unsigned featureSize
		)
		{
			//shape: [sampleCount, featureSize] -> [[gridDim.y, blockDim.y], [gridDim.x, blockDim.x]]
			//idx = sampleIdx * featureSize + featureIdx
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;
            const unsigned featureIdx = blockIdx.x * blockDim.x + tx;
            const unsigned batchIdx = blockIdx.y * blockDim.y + ty;
            const unsigned sharedIdx = ty * blockDim.x + tx;

			extern __shared__ moduleValueType sharedReduceVarData[];

            sharedReduceVarData[sharedIdx] = 0.f;
            if (featureIdx >= featureSize || batchIdx >= sampleCount)
                return;

			{
				const moduleValueType x = iFeat[batchIdx * featureSize + featureIdx] - iMean[batchIdx];
				sharedReduceVarData[sharedIdx] = x * x;
				__syncthreads();
			}

			if constexpr (blockSizeX >= 1024)
			{
				if (tx < 512) sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 512];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 512)
			{
				if (tx < 256) sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 256];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 256)
			{
				if (tx < 128) sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 128];
				__syncthreads();
			}
			if constexpr (blockSizeX >= 128)
			{
				if (tx < 64) sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 64];
				__syncthreads();
			}
			if (tx < 32)
			{
				if constexpr (blockSizeX >= 64)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 32];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 32)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 16];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 16)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 8];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 8)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 4];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 4)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 2];
					__syncthreads();
				}
				if constexpr (blockSizeX >= 2)
				{
					sharedReduceVarData[sharedIdx] += sharedReduceVarData[sharedIdx + 1];
					__syncthreads();
				}
                if (tx == 0) atomicAdd(oVar + batchIdx, sharedReduceVarData[(ptrdiff_t)ty * blockDim.x] / moduleValueType(featureSize));
			}
		}

        static __global__ void implNormalizeKernel(
            moduleValueType* ioFeat,
            const moduleValueType* iMean,
            const moduleValueType* iVar,
            unsigned sampleCount,
            unsigned featureSize,
            moduleValueType eps
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            if (x >= featureSize || y >= sampleCount)
	            return;

            extern __shared__ moduleValueType implNormalizeSharedMem[];

            const auto implNormalizeShared = implNormalizeSharedMem + 2ll * ty;

            if (threadIdx.x == 0)
            {
                implNormalizeShared[0] = iMean[y];
                implNormalizeShared[1] = sqrtf(iVar[y] + eps);
            }

            __syncthreads();

            const auto idx = y * featureSize + x;
            ioFeat[idx] = (ioFeat[idx] - implNormalizeShared[0]) / implNormalizeShared[1];
        }

		static void inplaceNorm(
			moduleValueType* ioFeat,
			moduleValueType* ioMean,
			moduleValueType* ioVar,
			unsigned sampleCount,
			unsigned featureSize,
			moduleValueType eps,
			cudaStream_t cudaStream
		)
		{
            cudaMemsetAsync(ioMean, 0, sizeof(moduleValueType) * sampleCount, cudaStream);
            cudaMemsetAsync(ioVar, 0, sizeof(moduleValueType) * sampleCount, cudaStream);

            dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE / 4, 4);
            dim3 gridSize(
                (featureSize + blockSize.x - 1) / blockSize.x,
                (sampleCount + blockSize.y - 1) / blockSize.y
            );

			constexpr auto sharedMemSize = DRAGONIANLIB_CUDA_BLOCK_SIZE * sizeof(moduleValueType);

			layerReduceMeanKernel<DRAGONIANLIB_CUDA_BLOCK_SIZE / 4><<<gridSize, blockSize, sharedMemSize, cudaStream>>>(
				ioFeat,
				ioMean,
                sampleCount,
				featureSize
				);

			layerReduceVarKernel<DRAGONIANLIB_CUDA_BLOCK_SIZE / 4><<<gridSize, blockSize, sharedMemSize, cudaStream>>>(
				ioFeat,
				ioMean,
                ioVar,
                sampleCount,
				featureSize
				);

			implNormalizeKernel<<<gridSize, blockSize, sharedMemSize, cudaStream>>>(
				ioFeat, 
				ioMean,
                ioVar,
                sampleCount,
				featureSize,
				eps
				);
		}

        static __global__ void implAffineKernel(
            moduleValueType* output,
            const moduleValueType* weight,
            unsigned batchSize,
            unsigned numChannel
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ moduleValueType implAffineSharedMem[];

            if (y >= batchSize || x >= numChannel)
                return;

            if (threadIdx.y == 0)
                implAffineSharedMem[threadIdx.x] = weight[x];

            __syncthreads();

            output[y * numChannel + x] *= implAffineSharedMem[threadIdx.x];
        }

        static __global__ void implAffineBiasKernel(
            moduleValueType* output,
            const moduleValueType* weight,
            const moduleValueType* bias,
            unsigned batchSize,
            unsigned numChannel
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ moduleValueType implAffineBiasSharedMem[];

            if (y >= batchSize || x >= numChannel)
                return;

            if (threadIdx.y == 0)
            {
                implAffineBiasSharedMem[threadIdx.x] = weight[x];
                implAffineBiasSharedMem[threadIdx.x + blockDim.x] = bias[x];
            }

            __syncthreads();

            (output[y * numChannel + x] *= implAffineBiasSharedMem[threadIdx.x]) +=
                implAffineBiasSharedMem[threadIdx.x + blockDim.x];
        }

        static __global__ void implAffineBias2DKernel(
            moduleValueType* output,
            const moduleValueType* weight,
            const moduleValueType* bias,
            unsigned numChannel,
            unsigned featureSize
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ moduleValueType implAffineBias2DSharedMem[];

            if (y >= numChannel || x >= featureSize)
                return;

            if (threadIdx.x == 0)
            {
                implAffineBias2DSharedMem[threadIdx.y] = weight[y];
                implAffineBias2DSharedMem[threadIdx.y + blockDim.y] = bias[y];
            }

            __syncthreads();

            (output[(blockIdx.z * numChannel + y) * featureSize + x] *= implAffineBias2DSharedMem[threadIdx.y]) +=
                implAffineBias2DSharedMem[threadIdx.y + blockDim.y];
        }

        static __global__ void implBias2DKernel(
            moduleValueType* output,
            const moduleValueType* bias,
            unsigned numChannel,
            unsigned featureSize
        )
        {
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            extern __shared__ moduleValueType implBias2DSharedMem[];

            if (y >= numChannel || x >= featureSize)
                return;

            if (threadIdx.x == 0)
	            implBias2DSharedMem[ty] = bias[y];

            __syncthreads();
            
        	output[(blockIdx.z * numChannel + y) * featureSize + x] += implBias2DSharedMem[ty];
        }

        Linear::Linear(
            Module* parent,
            const std::string& name,
            unsigned inFeatureDim,
            unsigned outFeatureDim,
            bool bias
        ) : Module(parent, name), InFeatureDim(inFeatureDim), OutFeatureDim(outFeatureDim), BiasEnabled(bias)
        {
            Weight = std::make_shared<Parameter>(
                this, "weight", Tensor<moduleValueType>(OutFeatureDim, InFeatureDim)
            );
            if (BiasEnabled)
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<moduleValueType>(OutFeatureDim)
                );
        }

        layerStatus_t Linear::Forward(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Linear " + Name, &output.Handle);
#endif

	        const unsigned inFeature = input.W;
            if (inFeature != InFeatureDim)
                return LAYER_STATUS_SIZE_MISMATCH;

	        const unsigned inputSize = input.N * input.C * input.H;
            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.H, OutFeatureDim);
            else if (input.Dim == 3)
                output.Resize(input.N, input.H, OutFeatureDim);
            else if (input.Dim == 2)
                output.Resize(input.H, OutFeatureDim);
            else
                output.Resize(OutFeatureDim);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            static constexpr moduleValueType Alpha = 1.f;
            static constexpr moduleValueType Beta = 0.f;

            if (auto Ret = cublasSgemm(
                cublasHandle_t(input.Handle), CUBLAS_OP_T, CUBLAS_OP_N,
                (int)OutFeatureDim, (int)inputSize, (int)InFeatureDim,
                &Alpha,
                Weight->GetTensor().Data, (int)InFeatureDim,
                input.Data, (int)InFeatureDim,
                &Beta,
                output.Data, (int)OutFeatureDim
            )) return static_cast<layerStatus_t>(Ret);

            if (BiasEnabled)
            {
                dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
                dim3 gridSize(
                    (OutFeatureDim + blockSize.x - 1) / blockSize.x,
                    (inputSize + blockSize.y - 1) / blockSize.y
                );
                unsigned sharedMemSize = blockSize.x * sizeof(moduleValueType);
                broadcastAddKernel<<<gridSize, blockSize, sharedMemSize, cudaStream_t(getHandleStream(input.Handle))>>>
                    (output.Data, Bias->GetTensor().Data, inputSize, OutFeatureDim);
            }

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        LayerNorm1D::LayerNorm1D(
            Module* parent,
            const std::string& name,
            unsigned numChannels,
            moduleValueType eps,
            bool affine,
            bool bias
        ) : Module(parent, name), NumChannels(numChannels), Epsilon(eps), BiasEnabled(bias), AffineEnabled(affine)
        {
            if (AffineEnabled)
            {
                Weight = std::make_shared<Parameter>(
                    this, "weight", Tensor<moduleValueType>(numChannels)
                );
                if (BiasEnabled)
                    Bias = std::make_shared<Parameter>(
                        this, "bias", Tensor<moduleValueType>(numChannels)
                    );
            }
        }

        layerStatus_t LayerNorm1D::Forward(
            // ReSharper disable once CppParameterMayBeConstPtrOrRef
            Tensor<moduleValueType>& output,
            Tensor<moduleValueType>& mean,
            Tensor<moduleValueType>& var
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("LayerNorm1D " + Name, &output.Handle);
#endif

            const auto featureDim = output.W;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            const unsigned sampleCountNorm = output.N * output.C * output.H;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);
            if (mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(mean.Handle));
            mean.Handle = output.Handle;
            if (var.Handle) CudaProvider::asyncCudaStream(getHandleStream(var.Handle));
            var.Handle = output.Handle;

            auto Stream = cudaStream_t(getHandleStream(output.Handle));

            inplaceNorm(
                output.Data,
                mean.Data,
                var.Data,
                sampleCountNorm,
                NumChannels,
                Epsilon,
                Stream
            );

            if (AffineEnabled)
            {
                dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
                dim3 gridSize(
                    (NumChannels + blockSize.x - 1) / blockSize.x,
                    (sampleCountNorm + blockSize.y - 1) / blockSize.y
                );
                unsigned sharedMemSize = blockSize.x * 2ull * sizeof(moduleValueType);
	            if (BiasEnabled)
                    implAffineBiasKernel<<<gridSize, blockSize, sharedMemSize, Stream>>>
						(output.Data, Weight->GetTensor().Data, Bias->GetTensor().Data, sampleCountNorm, NumChannels);
                else
                    implAffineKernel<<<gridSize, blockSize, sharedMemSize, Stream>>>
						(output.Data, Weight->GetTensor().Data, sampleCountNorm, NumChannels);
            }

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        GroupNorm1D::GroupNorm1D(
            Module* parent,
            const std::string& name,
            unsigned numGroups,
            unsigned numChannels,
            moduleValueType eps,
            bool affine
        ) : Module(parent, name), NumGroups(numGroups), NumChannels(numChannels), Epsilon(eps), AffineEnabled(affine)
        {
            if (NumChannels % NumGroups)
                throw std::logic_error("NumChannels must be exactly divisible by NumGroups");

            if (AffineEnabled)
            {
                Weight = std::make_shared<Parameter>(
                    this, "weight", Tensor<moduleValueType>(numChannels)
                );
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<moduleValueType>(numChannels)
                );
            }
        }

        layerStatus_t GroupNorm1D::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<moduleValueType>& output,
            Tensor<moduleValueType>& mean,
            Tensor<moduleValueType>& var
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("GroupNorm1D " + Name, &output.Handle);
#endif

	        const unsigned featureDim = output.H;
            if (featureDim != NumChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

	        const unsigned batchSize = output.N * output.C;
	        const unsigned featureSize = output.W;

	        const unsigned sampleCountNorm = batchSize * NumGroups;
	        const unsigned featureSizeNorm = output.H * output.W / NumGroups;
            mean.Resize(sampleCountNorm);
            var.Resize(sampleCountNorm);
            if (mean.Handle) CudaProvider::asyncCudaStream(getHandleStream(mean.Handle));
            mean.Handle = output.Handle;
            if (var.Handle) CudaProvider::asyncCudaStream(getHandleStream(var.Handle));
            var.Handle = output.Handle;

            auto Stream = cudaStream_t(getHandleStream(output.Handle));

            inplaceNorm(
                output.Data,
                mean.Data,
                var.Data,
                sampleCountNorm,
                featureSizeNorm,
                Epsilon,
                Stream
            );

            if (AffineEnabled)
            {
                dim3 blockSizeSp(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSizeSp(
                    (featureSize + blockSizeSp.x - 1) / blockSizeSp.x,
                    (NumChannels + blockSizeSp.y - 1) / blockSizeSp.y,
                    batchSize
                );
                unsigned sharedMemSize = blockSizeSp.y * 2ull * sizeof(moduleValueType);
	            implAffineBias2DKernel<<<gridSizeSp, blockSizeSp, sharedMemSize, Stream>>>(
                    output.Data,
                    Weight->GetTensor().Data,
                    Bias->GetTensor().Data,
                    NumChannels,
                    featureSize
                    );
            }

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __global__ void leakyReLUKernel(
            const moduleValueType* input,
            moduleValueType* output,
            const moduleValueType negativeSlope,
            const unsigned size)
    	{
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
                output[index] = input[index] > 0 ? input[index] : input[index] * negativeSlope;
        }

        layerStatus_t LeakyReLU::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<moduleValueType>& output
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("LeakyReLU", &output.Handle);
#endif

	        const auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            leakyReLUKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>
                (output.Data, output.Data, NegativeSlope, size);

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        Conv1D::Conv1D(
            Module* parent, const std::string& name,
            unsigned inputChannels, unsigned outputChannels, unsigned kernelSize,
            unsigned stride, unsigned padding, unsigned dilation, unsigned groups,
            bool bias
        ) : Module(parent, name), KernelSize(kernelSize), Stride(stride), Padding(padding), Dilation(dilation), Groups(groups), InputChannels(inputChannels), OutputChannels(outputChannels), BiasEnabled(bias)
        {
            if (inputChannels % groups)
                throw std::logic_error("InputChannels must be exactly divisible by Groups");
            if (outputChannels % groups)
                throw std::logic_error("OutputChannels must be exactly divisible by Groups");

            IsContiguous = KernelSize == 1 && Stride == 1 && Padding == 0 && Dilation == 1;
            UseGemm = IsContiguous && Groups == 1;
            IsDwConv = InputChannels == OutputChannels && Groups == InputChannels;

            Weight = std::make_shared<Parameter>(
                this, "weight", Tensor<moduleValueType>(outputChannels, inputChannels / groups, kernelSize)
            );
            if (BiasEnabled)
                Bias = std::make_shared<Parameter>(
                    this, "bias", Tensor<moduleValueType>(outputChannels)
                );
        }

        static __global__ void im2ColKernel(
            const moduleValueType* input,
            moduleValueType* output,
            unsigned inputLength,
            unsigned im2ColChannels,
            unsigned outputLength,
            unsigned kernelSize,
            unsigned stride,
            unsigned padding,
            unsigned dilation
        )
        {
			//im2ColChannels = groupSize * kernelSize, groupSize = inputChannel / groups
            //[batchSize, groups, groupSize, inputLength] ->  [batchSize, groups, im2ColChannels, outputLength]

            //unsigned bg = blockIdx.y;
            const unsigned outCh = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned outPos = blockIdx.x * blockDim.x + threadIdx.x;
            if (outCh >= im2ColChannels || outPos >= outputLength)
				return;
        	
            //const unsigned batchIdx = bg / groups;
            //const unsigned gIdx = bg % groups;

            const unsigned gcIdx = outCh / kernelSize;
            const unsigned kernelOffset = outCh % kernelSize;

			const int inPos = int(outPos * stride) - int(padding) + int(kernelOffset * dilation);
            //const unsigned oPos = ((batchIdx * groups + gIdx) * im2ColChannels + outCh) * outputLength + outPos;
            //const unsigned iPos = ((batchIdx * groups + gIdx) * groupSize + gcIdx) * inputLength + inPos;
            const unsigned oPos = outCh * outputLength + outPos;
            const unsigned iPos = gcIdx * inputLength + inPos;

            if (inPos >= 0 && inPos < int(inputLength))
                output[oPos] = input[iPos];
            else
                output[oPos] = 0.f;
        }

        layerStatus_t Conv1D::Forward(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output,
            Tensor<moduleValueType>& col
        ) const noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Conv1D " + Name, &output.Handle);
#endif

            if (input.H != InputChannels)
                return LAYER_STATUS_SIZE_MISMATCH;

            const unsigned batchSize = input.N * input.C;
			const unsigned iGroupSize = InputChannels / Groups;
            const unsigned oGroupSize = OutputChannels / Groups;
            const unsigned inputLength = input.W;

            const unsigned outputLength = (inputLength + 2 * Padding - Dilation * (KernelSize - 1) - 1) / Stride + 1;

            output.Resize(batchSize, OutputChannels, outputLength);
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            const unsigned im2ColChannels = iGroupSize * KernelSize;
            col.Resize(outputLength, im2ColChannels);
            if (col.Handle) CudaProvider::asyncCudaStream(getHandleStream(col.Handle));
            col.Handle = input.Handle;

            auto Stream = cudaStream_t(getHandleStream(input.Handle));

            static constexpr moduleValueType Alpha = 1.f;
            static constexpr moduleValueType Beta = 0.f;

            if (UseGemm)
            {
				//[BatchSize, InputChannels, InputLength/OutputLength, 1]^T * [1, OutputChannels, InputChannels, 1]^T
                if (auto Ret = cublasSgemmStridedBatched(
                    cublasHandle_t(input.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)outputLength, (int)OutputChannels, (int)InputChannels,
                    &Alpha,
                    input.Data, (int)outputLength, (ptrdiff_t)InputChannels * outputLength,
                    Weight->GetTensor().Data, (int)InputChannels, 0,
                    &Beta,
					output.Data, (int)outputLength, (ptrdiff_t)OutputChannels * outputLength,
                    (int)batchSize
                )) return static_cast<layerStatus_t>(Ret);
            }
            else
            {
	            dim3 blockSize(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSize(
                    (outputLength + blockSize.x - 1) / blockSize.x,
                    (im2ColChannels + blockSize.y - 1) / blockSize.y
                );

	            /*im2ColKernel<<<gridSize, blockSize, 0, Stream>>>(
	                input.Data,
	                col.Data,
	                Groups,
                    iGroupSize,
	                inputLength,
	                im2ColChannels,
	                outputLength,
	                KernelSize,
	                Stride,
	                Padding,
	                Dilation
					);*/

				//[batchSize, Groups, Im2ColChannels, OutputLength]^T * [1, Groups, oGroupSize, Im2ColChannels]^T
				//[batchSize, InputChannels, OutputLength]^T * [1, OutputChannels, InputChannels, KernelSize]^T
				//[batchSize, Groups, oGroupSize, OutputLength]^T
                for (unsigned b = 0; b < batchSize; ++b)
                    for (unsigned g = 0; g < Groups; ++g)
                    {
						//const unsigned iPos = ((b * Groups + g) * groupSize + 0) * inputLength + 0;
                        //const unsigned oPos = ((b * Groups + g) * im2ColChannels + 0) * outputLength + 0;
						const auto iPtr = input.Data + 
                            ptrdiff_t(b * Groups + g) * iGroupSize * inputLength;
                        im2ColKernel<<<gridSize, blockSize, 0, Stream>>>(
                            iPtr,
			                col.Data,
			                inputLength,
			                im2ColChannels,
			                outputLength,
			                KernelSize,
			                Stride,
			                Padding,
			                Dilation
						);
                        const auto oPtr = output.Data +
                            ptrdiff_t(b * Groups + g) * oGroupSize * outputLength;
                        const auto wPtr = Weight->GetTensor().Data +
							ptrdiff_t(g) * oGroupSize * im2ColChannels;
                        if (auto Ret = cublasSgemm(
                            cublasHandle_t(col.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)outputLength, (int)oGroupSize, (int)im2ColChannels,
                            &Alpha,
                            col.Data, (int)outputLength,
                            wPtr, (int)im2ColChannels,
                            &Beta,
                            oPtr, (int)outputLength
                        )) return static_cast<layerStatus_t>(Ret);
					}
                /*if (auto Ret = cublasSgemmStridedBatched(
                    cublasHandle_t(col.Handle), CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)outputLength, (int)oGroupSize, (int)im2ColChannels,
                    &Alpha,
                    col.Data + (ptrdiff_t)Groups * im2ColChannels * outputLength * b, (int)outputLength,
                    (ptrdiff_t)im2ColChannels * outputLength,
                    Weight->GetTensor().Data, (int)im2ColChannels, (ptrdiff_t)oGroupSize * im2ColChannels,
                    &Beta,
                    output.Data + (ptrdiff_t)OutputChannels * outputLength * b, (int)outputLength,
                    (ptrdiff_t)oGroupSize * outputLength,
                    (int)Groups
                )) return static_cast<layerStatus_t>(Ret);*/
            }

            //const unsigned iGroupSize = InputChannels / Groups;
            //const unsigned oGroupSize = OutputChannels / Groups;
            //const unsigned im2ColChannels = iGroupSize * KernelSize;
            //[batchSize, Groups, outputLength, im2ColChannels] * [Groups, oGroupSize, im2ColChannels]
            //[batchSize, Groups, oGroupSize, outputLength] -> [batchSize, OutputChannels, outputLength]
            /*static constexpr moduleValueType Alpha = 1.f;
            static constexpr moduleValueType Beta = 0.f;
            for (unsigned b = 0; b < 1; ++b)
                for (unsigned g = 0; g < 1; ++g)
                {
                    const moduleValueType* A = Weight->GetTensor().Data + (ptrdiff_t)g * oGroupSize * im2ColChannels;
                    const moduleValueType* B = col.Data + ptrdiff_t(b * Groups + g) * outputLength * im2ColChannels;
                    moduleValueType* C = output.Data + ptrdiff_t(b * Groups + g) * oGroupSize * outputLength;
                    if (auto Ret = cublasSgemm(
                        cublasHandle_t(input.Handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        (int)oGroupSize,
                        int(outputLength),
                        (int)im2ColChannels,
                        &Alpha,
                        A,
                        (int)im2ColChannels,
                        B,
                        (int)im2ColChannels,
                        &Beta,
                        C,
                        (int)oGroupSize
                    )) return static_cast<layerStatus_t>(Ret);
                }*/

            if (BiasEnabled)
            {
                dim3 blockSizeSp(DRAGONIANLIB_CUDA_BLOCK_SIZE / 32, 32);
                dim3 gridSizeSp(
                    (outputLength + blockSizeSp.x - 1) / blockSizeSp.x,
                    (OutputChannels + blockSizeSp.y - 1) / blockSizeSp.y,
                    batchSize
                );
                unsigned sharedMemSize = blockSizeSp.y * sizeof(moduleValueType);
	            implBias2DKernel<<<gridSizeSp, blockSizeSp, sharedMemSize, Stream>>>(
                    output.Data,
                    Bias->GetTensor().Data,
                    OutputChannels,
                    outputLength
                    );
            }

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __global__ void implInplaceAddKernel(
            moduleValueType* output,
            const moduleValueType* input,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
				output[index] += input[index];
        }

        layerStatus_t AddTensor(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<moduleValueType>& output,
            const Tensor<moduleValueType>& input
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Add", &output.Handle);
#endif

            if (output.Dim != input.Dim ||
                output.N != input.N ||
                output.C != input.C ||
                output.H != input.H ||
                output.W != input.W)
                return LAYER_STATUS_SIZE_MISMATCH;

            if (input.Handle) CudaProvider::asyncCudaStream(getHandleStream(input.Handle));

            const auto n = input.N * input.C * input.H * input.W;

			dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
			dim3 gridLength((n + blockLength.x - 1) / blockLength.x);

            implInplaceAddKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                input.Data,
                n
				);

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __device__ moduleValueType sigmoid(moduleValueType x)
        {
            return 1.0f / (1.0f + expf(-x));
        }

        static __global__ void implSigmoidKernel(
            const moduleValueType* input,
            moduleValueType* output,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < size)
                output[index] = sigmoid(input[index]);
        }

        layerStatus_t SigmoidTensor(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<moduleValueType>& output
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Sigmoid", &output.Handle);
#endif

            const auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            implSigmoidKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                output.Data,
                size
                );

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        layerStatus_t Transpose::Forward(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("Transpose", &output.Handle);
#endif

            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.W, input.H);
            else if (input.Dim == 3)
                output.Resize(input.N, input.W, input.H);
            else if (input.Dim == 2)
                output.Resize(input.W, input.H);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            if (input.Handle) output.Handle = input.Handle;

            const auto BatchSize = input.N * input.C;
            const auto BatchStride = input.H * input.W;

            static constexpr moduleValueType alpha = 1.f;
            static constexpr moduleValueType beta = 0.f;

            for (unsigned b = 0; b < BatchSize; ++b)
                if (auto Ret = cublasSgeam(
                    cublasHandle_t(output.Handle),
                    CUBLAS_OP_T,
                    CUBLAS_OP_T,
                    static_cast<int>(input.H),
                    static_cast<int>(input.W),
                    &alpha,
                    input.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.W),
                    &beta,
                    input.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.W),
                    output.Data + (ptrdiff_t)b * BatchStride,
                    static_cast<int>(input.H)
                )) return static_cast<layerStatus_t>(Ret);

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __global__ void implGLUKernel(
            const moduleValueType* input,
            moduleValueType* output,
            unsigned half,
            unsigned featureSize
        )
        {
            //[batch, 2, half, featureSize] -> [batch, half, featureSize]
            const unsigned bz = blockIdx.z;
            const unsigned ty = threadIdx.y;
            const unsigned tx = threadIdx.x;

            const unsigned y = blockIdx.y * blockDim.y + ty;
            const unsigned x = blockIdx.x * blockDim.x + tx;

            if (y < half && x < featureSize)
            {
                output[(bz * half + y) * featureSize + x] = 
                    input[((bz * 2 + 0) * half + y) * featureSize + x] * sigmoid(input[((bz * 2 + 1) * half + y) * featureSize + x]);
            }
        }

        layerStatus_t GLU::Forward(
            const Tensor<moduleValueType>& input,
            Tensor<moduleValueType>& output
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("GLU", &output.Handle);
#endif

            if (input.Dim == 4)
                output.Resize(input.N, input.C, input.H / 2, input.W);
            else if (input.Dim == 3)
                output.Resize(input.N, input.H / 2, input.W);
            else if (input.Dim == 2)
                output.Resize(input.H / 2, input.W);
            else
                return LAYER_STATUS_SIZE_MISMATCH;
            if (output.Handle) CudaProvider::asyncCudaStream(getHandleStream(output.Handle));
            output.Handle = input.Handle;

            const auto BatchSize = input.N * input.C;
            const auto Half = input.H / 2;
            const auto FeatureSize = input.W;
			

            dim3 blockSize(32, DRAGONIANLIB_CUDA_BLOCK_SIZE / 32);
            dim3 gridSize(
                (FeatureSize + blockSize.x - 1) / blockSize.x,
                (Half + blockSize.y - 1) / blockSize.y,
                BatchSize
            );

            implGLUKernel<<<gridSize, blockSize, 0, cudaStream_t(getHandleStream(input.Handle))>>>(
                input.Data,
                output.Data,
                Half,
                FeatureSize
            );

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

        static __global__ void implSiLUKernel(
            const moduleValueType* input,
            moduleValueType* output,
            const unsigned size
        )
        {
            const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index < size)
            {
	            const moduleValueType x = input[index];
                output[index] = x / (1.0f + expf(-x));
            }
        }

        layerStatus_t SiLU::Forward(
	        // ReSharper disable once CppParameterMayBeConstPtrOrRef
	        Tensor<moduleValueType>& output
        ) noexcept try
        {
#if DRAGONIANLIB_CUDA_EP_BENCHMARK
			auto __BENCH_TM_BEG = Timer("SiLU", &output.Handle);
#endif

            const auto size = output.N * output.C * output.H * output.W;

            dim3 blockLength(DRAGONIANLIB_CUDA_BLOCK_SIZE);
            dim3 gridLength((size + blockLength.x - 1) / blockLength.x);

            implSiLUKernel<<<gridLength, blockLength, 0, cudaStream_t(getHandleStream(output.Handle))>>>(
                output.Data,
                output.Data,
                size
                );

            return LAYER_STATUS_SUCCESS;
        }
        catch (std::exception& except)
        {
            CudaProvider::__LastError = except.what();
            return LAYER_STATUS_FATAL_ERROR;
        }

    }
}