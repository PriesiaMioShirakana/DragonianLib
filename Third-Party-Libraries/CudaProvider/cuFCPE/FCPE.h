#pragma once
#include "Layers/Modules/Base.h"

namespace DragonianLib
{
    namespace CudaModules
    {
        namespace FCPE
        {
            class ConformerConvModule : public Module
            {
            public:
                ConformerConvModule(
                    Module* parent,
                    const std::string& name,
                    unsigned dimModel,
                    unsigned expandFactor = 2,
                    unsigned kernelSize = 31
                );

                layerStatus_t Forward(
                    Tensor<moduleValueType>& output,
                    Tensor<moduleValueType>& mean,
                    Tensor<moduleValueType>& var,
                    Tensor<moduleValueType>& cache,
                    Tensor<moduleValueType>& col
                ) const noexcept;

            private:
                std::shared_ptr<LayerNorm1D> net_0;
                Transpose net_1;
                std::shared_ptr<Conv1D> net_2;
                GLU net_3;
                std::shared_ptr<Conv1D> net_4_conv;
                SiLU net_5;
                std::shared_ptr<Conv1D> net_6;
                Transpose net_7;
            };

            class CFNEncoderLayer : public Module
            {
            public:
                CFNEncoderLayer(
                    Module* parent,
                    const std::string& name,
                    unsigned dimModel,
                    unsigned numHeads,
                    bool useNorm = false,
                    bool convOnly = false
                );

                layerStatus_t Forward(
                    Tensor<moduleValueType>& output,
                    Tensor<moduleValueType>& mean,
                    Tensor<moduleValueType>& var,
                    Tensor<moduleValueType>& res,
                    Tensor<moduleValueType>& cache,
                    Tensor<moduleValueType>& col
                ) const noexcept;

            private:
                std::shared_ptr<ConformerConvModule> conformer;
                mutable Tensor<moduleValueType> conformerOut;
                std::shared_ptr<LayerNorm1D> norm;
            };


            class ConformerNaiveEncoder : public Module
            {
            public:
                ConformerNaiveEncoder(
                    Module* parent,
                    const std::string& name,
                    unsigned numLayers,
                    unsigned numHeads,
                    unsigned dimModel,
                    bool useNorm = false,
                    bool convOnly = false
                );

                layerStatus_t Forward(
                    Tensor<moduleValueType>& output,
                    Tensor<moduleValueType>& mean,
                    Tensor<moduleValueType>& var,
                    Tensor<moduleValueType>& res,
                    Tensor<moduleValueType>& cache,
                    Tensor<moduleValueType>& col
                ) const noexcept;

            private:
                std::vector<std::shared_ptr<CFNEncoderLayer>> encoder_layers;
            };

			class Model : public Module
            {
            public:
                struct CacheTensors
                {
                    CacheTensors() = default;
                    CacheTensors(Tensor<moduleValueType>&& _i) : input(std::move(_i)) {}
                    Tensor<moduleValueType> input;
                    Tensor<moduleValueType> output;
                    Tensor<moduleValueType> res;
                    friend Model;
                protected:
                    Tensor<moduleValueType> mean;
                    Tensor<moduleValueType> var;
                    Tensor<moduleValueType> col;
                };

                enum DECODER : uint8_t
                {
	                DECODER_ARGMAX,
                    DECODER_LOCAL_ARGMAX
                };

                Model(
                    unsigned inputChannels,
                    unsigned outputDims,
                    unsigned hiddenDims = 512,
                    unsigned numLayers = 6,
                    unsigned numHeads = 8,
                    moduleValueType f0Max = 1975.5f,
                    moduleValueType f0Min = 32.70f,
                    bool useFaNorm = false,
                    bool convOnly = true,
                    bool useHarmonicEmb = false
                );

                layerStatus_t Forward(
                    CacheTensors& caches
                ) const noexcept;

                layerStatus_t Infer(
                    CacheTensors& caches,
                    moduleValueType threshold = 0.05f,
                    DECODER decoder = DECODER_LOCAL_ARGMAX
                ) const noexcept;

                void LoadFromFile(const std::wstring& view);

            private:
                std::shared_ptr<Conv1D> input_stack_0;
                std::shared_ptr<GroupNorm1D> input_stack_1;
                LeakyReLU input_stack_2;
                std::shared_ptr<Conv1D> input_stack_3;
                std::shared_ptr<ConformerNaiveEncoder> net;
                std::shared_ptr<LayerNorm1D> norm;
                std::shared_ptr<Linear> output_proj;
                std::shared_ptr<Parameter> cent_table, gaussian_blurred_cent_mask;

                layerStatus_t Latent2Cents(
                    CacheTensors& caches,
                    moduleValueType threshold = 0.05f
                ) const noexcept;

                layerStatus_t Latent2CentsLocal(
                    CacheTensors& caches,
                    moduleValueType threshold = 0.05f
                ) const noexcept;
            };
        }
    }
}
