// ReSharper disable CppClangTidyClangDiagnosticCastFunctionTypeStrict
// ReSharper disable CppClangTidyClangDiagnosticMissingFieldInitializers
#include <filesystem>

#include "cuFCPE.h"
#include "FCPE.h"
#include "Layers/ThirdParty/Numpy.h"

using namespace DragonianLib::CudaModules::FCPE;

void cuFCPE_Test1(int argc, char** argv)
{
    bool testModel = true;
    std::filesystem::path modelPath;
    unsigned inputChannels = 114514;                            //-input_channels
    unsigned outputDims = 114514;                               //-output_dims
    unsigned hiddenDims = 512;                                  //-hidden_dims
    unsigned numLayers = 6;                                     //-layers
    unsigned numHeads = 8;                                      //-heads
    unsigned samplingRate = 16000;                              //-sampling_rate
    unsigned fftLength = 1024;                                  //-fft_length
    unsigned windowSize = 1024;                                 //-window_size
    unsigned hopSize = 160;                                     //-hop_size
    DragonianLib::CudaModules::moduleValueType freqMin = 0.f;   //-freq_min
    DragonianLib::CudaModules::moduleValueType freqMax = 8000.f;//-freq_max
    DragonianLib::CudaModules::moduleValueType clipVal = 1e-5f; //-clip_val
    DragonianLib::CudaModules::moduleValueType f0Max = 1975.5f; //-f0_max
    DragonianLib::CudaModules::moduleValueType f0Min = 32.70f;  //-f0_min
    bool useFaNorm = false;                                     //--use_fanorm
    bool convOnly = true;                                       //--has_attn
    bool useHarmonicEmb = false;                                //--use_harmonic
    if (argc < 2 && !testModel)
    {
        fprintf(stdout, "Usage: %s <model_path>\n\t[-input_channels=(int)]\n\t[-output_dims=(int)]\n\t[-hidden_dims=(int: 512)]\n\t[-layers=(int: 6)]\n\t[-heads=(int: 8)]\n\t[-sampling_rate=(int: 16000)]\n\t[-fft_length=(int: 1024)]\n\t[-window_size=(int: 1024)]\n\t[-hop_size=(int: 160)]\n\t[-freq_min=(float: 0.0)]\n\t[-freq_max=(float: 8000.0)]\n\t[-clip_val=(float: 1e-5)]\n\t[-f0_max=(float: 1975.5)]\n\t[-f0_min=(float: 32.70)]\n\t[--use_fanorm]\n\t[--has_attn]\n\t[--use_harmonic]\n", argv[0]);
        return;
    }
    --argc; ++argv;
    if (argc)
    {
        modelPath = std::filesystem::path(*argv);
        --argc; ++argv;
    }
    else
        modelPath = std::filesystem::path(R"(C:\DataSpace\torchfcpe\assets\fcpe)");

    while (argc)
    {
        std::string arg = *argv++;
        if (arg.find("-input_channels=") != std::string::npos)
            inputChannels = std::stoul(arg.substr(16));
        else if (arg.find("-output_dims=") != std::string::npos)
            outputDims = std::stoul(arg.substr(13));
        else if (arg.find("-hidden_dims=") != std::string::npos)
            hiddenDims = std::stoul(arg.substr(13));
        else if (arg.find("-layers=") != std::string::npos)
            numLayers = std::stoul(arg.substr(8));
        else if (arg.find("-heads=") != std::string::npos)
            numHeads = std::stoul(arg.substr(7));
        else if (arg.find("-sampling_rate=") != std::string::npos)
            samplingRate = std::stoul(arg.substr(15));
        else if (arg.find("-fft_length=") != std::string::npos)
            fftLength = std::stoul(arg.substr(12));
        else if (arg.find("-window_size=") != std::string::npos)
            windowSize = std::stoul(arg.substr(13));
        else if (arg.find("-hop_size=") != std::string::npos)
            hopSize = std::stoul(arg.substr(10));
        else if (arg.find("-freq_min=") != std::string::npos)
            freqMin = std::stof(arg.substr(10));
        else if (arg.find("-freq_max=") != std::string::npos)
            freqMax = std::stof(arg.substr(10));
        else if (arg.find("-clip_val=") != std::string::npos)
            clipVal = std::stof(arg.substr(10));
        else if (arg.find("-f0_max=") != std::string::npos)
            f0Max = std::stof(arg.substr(8));
        else if (arg.find("-f0_min=") != std::string::npos)
            f0Min = std::stof(arg.substr(8));
        else if (arg == "--use_fanorm")
            useFaNorm = true;
        else if (arg == "--has_attn")
            convOnly = false;
        else if (arg == "--use_harmonic")
            useHarmonicEmb = true;
        else
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        argc--;
    }

    if (inputChannels == 114514 || outputDims == 114514)
    {
        if (testModel)
        {
            inputChannels = 128;
            outputDims = 360;
        }
        else
        {
            fprintf(stderr, "Input channels and output dimensions must be specified.\n");
            return;
        }
    }

    unsigned melBins = inputChannels;

    try
    {
        Model model(
            inputChannels,
            outputDims,
            hiddenDims,
            numLayers,
            numHeads,
            f0Max,
            f0Min,
            useFaNorm,
            convOnly,
            useHarmonicEmb
        );
        DragonianLib::CudaModules::MelKernel mel(
            samplingRate,
            melBins,
            fftLength,
            windowSize,
            hopSize,
            freqMin,
            freqMax,
            clipVal
        );

        model.LoadFromFile(modelPath);
        fprintf(stdout, "Model initialized successfully with the provided parameters.\n");

        Model::CacheTensors cacheTensors;
        auto handle = DragonianLib::CudaModules::createHandle();
        auto stream = DragonianLib::CudaProvider::createCudaStream();
        setHandleStream(handle, stream);
        cacheTensors.res.Handle = handle;

        while (true)
        {
            char tmpPath[512];
            fprintf(stdout, "Enter audio file path (or 'exit' to quit): ");
            if (fscanf(stdin, "%s", tmpPath) != 1)
                continue;
            if (strcmp(tmpPath, "exit") == 0)
            {
                fprintf(stdout, "Exiting...\n");
                break;
            }
            auto path = std::filesystem::path(tmpPath);
            printf("Processing file: %s\n", path.string().c_str());
            auto audioData = DragonianLib::Util::LoadNumpyFile(path);
            // Cvt 2 2D array
            if (audioData.first.empty() || audioData.second.empty())
            {
                fprintf(stderr, "Failed to load audio data from %s\n", path.string().c_str());
                continue;
            }
            int64_t batchSize = 1;
            for (size_t i = 0; i < audioData.first.size() - 1; ++i)
                batchSize *= audioData.first[i];
            int64_t audioLength = audioData.first.back();

            printf("Loaded audio data with shape: [%lld, %lld]\n", batchSize, audioLength);

            cacheTensors.res.Resize(
                static_cast<unsigned>(batchSize),
                static_cast<unsigned>(audioLength)
            );
            DragonianLib::CudaProvider::cpy2Device(
                cacheTensors.res.Data,
                (const DragonianLib::CudaModules::moduleValueType*)audioData.second.data(),
                audioData.second.size() / sizeof(DragonianLib::CudaModules::moduleValueType),
                stream
            );

            auto timeBegin = std::chrono::high_resolution_clock::now();
            mel.Forward(cacheTensors.res, cacheTensors.input, cacheTensors.output);
            model.Infer(cacheTensors);
            auto output = cacheTensors.output.Cpy2Host(stream);
            DragonianLib::CudaProvider::asyncCudaStream(stream);
            printf("Processing time: %lld μs\n",
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - timeBegin
                ).count()
            );

            DragonianLib::Util::SaveNumpyFile(
                path.replace_extension("f0"),
                { cacheTensors.output.N, cacheTensors.output.C, cacheTensors.output.H, cacheTensors.output.W },
                output
            );
        }
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "Runtime error: %s\n", e.what());
    }
}

void cuFCPE_Test2(int argc, char** argv)
{
    bool testModel = true;
    std::filesystem::path modelPath;
    unsigned inputChannels = 114514;                            //-input_channels
    unsigned outputDims = 114514;                               //-output_dims
    unsigned batchSize = 1;                                     //-batch_size
    unsigned loopCount = 10; 								    //-loop_count
    unsigned audioLength = 10;                                  //-audio_length
    unsigned hiddenDims = 512;                                  //-hidden_dims
    unsigned numLayers = 6;                                     //-layers
    unsigned numHeads = 8;                                      //-heads
    unsigned samplingRate = 16000;                              //-sampling_rate
    unsigned fftLength = 1024;                                  //-fft_length
    unsigned windowSize = 1024;                                 //-window_size
    unsigned hopSize = 160;                                     //-hop_size
    DragonianLib::CudaModules::moduleValueType freqMin = 0.f;   //-freq_min
    DragonianLib::CudaModules::moduleValueType freqMax = 8000.f;//-freq_max
    DragonianLib::CudaModules::moduleValueType clipVal = 1e-5f; //-clip_val
    DragonianLib::CudaModules::moduleValueType f0Max = 1975.5f; //-f0_max
    DragonianLib::CudaModules::moduleValueType f0Min = 32.70f;  //-f0_min
    bool useFaNorm = false;                                     //--use_fanorm
    bool convOnly = true;                                       //--has_attn
    bool useHarmonicEmb = false;                                //--use_harmonic
    if (argc < 2 && !testModel)
    {
        fprintf(stdout, "Usage: %s <model_path>\n\t[-input_channels=(int)]\n\t[-output_dims=(int)]\n\t[-hidden_dims=(int: 512)]\n\t[-layers=(int: 6)]\n\t[-heads=(int: 8)]\n\t[-sampling_rate=(int: 16000)]\n\t[-fft_length=(int: 1024)]\n\t[-window_size=(int: 1024)]\n\t[-hop_size=(int: 160)]\n\t[-freq_min=(float: 0.0)]\n\t[-freq_max=(float: 8000.0)]\n\t[-clip_val=(float: 1e-5)]\n\t[-f0_max=(float: 1975.5)]\n\t[-f0_min=(float: 32.70)]\n\t[--use_fanorm]\n\t[--has_attn]\n\t[--use_harmonic]\n", argv[0]);
        return;
    }
    --argc; ++argv;
    if (argc)
    {
        modelPath = std::filesystem::path(*argv);
        --argc; ++argv;
    }
    else
        modelPath = std::filesystem::path(R"(C:\DataSpace\torchfcpe\assets\fcpe)");

    while (argc)
    {
        std::string arg = *argv++;
        if (arg.find("-input_channels=") != std::string::npos)
            inputChannels = std::stoul(arg.substr(16));
        else if (arg.find("-output_dims=") != std::string::npos)
            outputDims = std::stoul(arg.substr(13));
        else if (arg.find("-hidden_dims=") != std::string::npos)
            hiddenDims = std::stoul(arg.substr(13));
        else if (arg.find("-batch_size=") != std::string::npos)
            batchSize = std::stoul(arg.substr(12));
        else if (arg.find("-loop_count=") != std::string::npos)
            loopCount = std::stoul(arg.substr(12));
        else if (arg.find("-audio_length=") != std::string::npos)
            audioLength = std::stoul(arg.substr(14));
        else if (arg.find("-layers=") != std::string::npos)
            numLayers = std::stoul(arg.substr(8));
        else if (arg.find("-heads=") != std::string::npos)
            numHeads = std::stoul(arg.substr(7));
        else if (arg.find("-sampling_rate=") != std::string::npos)
            samplingRate = std::stoul(arg.substr(15));
        else if (arg.find("-fft_length=") != std::string::npos)
            fftLength = std::stoul(arg.substr(12));
        else if (arg.find("-window_size=") != std::string::npos)
            windowSize = std::stoul(arg.substr(13));
        else if (arg.find("-hop_size=") != std::string::npos)
            hopSize = std::stoul(arg.substr(10));
        else if (arg.find("-freq_min=") != std::string::npos)
            freqMin = std::stof(arg.substr(10));
        else if (arg.find("-freq_max=") != std::string::npos)
            freqMax = std::stof(arg.substr(10));
        else if (arg.find("-clip_val=") != std::string::npos)
            clipVal = std::stof(arg.substr(10));
        else if (arg.find("-f0_max=") != std::string::npos)
            f0Max = std::stof(arg.substr(8));
        else if (arg.find("-f0_min=") != std::string::npos)
            f0Min = std::stof(arg.substr(8));
        else if (arg == "--use_fanorm")
            useFaNorm = true;
        else if (arg == "--has_attn")
            convOnly = false;
        else if (arg == "--use_harmonic")
            useHarmonicEmb = true;
        else
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        argc--;
    }

    if (inputChannels == 114514 || outputDims == 114514)
    {
        if (testModel)
        {
            inputChannels = 128;
            outputDims = 360;
        }
        else
        {
            fprintf(stderr, "Input channels and output dimensions must be specified.\n");
            return;
        }
    }

    unsigned melBins = inputChannels;

    try
    {
        Model model(
            inputChannels,
            outputDims,
            hiddenDims,
            numLayers,
            numHeads,
            f0Max,
            f0Min,
            useFaNorm,
            convOnly,
            useHarmonicEmb
        );
        DragonianLib::CudaModules::MelKernel mel(
            samplingRate,
            melBins,
            fftLength,
            windowSize,
            hopSize,
            freqMin,
            freqMax,
            clipVal
        );

        model.LoadFromFile(modelPath);
        fprintf(stdout, "Model initialized successfully with the provided parameters.\n");

        Model::CacheTensors cacheTensors;
        auto handle = DragonianLib::CudaModules::createHandle();
        auto stream = DragonianLib::CudaProvider::createCudaStream();

        setHandleStream(handle, stream);
        cacheTensors.res.Handle = handle;
        for (unsigned i = 0; i < 5; ++i)
        {
            cacheTensors.res.Resize(batchSize, audioLength * 16000 * 60);
            mel.Forward(cacheTensors.res, cacheTensors.input, cacheTensors.output);
            model.Infer(cacheTensors);
            DragonianLib::CudaProvider::asyncCudaStream(stream);
        }
        int64_t totalTime = 0;
        auto i = loopCount;
        std::vector<float> inputData(16000ull * batchSize * audioLength * 60);
        for (size_t j = 0; j < inputData.size(); ++j)
            inputData[j] = float(j) / float(inputData.size());
        while (i--)
        {
            cacheTensors.res.Resize(batchSize, audioLength * 16000 * 60);
            DragonianLib::CudaProvider::cpy2Device(
                cacheTensors.res.Data,
                inputData.data(),
                inputData.size(),
                stream
            );
            auto timeBegin = std::chrono::high_resolution_clock::now();
            mel.Forward(cacheTensors.res, cacheTensors.input, cacheTensors.output);
            model.Infer(cacheTensors);
            auto output = cacheTensors.output.Cpy2Host(stream);
            DragonianLib::CudaProvider::asyncCudaStream(stream);
            const auto curTime = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - timeBegin
            ).count();
            printf("Processing time: %lld μs\n", curTime);
            totalTime += curTime;
        }
        const auto avgTime = totalTime / loopCount;
        printf("Average processing time: %lld μs\n", avgTime);
        printf("Average rtf: %f", (float(avgTime) / 1000.f) / float(audioLength * 60 * batchSize * 1000));
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "Runtime error: %s\n", e.what());
    }
}

void cuFCPE_Test3(int, char**)
{
    constexpr unsigned batchSize = 2;
    constexpr unsigned inputChannels = 128;
    constexpr unsigned frameCount = 810;
    auto modelPath = std::filesystem::path(R"(C:\DataSpace\torchfcpe\assets\fcpe)");
    Model model(inputChannels, 360);
    model.LoadFromFile(modelPath);

    Model::CacheTensors cacheTensors;
    auto handle = DragonianLib::CudaModules::createHandle();
    auto stream = DragonianLib::CudaProvider::createCudaStream();
    setHandleStream(handle, stream);
    cacheTensors.input.Handle = handle;
    cacheTensors.input.Resize(batchSize, inputChannels, frameCount);

    std::vector<DragonianLib::CudaModules::moduleValueType> inputData((size_t)batchSize * inputChannels * frameCount);
    for (size_t j = 0; j < inputData.size(); ++j)
        inputData[j] = DragonianLib::CudaModules::moduleValueType(j) / DragonianLib::CudaModules::moduleValueType(inputData.size());
    DragonianLib::CudaProvider::cpy2Device(
        cacheTensors.input.Data,
        inputData.data(),
        inputData.size(),
        stream
    );

    model.Infer(cacheTensors);
    auto output = cacheTensors.output.Cpy2Host(stream);
    DragonianLib::CudaProvider::asyncCudaStream(stream);
    printf("fin");
}
