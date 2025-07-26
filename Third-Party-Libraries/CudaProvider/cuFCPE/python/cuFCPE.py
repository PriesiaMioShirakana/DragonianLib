
import sys
if sys.platform == 'win32':
    from windllLoader import CudaDLLLoader
    cuda_loader = CudaDLLLoader()
    cuda_loader.preload_dlls()
    
import cuFCPEPythonImp

# 注意，这个是只读的！改config不会改模型配置！
class ModelConfig:
    def __init__(self, model_path, input_channels=128, output_dims=360, hidden_dims=512, num_layers=6, num_heads=8, f0_max=1975.5, f0_min=32.70, use_fa_norm=False, conv_only=True, use_harmonic_emb=False):
        self.model_path = model_path
        self.input_channels = input_channels
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.f0_max = f0_max
        self.f0_min = f0_min
        self.use_fa_norm = use_fa_norm
        self.conv_only = conv_only
        self.use_harmonic_emb = use_harmonic_emb
        self.model_config = cuFCPEPythonImp.ModelConfig(
            model_path, input_channels, output_dims, hidden_dims, num_layers, num_heads, f0_max, f0_min, use_fa_norm, conv_only, use_harmonic_emb
        )
        
# 注意，这个是只读的！改config不会改模型配置！
class PreProcessConfig:
    def __init__(self, sampling_rate=16000, fft_length=1024, window_size=1024, hop_size=160, mel_bins=128, freq_min=0.0, freq_max=8000.0, clip_val=1e-5):
        self.sampling_rate = sampling_rate
        self.fft_length = fft_length
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.clip_val = clip_val
        self.preprocess_config = cuFCPEPythonImp.PreProcessConfig(
            sampling_rate, fft_length, window_size, hop_size, mel_bins, freq_min, freq_max, clip_val
        )
        
def create_processor(model_config, preprocess_config):
    assert model_config.input_channels == preprocess_config.mel_bins, "Input channels must match mel bins!!!!"
    processor = cuFCPEPythonImp.cuFCPEProcessor(model_config.model_config, preprocess_config.preprocess_config)
    return processor
