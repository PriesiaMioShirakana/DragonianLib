"""
cuFCPE Python Implementation - Type Stubs

这个模块提供了 CUDA 加速的 FCPE 功能。
"""
import numpy as np
from numpy.typing import NDArray

class ModelConfig:
    """
    FCPE 模型配置类
    
    用于配置 FCPE 模型的各种参数，包括网络结构、频率范围等。
    """
    
    def __init__(
        self,
        modelPath: str,
        inputChannels: int = 128,
        outputDims: int = 360,
        hiddenDims: int = 512,
        numLayers: int = 6,
        numHeads: int = 8,
        f0Max: float = 1975.5,
        f0Min: float = 32.70,
        useFaNorm: bool = False,
        convOnly: bool = True,
        useHarmonicEmb: bool = False
    ) -> None:
        """
        初始化模型配置
        
        Args:
            modelPath: 模型文件路径
            inputChannels: 输入通道数，默认 128
            outputDims: 输出维度，默认 360
            hiddenDims: 隐藏层维度，默认 512
            numLayers: 网络层数，默认 6
            numHeads: 注意力头数，默认 8
            f0Max: 最大基频，默认 1975.5 Hz
            f0Min: 最小基频，默认 32.70 Hz
            useFaNorm: 是否使用 FaNorm，默认 False
            convOnly: 是否只使用卷积，默认 True
            useHarmonicEmb: 是否使用谐波嵌入，默认 False
        """
        ...
    
    @property
    def modelPath(self) -> str:
        """模型文件路径"""
        ...
    
    @property
    def inputChannels(self) -> int:
        """输入通道数"""
        ...
    
    @property
    def outputDims(self) -> int:
        """输出维度"""
        ...
    
    @property
    def hiddenDims(self) -> int:
        """隐藏层维度"""
        ...
    
    @property
    def numLayers(self) -> int:
        """网络层数"""
        ...
    
    @property
    def numHeads(self) -> int:
        """注意力头数"""
        ...
    
    @property
    def f0Max(self) -> float:
        """最大基频 (Hz)"""
        ...
    
    @property
    def f0Min(self) -> float:
        """最小基频 (Hz)"""
        ...
    
    @property
    def useFaNorm(self) -> bool:
        """是否使用 FaNorm"""
        ...
    
    @property
    def convOnly(self) -> bool:
        """是否只使用卷积"""
        ...
    
    @property
    def useHarmonicEmb(self) -> bool:
        """是否使用谐波嵌入"""
        ...


class PreProcessConfig:
    """
    预处理配置类
    
    用于配置音频预处理的各种参数，包括采样率、FFT 长度、Mel 滤波器组等。
    """
    
    def __init__(
        self,
        samplingRate: int = 16000,
        fftLength: int = 1024,
        windowSize: int = 1024,
        hopSize: int = 160,
        melBins: int = 128,
        freqMin: float = 0.0,
        freqMax: float = 8000.0,
        clipVal: float = 1e-5
    ) -> None:
        """
        初始化预处理配置
        
        Args:
            samplingRate: 采样率，默认 16000 Hz
            fftLength: FFT 长度，默认 1024
            windowSize: 窗口大小，默认 1024
            hopSize: 帧移，默认 160
            melBins: Mel 滤波器组数量，默认 128
            freqMin: 最小频率，默认 0.0 Hz
            freqMax: 最大频率，默认 8000.0 Hz
            clipVal: 裁剪值，默认 1e-5
        """
        ...
    
    @property
    def samplingRate(self) -> int:
        """采样率 (Hz)"""
        ...
    
    @property
    def fftLength(self) -> int:
        """FFT 长度"""
        ...
    
    @property
    def windowSize(self) -> int:
        """窗口大小"""
        ...
    
    @property
    def hopSize(self) -> int:
        """帧移"""
        ...
    
    @property
    def melBins(self) -> int:
        """Mel 滤波器组数量"""
        ...
    
    @property
    def freqMin(self) -> float:
        """最小频率 (Hz)"""
        ...
    
    @property
    def freqMax(self) -> float:
        """最大频率 (Hz)"""
        ...
    
    @property
    def clipVal(self) -> float:
        """裁剪值"""
        ...


class cuFCPEProcessor:
    """
    CUDA 加速的 FCPE 处理器
    
    这个类提供了基于 CUDA 的快速连续基频估计功能。
    支持批处理音频数据，并返回基频轨迹。
    """
    
    def __init__(
        self,
        modelConfig: ModelConfig,
        preProcessConfig: PreProcessConfig
    ) -> None:
        """
        初始化 FCPE 处理器
        
        Args:
            modelConfig: 模型配置对象
            preProcessConfig: 预处理配置对象
        
        Raises:
            RuntimeError: 如果 CUDA 初始化失败或模型加载失败
        """
        ...
    
    def execute(self, inputData: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        执行基频估计
        
        Args:
            inputData: 输入音频数据，形状为 [batch_size, audio_length]
                      数据类型必须为 float32
        
        Returns:
            基频轨迹，形状为 [batch_size, channels, height, width]
            数据类型为 float32
        
        Raises:
            RuntimeError: 如果输入数据类型不是 float32
            RuntimeError: 如果输入数据不是 2D 数组
        
        Example:
            >>> import numpy as np
            >>> processor = cuFCPEProcessor(model_config, preprocess_config)
            >>> audio = np.random.randn(1, 16000).astype(np.float32)  # 1秒音频
            >>> f0 = processor.execute(audio)
            >>> print(f0.shape)  # (1, channels, height, width)
        """
        ...
    
    @property
    def modelConfig(self) -> ModelConfig:
        """获取模型配置"""
        ...
    
    @property
    def preProcessConfig(self) -> PreProcessConfig:
        """获取预处理配置"""
        ...


def create_cuFCPEProcessor(
    modelConfig: ModelConfig,
    preProcessConfig: PreProcessConfig
) -> cuFCPEProcessor:
    """
    创建 cuFCPE 处理器实例
    
    这是一个便利函数，用于创建 cuFCPEProcessor 实例。
    
    Args:
        modelConfig: 模型配置对象
        preProcessConfig: 预处理配置对象
    
    Returns:
        cuFCPEProcessor 实例
    
    Example:
        >>> model_config = ModelConfig("path/to/model.pth")
        >>> preprocess_config = PreProcessConfig(samplingRate=22050)
        >>> processor = create_cuFCPEProcessor(model_config, preprocess_config)
    """
    ...
