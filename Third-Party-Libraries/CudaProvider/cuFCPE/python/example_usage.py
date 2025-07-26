#!/usr/bin/env python3
"""
cuFCPE 使用示例

这个脚本展示了如何使用 cuFCPE Python 绑定进行基频估计。
"""

import numpy as np
import sys
import os

try:
    import cuFCPE
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 cuFCPEPythonImp.pyd 文件在 Python 路径中")
    sys.exit(1)


def main():
    """主函数：演示 cuFCPE 的基本用法"""
    
    # 1. 创建模型配置
    model_config = cuFCPE.ModelConfig(
        model_path="C:/Users/lenovo/Downloads/fcpe/fcpe/model",  # 请替换为实际的模型路径
        input_channels=128,
        output_dims=360,
        hidden_dims=512,
        num_layers=6,
        num_heads=8,
        f0_max=1975.5,
        f0_min=32.7,
        use_fa_norm=False,
        conv_only=True,
        use_harmonic_emb=False
    )
    
    # 2. 创建预处理配置
    preprocess_config = cuFCPE.PreProcessConfig(
        sampling_rate=16000,
        fft_length=1024,
        window_size=1024,
        hop_size=160,
        mel_bins=128,
        freq_min=0.0,
        freq_max=8000.0,
        clip_val=1e-5
    )
    # 3. 创建处理器
    try:
        processor = cuFCPE.create_processor(model_config, preprocess_config)
        print("✓ 成功创建 cuFCPE 处理器")
    except Exception as e:
        print(f"✗ 创建处理器失败: {e}")
        return
    
    # 4. 准备测试音频数据
    import librosa
    audio_path = "e:/这是个音频.flac"
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_data = audio_data.astype(np.float32)  # 确保数据类型为 float32
        audio_data = audio_data.reshape(1, -1)  # 转换为 2D 数组
        audio_data = audio_data.repeat(2, axis=0)  # 扩展为 batch_size=1
    except Exception as e:
        print(f"✗ 加载音频数据失败: {e}")
        return
    
    print(f"✓ 生成测试音频数据，形状: {audio_data.shape}")
    
    # 5. 执行基频估计
    try:
        f0_result = processor.execute(audio_data)
        f0_result = f0_result.squeeze()  # 去除多余的维度
        print(f"✓ 基频估计成功，输出形状: {f0_result.shape}")
        # print(f0_result[0].tolist())  # 打印第一个样本的基频估计结果
        print(f"  输出数据类型: {f0_result.dtype}")
        print(f"  输出值范围: [{f0_result.min():.3f}, {f0_result.max():.3f}]")
    except Exception as e:
        print(f"✗ 基频估计失败: {e}")
        return
    
    # 6. 显示配置信息
    print("\n模型配置信息:")
    print(f"  模型路径: {processor.modelConfig.modelPath}")
    print(f"  输入通道: {processor.modelConfig.inputChannels}")
    print(f"  输出维度: {processor.modelConfig.outputDims}")
    print(f"  隐藏维度: {processor.modelConfig.hiddenDims}")
    print(f"  网络层数: {processor.modelConfig.numLayers}")
    print(f"  注意力头数: {processor.modelConfig.numHeads}")
    print(f"  基频范围: [{processor.modelConfig.f0Min:.1f}, {processor.modelConfig.f0Max:.1f}] Hz")
    
    print("\n预处理配置信息:")
    print(f"  采样率: {processor.preProcessConfig.samplingRate} Hz")
    print(f"  FFT 长度: {processor.preProcessConfig.fftLength}")
    print(f"  窗口大小: {processor.preProcessConfig.windowSize}")
    print(f"  跳跃大小: {processor.preProcessConfig.hopSize}")
    print(f"  Mel 滤波器组: {processor.preProcessConfig.melBins}")
    print(f"  频率范围: [{processor.preProcessConfig.freqMin:.1f}, {processor.preProcessConfig.freqMax:.1f}] Hz")



if __name__ == "__main__":
    print("cuFCPE Python 绑定测试")
    print("=" * 50)
    
    main()
    
    print("\n测试完成！")
