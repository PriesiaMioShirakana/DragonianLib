import os
import sys
import ctypes
from pathlib import Path

class CudaDLLLoader:
    """CUDA DLL 动态加载器"""
    
    def __init__(self):
        self.cuda_paths = os.environ.get('CUDA_PATH', '').split(os.pathsep)
        
        if not self.cuda_paths or self.cuda_paths == ['']:
            self.cuda_paths = [
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
            ]
        self.cuda_paths = [os.path.join(p, "bin") for p in self.cuda_paths]
        self.required_dlls = [
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cufft64_11.dll", 
            "cublasLt64_12.dll"
        ]
        self.loaded_dlls = {}
    
    def find_cuda_path(self):
        """查找 CUDA 安装路径"""
        for path in self.cuda_paths:
            if os.path.exists(path):
                return path
        return None
    
    def preload_dlls(self):
        """预加载所有必需的 CUDA DLL"""
        cuda_bin = self.find_cuda_path()
        if not cuda_bin:
            raise RuntimeError("No CUDA installation found in the expected paths.")

        # 将 CUDA bin 添加到当前进程的 PATH
        current_path = os.environ.get('PATH', '')
        if cuda_bin not in current_path:
            os.environ['PATH'] = cuda_bin + os.pathsep + current_path
        
        # 使用 Windows API 将目录添加到 DLL 搜索路径
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.SetDllDirectoryW(cuda_bin)
        except Exception as e:
            print(f"Set DLL fail: {e}")
            return False
        
        # 预加载每个 DLL
        success_count = 0
        for dll_name in self.required_dlls:
            dll_path = os.path.join(cuda_bin, dll_name)
            
            if not os.path.exists(dll_path):
                continue
            
            try:
                # 使用 ctypes 预加载 DLL
                dll_handle = ctypes.CDLL(dll_path)
                self.loaded_dlls[dll_name] = dll_handle
                success_count += 1
            except Exception as e:
                print(f"Set DLL fail: {e}")
                return False
            
        if success_count == len(self.required_dlls):
            return True
        else:
            return False