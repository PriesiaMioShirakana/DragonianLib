# cuFCPE

## 编译说明

### 环境要求
- C++17 编译器（GCC 7+ 或 Clang 5+ 或 MSVC 2017+）
- CMake 3.12+
- CUDA Toolkit

### 编译步骤

#### 使用 CMake（推荐）
```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build . --parallel

# 或者使用 make（在 Unix 系统上）
make -j$(nproc)
```

#### Debug 模式编译
```bash
mkdir build-debug
cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

#### 指定编译器
```bash
# 使用 GCC
cmake .. -DCMAKE_CXX_COMPILER=g++

# 使用 Clang
cmake .. -DCMAKE_CXX_COMPILER=clang++
```