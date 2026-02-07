# 解决 vLLM 安装问题：PyTorch 和 CUDA 版本匹配安装

---

## 🔧 问题分析

1. **PyTorch 未安装**：终端输出显示 `ModuleNotFoundError: No module named 'torch'`
2. **CUDA 版本不匹配**：你使用 CUDA 12.8，但安装 PyTorch 时未指定 CUDA 版本
3. **vLLM 版本与 PyTorch 不兼容**：你提到 vLLM 0.9.2 需要 torch==2.7.0，但你安装了 2.9.1

---

## ✅ 完整解决方案

### 1. 安装正确的 PyTorch 版本（CUDA 12.8）

```bash
# 确保激活你的虚拟环境
source ~/pgj/RDT2_5090_Py312/RDT2_5090_Py312_env/bin/activate

# 安装 PyTorch 2.9.1 with CUDA 12.8
pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 验证 PyTorch 安装

```bash
python3.12 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**预期输出**：
```
PyTorch: 2.9.1
CUDA available: True
CUDA version: 12.8
```

### 3. 安装 Flash Attention

```bash
# 确保你有正确的预编译包
pip3.12 install flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

### 4. 安装 vLLM（关键步骤）

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python3.12 use_existing_torch.py
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --no-build-isolation --refresh --verbose -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 💡 **为什么需要 `--no-build-isolation`？**  
> 由于你已经安装了 PyTorch，这个参数告诉构建系统使用当前环境中的 PyTorch，而不是尝试重新安装。

### 5. 验证 vLLM 安装

```bash
python3.12 -c "from vllm import LLM; print('vLLM导入成功'); llm = LLM(model='facebook/opt-125m', gpu_memory_utilization=0.1); print('LLM初始化成功')"
```

**预期输出**：
```
vLLM导入成功
LLM初始化成功
```

---

## 📌 重要说明

### 为什么 vLLM 0.9.2 与 PyTorch 2.9.1 不兼容？
- vLLM 0.9.2 确实要求 torch==2.7.0
- 但你安装的是 vLLM 最新版本（不是 0.9.2），**最新 vLLM 版本支持 PyTorch 2.9.1**

### 正确的 vLLM 版本选择
- 如果你想使用最新功能，安装 vLLM 最新版本（0.14.0+）
- 如果必须使用 vLLM 0.9.2，需要安装 PyTorch 2.7.0：
  ```bash
  pip3.12 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

---

## 🔍 诊断命令

运行以下命令验证环境：

```bash
# 1. PyTorch 版本
python3.12 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 2. CUDA 版本
python3.12 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 3. Flash Attention 版本
pip3.12 show flash-attn

# 4. vLLM 版本
pip3.12 show vllm
```

---

## 🚫 常见错误及解决

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch 未安装 | 安装 `torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| `vLLM 0.9.2 requires torch==2.7.0` | 使用了不兼容的 vLLM 版本 | 升级 vLLM 或安装 PyTorch 2.7.0 |
| `CUDA version mismatch` | PyTorch 和 CUDA 版本不匹配 | 确保使用 `--index-url https://download.pytorch.org/whl/cu128` |

---

## 💡 额外建议

1. **使用最新 vLLM**：
   ```bash
   # 确保你使用的是最新 vLLM 版本
   cd ~/pgj/RDT2_5090_Py312/rdt2/vllm
   git pull origin main
   ```

2. **使用正确版本的 Flash Attention**：
   - 从 [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) 下载与 PyTorch 2.9.1 和 CUDA 12.8 匹配的版本

3. **检查 nvcc 版本**：
   ```bash
   nvcc --version
   ```
   确保输出包含 `release 12.8`，否则需要安装 CUDA Toolkit 12.8

---

## ✅ 最终验证

```bash
# 1. 确认 PyTorch 安装
python3.12 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA version: {torch.version.cuda}')"

# 2. 确认 vLLM 安装
python3.12 -c "from vllm import LLM; print('vLLM is installed successfully'); llm = LLM(model='facebook/opt-125m'); print('vLLM model loaded successfully')"
```

如果看到 `vLLM is installed successfully` 和 `vLLM model loaded successfully`，说明安装成功！

---

## 📌 总结

1. **先安装 PyTorch 2.9.1 with CUDA 12.8**
2. **然后安装 Flash Attention**
3. **最后使用 `--no-build-isolation` 安装 vLLM**

这样就能解决 `ModuleNotFoundError: No module named 'torch'` 问题，成功安装 vLLM。

<img width="3772" height="1360" alt="截图 2025-12-25 15-55-42" src="https://github.com/user-attachments/assets/f92da168-7053-42b9-b951-022887e881ed" />

<img width="2499" height="2059" alt="截图 2025-12-25 15-59-55" src="https://github.com/user-attachments/assets/efb30a92-e5be-4f9f-8811-4d8db9ef966f" />

<img width="3067" height="1608" alt="截图 2025-12-25 16-21-15" src="https://github.com/user-attachments/assets/19f95548-9a61-475b-8ce9-ccad6e9fc4b1" />
