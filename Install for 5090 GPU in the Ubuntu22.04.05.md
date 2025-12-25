

# rdt2_5090_env for flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl


## 1. 更新软件包列表并安装预备工具

在终端中执行：

```bash
sudo apt update
sudo apt install software-properties-common -y
```

## 2. 添加 `deadsnakes` PPA 并安装 Python 3.12

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```

* `python3.12-venv`：用于创建虚拟环境的关键模块
* `python3.12-dev`：包含开发头文件，后续某些 Python 包（如通过 `pip` 编译的包）可能会用到

## 3. 验证安装

安装完成后，运行以下命令检查：

```bash
python3.12 --version
```

如果输出 `Python 3.12.x` 则表示安装成功。

## 4. 安装编译工具和依赖库

`ur-rtde` 是一个与 Universal Robots 进行通信的库，它包含需要编译的 C++ 扩展。要成功安装它，你的系统必须安装必要的**编译工具链**。

```bash
sudo apt update
sudo apt install -y cmake build-essential
```

安装 Boost 开发库（包含头文件和动态链接库），ur-rtde==1.5.6 需要：

```bash
sudo apt install libboost-all-dev
```

## 5. 创建和激活虚拟环境

```bash
python3.12 -m venv rdt2_5090_env
source ~/pgj/RDT2_5090/rdt2/rdt2_5090_env/bin/activate  # 激活环境
```

## 6. 设置 CUDA 环境变量

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
```

## 7. 验证 CUDA 安装

检查 nvcc 版本：

```bash
nvcc --version
```
输出末尾应显示 "release 12.8, ..."



## 8. 安装 PyTorch 和 Flash Attention

从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令：

```bash
pip3 install torch torchvision
```

在 Python 中检查 PyTorch 使用的 CUDA 版本：

```bash
python3.12 -c "import torch; print(torch.version.cuda)"
```

从 [Flash Attention GitHub Releases](https://github.com/Dao-AILab/flash-attention/releases) 下载并安装：

```bash
pip3 install /home/ghzn/Downloads/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

## 9. 安装其他依赖

```bash
pip3 install -r requirements.txt
```

## 10. 验证 transformers 版本

确保安装了正确版本的 transformers (4.51.3)：

```bash
pip3 list | grep transformers
```

## 11. 部署到 Franka Research 3 的额外依赖

```bash
pip3 install -r requirements/franka_research_3.txt
```

## 注意事项

### franka_research_3.txt 调整建议

以下是 `franka_research_3.txt` 文件可能需要包含的包：

```
ur-rtde==1.5.6
pynput==1.7.6
imagecodecs==2023.9.18
atomics==1.0.2
minimalmodbus
zerorpc
openvr
```



<img width="1492" height="523" alt="1" src="https://github.com/user-attachments/assets/a7df2eed-f26b-4af0-84d4-77646dc2d928" />

<img width="1729" height="892" alt="2" src="https://github.com/user-attachments/assets/26a059ff-c6bd-45d8-95bb-99345bc18623" />

<img width="1742" height="913" alt="3" src="https://github.com/user-attachments/assets/375e0a6e-e2ce-4fe3-bed3-275733ce61fc" />
