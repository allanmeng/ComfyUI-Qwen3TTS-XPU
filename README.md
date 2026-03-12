# ComfyUI-Qwen3TTS-XPU

<div align="center">

![Version](https://img.shields.io/badge/版本-1.2.5-blue)
![License](https://img.shields.io/badge/许可证-MIT-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange)
![XPU](https://img.shields.io/badge/Intel_Arc-XPU专用-0071C5)

**为 Intel Arc GPU 用户设计的 ComfyUI TTS 插件 — 基于阿里巴巴 Qwen3-TTS**

[English README](./README_EN.md)

> ⚠️ **本插件专为 Intel Arc GPU（XPU）用户设计。**
> 不支持 CUDA 或 Apple Silicon 平台。如需在其他平台使用，请参考原版 [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS)。

</div>

---

## 📖 简介

本项目基于 [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS) 深度改造，专为 Intel Arc GPU（XPU）加速推理而设计，将阿里巴巴 Qwen3-TTS 的高质量语音合成能力带入 ComfyUI。

相比原版的主要改进：
- 修复所有 XPU 设备挂载问题
- 手动 Code Predictor 推理循环（替换 HF `generate()`），速度提升 2×
- 支持 `torch.compile` inductor 后端，较 CPU 提速 4.6×
- 每个节点新增 5 个 XPU 专属面板选项

**依赖：** `torch 2.x+xpu`（Intel PyTorch 构建版本），**无需** `intel_extension_for_pytorch`。

---

## 🖥️ 硬件要求

| 项目 | 要求 |
|------|------|
| GPU | Intel Arc 独立显卡（A 系列或 B 系列） |
| 显存 | 最低 8 GB，1.7B 模型推荐 12 GB |
| PyTorch | `torch 2.x+xpu`（Intel 构建版本） |
| 操作系统 | Windows / Linux（需 Level Zero 驱动） |
| torch.compile | 需要 Intel oneAPI（icx / cl.exe）才能使用 inductor 后端 |

---

## 🤖 模型列表

所有模型放置于 `ComfyUI/models/TTS/Qwen3-TTS/` 目录下。

| 节点 | 推荐模型 | HuggingFace | 国内镜像（hf-mirror） | ModelScope | 说明 |
|------|----------|-------------|----------------------|------------|------|
| 🎵 **CustomVoice** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | [下载](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | [镜像](https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | [下载](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | 预设音色，质量最佳 |
| 🎵 **CustomVoice**（快速） | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | [下载](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | [镜像](https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | [下载](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | 速度更快，质量略低 |
| 🎭 **VoiceClone** | `Qwen3-TTS-12Hz-1.7B-Base` | [下载](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | [镜像](https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | [下载](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 克隆精度最佳 |
| 🎭 **VoiceClone**（快速） | `Qwen3-TTS-12Hz-0.6B-Base` | [下载](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | [镜像](https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | [下载](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | 速度更快 |
| 🎨 **VoiceDesign** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | [下载](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | [镜像](https://hf-mirror.com/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | [下载](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | **仅支持 1.7B**，无 0.6B 版本 |

模型目录结构示例：

```
ComfyUI/models/TTS/Qwen3-TTS/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-0.6B-Base/
└── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
```

> 默认为离线模式。如需自动从 HuggingFace 下载，请在启动 ComfyUI 前设置环境变量 `QWEN_TTS_ALLOW_DOWNLOAD=1`。

---

## ✨ 功能特性

### 🔷 XPU 专属面板选项（本 fork 新增）

三个节点均包含以下五个新选项：

| 选项 | 默认值 | 说明 |
|------|--------|------|
| 🔷 **torch_compile** | 关 | 使用 `inductor` 后端编译 talker 和 code predictor。首次运行需约 2.5 分钟热身，之后稳定速度 **6.5 tok/s**（较 CPU 提速 4.6×）。需要 Intel oneAPI / MSVC cl.exe 在系统 PATH 中。 |
| 🔷 **quantize_int8** | 关 | 对所有 Linear 层进行 weight-only INT8 量化，节省约 1.5 GB 显存，速度略有下降。不建议与 torch_compile 同时开启。 |
| 🔷 **full_text_prefill** | 开 | 将完整文本一次性输入 talker（推荐）。关闭则为逐 token 流式输入，速度相同但质量略低。 |
| 🔷 **cache_clean** | 开 | 每次生成前后清理 XPU allocator 缓存。防止 torch_compile 残留缓冲区累积导致第二次生成时出现 `UR_RESULT_ERROR_OUT_OF_RESOURCES` OOM。**不会**清除模型权重或已编译内核——第二次生成仍为热启动。 |
| **trailing_pad** | `…………………` | 追加在输入文本末尾，防止模型在读完最后几个字之前提前生成 EOS 导致截断。若仍有截断，可适当增加长度。 |

**Intel Arc B580（12 GB 显存）实测性能：**

| 配置 | 速度 |
|------|------|
| CPU 基线 | ~0.1 tok/s |
| XPU 基础修复 | 1.4 tok/s |
| + 手动 Code Predictor 循环 | 2.9 tok/s |
| + torch.compile inductor（热启动） | **6.5 tok/s** |

---

### 三个节点

> 基于 [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS)，保留全部原版功能。

#### 🎵 Qwen3-TTS CustomVoice
使用预设音色合成语音，支持风格指令：
- 9 种预设音色：Aiden、Dylan、Eric、Ono_anna、Ryan、Serena、Sohee、Uncle_fu、Vivian
- 可选 instruct 字段进行风格调整（如"请用慢速沉稳的语气朗读"）
- 支持 0.6B 和 1.7B 模型

#### 🎭 Qwen3-TTS VoiceClone
从参考音频克隆声音：
- 上传参考音频，可附带文字转写以提升精度
- 支持声音特征预提取复用（VoiceClonePrompt 节点）
- X-vector 模式：仅提取音色特征，速度更快
- 支持 0.6B 和 1.7B 模型

#### 🎨 Qwen3-TTS VoiceDesign
通过文字描述生成自定义声音：
- 用自然语言描述目标音色（如"沉稳的男性新闻播音员音色"）
- 仅支持 1.7B 模型

---

## 📦 安装

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU.git
cd ComfyUI-Qwen3TTS-XPU
pip install -r requirements.txt
```

**依赖：**
- `torch 2.x+xpu`（Intel PyTorch 构建版本，**非**标准 PyTorch）
- `transformers`、`librosa`、`soundfile`
- Intel oneAPI（可选，仅 `torch_compile` 需要）

---

## 🎮 使用方法

1. 将模型文件夹放置于 `ComfyUI/models/TTS/Qwen3-TTS/`
2. 在 ComfyUI 中从 `Qwen3TTS-XPU` 分类添加节点
3. 设置 **device** 为 `xpu`，**precision** 为 `bf16`
4. 如需最高速度，开启 🔷 **torch_compile**（首次运行较慢）
5. 输入文字，运行工作流

---

## 🔧 常见问题

### 第二次生成时 OOM 崩溃
开启 🔷 **cache_clean**（默认已开启）。该选项在每次生成之间清理 torch.compile 残留显存，无需重新加载模型。

### 语音末尾被截断
增加 **trailing_pad** 的长度（默认 `…………………`）。若最后几个字仍缺失，追加更多 `……`。

### torch.compile 报错
确保在启动 ComfyUI 前，`cl.exe`（MSVC）或 `icx`（Intel oneAPI）已在系统 PATH 中。可先运行 Intel oneAPI 的 `setvars.bat`。

### 显存不足
- 切换为 0.6B 模型
- 开启 🔷 **quantize_int8**（节省约 1.5 GB 显存）
- 减小 `max_new_tokens`

### 找不到模型
检查模型文件夹是否直接放置于 `ComfyUI/models/TTS/Qwen3-TTS/` 下，且文件夹名称与模型名完全一致。

---

## 🙏 致谢

- [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS) — 原版 ComfyUI 集成
- [阿里巴巴 Qwen 团队](https://github.com/QwenLM) — Qwen3-TTS 模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — UI 框架

---

## 📮 问题反馈

- **Issues**：[GitHub Issues](https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU/issues)

---

<div align="center">

**觉得有用请点 Star ⭐**

</div>
