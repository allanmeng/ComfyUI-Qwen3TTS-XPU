# ComfyUI-Qwen3-TTS XPU 优化项目总结

## 项目背景

为 Intel Arc B580 GPU (XPU) 启用 ComfyUI-Qwen3-TTS 插件的 GPU 加速推理。

## 硬件/软件环境

| 项目 | 详情 |
|------|------|
| GPU | Intel Arc B580 (xpu:0), 12GB VRAM |
| PyTorch | torch 2.11.0+xpu（Intel 官方版，内置 XPU 支持） |
| Triton | pytorch-triton-xpu 3.3.0 |
| Python | 3.13.11 MSC v.1944 (Windows 11) |
| IPEX | 不可用（已被 torch+xpu 取代，无法安装） |
| oneAPI | 已安装（setvars.bat + icx.exe + cl.exe） |

## 模型信息

| 模型 | 用途 | 状态 |
|------|------|------|
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 预设音色 TTS | 已有 |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 预设音色 TTS（轻量） | 已有 |
| Qwen3-TTS-12Hz-1.7B-Base | VoiceClone 声音克隆 | 待下载 |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 文字描述设计声音 | 待下载 |

模型路径：`F:\ComfyUI-aki-v3\ComfyUI\models\TTS\Qwen3-TTS\`

## 模型架构

- **Talker**（1.7B Transformer）：主语言模型，生成 token 序列
- **Code Predictor**：每个主 token 需调用 6 次，生成音频编码
- **Mimi Decoder**（Speech Tokenizer）：将音频编码解码为波形
- 每个主 token = 7 次 forward pass（1 Talker + 6 Code Predictor）

---

## 已完成的代码修改

### 1. `qwen_tts/core/models/modeling_qwen3_tts.py`

| 修改 | 位置 | 说明 |
|------|------|------|
| autocast XPU 兼容 | 行 516, 549 | `not in ("mps", "xpu")` 排除 XPU |
| 手动 Code Predictor 循环 | 行 1634-1692 | 替换 HF generate() 为 forward()+KV cache |
| XPU tensor device 修复 | 行 2230, 2242 | original_lengths / indices 加 device= 参数 |

### 2. `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`

| 修改 | 位置 | 说明 |
|------|------|------|
| autocast XPU 兼容 | 行 271 | 同上 |

### 3. `qwen_tts/core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py`

| 修改 | 位置 | 说明 |
|------|------|------|
| autocast XPU 兼容 | 行 112 | 同上 |

### 4. `nodes.py`

| 修改 | 说明 |
|------|------|
| `_ensure_on_xpu()` | 强制主模型和 speech tokenizer 移到 XPU |
| `_try_set_cc_for_windows()` | 自动检测 C 编译器，设置 CC 环境变量供 Triton 使用 |
| `_apply_torch_compile()` | 编译 talker.model 和 code_predictor.model |
| `_apply_int8_weight_quantization()` | weight-only INT8 量化，250 层 Linear |
| `load_qwen_model()` | 新增 torch_compile、quantize_int8 参数 |
| UI 开关 | 三个节点新增：torch_compile、quantize_int8、full_text_prefill |
| Verbose 日志 | 每阶段计时 + tensor device 信息 |
| 模型路径 | 统一到 models/TTS/Qwen3-TTS/ |
| 清理 | 删除无效的 check_and_download_tokenizer() 函数 |
| 中文 tooltip | 所有参数添加中文悬停说明 |

---

## 速度测试结果

| 配置 | 速度 | 备注 |
|------|------|------|
| 原始（CPU 推理） | ~0.1 tok/s | 无 XPU 支持 |
| XPU 基础修复后 | 1.4 tok/s | autocast + device 修复 |
| Code Predictor 手动循环 | 2.9 tok/s | 核心优化，速度 2x |
| + INT8 量化 | 1.9 tok/s | 速度略降，省 1.5GB VRAM |
| + torch.compile inductor（首次含编译） | 0.14 tok/s | 编译耗时 ~12min |
| + torch.compile inductor（同 session 稳态） | 6.5 tok/s | 比基准快 2.2x |
| + torch.compile inductor（重启后热身） | ~2.5min 热身后 6.5 tok/s | Triton kernel 缓存命中，FX 图需重追踪 |

理论峰值 Arc B580：~50 tok/s（差距原因：SYCL kernel dispatch 延迟）

---

## UI 参数说明

### 三个节点共有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| device | auto | 自动选择 GPU |
| precision | bf16 | 推荐，速度快显存少 |
| seed | 0 | 固定=可复现，randomize=每次不同 |
| max_new_tokens | 2048 | 控制最长音频时长 |
| temperature | 1.0 | 越低越稳定 |
| top_p | 0.8 | Nucleus 采样阈值 |
| top_k | 20 | 每步候选 token 数 |
| repetition_penalty | 1.05 | 防止音频重复 |
| attention | auto | sdpa 通常最快 |
| unload_model_after_generate | False | 生成后释放显存 |
| torch_compile | False | 每次启动后首次编译需约 2.5 分钟，之后同一会话内速度提升约 2 倍 |
| quantize_int8 | False | 省 1.5GB VRAM，速度略降，不建议与 torch_compile 同时开启 |
| full_text_prefill | True | True 推荐，全文一次性输入 |

### 节点区别

| 节点 | 用途 | 特有输入 |
|------|------|---------|
| CustomVoice | 使用内置预设音色 | speaker（9种预设人声） |
| VoiceClone | 克隆参考音频的声音 | ref_audio + ref_text |
| VoiceDesign | 用文字描述设计声音 | instruct（如"温柔的女声"） |

---

## 讨论过的方向

| 讨论主题 | 分析现状 | 结论 |
|----------|----------|------|
| XPU 设备兼容性 | autocast 不支持 xpu，tensor 默认在 CPU 创建 | 修复 autocast 排除条件 + 显式指定 device |
| Code Predictor 瓶颈 | 每个主 token 调用一次完整 HF generate()，开销极大 | 替换为手动 forward()+KV cache 循环，速度 2x |
| Speech Tokenizer 位置 | 普通 Python 对象，`.to()` 不自动移动内部 module | 显式 `.to("xpu")`，CPU 解码需 >5 分钟 |
| IPEX 是否可用 | torch 2.11+xpu 已内置 XPU 支持，IPEX 无法安装 | 不可用，跳过 |
| sdpa vs eager attention | 两种实现都支持 XPU | 速度无差异，保持 sdpa 默认 |
| OpenVINO Mimi Decoder | Audio Decode 占总时间 22%，有自定义算子兼容问题 | 收益低难度高，不值得 |
| intel-extension-for-transformers | 需要 IPEX，而 IPEX 不可用 | 不适用 |
| XMX 矩阵加速是否启用 | torch+xpu 理论上自动使用 XMX，无法手动强制 | 已自动启用，无需操作 |
| INT8 量化 | weight-only per-channel，250 层 Linear 替换 | 速度略降，省 1.5GB VRAM，质量无明显影响 |
| full_text_prefill | non_streaming_mode 代码已有，通过 UI 暴露 | True/False 速度无差异，默认 True |
| llama.cpp SYCL | 需要 GGUF 格式，Qwen3-TTS 官方无此格式 | 暂无意义 |
| torch.compile aot_eager | 无需编译器，但只做图追踪不生成优化 kernel | 速度无提升，无实用价值 |
| torch.compile inductor | 需要 MSVC cl.exe + setvars.bat | 稳态 2.9→6.5 tok/s，多次使用值得开启 |
| inductor 缓存持久化 | Triton kernel 缓存到磁盘，FX 图每次重启需重追踪 | 无法完全消除热身，属 PyTorch 当前限制 |
| 模型路径管理 | models/qwen-tts/ 与 models/TTS/Qwen3-TTS/ 两处重复 | 统一到 models/TTS/Qwen3-TTS/ |
| 三节点模型需求 | 当前只有两个 CustomVoice 模型 | VoiceClone 需 Base 模型，VoiceDesign 需 VoiceDesign 模型 |

---

## 未来优化方向

| 方向 | 难度 | 收益 | 建议 |
|------|------|------|------|
| 下载 Base/VoiceDesign 模型 | 低 | 高（解锁另外两个节点） | 推荐 |
| inductor 持久化缓存 | 高（PyTorch 限制） | 高（消除 2.5min 热身） | 待 PyTorch 版本更新 |
| 更大批量/并行生成 | 中 | 中（适合批量场景） | 可选 |
| OpenVINO Mimi Decoder | 高 | 低（仅占 22%） | 不推荐 |

---

*生成日期：2026-03-11*
*项目路径：`F:\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI-Qwen3TTS-XPU`*
