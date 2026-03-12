# ComfyUI-Qwen3TTS-XPU

<div align="center">

![Version](https://img.shields.io/badge/version-1.2.5-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange)
![XPU](https://img.shields.io/badge/Intel_Arc-XPU_only-0071C5)

**ComfyUI TTS plugin for Intel Arc GPU users — powered by Alibaba Qwen3-TTS**

> ⚠️ **This plugin is designed exclusively for Intel Arc GPU (XPU) users.**
> It does not target CUDA or Apple Silicon. For those platforms, use the original [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS).

</div>

---

## 📖 Overview

Forked and heavily modified from [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS) to bring full Intel Arc XPU acceleration to Qwen3-TTS in ComfyUI.

Key changes over the original:
- All XPU device placement bugs fixed
- Manual Code Predictor inference loop (replaces HF `generate()`) for 2× speed
- `torch.compile` inductor support for 4.6× speed vs CPU
- 5 new XPU-specific panel options on every node

**Requires:** `torch 2.x+xpu` (Intel PyTorch build). Does **not** require `intel_extension_for_pytorch`.

---

## 🖥️ Hardware Requirements

| Item | Requirement |
|------|-------------|
| GPU | Intel Arc discrete GPU (A-series or B-series) |
| VRAM | 8 GB minimum, 12 GB recommended for 1.7B models |
| PyTorch | `torch 2.x+xpu` (Intel build) |
| OS | Windows / Linux with Level Zero driver |
| torch.compile | Intel oneAPI (icx / cl.exe) required for inductor backend |

---

## 🤖 Models

Each node requires a specific model variant. Place all models under `ComfyUI/models/TTS/Qwen3-TTS/`.

| Node | Recommended Model | Download | Notes |
|------|-------------------|----------|-------|
| 🎵 **CustomVoice** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Best quality for preset speakers |
| 🎵 **CustomVoice** (fast) | `Qwen3-TTS-12Hz-0.6B-CustomVoice` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | Faster, slightly lower quality |
| 🎭 **VoiceClone** | `Qwen3-TTS-12Hz-1.7B-Base` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Best clone accuracy |
| 🎭 **VoiceClone** (fast) | `Qwen3-TTS-12Hz-0.6B-Base` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Faster clone |
| 🎨 **VoiceDesign** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | **1.7B only**, no 0.6B variant |

Download from HuggingFace (Qwen org) and place each model folder directly under `ComfyUI/models/TTS/Qwen3-TTS/`:

```
ComfyUI/models/TTS/Qwen3-TTS/
├── Qwen3-TTS-12Hz-1.7B-CustomVoice/
├── Qwen3-TTS-12Hz-0.6B-CustomVoice/
├── Qwen3-TTS-12Hz-1.7B-Base/
├── Qwen3-TTS-12Hz-0.6B-Base/
└── Qwen3-TTS-12Hz-1.7B-VoiceDesign/
```

> Set `QWEN_TTS_ALLOW_DOWNLOAD=1` to enable automatic download on first run. Default is offline mode.

---

## ✨ Features

### 🔷 XPU Panel Options (new in this fork)

All five options are available on every node:

| Option | Default | Description |
|--------|---------|-------------|
| 🔷 **torch_compile** | Off | Compiles talker + code predictor with `inductor` backend. First run ~2.5 min warm-up, then **6.5 tok/s** (4.6× vs CPU). Requires Intel oneAPI / MSVC cl.exe in PATH. |
| 🔷 **quantize_int8** | Off | Weight-only INT8 quantization on all Linear layers. Saves ~1.5 GB VRAM with slight speed trade-off. Not recommended together with torch_compile. |
| 🔷 **full_text_prefill** | On | Feed the entire text to the talker at once (recommended). Off = streaming token-by-token, same speed but lower quality. |
| 🔷 **cache_clean** | On | Flushes XPU allocator cache before/after each generation. Prevents `UR_RESULT_ERROR_OUT_OF_RESOURCES` OOM on second run when torch_compile is active. Does **not** evict model weights or compiled kernels — second run is still a hot-start. |
| **trailing_pad** | `…………………` | Appended to input text to prevent the model from generating EOS before finishing the last few characters. Increase length if truncation persists. |

**Measured performance on Intel Arc B580 (12 GB VRAM):**

| Configuration | Speed |
|---------------|-------|
| CPU baseline | ~0.1 tok/s |
| XPU basic | 1.4 tok/s |
| + Manual Code Predictor loop | 2.9 tok/s |
| + torch.compile inductor (warm) | **6.5 tok/s** |

---

### Three Nodes

> Based on [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS) — all original functionality preserved.

#### 🎵 Qwen3-TTS CustomVoice
Use preset high-quality voices with optional style instructions:
- 9 preset speakers: Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
- Optional instruct field for style modulation (e.g. "speak slowly and calmly")
- Available in 0.6B and 1.7B

#### 🎭 Qwen3-TTS VoiceClone
Clone any voice from a reference audio sample:
- Upload reference audio + optional transcript for better accuracy
- Reusable voice prompt extraction (VoiceClonePrompt node)
- X-vector only mode for lightweight speaker embedding
- Available in 0.6B and 1.7B

#### 🎨 Qwen3-TTS VoiceDesign
Generate a custom voice from a text description:
- Describe the voice (e.g. "A calm male news anchor voice")
- 1.7B model only

---

## 📦 Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU.git
cd ComfyUI-Qwen3TTS-XPU
pip install -r requirements.txt
```

**Requirements:**
- `torch 2.x+xpu` (Intel PyTorch build, **not** standard PyTorch)
- `transformers`, `librosa`, `soundfile`
- Intel oneAPI (optional, needed only for `torch_compile`)

---

## 🎮 Usage

1. Place model folders under `ComfyUI/models/TTS/Qwen3-TTS/`
2. Add a node from the `Qwen3TTS-XPU` category
3. Set **device** to `xpu`, **precision** to `bf16`
4. Enable 🔷 **torch_compile** for best speed (first run will be slow)
5. Enter text and run

---

## 🔧 Troubleshooting

### OOM on second generation
Enable 🔷 **cache_clean** (default On). This flushes torch.compile residual VRAM between runs without reloading the model.

### End of speech truncated
Increase **trailing_pad** (default `…………………`). Add more `……` if the last few characters are still missing.

### torch.compile fails
Ensure `cl.exe` (MSVC) or `icx` (Intel oneAPI) is in PATH before launching ComfyUI. Run `setvars.bat` from your Intel oneAPI installation first.

### Out of Memory
- Switch to 0.6B model
- Enable 🔷 **quantize_int8** (~1.5 GB savings)
- Reduce `max_new_tokens`

### Model not found
Check that model folders are directly under `ComfyUI/models/TTS/Qwen3-TTS/` with the correct folder names.

---

## 🙏 Acknowledgments

- [ai-joe-git/ComfyUI-Qwen3-TTS](https://github.com/ai-joe-git/ComfyUI-Qwen3-TTS) — original ComfyUI integration
- [Alibaba Qwen Team](https://github.com/QwenLM) — Qwen3-TTS models
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — UI framework

---

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU/issues)

---

<div align="center">

**Star ⭐ this repo if you find it useful!**

</div>
