# TTS 基于 OpenVINO 优化的价值验证

> 硬件环境：Intel Arc B580（XPU，12GB VRAM）
> 软件环境：torch 2.11.0+xpu、OpenVINO 2025.4.1、Python 3.13
> 模型：Qwen3-TTS-12Hz-1.7B-CustomVoice
> 测试日期：2026-03-12 ~ 2026-03-13

---

## 一、背景与动机

Qwen3-TTS 的推理管线由三个主要阶段组成：

| 阶段 | 模块 | 说明 |
|------|------|------|
| 文本编码 + 语音生成 | Talker（1.7B）+ Code Predictor | 自回归生成 audio codes |
| 音频解码 | Mimi Decoder | audio codes → 波形 |

在完成 XPU 基础移植（autocast 修复、Code Predictor 手动循环）和 torch.compile inductor 加速后，整体速度达到 **6.5 tok/s**。此时各阶段耗时占比约为：

- Talker + Code Predictor 生成：**~78%**
- Mimi Decoder 音频解码：**~22%**

Mimi Decoder 占 22% 看似有优化空间，因此开始评估 OpenVINO 是否可以进一步提速。

---

## 二、OpenVINO 导出实验

### 2.1 Mimi Decoder Conv 部分

Mimi Decoder 的 `forward()` 由以下阶段组成：

```
audio_codes → quantizer.decode → pre_conv → pre_transformer → upsample(ConvTranspose1d) → decoder(Conv1d) → wav
```

由于 `pre_transformer`（HuggingFace Transformer）使用 vmap-based causal mask，导出时会产生 tracing 问题。因此采用分段导出策略，**只导出 conv-only 部分**（upsample + decoder），绕过 Transformer。

**导出结果：**

| 文件 | 大小 |
|------|------|
| `mimi_decoder_ov/mimi_conv_decoder.xml` | 584 KB |
| `mimi_decoder_ov/mimi_conv_decoder.bin` | ~140 MB |

- `torch.export.export()` 成功（含动态 T 维度）
- `ov.convert_model()` 成功
- CPU 推理验证通过，PyTorch vs OpenVINO 最大误差 < 1e-3 ✅

### 2.2 Code Predictor

Code Predictor 分两个阶段：Prefill（seq_len=2，无 KV cache）和 Decode（带 KV cache，每主 token 调用 15 次）。

**Prefill 导出：**

| 文件 | 大小 |
|------|------|
| `mimi_decoder_ov/cp_prefill.xml` | 391 KB |
| `mimi_decoder_ov/cp_prefill.bin` | ~155 MB |

- `torch.export.export()` 成功
- OV 编译及 CPU 推理验证通过 ✅

**Decode 导出：**

- ❌ 失败：`past_key_values`（`unordered_map<K,T>` 结构）无法序列化
- JIT tracing 无法处理自回归 KV cache 的动态字典结构
- 需要 OV StatefulModel API 才能支持，实现复杂度高

### 2.3 Intel Arc GPU 测试

受时间限制（测试前机器死机），**未完成 GPU（`device="GPU"`）端到端性能测试**。CPU 基准验证通过，Arc GPU 性能数据待补充。

---

## 三、关键发现：性能瓶颈的真实原因

在评估 OpenVINO 价值的过程中，通过对 `forward()` 添加 XPU-synchronized 分段计时，发现了一个关键现象：

**Mimi Decoder conv 的执行时间在 0.02s 和 8~9s 之间随机跳变。**

经过系统分析，确认根本原因为：

> **Intel XPU（Level Zero / OpenCL）的 kernel JIT 编译按 input shape 缓存。每遇到新的 T 值（token 序列长度），首次执行触发驱动层 OpenCL/SYCL kernel 编译，耗时约 8~10 秒；同一 T 值第二次直接走缓存，耗时约 0.02 秒。**

这意味着原先观测到的"Mimi Decoder 占 22% 耗时"包含了隐性的 JIT 编译开销，**实际计算时间远小于 22%**。

### 修复方案

**Bucket Padding**：将 audio codes 填充到 64 的整数倍，限制唯一 shape 数量；模型加载时对 T=64/128/192 做一次 dummy decode 预热，后续所有生成均走 kernel 缓存。

**修复效果（短/中文本，T ≤ 300）：**

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| Audio Decode 耗时 | 0.02s ↔ **8~9s（随机）** | **0.07~0.12s（稳定）** |
| 整体稳定性 CV | 35% ⚠️ | **6.4% ✅** |
| 整体平均速度 | 3.5 tok/s | **5.8 tok/s** |

**长文本（T > 300）**：`chunked_decode` 将完整 codes 切成 chunk_size=300 的片段，chunk shape 不是 64 的整数倍，仍会触发 JIT。该问题需将 bucket padding 移入 `forward()` 内部解决（待办）。

---

## 四、OpenVINO 各模块价值评估

基于上述发现，重新评估各模块的 OpenVINO 优化价值：

### 4.1 Mimi Decoder

| 项目 | 评估 |
|------|------|
| 导出可行性 | ✅ 已验证（conv 部分） |
| 实际计算时间 | **< 0.1s**（bucket padding 修复后） |
| 占总耗时比例 | **< 1%** |
| **OV 优化收益** | **极低，不值得集成** |

**结论**：Mimi Decoder 的性能问题已通过 bucket padding 解决，OV 路线优先级最低。

### 4.2 Code Predictor

| 项目 | 评估 |
|------|------|
| Prefill 导出 | ✅ 已验证 |
| Decode 导出 | ❌ KV cache 无法序列化 |
| 实际影响 | Code Predictor 每主 token 调用 15 次，约占生成时间 30~40% |
| **OV 优化收益** | **中等，但实现成本高** |

**结论**：需要使用 OV StatefulModel API 重构 Decode 阶段，工程量大；且 torch.compile inductor 已对其提速，优先级中等。

### 4.3 Talker（1.7B 主模型）

| 项目 | 评估 |
|------|------|
| 导出可行性 | ❌ 与 Code Predictor 相同的 KV cache 问题 |
| 硬件带宽利用率 | ~6%（B580 理论 336GB/s，实际 ~20GB/s） |
| torch.compile 现状 | 已达 6.5 tok/s 稳态 |
| **OV 优化收益** | **潜在较大，但实现极复杂** |

**结论**：OV INT8 量化（Calibrated PTQ）若成功，带宽需求减半，理论速度可达 10~15 tok/s；但自回归生成 + KV cache 的 OV 适配是主要障碍。

---

## 五、总体结论

| 模块 | 实现成本 | 预期收益 | 优先级 |
|------|---------|---------|-------|
| Mimi Decoder OV | 低（已完成） | **无意义**（< 1%） | ⛔ 暂不集成 |
| Code Predictor Decode OV | 高 | 中 | 🔶 低优先级 |
| Talker OV（StatefulModel） | 极高 | 大（不确定） | 🔶 中优先级 |

**核心结论：**

1. **OpenVINO 对 Mimi Decoder 无实质价值**——其性能问题是 XPU kernel JIT 导致的，与计算框架无关，已通过 bucket padding 解决。
2. **OpenVINO 对 Code Predictor 和 Talker 有潜在价值**，但实现门槛在于 KV cache 序列化，需要 OV StatefulModel API。
3. **当前阶段的最优路线仍是 torch.compile inductor + XPU kernel 预热**，在 Intel Arc B580 上已实现 5.8~6.5 tok/s 的稳定性能。
4. **下一步更值得探索的方向**：修复长文本 Audio Decode（将 bucket padding 移入 `forward()`），以及探索 OV Calibrated INT8 量化对 Talker 的加速效果。

---

## 附录：相关文件

| 文件 | 说明 |
|------|------|
| `test_ov_mimi_decoder.py` | Mimi Decoder OV 导出 + 基准测试脚本 |
| `test_ov_code_predictor.py` | Code Predictor OV 导出 + 基准测试脚本 |
| `test_e2e_benchmark.py` | E2E 端到端基准测试脚本 |
| `mimi_decoder_ov/mimi_conv_decoder.xml/.bin` | Mimi Decoder conv OV IR 模型 |
| `mimi_decoder_ov/cp_prefill.xml/.bin` | Code Predictor Prefill OV IR 模型 |
