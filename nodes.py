# ComfyUI-Qwen-TTS Node Implementation
# Based on the open-source Qwen3-TTS project by Alibaba Qwen team
# XPU Intel Arc GPU compatibility + CPU multi-core optimization added

import os
import sys
import torch
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import folder_paths
import types
from comfy import model_management
from comfy.utils import ProgressBar

from qwen_tts.core.rope_compat import patch_rope_default_if_missing
patch_rope_default_if_missing()


# Optional Intel Extension for PyTorch support
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    print("✅ [Qwen3-TTS-XPU] Intel Extension for PyTorch (IPEX) loaded - XPU optimizations enabled")
except ImportError:
    IPEX_AVAILABLE = False

# CPU Core Optimization - Auto-detect and configure (ONE-TIME, AT IMPORT)
CPU_CORES = os.cpu_count() or 4
try:
    # Set PyTorch to use all CPU cores for inference
    torch.set_num_threads(CPU_CORES)
    # IMPORTANT: we do NOT call set_num_interop_threads here to avoid runtime warnings
    os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
    os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)
    os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_CORES)
    print(f"✅ [Qwen3-TTS-XPU] CPU multi-core optimization enabled - Using {CPU_CORES} cores")
except Exception as e:
    print(f"⚠️ [Qwen3-TTS-XPU] CPU threading setup warning: {e}")

def optimize_cpu_performance():
    """Re-apply light CPU hints without touching inter-op thread count."""
    try:
        torch.set_num_threads(CPU_CORES)
        os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
        os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)
        os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_CORES)
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS-XPU] CPU optimization warning: {e}")

# Common languages list for UI
DEMO_LANGUAGES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]

# Language mapping dictionary to engine codes
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

# Model family options for UI (0.6B / 1.7B)
MODEL_FAMILIES = ["0.6B", "1.7B"]

# Mapping of family to default HuggingFace repo ID
MODEL_FAMILY_TO_HF = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

# All required models for batch download
ALL_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
]


_MODEL_CACHE: Dict[Any, Any] = {}

# Handle qwen_tts package import
current_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_dir = os.path.join(current_dir, "qwen_tts")

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if qwen_tts_dir not in sys.path:
    sys.path.insert(0, qwen_tts_dir)

try:
    import qwen_tts
    Qwen3TTSModel = qwen_tts.Qwen3TTSModel
    VoiceClonePromptItem = qwen_tts.VoiceClonePromptItem
except ImportError:
    try:
        from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
    except ImportError as e:
        import traceback
        print(f"\n❌ [Qwen3-TTS-XPU] Critical Import Error: {e}")
        if not os.path.exists(qwen_tts_dir):
            print(f"   Missing directory: {qwen_tts_dir}")
            print("   Please clone the repository with submodules or ensure 'qwen_tts' folder exists.")
        else:
            print("   Traceback for debugging:")
            traceback.print_exc()
            print("\n   Common fix: run 'pip install -r requirements.txt' in your ComfyUI environment.")
        Qwen3TTSModel = None
        VoiceClonePromptItem = None

ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]


def check_attention_implementation():
    """Check available attention implementations and return in priority order."""
    available = []

    try:
        from sageattention import sageattn  # noqa: F401
        available.append("sage_attn")
    except ImportError:
        pass

    try:
        import flash_attn  # noqa: F401
        available.append("flash_attn")
    except ImportError:
        pass

    available.append("sdpa")
    available.append("eager")

    return available


def get_attention_implementation(selection: str) -> str:
    """Get the actual attention implementation based on selection and availability."""
    available = check_attention_implementation()

    if selection == "auto":
        priority = ["sage_attn", "flash_attn", "sdpa", "eager"]
        for attn in priority:
            if attn in available:
                print(f"🔧 [Qwen3-TTS-XPU] Auto-selected attention: {attn}")
                return attn
        return "eager"
    else:
        if selection in available:
            print(f"🔧 [Qwen3-TTS-XPU] Using attention: {selection}")
            return selection
        else:
            print(f"⚠️ [Qwen3-TTS-XPU] Requested attention '{selection}' not available, falling back to sdpa")
            if "sdpa" in available:
                return "sdpa"
            return "eager"


def unload_cached_model(cache_key=None):
    """Unload cached model(s) and clear GPU memory."""
    global _MODEL_CACHE

    if cache_key and cache_key in _MODEL_CACHE:
        print(f"🔧 [Qwen3-TTS-XPU] Unloading model: {cache_key}...")
        del _MODEL_CACHE[cache_key]
    elif _MODEL_CACHE:
        print(f"🔧 [Qwen3-TTS-XPU] Unloading {len(_MODEL_CACHE)} cached model(s)...")
        _MODEL_CACHE.clear()

    # XPU cleanup
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS-XPU] XPU cleanup warning: {e}")

    # CUDA cleanup
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS-XPU] CUDA cleanup warning: {e}")

    # ComfyUI cleanup
    model_management.soft_empty_cache()

    import gc
    gc.collect()
    gc.collect()

    print("✅ [Qwen3-TTS-XPU] Model cache and GPU memory cleared")


def _ensure_on_xpu(model):
    """
    Verify and force-place all model components onto XPU (Intel Arc).

    device_map='xpu' may silently fall back to CPU on older accelerate versions.
    The speech_tokenizer is a plain Python object (not nn.Module), so it is NOT
    moved by model.to('xpu') and must be handled explicitly.
    """
    xpu_dev = torch.device("xpu")

    # --- main transformer body ---
    try:
        current = next(model.model.parameters()).device
        if current.type != "xpu":
            print(f"⚠️ [Qwen3-TTS-XPU] Main model is on {current}, moving to XPU ...")
            model.model.to(xpu_dev)
            print("✅ [Qwen3-TTS-XPU] Main model moved to XPU")
        else:
            print(f"✅ [Qwen3-TTS-XPU] Main model confirmed on XPU ({current})")
    except StopIteration:
        pass

    # update the wrapper's cached device attribute
    model.device = xpu_dev

    # --- speech_tokenizer (plain Python wrapper, NOT an nn.Module submodule) ---
    st = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    if st is not None:
        st_model = getattr(st, "model", None)
        if st_model is not None:
            try:
                st_current = next(st_model.parameters()).device
                if st_current.type != "xpu":
                    print(f"⚠️ [Qwen3-TTS-XPU] Speech tokenizer is on {st_current}, moving to XPU ...")
                    st_model.to(xpu_dev)
                    print("✅ [Qwen3-TTS-XPU] Speech tokenizer moved to XPU")
                else:
                    print(f"✅ [Qwen3-TTS-XPU] Speech tokenizer confirmed on XPU ({st_current})")
            except StopIteration:
                print("⚠️ [Qwen3-TTS-XPU] Speech tokenizer has no parameters to check")
        else:
            print("⚠️ [Qwen3-TTS-XPU] Speech tokenizer inner model not found (may be loaded lazily)")
        # always sync the Python-level device attribute
        st.device = xpu_dev




def _xpu_cleanup():
    """Sync and flush XPU allocator cache between generations.

    torch.compile (inductor) accumulates autotuning / kernel-compilation
    buffers in VRAM between runs.  Calling this before/after each generation
    returns those freed-but-unclaimed pages to Level Zero, preventing the
    UR_RESULT_ERROR_OUT_OF_RESOURCES OOM on the second run.

    This does NOT touch model weights, compiled kernel code, or FX-graph
    caches — those all live in CPU-side Python structures and are unaffected
    by empty_cache().  The second generation is therefore still a hot-start.
    """
    import gc
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        except Exception as e:
            print(f"⚠️ [Qwen3-TTS-XPU] XPU cache cleanup warning: {e}")
    gc.collect()


def _try_set_cc_for_windows():
    """On Windows, help Triton find a C compiler if CC is not already set."""
    if sys.platform != "win32" or "CC" in os.environ:
        return
    import shutil
    for compiler in ("cl", "icx", "icpx", "gcc", "clang"):
        if shutil.which(compiler):
            os.environ["CC"] = compiler
            print(f"🔧 [Qwen3-TTS-XPU] Auto-set CC={compiler} for Triton compilation")
            return
    print("⚠️ [Qwen3-TTS-XPU] No C compiler found in PATH (cl/icx/gcc/clang). "
          "inductor backend may fail. Try loading Intel oneAPI setvars.bat before starting ComfyUI, "
          "then set CC=icx, or use aot_eager backend instead.")


def _apply_torch_compile(model, dynamic: bool = True, backend: str = "aot_eager"):
    """Apply torch.compile to the inner transformer bodies for faster inference."""
    try:
        if backend == "inductor":
            _try_set_cc_for_windows()
        compile_kwargs = {"dynamic": dynamic, "fullgraph": False, "backend": backend}
        talker = getattr(getattr(model, "model", None), "talker", None)
        if talker is not None:
            inner = getattr(talker, "model", None)
            if inner is not None:
                talker.model = torch.compile(inner, **compile_kwargs)
                print(f"✅ [Qwen3-TTS-XPU] torch.compile applied to talker.model (backend={backend}, dynamic={dynamic})")
            cp = getattr(talker, "code_predictor", None)
            if cp is not None:
                cp_inner = getattr(cp, "model", None)
                if cp_inner is not None:
                    cp.model = torch.compile(cp_inner, **compile_kwargs)
                    print(f"✅ [Qwen3-TTS-XPU] torch.compile applied to code_predictor.model (backend={backend}, dynamic={dynamic})")
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS-XPU] torch.compile failed (inference will continue without it): {e}")


def _apply_int8_weight_quantization(model_wrapper):
    """
    Weight-only INT8 quantization for Linear layers in the talker.
    Replaces nn.Linear with a quantized version: weights stored as int8,
    dequantized to input dtype on each forward pass.
    Pure PyTorch ops — XPU compatible, no extra libraries needed.
    Expected benefit: ~50% memory bandwidth reduction per forward pass.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class _QuantizedLinear(nn.Module):
        def __init__(self, w_int8, scale, bias):
            super().__init__()
            self.register_buffer("weight_int8", w_int8)  # (out, in) int8
            self.register_buffer("scale", scale)          # (out, 1) same dtype as original weight
            self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        def forward(self, x):
            w = self.weight_int8.to(x.dtype) * self.scale
            return F.linear(x, w, self.bias)

    def _quantize_children(parent, min_numel=1024):
        count = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child.weight is not None and child.weight.numel() >= min_numel:
                try:
                    w = child.weight.data.float()
                    scale = (w.abs().max(dim=1, keepdim=True)[0] / 127.0).to(child.weight.dtype)
                    w_int8 = (w / scale.float()).round().clamp(-128, 127).to(torch.int8)
                    q_lin = _QuantizedLinear(w_int8, scale, child.bias)
                    setattr(parent, name, q_lin)
                    count += 1
                except Exception as e:
                    print(f"⚠️ [Qwen3-TTS-XPU] INT8: skipping layer '{name}': {e}")
            else:
                count += _quantize_children(child, min_numel)
        return count

    talker = getattr(getattr(model_wrapper, "model", None), "talker", None)
    if talker is None:
        print("⚠️ [Qwen3-TTS-XPU] INT8: talker not found, skipping quantization")
        return 0

    count = _quantize_children(talker)
    print(f"✅ [Qwen3-TTS-XPU] INT8 weight quantization: {count} Linear layers → int8 (~50% memory bandwidth saving)")
    return count


def apply_qwen3_patches(model):
    """Apply stability and compatibility patches to the model instance."""
    if model is None:
        return

    def _safe_normalize(self, audios):
        if isinstance(audios, list):
            items = audios
        elif isinstance(audios, tuple) and len(audios) == 2 and isinstance(audios[0], np.ndarray):
            items = [audios]
        else:
            items = [audios]

        out = []
        for a in items:
            if a is None:
                continue

            if isinstance(a, str):
                wav, sr = self._load_audio_to_np(a)
                out.append([wav.astype(np.float32), int(sr)])
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, list) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append([a[0].astype(np.float32), int(a[1])])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                print(f"⚠️ [Qwen3-TTS-XPU] Unknown audio input type: {type(a)}")
                continue

        # Ensure mono
        for i in range(len(out)):
            wav, sr = out[i][0], out[i][1]
            if wav.ndim > 1:
                out[i][0] = np.mean(wav, axis=-1).astype(np.float32)

        return out

    try:
        model._normalize_audio_inputs = types.MethodType(_safe_normalize, model)
    except Exception as e:
        print(f"⚠️ [Qwen3-TTS-XPU] Failed to apply audio normalization patch: {e}")


def download_model_if_needed(model_id: str, target_dir: str) -> str | None:
    """Offline-first: only use local files.

    If QWEN_TTS_ALLOW_DOWNLOAD=1 is set, we'll allow HuggingFace downloads as a fallback.
    Otherwise we *never* download and we return None when the folder doesn't exist.
    """
    # If user explicitly allows downloads, keep the old behavior (snapshot_download).
    allow_dl = os.environ.get("QWEN_TTS_ALLOW_DOWNLOAD", "").strip() in ("1", "true", "True", "yes", "YES")
    if os.path.isdir(target_dir):
        return target_dir

    if not allow_dl:
        # Offline mode: don't create folders, don't download anything.
        print(f"🛑 [Qwen3-TTS-XPU] Offline mode: refusing to download '{model_id}'. Missing folder: {target_dir}")
        return None

    # ---- Online fallback (only when explicitly enabled) ----
    print(f"⬇️ [Qwen3-TTS-XPU] Downloading '{model_id}' to: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        return target_dir
    except Exception as e:
        print(f"❌ [Qwen3-TTS-XPU] Download failed for '{model_id}': {e}")
        return None


def load_qwen_model(
    model_type: str,
    model_choice: str,
    device: str,
    precision: str,
    attention: str = "auto",
    unload_after: bool = False,
    previous_attention: str = None,
    custom_model_path: Optional[str] = None,
    torch_compile: bool = False,
    quantize_int8: bool = False,
):
    """Shared model loading logic with caching and local path priority - XPU + CPU optimized."""
    global _MODEL_CACHE

    if previous_attention is not None and previous_attention != attention:
        print(f"🔄 [Qwen3-TTS-XPU] Attention changed from '{previous_attention}' to '{attention}', clearing cache...")
        unload_cached_model()

    attn_impl = get_attention_implementation(attention)


    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    # CPU optimization
    if device == "cpu":
        print(f"🔧 [Qwen3-TTS-XPU] CPU mode detected - Optimizing for {CPU_CORES} cores...")
        optimize_cpu_performance()
        dtype = torch.float32
        print("🔧 [Qwen3-TTS-XPU] CPU precision set to float32 for compatibility")
    elif device == "mps" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    elif device == "xpu" and precision == "bf16":
        dtype = torch.bfloat16
    elif device == "xpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    # VoiceDesign restriction
    if model_type == "VoiceDesign" and model_choice == "0.6B":
        raise RuntimeError("❌ VoiceDesign only supports 1.7B models!")

    # Cache key includes attention implementation, custom model path, and compile settings
    cache_key = (model_type, model_choice, device, precision, attn_impl, custom_model_path, torch_compile, quantize_int8)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Clear old cache
    if _MODEL_CACHE:
        _MODEL_CACHE.clear()

    # Search for models
    base_paths: List[str] = []
    try:
        comfy_root = os.path.dirname(os.path.abspath(folder_paths.__file__))
        primary = os.path.join(comfy_root, "models", "TTS", "Qwen3-TTS")
        if os.path.exists(primary):
            base_paths.append(primary)
    except Exception:
        pass
    try:
        for p in folder_paths.get_folder_paths("TTS") or []:
            qwen_subdir = os.path.join(p, "Qwen3-TTS")
            if os.path.exists(qwen_subdir) and qwen_subdir not in base_paths:
                base_paths.append(qwen_subdir)
            elif p not in base_paths:
                base_paths.append(p)
    except Exception:
        pass

    HF_MODEL_MAP = {
        ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    }

    final_source = HF_MODEL_MAP.get((model_type, model_choice)) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    found_local = None

    if custom_model_path and isinstance(custom_model_path, str) and custom_model_path.strip():
        if os.path.exists(custom_model_path) and os.path.isdir(custom_model_path):
            print(f"🔧 [Qwen3-TTS-XPU] Using custom model path: {custom_model_path}")
            found_local = custom_model_path
        else:
            print(f"⚠️ [Qwen3-TTS-XPU] Custom model path not found or invalid: {custom_model_path}")

    if not found_local:
        for base in base_paths:
            try:
                if not os.path.isdir(base):
                    continue
                subdirs = os.listdir(base)
                for d in subdirs:
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand):
                        if model_choice in d and model_type.lower() in d.lower():
                            found_local = cand
                            break
                if found_local:
                    break
            except Exception:
                pass

    if found_local:
        final_source = found_local
        print(f"✅ [Qwen3-TTS-XPU] Loading local model: {os.path.basename(final_source)}")
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfy_models_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models")
        qwen_root = os.path.join(comfy_models_path, "TTS", "Qwen3-TTS")
        downloaded_path = download_model_if_needed(final_source, qwen_root)
        if downloaded_path:
            final_source = downloaded_path
            print(f"✅ [Qwen3-TTS-XPU] Loading downloaded model: {os.path.basename(final_source)}")
        else:
            print(f"🌐 [Qwen3-TTS-XPU] Loading remote model: {final_source}")

    if Qwen3TTSModel is None:
        raise RuntimeError(
            "❌ [Qwen3-TTS-XPU] Model class is not loaded because the 'qwen_tts' package failed to import."
        )

    # Map attention implementation
    attn_param = None
    use_sage_attn = False

    if attn_impl == "flash_attn":
        attn_param = "flash_attention_2"
    elif attn_impl == "sage_attn":
        use_sage_attn = True
    elif attn_impl == "sdpa":
        attn_param = "sdpa"
    elif attn_impl == "eager":
        attn_param = "eager"

    # Handle sage_attn
    if use_sage_attn:
        try:
            from sageattention import sageattn

            print("🔧 [Qwen3-TTS-XPU] Loading model with sage_attn (sageattention)")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, local_files_only=True)

            patched_count = 0
            for name, module in model.model.named_modules():
                if hasattr(module, "forward") and (
                    "Attention" in type(module).__name__ or "attn" in name.lower()
                ):
                    try:
                        original_forward = module.forward

                        def make_sage_forward(orig_forward, mod):
                            def sage_forward(*args, **kwargs):
                                if len(args) >= 3:
                                    q, k, v = args[0], args[1], args[2]
                                else:
                                    return orig_forward(*args, **kwargs)

                                attn_mask = kwargs.get("attention_mask", None)
                                out = sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                                return out

                            return sage_forward

                        module.forward = make_sage_forward(original_forward, module)
                        patched_count += 1
                    except Exception:
                        pass

            print(f"🔧 [Qwen3-TTS-XPU] Patched {patched_count} attention modules with sage_attn")
        except (ImportError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS-XPU] Failed with sage_attn, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, local_files_only=True)
    else:
        try:
            print(f"🔧 [Qwen3-TTS-XPU] Loading model with attention: {attn_impl}")
            if attn_param:
                model = Qwen3TTSModel.from_pretrained(
                    final_source, device_map=device, dtype=dtype, attn_implementation=attn_param, local_files_only=True)
            else:
                model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, local_files_only=True)
        except (ImportError, ValueError, Exception) as e:
            print(f"⚠️ [Qwen3-TTS-XPU] Failed with {attn_impl}, falling back to default attention: {e}")
            model = Qwen3TTSModel.from_pretrained(final_source, device_map=device, dtype=dtype, local_files_only=True)

    apply_qwen3_patches(model)

    # XPU: verify all components are on the Intel GPU (device_map="xpu" may
    # silently fall back to CPU on older accelerate versions)
    if device == "xpu" and model is not None:
        _ensure_on_xpu(model)

    if quantize_int8 and model is not None:
        print("🔧 [Qwen3-TTS-XPU] Applying INT8 weight quantization (this may take 10-30s on first load)...")
        _apply_int8_weight_quantization(model)

    if torch_compile and model is not None:
        print(f"🔧 [Qwen3-TTS-XPU] Applying torch.compile (backend=inductor, dynamic=True) — first run will be slow...")
        _apply_torch_compile(model, dynamic=True, backend="inductor")

    _MODEL_CACHE[cache_key] = model

    if unload_after:
        def unload_callback():
            unload_cached_model()
        model._unload_callback = unload_callback
    else:
        model._unload_callback = None

    return model


class VoiceDesignNode:
    """VoiceDesign Node: Generate custom voices based on text descriptions."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize", "tooltip": "要合成的文字内容"}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (required for VoiceDesign)", "tooltip": "声音风格描述，例如：温柔的女声、沉稳的男声、活泼的童声"}),
                "model_choice": (["1.7B"], {"default": "1.7B", "tooltip": "模型大小，VoiceDesign 仅支持 1.7B"}),
                "device": (["auto", "cuda", "mps", "xpu", "cpu"], {"default": "auto", "tooltip": "推理设备，auto 自动选择可用 GPU（cuda/xpu），无 GPU 则用 CPU"}),
                "precision": (["bf16", "fp32"], {"default": "bf16", "tooltip": "计算精度，bf16 速度更快显存更少（推荐），fp32 精度略高但慢"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto", "tooltip": "输出语言，Auto 自动检测输入文字语言"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子，固定后输出稳定可复现；设为 randomize 每次不同"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256, "tooltip": "最大生成 token 数，控制音频最长时长，过短会截断长文本"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus 采样阈值，越小越保守稳定，越大越多样"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "每步候选 token 数，越小越保守，0 表示不限制"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "采样温度，越低越稳定（接近贪心），越高越随机有创意"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "重复惩罚系数，防止音频内容重复，建议保持默认 1.05"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "注意力实现方式，auto 自动选择，sdpa 通常最快"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "生成后立即卸载模型释放显存，下次使用需重新加载（约 10-20s）"}),
                "torch_compile": ("BOOLEAN", {"default": False, "display_name": "🔷 torch_compile", "tooltip": "每次启动后首次编译需约2.5分钟，之后同一会话内速度提升约2倍"}),
                "quantize_int8": ("BOOLEAN", {"default": False, "display_name": "🔷 quantize_int8", "tooltip": "INT8 权重量化，减少约 1.5GB 显存占用，速度略有下降，不建议与 torch_compile 同时开启"}),
                "full_text_prefill": ("BOOLEAN", {"default": True, "display_name": "🔷 full_text_prefill", "tooltip": "True（推荐）：全文一次性输入，生成质量更好。False：逐步输入模拟流式，速度无差异"}),
                "cache_clean": ("BOOLEAN", {"default": True, "display_name": "🔷 cache_clean", "tooltip": "每次生成前后清理 XPU allocator 缓存，防止 torch_compile 残留显存累积导致第二次生成 OOM。不影响编译缓存和模型权重，第二次仍为热启动"}),
                "trailing_pad": ("STRING", {"default": "…………………", "tooltip": "追加在文字末尾的填充内容，防止模型在朗读最后几个字之前提前生成 EOS 导致截断。默认 …… 可见且有效，留空则不追加"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "VoiceDesign: Generate custom voices from descriptions."

    def generate(
        self,
        text: str,
        instruct: str,
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        seed: int = 0,
        max_new_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.05,
        attention: str = "auto",
        unload_model_after_generate: bool = False,
        torch_compile: bool = False,
        quantize_int8: bool = False,
        full_text_prefill: bool = True,
        cache_clean: bool = True,
        trailing_pad: str = "…………………",
    ) -> Tuple[Dict[str, Any]]:
        if not text or not instruct:
            raise RuntimeError("Text and instruction description are required")

        pbar = ProgressBar(3)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "VoiceDesign":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.update_absolute(1, 3, None)

        model = load_qwen_model(
            "VoiceDesign",
            model_choice,
            device,
            precision,
            attention,
            unload_model_after_generate,
            previous_attention,
            torch_compile=torch_compile,
            quantize_int8=quantize_int8,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.update_absolute(2, 3, None)

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        if trailing_pad:
            text = text + trailing_pad

        if cache_clean:
            _xpu_cleanup()

        wavs, sr = model.generate_voice_design(
            text=text,
            language=mapped_lang,
            instruct=instruct,
            non_streaming_mode=full_text_prefill,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        if cache_clean:
            _xpu_cleanup()

        pbar.update_absolute(3, 3, None)

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class VoiceCloneNode:
    """VoiceClone (Base) Node: Create clones from reference audio and synthesize target text."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "target_text": ("STRING", {"multiline": True, "default": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B", "tooltip": "模型大小，0.6B 速度更快，1.7B 质量更好"}),
                "device": (["auto", "cuda", "mps", "xpu", "cpu"], {"default": "auto", "tooltip": "推理设备，auto 自动选择可用 GPU（cuda/xpu），无 GPU 则用 CPU"}),
                "precision": (["bf16", "fp32"], {"default": "bf16", "tooltip": "计算精度，bf16 速度更快显存更少（推荐），fp32 精度略高但慢"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto", "tooltip": "输出语言，Auto 自动检测输入文字语言"}),
            },
            "optional": {
                "ref_audio": ("AUDIO", {"tooltip": "参考音频，用于提取说话人声音特征进行克隆"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)", "tooltip": "参考音频对应的文字（可选），提供后可提升克隆精度"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT", {"tooltip": "预先提取的声音克隆提示，可复用，无需每次提供参考音频"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子，固定后输出稳定可复现；设为 randomize 每次不同"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256, "tooltip": "最大生成 token 数，控制音频最长时长，过短会截断长文本"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus 采样阈值，越小越保守稳定，越大越多样"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "每步候选 token 数，越小越保守，0 表示不限制"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "采样温度，越低越稳定（接近贪心），越高越随机有创意"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "重复惩罚系数，防止音频内容重复，建议保持默认 1.05"}),
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "仅使用 x-vector 声纹特征（不含内容信息），适合只需要音色而不关注内容的场景"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "注意力实现方式，auto 自动选择，sdpa 通常最快"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "生成后立即卸载模型释放显存，下次使用需重新加载（约 10-20s）"}),
                "custom_model_path": ("STRING", {"default": "", "placeholder": "Absolute path to local fine-tuned model", "tooltip": "本地微调模型的绝对路径，留空则使用默认模型"}),
                "torch_compile": ("BOOLEAN", {"default": False, "display_name": "🔷 torch_compile", "tooltip": "每次启动后首次编译需约2.5分钟，之后同一会话内速度提升约2倍"}),
                "quantize_int8": ("BOOLEAN", {"default": False, "display_name": "🔷 quantize_int8", "tooltip": "INT8 权重量化，减少约 1.5GB 显存占用，速度略有下降，不建议与 torch_compile 同时开启"}),
                "full_text_prefill": ("BOOLEAN", {"default": True, "display_name": "🔷 full_text_prefill", "tooltip": "True（推荐）：全文一次性输入，生成质量更好。False：逐步输入模拟流式，速度无差异"}),
                "cache_clean": ("BOOLEAN", {"default": True, "display_name": "🔷 cache_clean", "tooltip": "每次生成前后清理 XPU allocator 缓存，防止 torch_compile 残留显存累积导致第二次生成 OOM。不影响编译缓存和模型权重，第二次仍为热启动"}),
                "trailing_pad": ("STRING", {"default": "…………………", "tooltip": "追加在文字末尾的填充内容，防止模型在朗读最后几个字之前提前生成 EOS 导致截断。默认 …… 可见且有效，留空则不追加"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "VoiceClone: Clone voice from reference audio."

    def _audio_tensor_to_tuple(self, audio_tensor: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        waveform = None
        sr = None

        try:
            if isinstance(audio_tensor, dict):
                if "waveform" in audio_tensor:
                    waveform = audio_tensor.get("waveform")
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr") or audio_tensor.get("sampling_rate")
                elif "data" in audio_tensor and "sampling_rate" in audio_tensor:
                    waveform = np.asarray(audio_tensor.get("data"))
                    sr = audio_tensor.get("sampling_rate")
            elif isinstance(audio_tensor, tuple) and len(audio_tensor) == 2:
                a0, a1 = audio_tensor
                if isinstance(a0, (int, float)) and isinstance(a1, (list, np.ndarray, torch.Tensor)):
                    sr = int(a0)
                    waveform = np.asarray(a1)
                elif isinstance(a1, (int, float)) and isinstance(a0, (list, np.ndarray, torch.Tensor)):
                    sr = int(a1)
                    waveform = np.asarray(a0)
        except Exception:
            pass

        if isinstance(waveform, torch.Tensor):
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
            if waveform.dim() > 1:
                waveform = torch.mean(waveform, dim=0)
            waveform = waveform.cpu().numpy()

        if isinstance(waveform, np.ndarray):
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform)
            if waveform.ndim > 1:
                if waveform.shape[0] < waveform.shape[1]:
                    waveform = np.mean(waveform, axis=0)
                else:
                    waveform = np.mean(waveform, axis=1)
            waveform = waveform.astype(np.float32)

        if waveform is not None and waveform.ndim > 1:
            waveform = waveform.flatten()

        if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0:
            raise RuntimeError("Failed to parse reference audio waveform")

        min_samples = 1024
        if waveform.size < min_samples:
            pad_amount = min_samples - waveform.size
            waveform = np.concatenate([waveform, np.zeros(pad_amount, dtype=np.float32)])

        return (waveform, int(sr))

    def generate(
        self,
        target_text: str,
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        ref_audio: Optional[Dict[str, Any]] = None,
        ref_text: str = "",
        voice_clone_prompt: Optional[Any] = None,
        seed: int = 0,
        max_new_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.05,
        x_vector_only: bool = False,
        attention: str = "auto",
        unload_model_after_generate: bool = False,
        custom_model_path: str = "",
        torch_compile: bool = False,
        quantize_int8: bool = False,
        full_text_prefill: bool = True,
        cache_clean: bool = True,
        trailing_pad: str = "…………………",
    ) -> Tuple[Dict[str, Any]]:
        if ref_audio is None and voice_clone_prompt is None:
            raise RuntimeError("Either reference audio or voice clone prompt is required")

        pbar = ProgressBar(3)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.update_absolute(1, 3, None)

        model = load_qwen_model(
            "Base",
            model_choice,
            device,
            precision,
            attention,
            unload_model_after_generate,
            previous_attention,
            custom_model_path,
            torch_compile=torch_compile,
            quantize_int8=quantize_int8,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.update_absolute(2, 3, None)

        audio_tuple = None
        if ref_audio is not None:
            audio_tuple = self._audio_tensor_to_tuple(ref_audio)

        try:
            mapped_lang = LANGUAGE_MAP.get(language, "auto")
            voice_clone_prompt_param = None
            ref_audio_param = None

            if trailing_pad:
                target_text = target_text + trailing_pad

            if voice_clone_prompt is not None:
                voice_clone_prompt_param = voice_clone_prompt
            elif ref_audio is not None:
                ref_audio_param = audio_tuple
            else:
                raise RuntimeError("Either ref_audio or voice_clone_prompt must be provided")

            if cache_clean:
                _xpu_cleanup()

            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=ref_audio_param,
                ref_text=ref_text if ref_text and ref_text.strip() else None,
                voice_clone_prompt=voice_clone_prompt_param,
                x_vector_only_mode=x_vector_only,
                non_streaming_mode=full_text_prefill,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            if cache_clean:
                _xpu_cleanup()
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

        pbar.update_absolute(3, 3, None)

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class CustomVoiceNode:
    """CustomVoice (TTS) Node: Generate text-to-speech using preset speakers."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello world", "placeholder": "Enter text to synthesize", "tooltip": "要合成的文字内容"}),
                "speaker": (
                    ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"],
                    {"default": "Ryan", "tooltip": "预设说话人，每个音色风格不同"},
                ),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B", "tooltip": "模型大小，0.6B 速度更快，1.7B 质量更好"}),
                "device": (["auto", "cuda", "mps", "xpu", "cpu"], {"default": "auto", "tooltip": "推理设备，auto 自动选择可用 GPU（cuda/xpu），无 GPU 则用 CPU"}),
                "precision": (["bf16", "fp32"], {"default": "bf16", "tooltip": "计算精度，bf16 速度更快显存更少（推荐），fp32 精度略高但慢"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto", "tooltip": "输出语言，Auto 自动检测输入文字语言"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "随机种子，固定后输出稳定可复现；设为 randomize 每次不同"}),
                "instruct": ("STRING", {"multiline": True, "default": "", "placeholder": "Style instruction (optional)", "tooltip": "额外风格指令（可选），如：慢速朗读、带情绪"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256, "tooltip": "最大生成 token 数，控制音频最长时长，过短会截断长文本"}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Nucleus 采样阈值，越小越保守稳定，越大越多样"}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "每步候选 token 数，越小越保守，0 表示不限制"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "采样温度，越低越稳定（接近贪心），越高越随机有创意"}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05, "tooltip": "重复惩罚系数，防止音频内容重复，建议保持默认 1.05"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto", "tooltip": "注意力实现方式，auto 自动选择，sdpa 通常最快"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False, "tooltip": "生成后立即卸载模型释放显存，下次使用需重新加载（约 10-20s）"}),
                "torch_compile": ("BOOLEAN", {"default": False, "display_name": "🔷 torch_compile", "tooltip": "每次启动后首次编译需约2.5分钟，之后同一会话内速度提升约2倍"}),
                "quantize_int8": ("BOOLEAN", {"default": False, "display_name": "🔷 quantize_int8", "tooltip": "INT8 权重量化，减少约 1.5GB 显存占用，速度略有下降，不建议与 torch_compile 同时开启"}),
                "full_text_prefill": ("BOOLEAN", {"default": True, "display_name": "🔷 full_text_prefill", "tooltip": "True（推荐）：全文一次性输入，生成质量更好。False：逐步输入模拟流式，速度无差异"}),
                "cache_clean": ("BOOLEAN", {"default": True, "display_name": "🔷 cache_clean", "tooltip": "每次生成前后清理 XPU allocator 缓存，防止 torch_compile 残留显存累积导致第二次生成 OOM。不影响编译缓存和模型权重，第二次仍为热启动"}),
                "trailing_pad": ("STRING", {"default": "…………………", "tooltip": "追加在文字末尾的填充内容，防止模型在朗读最后几个字之前提前生成 EOS 导致截断。默认 …… 可见且有效，留空则不追加"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "CustomVoice: Generate speech using preset speakers."

    def generate(
        self,
        text: str,
        speaker: str,
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        seed: int = 0,
        instruct: str = "",
        max_new_tokens: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.05,
        attention: str = "auto",
        unload_model_after_generate: bool = False,
        torch_compile: bool = False,
        quantize_int8: bool = False,
        full_text_prefill: bool = True,
        cache_clean: bool = True,
        trailing_pad: str = "…………………",
    ) -> Tuple[Dict[str, Any]]:
        if not text or not speaker:
            raise RuntimeError("Text and speaker are required")

        pbar = ProgressBar(3)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "CustomVoice":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.update_absolute(1, 3, None)

        model = load_qwen_model(
            "CustomVoice",
            model_choice,
            device,
            precision,
            attention,
            unload_model_after_generate,
            previous_attention,
            torch_compile=torch_compile,
            quantize_int8=quantize_int8,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        pbar.update_absolute(2, 3, None)

        mapped_lang = LANGUAGE_MAP.get(language, "auto")

        if trailing_pad:
            text = text + trailing_pad

        if cache_clean:
            _xpu_cleanup()

        wavs, sr = model.generate_custom_voice(
            text=text,
            language=mapped_lang,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct if instruct and instruct.strip() else None,
            non_streaming_mode=full_text_prefill,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        if cache_clean:
            _xpu_cleanup()

        pbar.update_absolute(3, 3, None)

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
                model._unload_callback()

            return (audio_data,)

        raise RuntimeError("Invalid audio data generated")


class VoiceClonePromptNode:
    """VoiceClonePrompt Node: Extract voice features from reference audio for reuse."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "ref_audio": ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (highly recommended for better quality)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device": (["auto", "cuda", "mps", "xpu", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False, "tooltip": "If True, only speaker embedding is extracted (ref_text not needed)"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "VoiceClonePrompt: Extract and cache voice features for reuse in VoiceClone node."

    def create_prompt(
        self,
        ref_audio: Dict[str, Any],
        ref_text: str,
        model_choice: str,
        device: str,
        precision: str,
        attention: str,
        x_vector_only: bool = False,
        unload_model_after_generate: bool = False,
    ) -> Tuple[Any]:
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")

        pbar = ProgressBar(3)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.update_absolute(1, 3, None)

        model = load_qwen_model(
            "Base",
            model_choice,
            device,
            precision,
            attention,
            unload_model_after_generate,
            previous_attention,
        )

        pbar.update_absolute(2, 3, None)

        vcn = VoiceCloneNode()
        audio_tuple = vcn._audio_tensor_to_tuple(ref_audio)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=audio_tuple,
            ref_text=ref_text if ref_text and ref_text.strip() else None,
            x_vector_only_mode=x_vector_only,
        )

        pbar.update_absolute(3, 3, None)

        if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
            model._unload_callback()

        return (prompt_items,)


class RoleBankNode:
    """RoleBank Node: Manage a collection of voice prompts mapped to names. Supports up to 8 roles per node."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        inputs: Dict[str, Dict[str, Any]] = {
            "required": {},
            "optional": {},
        }

        for i in range(1, 9):
            inputs["optional"][f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            inputs["optional"][f"prompt_{i}"] = ("VOICE_CLONE_PROMPT",)

        return inputs

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "create_bank"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "RoleBank: Collect multiple voice prompts into a named registry for dialogue generation."

    def create_bank(self, **kwargs) -> Tuple[Dict[str, Any]]:
        bank: Dict[str, Any] = {}

        for i in range(1, 9):
            name = kwargs.get(f"role_name_{i}", "").strip()
            prompt = kwargs.get(f"prompt_{i}")

            if name and prompt is not None:
                bank[name] = prompt

        return (bank,)


class DialogueInferenceNode:
    """DialogueInference Node: Generate multi-role continuous dialogue from a script."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.", "placeholder": "Format: RoleName: Text"}),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "mps", "xpu", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "bf16"}),
                "language": (DEMO_LANGUAGES, {"default": "Auto"}),
                "pause_linebreak": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration between dialogue lines"}),
                "period_pause": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after periods (.)"}),
                "comma_pause": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after commas (,)"})
                ,
                "question_pause": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after question marks (?)"}),
                "hyphen_pause": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Silence duration after hyphens (-)"}),
                "merge_outputs": ("BOOLEAN", {"default": True, "tooltip": "Merge all dialogue segments into a single long audio"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Number of lines to process in parallel"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_new_tokens_per_line": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
                "attention": (ATTENTION_OPTIONS, {"default": "auto"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_dialogue"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "DialogueInference: Execute a script with multiple roles and generate continuous speech."

    def generate_dialogue(
        self,
        script: str,
        role_bank: Dict[str, Any],
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        pause_linebreak: float,
        period_pause: float,
        comma_pause: float,
        question_pause: float,
        hyphen_pause: float,
        merge_outputs: bool,
        batch_size: int,
        seed: int = 0,
        max_new_tokens_per_line: int = 2048,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 1.0,
        repetition_penalty: float = 1.05,
        attention: str = "auto",
        unload_model_after_generate: bool = False,
    ) -> Tuple[Dict[str, Any]]:
        if not script or not role_bank:
            raise RuntimeError("Script and Role Bank are required")

        pbar = ProgressBar(3)

        global _MODEL_CACHE
        previous_attention = None
        for key in _MODEL_CACHE:
            if key[0] == "Base":
                previous_attention = key[4] if len(key) > 4 else None
                break

        pbar.update_absolute(1, 3, None)

        model = load_qwen_model(
            "Base",
            model_choice,
            device,
            precision,
            attention,
            unload_model_after_generate,
            previous_attention,
        )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
        np.random.seed(seed % (2**32))

        lines = script.strip().split("\n")
        texts_to_gen: List[str] = []
        prompts_to_gen: List[Any] = []
        langs_to_gen: List[str] = []
        pauses_to_gen: List[float] = []

        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        pause_pattern = r"\[break=([\d\.]+)\]"

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or (":" not in line and "：" not in line):
                continue

            pos_en = line.find(":")
            pos_cn = line.find("：")

            if pos_en == -1 and pos_cn == -1:
                continue

            if pos_en != -1 and (pos_cn == -1 or pos_en < pos_cn):
                role_name, text = line.split(":", 1)
            else:
                role_name, text = line.split("：", 1)

            role_name = role_name.strip()
            text = text.strip()

            if role_name not in role_bank:
                continue

            role_data = role_bank[role_name]
            current_prompt = role_data[0] if isinstance(role_data, list) else role_data

            if period_pause > 0:
                text = re.sub(r"\.(?!\d)", f". [break={period_pause}]", text)
            if comma_pause > 0:
                text = re.sub(r",(?!\d)", f", [break={comma_pause}]", text)
            if question_pause > 0:
                text = re.sub(r"\?(?!\d)", f"? [break={question_pause}]", text)
            if hyphen_pause > 0:
                text = re.sub(r"-(?!\d)", f"- [break={hyphen_pause}]", text)

            parts = re.split(pause_pattern, text)

            for i in range(0, len(parts), 2):
                segment_text = parts[i].strip()
                if not segment_text:
                    continue

                current_segment_pause = 0.0
                if i + 1 < len(parts):
                    try:
                        current_segment_pause = float(parts[i + 1])
                    except ValueError:
                        pass

                texts_to_gen.append(segment_text)
                prompts_to_gen.append(current_prompt)
                langs_to_gen.append(mapped_lang)
                pauses_to_gen.append(current_segment_pause)

        if pauses_to_gen:
            pauses_to_gen[-1] += pause_linebreak

        if not texts_to_gen:
            raise RuntimeError("No valid dialogue lines found matching Role Bank.")

        num_lines = len(texts_to_gen)
        num_chunks = (num_lines + batch_size - 1) // batch_size
        total_stages = num_chunks + 1

        pbar = ProgressBar(total_stages)
        pbar.update_absolute(1, total_stages, None)

        try:
            results: List[torch.Tensor] = []
            sr = 24000

            for i in range(0, num_lines, batch_size):
                chunk_texts = texts_to_gen[i : i + batch_size]
                chunk_prompts = prompts_to_gen[i : i + batch_size]
                chunk_langs = langs_to_gen[i : i + batch_size]
                chunk_pauses = pauses_to_gen[i : i + batch_size]

                current_chunk = i // batch_size + 1
                print(f"🎬 [Qwen3-TTS-XPU] Running batched inference for chunk {current_chunk} of {num_chunks}...")

                wavs_list, sr = model.generate_voice_clone(
                    text=chunk_texts,
                    language=chunk_langs,
                    voice_clone_prompt=chunk_prompts,
                    max_new_tokens=max_new_tokens_per_line,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )

                for j, wav in enumerate(wavs_list):
                    waveform = torch.from_numpy(wav).float()
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0).unsqueeze(0)
                    elif waveform.ndim == 2:
                        waveform = waveform.unsqueeze(0)

                    if waveform.shape[1] > 1:
                        waveform = torch.mean(waveform, dim=1, keepdim=True)

                    results.append(waveform)

                    this_pause = chunk_pauses[j]
                    if this_pause > 0:
                        silence_len = int(this_pause * sr)
                        silence = torch.zeros((1, 1, silence_len))
                        results.append(silence)

                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    torch.xpu.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                pbar.update_absolute(current_chunk + 1, total_stages, None)

        except Exception as e:
            raise RuntimeError(f"Dialogue generation failed during chunked inference: {e}")

        if not results:
            raise RuntimeError("No dialogue lines were successfully generated.")

        pbar.update_absolute(total_stages, total_stages, None)

        if merge_outputs:
            merged_waveform = torch.cat(results, dim=-1)
            audio_data = {"waveform": merged_waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
                model._unload_callback()

            return (audio_data,)
        else:
            max_len = max(w.shape[-1] for w in results)
            padded_results: List[torch.Tensor] = []

            for w in results:
                curr_len = w.shape[-1]
                if curr_len < max_len:
                    padding = torch.zeros((w.shape[0], w.shape[1], max_len - curr_len))
                    w = torch.cat([w, padding], dim=-1)
                padded_results.append(w)

            batched_waveform = torch.cat(padded_results, dim=0)
            audio_data = {"waveform": batched_waveform, "sample_rate": sr}

            if unload_model_after_generate and hasattr(model, "_unload_callback") and model._unload_callback:
                model._unload_callback()

            return (audio_data,)


class SaveVoiceNode:
    """SaveVoice Node: Persist extracted voice features to a file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",),
                "filename": ("STRING", {"default": "my_custom_voice"}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "ref_text": ("STRING", {"multiline": True, "default": "", "placeholder": "Reference audio text (optional)"}),
            },
        }

    RETURN_TYPES: Tuple = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "SaveVoice: Save voice clone prompt features to disk for later use."

    def save(self, voice_clone_prompt, filename, audio=None, ref_text: str = ""):
        import soundfile as sf
        import json

        if not filename.endswith(".qvp"):
            filename_qvp = filename + ".qvp"
            filename_wav = filename + ".wav"
            filename_json = filename + ".json"
        else:
            filename_qvp = filename
            filename_wav = filename.replace(".qvp", ".wav")
            filename_json = filename.replace(".qvp", ".json")

        output_dir = os.path.join(folder_paths.models_dir, "qwen-tts", "voices")
        os.makedirs(output_dir, exist_ok=True)

        path_qvp = os.path.join(output_dir, filename_qvp)
        try:
            torch.save(voice_clone_prompt, path_qvp)
            print(f"✅ [Qwen3-TTS-XPU] Voice prompt features saved to: {path_qvp}")
        except Exception as e:
            print(f"❌ [Qwen3-TTS-XPU] Failed to save voice prompt: {e}")

        metadata = {
            "ref_text": ref_text.strip() if ref_text else "",
            "source": "SaveVoiceNode",
            "version": "1.0",
        }

        path_json = os.path.join(output_dir, filename_json)
        try:
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)
            print(f"✅ [Qwen3-TTS-XPU] Voice metadata saved to: {path_json}")
        except Exception as e:
            print(f"❌ [Qwen3-TTS-XPU] Failed to save metadata JSON: {e}")

        if audio is not None:
            try:
                waveform = audio.get("waveform")
                sr = audio.get("sample_rate")

                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform)

                if waveform_np.ndim == 3:
                    waveform_np = waveform_np[0].T

                wav_path = os.path.join(output_dir, filename_wav)
                sf.write(wav_path, waveform_np, sr)
                print(f"✅ [Qwen3-TTS-XPU] Reference audio (Speaker) saved to: {wav_path}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS-XPU] Failed to save reference audio: {e}")

        return ()


class LoadSpeakerNode:
    """LoadSpeaker Node: Directly load a WAV/Speaker file and its associated features."""

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = os.path.join(folder_paths.models_dir, "qwen-tts", "voices")
        os.makedirs(output_dir, exist_ok=True)

        try:
            files = os.listdir(output_dir)
            wav_files = [f for f in files if f.endswith(".wav")]

            if not wav_files:
                wav_files = ["(no speakers found)"]
        except Exception:
            wav_files = ["(no speakers found)"]

        return {
            "required": {
                "filename": (wav_files, {"default": wav_files[0] if wav_files else None}),
            }
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT", "AUDIO", "STRING")
    RETURN_NAMES = ("voice_clone_prompt", "audio", "ref_text")
    FUNCTION = "load_speaker"
    CATEGORY = "Qwen3TTS-XPU"
    DESCRIPTION = "LoadSpeaker: Load saved WAV audio and its metadata. Fast-loads .qvp features if available."

    def load_speaker(self, filename):
        if filename == "(no speakers found)" or filename is None:
            raise RuntimeError("No speaker files found to load.")

        voices_dir = os.path.join(folder_paths.models_dir, "qwen-tts", "voices")
        wav_path = os.path.join(voices_dir, filename)

        qvp_file = filename.replace(".wav", ".qvp")
        json_file = filename.replace(".wav", ".json")

        qvp_path = os.path.join(voices_dir, qvp_file)
        json_path = os.path.join(voices_dir, json_file)

        prompt_items = None
        if os.path.exists(qvp_path):
            try:
                data = torch.load(qvp_path, weights_only=False)
                if isinstance(data, dict) and "prompt" in data:
                    prompt_items = data["prompt"]
                else:
                    prompt_items = data
                print(f"✅ [Qwen3-TTS-XPU] Fast-loaded pre-computed features from: {qvp_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS-XPU] Failed to fast-load .qvp: {e}")

        import soundfile as sf

        try:
            waveform_np, sr = sf.read(wav_path)

            if waveform_np.ndim == 1:
                waveform_np = waveform_np.reshape(1, 1, -1)
            elif waveform_np.ndim == 2:
                waveform_np = waveform_np.T
                waveform_np = waveform_np.reshape(1, waveform_np.shape[0], waveform_np.shape[1])

            waveform = torch.from_numpy(waveform_np).float()
            audio_preview = {"waveform": waveform, "sample_rate": sr}
        except Exception as e:
            raise RuntimeError(f"Failed to load speaker audio: {e}")

        ref_text = ""
        if os.path.exists(json_path):
            try:
                import json

                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    if not ref_text or not ref_text.strip():
                        ref_text = metadata.get("ref_text", "")
                    if ref_text:
                        print(f"✅ [Qwen3-TTS-XPU] Loaded metadata from JSON: {json_file}")
            except Exception as e:
                print(f"⚠️ [Qwen3-TTS-XPU] Failed to load metadata JSON: {e}")

        return (prompt_items, audio_preview, ref_text)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "Qwen3TTS_XPU_VoiceDesignNode": VoiceDesignNode,
    "Qwen3TTS_XPU_VoiceCloneNode": VoiceCloneNode,
    "Qwen3TTS_XPU_CustomVoiceNode": CustomVoiceNode,
    "Qwen3TTS_XPU_VoiceClonePromptNode": VoiceClonePromptNode,
    "Qwen3TTS_XPU_RoleBankNode": RoleBankNode,
    "Qwen3TTS_XPU_DialogueInferenceNode": DialogueInferenceNode,
    "Qwen3TTS_XPU_SaveVoiceNode": SaveVoiceNode,
    "Qwen3TTS_XPU_LoadSpeakerNode": LoadSpeakerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS_XPU_VoiceDesignNode": "Qwen3TTS_XPU_VoiceDesign",
    "Qwen3TTS_XPU_VoiceCloneNode": "Qwen3TTS_XPU_VoiceClone",
    "Qwen3TTS_XPU_CustomVoiceNode": "Qwen3TTS_XPU_CustomVoice",
    "Qwen3TTS_XPU_VoiceClonePromptNode": "Qwen3TTS_XPU_VoiceClonePrompt",
    "Qwen3TTS_XPU_RoleBankNode": "Qwen3TTS_XPU_RoleBank",
    "Qwen3TTS_XPU_DialogueInferenceNode": "Qwen3TTS_XPU_DialogueInference",
    "Qwen3TTS_XPU_SaveVoiceNode": "Qwen3TTS_XPU_SaveVoice",
    "Qwen3TTS_XPU_LoadSpeakerNode": "Qwen3TTS_XPU_LoadSpeaker",
}
