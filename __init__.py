# ComfyUI-Qwen3TTS-XPU Custom Nodes

# Based on the open-source Qwen3-TTS project by Alibaba Qwen team

import os
import sys
import torch

# Add current directory to path for qwen_tts package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import nodes
from .nodes import (
    VoiceDesignNode,
    VoiceCloneNode,
    CustomVoiceNode,
)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTS_XPU_VoiceCloneNode": VoiceCloneNode,
    "Qwen3TTS_XPU_VoiceDesignNode": VoiceDesignNode,
    "Qwen3TTS_XPU_CustomVoiceNode": CustomVoiceNode,
}

# Display names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTS_XPU_VoiceCloneNode": "Qwen3TTS_XPU_VoiceClone",
    "Qwen3TTS_XPU_VoiceDesignNode": "Qwen3TTS_XPU_VoiceDesign",
    "Qwen3TTS_XPU_CustomVoiceNode": "Qwen3TTS_XPU_CustomVoice",
}

# Version information
__version__ = "1.2.5"

# Web directory for custom UI components (if any)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__"]

print(f"✅ ComfyUI-Qwen3-TTS-XPU v{__version__} loaded")
