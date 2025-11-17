# Bug Fix: Invalid Device String for Hugging Face Pipeline

**Date**: 2025-01-17
**Type**: Bug Fix
**Severity**: High
**Status**: ✅ Fixed

## Problem

Hebrew model loading failed with error:
```
⚠️  Hebrew model failed (Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl,
opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia,
privateuseone device type at start of device string: auto)
```

Script always fell back to Whisper large-v3-turbo instead of using the faster Hebrew-optimized model.

## Root Cause

**File**: `src/scripts/transcribe_advanced.py`
**Line**: 47 (original)

```python
transcriber = pipeline(
    "automatic-speech-recognition",
    model="imvladikon/wav2vec2-large-xlsr-53-hebrew",
    device="auto"  # ❌ Not a valid device string
)
```

Hugging Face `pipeline` doesn't accept `device="auto"`. Valid values are:
- `0`, `1`, `2`, etc. for CUDA GPUs
- `"mps"` for Apple Silicon
- `-1` for CPU

## Solution

**File**: `src/scripts/transcribe_advanced.py`
**Lines**: 45-58

Added proper device detection using PyTorch:

```python
def load_optimal_model():
    try:
        print("🇮🇱 Loading Hebrew-optimized model from Hugging Face...")

        # Detect available device
        import torch
        if torch.cuda.is_available():
            device = 0  # First CUDA GPU
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = -1  # CPU

        transcriber = pipeline(
            "automatic-speech-recognition",
            model="imvladikon/wav2vec2-large-xlsr-53-hebrew",
            device=device
        )
        print("✅ Hebrew-optimized model loaded successfully!")
        return transcriber, "huggingface"
    except Exception as e:
        print(f"⚠️  Hebrew model failed ({e}), trying Whisper large-v3-turbo...")
        # ... fallback logic
```

## Device Detection Logic

| Hardware | Detection | Device Value |
|----------|-----------|--------------|
| NVIDIA GPU | `torch.cuda.is_available()` | `0` (first GPU) |
| Apple M1/M2/M3 | `torch.backends.mps.is_available()` | `"mps"` |
| CPU (fallback) | Always available | `-1` |

## Testing

Tested on M1 Mac:
```
🇮🇱 Loading Hebrew-optimized model from Hugging Face...
✅ Hebrew-optimized model loaded successfully!
```

Model now loads correctly and uses MPS backend for hardware acceleration.

## Impact

- Hebrew-optimized model works on Apple Silicon Macs
- Faster transcription (Hebrew model is optimized for Hebrew/English)
- Proper hardware acceleration (MPS on M1/M2/M3)
- Better fallback behavior if Hebrew model fails

## Related Fixes

This fix was discovered while fixing the Hugging Face pipeline `KeyError`.
- See: `01_huggingface_pipeline_fix.md`
