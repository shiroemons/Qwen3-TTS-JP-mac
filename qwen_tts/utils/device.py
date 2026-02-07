# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Device detection utilities for macOS Apple Silicon (MPS), CUDA, and CPU.
"""

import os

import torch


def detect_device() -> str:
    """Auto-detect the best available device: mps > cuda > cpu."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def detect_dtype(device: str) -> torch.dtype:
    """Return the recommended dtype for the given device.

    - CUDA: bfloat16 (best performance)
    - MPS / CPU: float32 (MPS has limited fp16/bf16 support)
    """
    if "cuda" in device:
        return torch.bfloat16
    return torch.float32


def detect_attn_implementation(device: str) -> str:
    """Return the recommended attention implementation for the given device.

    - CUDA: flash_attention_2 (if available)
    - MPS / CPU: sdpa (Scaled Dot-Product Attention)
    """
    if "cuda" in device:
        return "flash_attention_2"
    return "sdpa"


def setup_mps_env() -> None:
    """Set environment variables required for MPS backend.

    PYTORCH_ENABLE_MPS_FALLBACK=1 allows unsupported MPS operations
    to fall back to CPU transparently.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def sync_device(device: str) -> None:
    """Synchronize the given device (for timing measurements).

    - CUDA: torch.cuda.synchronize()
    - MPS: torch.mps.synchronize()
    - CPU: no-op
    """
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "mps" in device:
        torch.mps.synchronize()
