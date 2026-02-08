# Copyright 2025 Alibaba Inc. and the authors of Qwen3-TTS-JP-mac.
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

"""Gradio hot-reload wrapper for Qwen3-TTS demo development.

Usage (via devbox):
    devbox run dev:clone
    devbox run dev:custom
    devbox run dev:design

Or directly:
    QWEN_TTS_CHECKPOINT=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      uv run gradio scripts/dev_demo.py --watch-dirs qwen_tts

How it works:
    - The `gradio` CLI watches this file and ``--watch-dirs`` for changes.
    - Code inside ``if gr.NO_RELOAD:`` runs **once** (model loading).
    - Everything outside that block re-executes on every reload, picking up
      UI changes in ``qwen_tts/cli/demo.py`` without reloading the model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that ``qwen_tts`` is importable
# when this script is executed via ``uv run gradio scripts/dev_demo.py``.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr

from qwen_tts.utils.device import (
    detect_attn_implementation,
    detect_device,
    detect_dtype,
    setup_mps_env,
)

# ── Model loading (runs only once) ──────────────────────────────────────────
if gr.NO_RELOAD:
    from qwen_tts import Qwen3TTSModel

    ckpt = os.environ.get(
        "QWEN_TTS_CHECKPOINT", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )

    device = os.environ.get("QWEN_TTS_DEVICE", "") or detect_device()
    if "mps" in device:
        setup_mps_env()

    dtype = detect_dtype(device)
    attn_impl = detect_attn_implementation(device)

    print(f"[dev] Loading checkpoint: {ckpt}")
    print(f"[dev] Device: {device}, Dtype: {dtype}, Attention: {attn_impl}")

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print("[dev] Model loaded – ready for hot reload")

# ── UI construction (re-runs on every reload) ───────────────────────────────
from qwen_tts.cli.demo import build_demo  # noqa: E402

demo, extra_launch_kwargs = build_demo(tts, ckpt, {})
demo.queue().launch(inbrowser=True, **extra_launch_kwargs)
