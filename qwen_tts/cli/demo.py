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
Qwen3-TTS-JP-mac Gradio デモ
"""

import argparse
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from .. import Qwen3TTSModel, VoiceClonePromptItem
from ..utils.device import detect_device, detect_dtype, detect_attn_implementation, setup_mps_env

# Whisper integration (optional)
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

_whisper_model_cache = {}

def _get_whisper_model(model_size: str = "base") -> "WhisperModel":
    """Load a faster-whisper model. MPS is not supported by faster-whisper, so use CPU + int8."""
    if model_size in _whisper_model_cache:
        return _whisper_model_cache[model_size]
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    _whisper_model_cache[model_size] = model
    return model

def _transcribe_audio(audio, model_size: str = "base") -> str:
    """Transcribe audio using faster-whisper. Returns text."""
    if not WHISPER_AVAILABLE:
        return "Error: faster-whisper is not installed. Run: pip install faster-whisper"

    import soundfile as _sf

    at = _audio_to_tuple(audio)
    if at is None:
        return ""
    wav, sr = at

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _sf.write(tmp_path, wav, sr)
        model = _get_whisper_model(model_size)
        segments, info = model.transcribe(tmp_path, beam_size=5)
        text = "".join(seg.text for seg in segments).strip()
        return text
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


LANG_LABELS_JA: Dict[str, str] = {
    "auto": "自動検出",
    "chinese": "中国語",
    "english": "英語",
    "japanese": "日本語",
    "korean": "韓国語",
    "german": "ドイツ語",
    "french": "フランス語",
    "russian": "ロシア語",
    "portuguese": "ポルトガル語",
    "spanish": "スペイン語",
    "italian": "イタリア語",
}

VOICE_CLONE_TOUR_CLICK_JS = """\
() => {
  function startTour() {
    function ensureTab(idx) {
      let btn;
      if (idx <= 9) btn = '#vc-tab-clone-button';
      else btn = '#vc-tab-save-load-button';
      document.querySelector(btn)?.click();
    }
    function getStartIndex() {
      if (document.querySelector('#vc-tab-save-load-button[aria-selected="true"]')) return 10;
      return 0;
    }
    const d = window.driver.js.driver({
      showProgress: true,
      progressText: '{{current}} / {{total}}',
      nextBtnText: '次へ',
      prevBtnText: '戻る',
      doneBtnText: '完了',
      onNextClick: () => {
        ensureTab(d.getActiveIndex() + 1);
        requestAnimationFrame(() => d.moveNext());
      },
      onPrevClick: () => {
        ensureTab(d.getActiveIndex() - 1);
        requestAnimationFrame(() => d.movePrevious());
      },
      steps: [
        {
          element: '#vc-ref-audio',
          popover: {
            title: '① 参照音声（クローン元の音声）',
            description: 'クローン元となる3秒以上の音声をアップロードしてください（WAV/MP3対応）',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-whisper-model',
          popover: {
            title: '文字起こし精度',
            description: '精度と速度のバランスで選択します。base が推奨です。large-v3 は高精度ですが処理が遅くなります',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-whisper-btn',
          popover: {
            title: '② 自動文字起こし',
            description: '参照音声をアップロード後にクリックすると、Whisper が音声をテキストに変換し参照テキストに自動入力します',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-ref-text',
          popover: {
            title: '③ 参照テキスト',
            description: '参照音声の書き起こしを入力します。自動文字起こしボタンで自動入力も可能です',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-xvec-only',
          popover: {
            title: '簡易モード(x-vectorのみ)',
            description: '有効にすると参照テキスト不要で簡易クローンが可能ですが、品質は低下します',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-synth-text',
          popover: {
            title: '④ 合成するテキスト',
            description: 'クローンした声で読み上げたい文章を入力してください',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-lang',
          popover: {
            title: '⑤ 言語',
            description: 'テキストの言語を選択します',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-generate-btn',
          popover: {
            title: '⑥ 音声生成',
            description: 'すべて入力したらクリックして音声を合成します',
            side: 'top', align: 'center'
          }
        },
        {
          element: '#vc-audio-out',
          popover: {
            title: '生成された音声',
            description: '生成された音声がここに表示されます。再生・ダウンロードが可能です',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-status',
          popover: {
            title: '結果',
            description: '処理の進行状況やエラーメッセージがここに表示されます',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-tab-save-load-button',
          popover: {
            title: '音声データの保存・読込タブ',
            description: '音声データを .pt ファイルとして保存・読込できます。同じ話者で繰り返し生成する場合に便利です',
            side: 'bottom', align: 'center'
          }
        },
        {
          element: '#vc-save-ref-audio',
          popover: {
            title: '① 参照音声（クローン元の音声・保存用）',
            description: '音声データとして保存したい話者の音声サンプルをアップロードします',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-save-ref-text',
          popover: {
            title: '② 参照テキスト（保存用）',
            description: '参照音声の書き起こしテキストを入力します。簡易モードが無効の場合は必須です',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-save-xvec-only',
          popover: {
            title: '簡易モード(x-vectorのみ)（保存用）',
            description: '有効にすると参照テキスト不要で音声データを保存できますが、生成品質は低下します',
            side: 'right', align: 'start'
          }
        },
        {
          element: '#vc-save-btn',
          popover: {
            title: '③ 保存',
            description: '入力した参照音声・テキストから音声データを作成し、.pt ファイルとして保存します',
            side: 'top', align: 'center'
          }
        },
        {
          element: '#vc-save-output',
          popover: {
            title: '保存された音声データ',
            description: '生成された .pt ファイルがここに表示されます。クリックしてダウンロードしてください',
            side: 'top', align: 'center'
          }
        },
        {
          element: '#vc-load-file',
          popover: {
            title: '① 音声データ (.pt)',
            description: '保存済みの .pt ファイルをここにアップロードします。参照音声を再度用意する必要はありません',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-load-text',
          popover: {
            title: '② 合成するテキスト（読込用）',
            description: '読み込んだ音声データの声で読み上げたい文章を入力してください',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-load-lang',
          popover: {
            title: '③ 言語（読込用）',
            description: 'テキストの言語を選択します',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-load-gen-btn',
          popover: {
            title: '④ 音声生成',
            description: 'アップロードした音声データとテキストを使ってクローン音声を生成します',
            side: 'top', align: 'center'
          }
        },
        {
          element: '#vc-load-audio-out',
          popover: {
            title: '生成された音声（読込用）',
            description: '読み込んだ音声データから生成された音声がここに表示されます。再生・ダウンロードが可能です',
            side: 'left', align: 'start'
          }
        },
        {
          element: '#vc-load-status',
          popover: {
            title: '結果（読込用）',
            description: '保存・読込・生成の進行状況やエラーメッセージがここに表示されます',
            side: 'left', align: 'start'
          }
        }
      ]
    });
    setTimeout(() => d.drive(getStartIndex()), 100);
  }
  if (window.driver && window.driver.js) {
    startTour();
    return;
  }
  if (!document.querySelector('link[href*="driver.js"]')) {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://cdn.jsdelivr.net/npm/driver.js@1.4.0/dist/driver.css';
    document.head.appendChild(link);
  }
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/driver.js@1.4.0/dist/driver.js.iife.js';
  s.onload = startTour;
  document.head.appendChild(s);
}
"""


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _build_lang_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [LANG_LABELS_JA.get(x.lower(), _title_case_display(x)) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s == "auto":
        return torch.float32  # safe default
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-demo",
        description=(
            "Launch a Gradio demo for Qwen3 TTS models (CustomVoice / VoiceDesign / Base).\n\n"
            "Examples:\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --port 8000 --ip 127.0.0.1\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --device cuda:0\n"
            "  qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --dtype bfloat16 --no-flash-attn\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )

    # Positional checkpoint (also supports -c/--checkpoint)
    parser.add_argument(
        "checkpoint_pos",
        nargs="?",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (positional).",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model checkpoint path or HuggingFace repo id (optional if positional is provided).",
    )

    # Model loading / from_pretrained args
    parser.add_argument(
        "--device",
        default="auto",
        help="デバイス指定。auto で MPS > CUDA > CPU を自動検出 (デフォルト: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="データ型。auto でデバイスに応じて自動選択 (デフォルト: auto)",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="FlashAttention-2 を使用 (デフォルト: 自動検出。CUDA 環境のみ有効)",
    )

    # Gradio server args
    parser.add_argument(
        "--ip",
        default="0.0.0.0",
        help="Server bind IP for Gradio (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for Gradio (default: 8000).",
    )
    parser.add_argument(
        "--share/--no-share",
        dest="share",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to create a public Gradio link (default: disabled).",
    )
    parser.add_argument(
        "--inbrowser/--no-inbrowser",
        dest="inbrowser",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to auto-open browser on launch (default: enabled).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Gradio queue concurrency (default: 16).",
    )

    # HTTPS args
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file for HTTPS (optional).",
    )
    parser.add_argument(
        "--ssl-verify/--no-ssl-verify",
        dest="ssl_verify",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to verify SSL certificate (default: enabled).",
    )

    # Optional generation args
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional, only for tokenizer v2).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional, only for tokenizer v2).")
    parser.add_argument(
        "--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional, only for tokenizer v2)."
    )

    return parser


def _resolve_checkpoint(args: argparse.Namespace) -> str:
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt:
        raise SystemExit(0)  # main() prints help
    return ckpt


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)

        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid

    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0

        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    wav = (wav * 32767).astype(np.int16)
    return sr, wav


def _detect_model_kind(ckpt: str, tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    else:
        raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


DISCLAIMER_TEXT = """\
**免責事項**

○ この音声はAIモデルによって自動生成/合成されたものであり、モデルの機能を示すためのデモンストレーション目的でのみ提供されています。不正確または不適切な内容が含まれる場合があります。この音声は開発者/運営者の見解を代表するものではなく、専門的なアドバイスを構成するものでもありません。

○ ユーザーは、この音声の評価、使用、配布、または拡散に関するすべてのリスクと責任を自ら負うものとします。適用法が許容する最大限の範囲において、開発者/運営者は、この音声の使用または使用不能から生じる直接的、間接的、偶発的、または結果的な損害について責任を負いません（法律で免責が認められない場合を除く）。

○ 本サービスを使用して、違法、有害、名誉毀損、詐欺、ディープフェイク、プライバシー/肖像権/著作権/商標を侵害するコンテンツを意図的に生成または複製することは禁止されています。ユーザーがプロンプト、素材、その他の手段によって違法または侵害行為を実施または促進した場合、その法的責任はすべてユーザーが負い、開発者/運営者は一切の責任を負いません。\
"""


def build_demo(tts: Qwen3TTSModel, ckpt: str, gen_kwargs_default: Dict[str, Any]) -> tuple[gr.Blocks, Dict[str, Any]]:
    model_kind = _detect_model_kind(ckpt, tts)

    supported_langs_raw = None
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported_langs_raw = tts.model.get_supported_languages()

    supported_spks_raw = None
    if callable(getattr(tts.model, "get_supported_speakers", None)):
        supported_spks_raw = tts.model.get_supported_speakers()

    lang_choices_disp, lang_map = _build_lang_choices_and_map([x for x in (supported_langs_raw or [])])
    spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])

    def _gen_common_kwargs() -> Dict[str, Any]:
        return dict(gen_kwargs_default)

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = ".gradio-container {max-width: none !important;} #vc-tour-btn {max-width: 160px; margin-left: auto;} .prose h3 {border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; margin-bottom: 12px;}"

    with gr.Blocks() as demo:
        gr.Markdown(
            f"""
# Qwen3-TTS-JP-mac デモ
**チェックポイント:** `{ckpt}`

**モデルタイプ:** `{model_kind}`
"""
        )

        if model_kind == "custom_voice":
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### テキストと話者の設定")
                    text_in = gr.Textbox(
                        label="① 合成するテキスト",
                        lines=4,
                        placeholder="読み上げたい文章を入力",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="② 言語",
                            choices=lang_choices_disp,
                            value="日本語",
                            interactive=True,
                        )
                        spk_in = gr.Dropdown(
                            label="③ 話者",
                            choices=spk_choices_disp,
                            value="Vivian",
                            interactive=True,
                        )
                    instruct_in = gr.Textbox(
                        label="④ 話し方の指示",
                        info="任意。口調や感情を指定できます",
                        lines=2,
                        placeholder="例：怒った口調で話してください",
                    )
                    btn = gr.Button("⑤ 音声生成", variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("### 生成結果")
                    audio_out = gr.Audio(label="生成された音声", type="numpy")
                    err = gr.Textbox(label="結果", lines=1)

            def run_instruct(text: str, lang_disp: str, spk_disp: str, instruct: str, progress=gr.Progress()):
                try:
                    if not text or not text.strip():
                        return None, "テキストを入力してください。"
                    if not spk_disp:
                        return None, "話者を選択してください。"
                    language = lang_map.get(lang_disp, "Auto")
                    speaker = spk_map.get(spk_disp, spk_disp)
                    progress(0.0, desc="音声を生成中...")
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_custom_voice(
                        text=text.strip(),
                        language=language,
                        speaker=speaker,
                        instruct=(instruct or "").strip() or None,
                        **kwargs,
                    )
                    progress(0.9, desc="波形をデコード中...")
                    result = _wav_to_gradio_audio(wavs[0], sr)
                    progress(1.0, desc="完了")
                    return result, "生成完了"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_instruct, inputs=[text_in, lang_in, spk_in, instruct_in], outputs=[audio_out, err])

        elif model_kind == "voice_design":
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### テキストと声のデザイン")
                    text_in = gr.Textbox(
                        label="① 合成するテキスト",
                        lines=4,
                        value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!",
                        placeholder="読み上げたい文章を入力",
                    )
                    with gr.Row():
                        lang_in = gr.Dropdown(
                            label="② 言語",
                            choices=lang_choices_disp,
                            value="日本語",
                            interactive=True,
                        )
                    design_in = gr.Textbox(
                        label="③ 声のデザイン",
                        lines=3,
                        value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.",
                        info="どんな声で読み上げるか自由に指示できます",
                        placeholder="例：落ち着いた低めの男性の声で、ゆっくり丁寧に",
                    )
                    btn = gr.Button("④ 音声生成", variant="primary")
                with gr.Column(scale=3):
                    gr.Markdown("### 生成結果")
                    audio_out = gr.Audio(label="生成された音声", type="numpy")
                    err = gr.Textbox(label="結果", lines=1)

            def run_voice_design(text: str, lang_disp: str, design: str, progress=gr.Progress()):
                try:
                    if not text or not text.strip():
                        return None, "テキストを入力してください。"
                    if not design or not design.strip():
                        return None, "音声デザイン指示を入力してください。"
                    language = lang_map.get(lang_disp, "Auto")
                    progress(0.0, desc="音声を生成中...")
                    kwargs = _gen_common_kwargs()
                    wavs, sr = tts.generate_voice_design(
                        text=text.strip(),
                        language=language,
                        instruct=design.strip(),
                        **kwargs,
                    )
                    progress(0.9, desc="波形をデコード中...")
                    result = _wav_to_gradio_audio(wavs[0], sr)
                    progress(1.0, desc="完了")
                    return result, "生成完了"
                except Exception as e:
                    return None, f"{type(e).__name__}: {e}"

            btn.click(run_voice_design, inputs=[text_in, lang_in, design_in], outputs=[audio_out, err])

        else:  # voice_clone for base
            tour_btn = gr.Button(
                "使い方ガイド", size="sm", variant="secondary",
                elem_id="vc-tour-btn",
            )
            tour_btn.click(None, None, None, js=VOICE_CLONE_TOUR_CLICK_JS)
            with gr.Tabs():
                with gr.Tab("ボイスクローン", elem_id="vc-tab-clone"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### ステップ1：参照音声の設定")
                            ref_audio = gr.Audio(
                                label="① 参照音声（クローン元の音声）",
                                elem_id="vc-ref-audio",
                            )
                            with gr.Group(visible=WHISPER_AVAILABLE):
                                whisper_model_dd = gr.Dropdown(
                                    label="文字起こし精度",
                                    info="base推奨。large-v3は高精度だが低速",
                                    choices=WHISPER_MODELS, value="base",
                                    interactive=True, elem_id="vc-whisper-model",
                                )
                                whisper_btn = gr.Button(
                                    "② 自動文字起こし", variant="secondary",
                                    elem_id="vc-whisper-btn",
                                )
                            ref_text = gr.Textbox(
                                label="③ 参照テキスト",
                                info="参照音声の書き起こし。自動文字起こしで入力可",
                                lines=2,
                                placeholder="参照音声の書き起こしテキストを入力してください",
                                elem_id="vc-ref-text",
                            )
                            xvec_only = gr.Checkbox(
                                label="簡易モード(x-vectorのみ)",
                                info="テキスト不要で簡易クローン。品質は低下します",
                                value=False,
                                elem_id="vc-xvec-only",
                            )

                        with gr.Column(scale=2):
                            gr.Markdown("### ステップ2：テキストの入力")
                            text_in = gr.Textbox(
                                label="④ 合成するテキスト",
                                lines=4,
                                placeholder="クローンした声で読み上げたい文章を入力",
                                elem_id="vc-synth-text",
                            )
                            lang_in = gr.Dropdown(
                                label="⑤ 言語",
                                choices=lang_choices_disp,
                                value="日本語",
                                interactive=True,
                                elem_id="vc-lang",
                            )
                            btn = gr.Button("⑥ 音声生成", variant="primary", elem_id="vc-generate-btn")

                        with gr.Column(scale=3):
                            gr.Markdown("### 生成結果")
                            audio_out = gr.Audio(label="生成された音声", type="numpy", elem_id="vc-audio-out")
                            err = gr.Textbox(label="結果", lines=1, elem_id="vc-status")

                    def run_voice_clone(ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, progress=gr.Progress()):
                        try:
                            if not text or not text.strip():
                                return None, "合成テキストを入力してください。"
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "参照音声をアップロードしてください。"
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, "簡易モードが無効の場合は、参照テキストが必要です。"
                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            progress(0.0, desc="音声の特徴を抽出中...")
                            progress(0.1, desc="音声コードを生成中...")
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                                **kwargs,
                            )
                            progress(0.9, desc="波形をデコード中...")
                            result = _wav_to_gradio_audio(wavs[0], sr)
                            progress(1.0, desc="完了")
                            return result, "生成完了"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    btn.click(
                        run_voice_clone,
                        inputs=[ref_audio, ref_text, xvec_only, text_in, lang_in],
                        outputs=[audio_out, err],
                    )

                    def run_whisper_and_fill(ref_aud, model_size: str, progress=gr.Progress()):
                        if ref_aud is None:
                            return "参照音声をアップロードしてください。"
                        progress(0.0, desc="Whisperモデルを読み込み中...")
                        progress(0.3, desc="文字起こし中...")
                        result = _transcribe_audio(ref_aud, model_size)
                        progress(1.0, desc="完了")
                        return result

                    if WHISPER_AVAILABLE:
                        whisper_btn.click(
                            run_whisper_and_fill,
                            inputs=[ref_audio, whisper_model_dd],
                            outputs=[ref_text],
                        )

                with gr.Tab("音声データの保存・読込", elem_id="vc-tab-save-load"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### 音声データの保存
参照音声から再利用可能な音声データを作成します。
"""
                            )
                            ref_audio_s = gr.Audio(label="① 参照音声（クローン元の音声）", type="numpy", elem_id="vc-save-ref-audio")
                            ref_text_s = gr.Textbox(
                                label="② 参照テキスト",
                                info="簡易モードが無効の場合は必須",
                                lines=2,
                                placeholder="参照音声の書き起こしテキストを入力してください",
                                elem_id="vc-save-ref-text",
                            )
                            xvec_only_s = gr.Checkbox(
                                label="簡易モード(x-vectorのみ)",
                                info="テキスト不要で簡易クローン。品質は低下します",
                                value=False,
                                elem_id="vc-save-xvec-only",
                            )
                            save_btn = gr.Button("③ 保存", variant="primary", elem_id="vc-save-btn")
                            prompt_file_out = gr.File(label="保存された音声データ", elem_id="vc-save-output")

                        with gr.Column(scale=2):
                            gr.Markdown(
                                """
### 音声データから生成
保存済みの .pt ファイルで音声を合成します。
"""
                            )
                            prompt_file_in = gr.File(label="① 音声データ (.pt)", elem_id="vc-load-file")
                            text_in2 = gr.Textbox(
                                label="② 合成するテキスト",
                                lines=4,
                                placeholder="読み上げたい文章を入力",
                                elem_id="vc-load-text",
                            )
                            lang_in2 = gr.Dropdown(
                                label="③ 言語",
                                choices=lang_choices_disp,
                                value="日本語",
                                interactive=True,
                                elem_id="vc-load-lang",
                            )
                            gen_btn2 = gr.Button("④ 音声生成", variant="primary", elem_id="vc-load-gen-btn")

                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(label="生成された音声", type="numpy", elem_id="vc-load-audio-out")
                            err2 = gr.Textbox(label="結果", lines=1, elem_id="vc-load-status")

                    def save_prompt(ref_aud, ref_txt: str, use_xvec: bool, progress=gr.Progress()):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None:
                                return None, "参照音声をアップロードしてください。"
                            if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                                return None, "簡易モードが無効の場合は、参照テキストが必要です。"
                            progress(0.0, desc="音声の特徴を抽出中...")
                            items = tts.create_voice_clone_prompt(
                                ref_audio=at,
                                ref_text=(ref_txt.strip() if ref_txt else None),
                                x_vector_only_mode=bool(use_xvec),
                            )
                            payload = {
                                "items": [asdict(it) for it in items],
                            }
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            progress(0.5, desc="プロファイルを保存中...")
                            torch.save(payload, out_path)
                            progress(1.0, desc="完了")
                            return out_path, "生成完了"
                        except Exception as e:
                            return None, f"{type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text: str, lang_disp: str, progress=gr.Progress()):
                        try:
                            if file_obj is None:
                                return None, "音声ファイルをアップロードしてください。"
                            if not text or not text.strip():
                                return None, "合成テキストを入力してください。"

                            progress(0.0, desc="プロファイルを読み込み中...")
                            path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload, dict) or "items" not in payload:
                                return None, "ファイル形式が正しくありません。"

                            items_raw = payload["items"]
                            if not isinstance(items_raw, list) or len(items_raw) == 0:
                                return None, "音声データが空です。"

                            items: List[VoiceClonePromptItem] = []
                            for d in items_raw:
                                if not isinstance(d, dict):
                                    return None, "ファイル内部の形式が正しくありません。"
                                ref_code = d.get("ref_code", None)
                                if ref_code is not None and not torch.is_tensor(ref_code):
                                    ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding", None)
                                if ref_spk is None:
                                    return None, "話者ベクトルが見つかりません。"
                                if not torch.is_tensor(ref_spk):
                                    ref_spk = torch.tensor(ref_spk)

                                items.append(
                                    VoiceClonePromptItem(
                                        ref_code=ref_code,
                                        ref_spk_embedding=ref_spk,
                                        x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                                        icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                                        ref_text=d.get("ref_text", None),
                                    )
                                )

                            language = lang_map.get(lang_disp, "Auto")
                            kwargs = _gen_common_kwargs()
                            progress(0.2, desc="音声コードを生成中...")
                            wavs, sr = tts.generate_voice_clone(
                                text=text.strip(),
                                language=language,
                                voice_clone_prompt=items,
                                **kwargs,
                            )
                            progress(0.9, desc="波形をデコード中...")
                            result = _wav_to_gradio_audio(wavs[0], sr)
                            progress(1.0, desc="完了")
                            return result, "生成完了"
                        except Exception as e:
                            return None, (
                                f"音声ファイルの読み込みまたは使用に失敗しました。ファイルの形式や内容を確認してください。\n"
                                f"{type(e).__name__}: {e}"
                            )

                    save_btn.click(save_prompt, inputs=[ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in, text_in2, lang_in2], outputs=[audio_out2, err2])

        gr.Markdown(DISCLAIMER_TEXT)

    return demo, {"theme": theme, "css": css}


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0

    ckpt = _resolve_checkpoint(args)

    # Device detection
    if args.device == "auto":
        device = detect_device()
    else:
        device = args.device

    # Setup MPS environment if needed
    if "mps" in device:
        setup_mps_env()

    # Dtype detection
    if args.dtype == "auto":
        dtype = detect_dtype(device)
    else:
        dtype = _dtype_from_str(args.dtype)

    # Attention implementation
    if args.flash_attn is True:
        attn_impl = "flash_attention_2"
    elif args.flash_attn is False:
        attn_impl = "sdpa"
    else:  # None = auto
        attn_impl = detect_attn_implementation(device)

    print(f"Device: {device}, Dtype: {dtype}, Attention: {attn_impl}")

    tts = Qwen3TTSModel.from_pretrained(
        ckpt,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    gen_kwargs_default = _collect_gen_kwargs(args)
    demo, extra_launch_kwargs = build_demo(tts, ckpt, gen_kwargs_default)

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        ssl_verify=True if args.ssl_verify else False,
    )
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs, **extra_launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
