# Qwen3-TTS-JP-mac

macOS Apple Silicon 向け Qwen3-TTS 日本語対応フォーク

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## 概要

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) を macOS Apple Silicon (M1/M2/M3/M4) で簡単に使えるようにした日本語ローカライズ版フォークです。

主な特徴:
- **macOS Apple Silicon ネイティブ対応** — MPS バックエンドによる GPU アクセラレーション
- **日本語 GUI** — Gradio デモ UI を完全日本語化
- **Whisper 自動文字起こし** — 音声クローン用の参照テキストを自動生成
- **簡単セットアップ** — devbox / uv によるワンコマンド環境構築
- **10言語対応** — 中国語, 英語, 日本語, 韓国語, ドイツ語, フランス語, ロシア語, ポルトガル語, スペイン語, イタリア語

### 3つの音声生成モード

| モード | モデル | 説明 |
|---|---|---|
| **CustomVoice** | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 9種類のプレミアム音声 + 指示による制御 |
| **VoiceDesign** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 自然言語の説明から音声を生成 |
| **VoiceClone** | `Qwen3-TTS-12Hz-1.7B-Base` | 3秒の音声サンプルから話者をクローン |

## 動作要件

| 項目 | 要件 |
|---|---|
| OS | macOS 14 (Sonoma) 以降 |
| チップ | Apple Silicon (M1/M2/M3/M4) |
| Python | 3.12 |
| メモリ | 16GB 以上（1.7B モデル推奨） |
| ストレージ | モデルごとに約 3-7GB |

## クイックスタート

### devbox を使う場合（推奨）

```bash
# devbox のインストール（未導入の場合）
curl -fsSL https://get.jetify.com/devbox | bash

# セットアップ
git clone https://github.com/shiroemons/Qwen3-TTS-JP-mac.git
cd Qwen3-TTS-JP-mac
devbox install
devbox run setup

# MPS 確認
devbox run check-mps

# デモ起動
devbox run demo:custom    # CustomVoice
devbox run demo:design    # VoiceDesign
devbox run demo:clone     # VoiceClone (Base)
```

### セットアップスクリプト

```bash
git clone https://github.com/shiroemons/Qwen3-TTS-JP-mac.git
cd Qwen3-TTS-JP-mac
bash setup_mac.sh

# デモ起動
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

### 手動セットアップ (uv)

```bash
# 前提: uv, sox, ffmpeg がインストール済み
# brew install uv sox ffmpeg

git clone https://github.com/shiroemons/Qwen3-TTS-JP-mac.git
cd Qwen3-TTS-JP-mac

# 依存関係インストール（Whisper 含む）
uv sync --extra whisper

# MPS 確認
uv run python -c "import torch; print(torch.backends.mps.is_available())"
```

## 使い方

> **Note:** モデルは初回実行時に HuggingFace から自動ダウンロードされます（1モデルあたり約 3〜7GB）。初回は時間がかかる場合があります。

### Web UI デモ

```bash
# CustomVoice モデル（推奨: 初回利用）
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

# VoiceDesign モデル
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# VoiceClone（Base）モデル
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base

# オプション
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --ip 127.0.0.1 --port 8000 --device auto --dtype auto
```

devbox 環境の場合:

```bash
devbox run demo:custom    # CustomVoice
devbox run demo:design    # VoiceDesign
devbox run demo:clone     # VoiceClone (Base)

# または devbox shell 内で
devbox shell
uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

ブラウザで `http://localhost:8000` を開いてください。

### Python API

#### CustomVoice

```python
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from qwen_tts.utils.device import detect_device, detect_dtype, detect_attn_implementation, setup_mps_env

setup_mps_env()
device = detect_device()

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map=device,
    dtype=detect_dtype(device),
    attn_implementation=detect_attn_implementation(device),
)

wavs, sr = model.generate_custom_voice(
    text="こんにちは、音声合成のテストです。",
    language="Japanese",
    speaker="Ono_Anna",
)
sf.write("output.wav", wavs[0], sr)
```

#### VoiceDesign

```python
wavs, sr = model.generate_voice_design(
    text="今日はいい天気ですね。",
    language="Japanese",
    instruct="明るく元気な若い女性の声で話してください。",
)
sf.write("output_design.wav", wavs[0], sr)
```

#### VoiceClone

```python
wavs, sr = model.generate_voice_clone(
    text="合成したいテキストをここに入力します。",
    language="Japanese",
    ref_audio="reference.wav",
    ref_text="参照音声のテキスト",
)
sf.write("output_clone.wav", wavs[0], sr)
```

## Whisper 自動文字起こし

VoiceClone モードでは、参照音声のテキストが必要です。Web UI の「Whisper 文字起こし」タブで自動生成できます。

```bash
# Whisper サポートをインストール
uv sync --extra whisper
```

対応モデル: `tiny`, `base`, `small`, `medium`, `large-v3`

> **Note:** faster-whisper は MPS 非対応のため CPU で動作します。

## トラブルシューティング

### MPS 関連

**「MPS backend out of memory」エラー**
- 0.6B モデルに切り替えてください
- `--dtype float32` で実行してください
- 他のメモリ消費アプリを閉じてください

**MPS フォールバック警告**
```
UserWarning: MPS: no support for ... falling back to CPU
```
これは正常です。`PYTORCH_ENABLE_MPS_FALLBACK=1` により未サポートの操作は CPU にフォールバックされます。

### FlashAttention

```
FlashAttention is not available
```
macOS では FlashAttention-2 は使えません。本フォークでは自動的に SDPA (Scaled Dot-Product Attention) を使用します。

### sox が見つからない

```bash
brew install sox
# または devbox を使用
```

### モデルのダウンロード

モデルは初回実行時に自動ダウンロードされます。手動でダウンロードする場合:

```bash
uv run huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
uv run huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## Fine-tuning

macOS (MPS) での Fine-tuning は**実験的**です。

```bash
# データ準備
uv run python finetuning/prepare_data.py \
  --device auto \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl

# SFT 実行
uv run python finetuning/sft_12hz.py \
  --device auto \
  --no-flash-attn \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 --lr 2e-6 --num_epochs 3
```

> **Warning:** MPS では mixed precision (bf16) が無効になるため、トレーニングは CUDA 環境と比べて大幅に遅くなります。本番のトレーニングには CUDA 環境を推奨します。

## 元プロジェクトとの差分

| 変更点 | 詳細 |
|---|---|
| デバイス自動検出 | `qwen_tts/utils/device.py` — MPS > CUDA > CPU |
| FlashAttention → SDPA | macOS では自動的に SDPA を使用 |
| 日本語 GUI | Gradio デモの全 UI を日本語化 |
| Whisper 統合 | `faster-whisper` によるオプション文字起こし |
| 開発環境 | devbox + uv によるセットアップ自動化 |
| sox 依存 | Python パッケージから除外（システム依存として管理） |

## ライセンス

Apache License 2.0 — 詳細は [LICENSE](LICENSE) を参照

### 謝辞

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba Qwen Team
- [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — hiroki-abe-58
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — SYSTRAN
