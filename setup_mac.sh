#!/usr/bin/env bash
set -euo pipefail

echo "=== Qwen3-TTS-JP-mac セットアップ ==="

# macOS + Apple Silicon チェック
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Error: このスクリプトは macOS 専用です。"
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    echo "Warning: Apple Silicon (arm64) ではありません (検出: $ARCH)"
    echo "MPS アクセラレーションは利用できない可能性があります。"
fi

# devbox が利用可能な場合
if command -v devbox &> /dev/null; then
    echo "devbox が見つかりました。devbox でセットアップします..."
    devbox install
    devbox run setup
    echo ""
    echo "=== セットアップ完了 ==="
    echo "使い方:"
    echo "  devbox shell                    # 開発環境に入る"
    echo "  devbox run check-mps            # MPS 利用可能か確認"
    echo "  devbox run demo:custom          # CustomVoice デモ起動"
    echo "  devbox run demo:design          # VoiceDesign デモ起動"
    echo "  devbox run demo:clone           # VoiceClone デモ起動"
    exit 0
fi

echo "devbox が見つかりませんでした。手動セットアップを行います..."

# Homebrew チェック
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew がインストールされていません。"
    echo "インストール: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# システム依存パッケージ
echo "sox, ffmpeg, git-lfs をインストール中..."
for pkg in sox ffmpeg git-lfs; do
    if ! brew install "$pkg"; then
        echo "Warning: $pkg のインストールに失敗しました。手動でインストールしてください: brew install $pkg"
    fi
done

# Git LFS の初期化
git lfs install

# uv チェック
if ! command -v uv &> /dev/null; then
    echo "uv をインストール中..."
    if ! brew install uv 2>/dev/null; then
        echo "Homebrew での uv インストールに失敗しました。公式インストーラーを使用します..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Python 環境セットアップ
echo "Python 依存関係をインストール中..."
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv sync --extra whisper

# MPS チェック
echo ""
echo "MPS バックエンド確認中..."
uv run python -c "
import torch
mps = torch.backends.mps.is_available()
print(f'  MPS 利用可能: {mps}')
if mps:
    print(f'  MPS デバイス: {torch.device(\"mps\")}')
from qwen_tts.utils.device import detect_device
print(f'  自動検出デバイス: {detect_device()}')
"

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "次のステップ:"
echo "  1. モデルをダウンロード:"
echo "     uv run huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
echo ""
echo "  2. デモを起動:"
echo "     uv run qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
echo ""
echo "  3. ブラウザで http://localhost:8000 を開く"
