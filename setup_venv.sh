#!/bin/bash
# MASt3R環境構築スクリプト（venv使用版）
# condaが不要な環境構築手順

set -e  # エラー時に停止

echo "======================================"
echo "MASt3R 環境構築スクリプト（venv版）"
echo "======================================"
echo ""

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# Python 3.11が利用可能か確認（推奨）または3.12を使用
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
    echo "✅ Python 3.11 が見つかりました"
elif command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
    echo "⚠️  Python 3.12 を使用します（3.11が推奨ですが動作します）"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "⚠️  Python $PYTHON_VERSION を使用します（3.11が推奨）"
else
    echo "❌ エラー: Python3 が見つかりません"
    exit 1
fi

echo "使用するPython: $($PYTHON_CMD --version)"
echo ""

# 仮想環境の作成
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  既存の仮想環境 '$VENV_DIR' が見つかりました"
    read -p "削除して再作成しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "✅ 既存の環境を削除しました"
    else
        echo "既存の環境を使用します"
        SKIP_VENV=true
    fi
fi

if [ "$SKIP_VENV" != "true" ]; then
    echo "📦 仮想環境を作成しています..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "✅ 仮想環境を作成しました"
fi

# 仮想環境をアクティベート
echo "🔧 仮想環境をアクティベートしています..."
source "$VENV_DIR/bin/activate"

# pipをアップグレード
echo "📦 pipをアップグレードしています..."
pip install --upgrade pip setuptools wheel

# CMakeのインストール確認
if ! command -v cmake &> /dev/null; then
    echo "⚠️  CMakeがインストールされていません。システムパッケージマネージャーでインストールしてください："
    echo "   Ubuntu/Debian: sudo apt-get install cmake"
    echo "   続行しますか？ (CMakeが必要な場合があります)"
    read -p "続行しますか？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# PyTorchのインストール
echo "📦 PyTorchをインストールしています..."
# CUDA 12.1用のPyTorchをインストール（システムに合わせて調整してください）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# または、システムのCUDAバージョンに応じて以下を選択：
# CUDA 12.4: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.8: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only: pip install torch torchvision

# requirements.txtのインストール
if [ -f "requirements.txt" ]; then
    echo "📦 requirements.txtをインストールしています..."
    pip install -r requirements.txt
else
    echo "⚠️  requirements.txtが見つかりません"
fi

# dust3r/requirements.txtのインストール
if [ -f "dust3r/requirements.txt" ]; then
    echo "📦 dust3r/requirements.txtをインストールしています..."
    pip install -r dust3r/requirements.txt
else
    echo "⚠️  dust3r/requirements.txtが見つかりません（サブモジュールが初期化されていない可能性があります）"
    echo "   親ディレクトリのdust3rを使用します..."
    if [ -f "../dust3r/requirements.txt" ]; then
        pip install -r ../dust3r/requirements.txt
        echo "✅ ../dust3r/requirements.txtをインストールしました"
    fi
fi

# オプショナルパッケージ（visloc.py用など）
if [ -f "dust3r/requirements_optional.txt" ]; then
    echo "📦 dust3r/requirements_optional.txtをインストールしています..."
    pip install -r dust3r/requirements_optional.txt
elif [ -f "../dust3r/requirements_optional.txt" ]; then
    echo "📦 ../dust3r/requirements_optional.txtをインストールしています..."
    pip install -r ../dust3r/requirements_optional.txt
fi

# ASMKのインストール（オプション、必要に応じて）
echo ""
read -p "ASMKをインストールしますか？（画像検索に使用）(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 ASMKをインストールしています..."
    pip install cython
    if [ ! -d "asmk" ]; then
        git clone https://github.com/jenicek/asmk
    fi
    cd asmk/cython/
    cythonize *.pyx
    cd ..
    pip install .
    cd ..
    echo "✅ ASMKをインストールしました"
fi

# CuRoPE CUDAカーネルのコンパイル（オプション、高速化のため）
echo ""
read -p "CuRoPE CUDAカーネルをコンパイルしますか？（高速化のため、推奨）(Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "🔧 CuRoPE CUDAカーネルをコンパイルしています..."
    if [ -d "dust3r/croco/models/curope" ]; then
        cd dust3r/croco/models/curope/
        python setup.py build_ext --inplace
        cd ../../../../
        echo "✅ CuRoPEカーネルをコンパイルしました"
    elif [ -d "../dust3r/croco/models/curope" ]; then
        cd ../dust3r/croco/models/curope/
        python setup.py build_ext --inplace
        cd ../../../../mast3r/
        echo "✅ CuRoPEカーネルをコンパイルしました"
    else
        echo "⚠️  CuRoPEディレクトリが見つかりません"
    fi
fi

echo ""
echo "======================================"
echo "✅ 環境構築が完了しました！"
echo "======================================"
echo ""
echo "📝 環境をアクティベートする方法："
echo "   source venv/bin/activate"
echo ""
echo "📝 環境を非アクティベートする方法："
echo "   deactivate"
echo ""
echo "📝 環境の確認："
echo "   python --version"
echo "   python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo ""
echo "次のステップ："
echo "1. 事前学習済みモデルをダウンロード:"
echo "   mkdir -p checkpoints"
echo "   wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/"
echo ""
echo "2. 学習を開始:"
echo "   bash train_blender.sh"
echo ""

