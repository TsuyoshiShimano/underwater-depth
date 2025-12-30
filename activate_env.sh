#!/bin/bash
# MASt3R環境アクティベーションスクリプト

# スクリプトのディレクトリに移動
cd "$(dirname "$0")"

# 仮想環境の存在確認
if [ ! -d "venv" ]; then
    echo "❌ エラー: 仮想環境 'venv' が見つかりません"
    echo "環境構築を行ってください: bash setup_venv.sh"
    exit 1
fi

# 仮想環境をアクティベート
source venv/bin/activate

echo ""
echo "======================================"
echo "MASt3R 環境がアクティベートされました"
echo "======================================"
echo ""
echo "Python: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""
echo "使用可能なコマンド:"
echo "  - 学習: bash train_blender.sh"
echo "  - 推論: python inference_blender.py"
echo "  - 環境を終了: deactivate"
echo ""


