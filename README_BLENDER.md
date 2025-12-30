# MASt3R学習・推論ガイド（Blenderデータセット用）

このドキュメントでは、MASt3Rを基にしたBlenderデータセットの学習と推論の方法を説明します。

## 概要

- **学習データ**: `/workspace/underwater/dust3r/data/blender_processed`
- **深度不明点の処理**: `.exr`ファイルで深度不明点を`1e10`で出力しているため、それらの点は除外されます
- **推論データ**: `/workspace/dust3r/test/5a489fb1c7dab83a7d7b1070/down_caustics`

## データセット

`BlenderProcessed`データセットクラスは、深度不明点（`1e10`）を自動的に除外します。`_get_views`メソッドで深度マップを読み込んだ後、`>= 1e9`の値をNaNに設定し、後続の処理で`valid_mask`によって除外されます。

## 学習コマンド

### 基本的な学習コマンド

```bash
cd /workspace/underwater/mast3r
bash train_blender.sh
```

### 詳細な学習コマンド

```bash
torchrun --nproc_per_node=1 train.py \
  --model "AsymmetricMASt3R(patch_embed_cls='ManyAR_PatchEmbed', enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True)" \
  --pretrained naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --auto_download_pretrained \
  --train_dataset "BlenderProcessed(ROOT='/workspace/underwater/dust3r/data/blender_processed', split='train', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop='auto', aug_monocular=0.005, aug_rot90='diff', transform=ColorJitter, n_corres=8192, nneg=0.5)" \
  --test_dataset "BlenderProcessed(ROOT='/workspace/underwater/dust3r/data/blender_processed', split='val', resolution=(512,384), n_corres=1024, seed=777)" \
  --train_criterion "ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', temperature=0.05), negatives_padding=0, blocksize=8192), alpha=10.0, confmode='mean')" \
  --test_criterion "Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + -1.*MatchingLoss(APLoss(nq='torch', fp=torch.float16), negatives_padding=12288)" \
  --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 4 --accum_iter 4 \
  --save_freq 1 --keep_freq 5 --eval_freq 1 --disable_cudnn_benchmark \
  --output_dir "checkpoints/mast3r_blender"
```

### パラメータ説明

- `--model`: MASt3Rモデルの設定
- `--pretrained`: 事前学習済みモデル（MASt3Rの公開モデル）
- `--train_dataset`: 学習データセット（BlenderProcessed）
- `--test_dataset`: 検証データセット（BlenderProcessed）
- `--train_criterion`: 学習時の損失関数
- `--test_criterion`: 評価時の損失関数
- `--lr`: 学習率
- `--epochs`: エポック数
- `--batch_size`: バッチサイズ
- `--output_dir`: 出力ディレクトリ

## 推論コマンド

### 基本的な推論コマンド

```bash
cd /workspace/underwater/mast3r
python inference_blender.py
```

### 推論スクリプトの動作

1. 事前学習済みMASt3Rモデルを読み込み
2. `/workspace/dust3r/test/5a489fb1c7dab83a7d7b1070/down_caustics`から画像を読み込み
3. 画像ペアを作成（complete graph）
4. 推論を実行
5. 深度マップと信頼度マップを保存

### 出力ファイル

- `output_depth1.png`: 1つ目のビューの深度マップ
- `output_depth2.png`: 2つ目のビューの深度マップ
- `output_conf1.png`: 1つ目のビューの信頼度マップ

## データセット構造

### 学習データ

```
/workspace/underwater/dust3r/data/blender_processed/
├── {scene_id}/
│   ├── down/
│   │   ├── {view_index:08d}.jpg
│   │   ├── {view_index:08d}.exr
│   │   └── {view_index:08d}.npz
│   └── up/
│       ├── {view_index:08d}.jpg
│       ├── {view_index:08d}.exr
│       └── {view_index:08d}.npz
└── blender_pairs.npy
```

### 推論データ

```
/workspace/dust3r/test/5a489fb1c7dab83a7d7b1070/down_caustics/
├── view_0000/
│   └── color/
│       └── frame_0000_caustics_0001.png
├── view_0001/
│   └── color/
│       └── frame_0001_caustics_0001.png
└── ...
```

## 深度不明点の処理

`.exr`ファイルで深度不明点が`1e10`で出力されている場合、`BlenderProcessed`データセットクラスが自動的に処理します：

1. 深度マップを読み込み
2. `>= 1e9`の値をNaNに設定
3. `valid_mask`で除外（後続の処理で使用されない）

これにより、深度不明点が学習や推論に影響を与えません。

## 注意事項

1. **GPUメモリ**: バッチサイズはGPUメモリに応じて調整してください
2. **データパス**: データパスは環境に応じて変更してください
3. **事前学習モデル**: `--auto_download_pretrained`を使用すると、事前学習モデルが自動的にダウンロードされます

## トラブルシューティング

### エラー: "No images found!"

- データパスが正しいか確認してください
- 画像ファイルの拡張子が`.png`または`.jpg`であることを確認してください

### エラー: "CUDA out of memory"

- `--batch_size`を小さくしてください
- `--accum_iter`を大きくしてください

### エラー: "Invalid depth values"

- `.exr`ファイルが正しく読み込めているか確認してください
- 深度不明点が`1e10`で出力されているか確認してください


