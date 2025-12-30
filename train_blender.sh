#!/bin/bash
# MASt3R training script for Blender processed data
# Filters out invalid depth points (1e10) from .exr files

cd "$(dirname "$0")"

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

