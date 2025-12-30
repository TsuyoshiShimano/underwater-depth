#!/bin/bash
# tmuxセッションで学習を実行するスクリプト

SESSION_NAME="mast3r_training"
SCRIPT_DIR="/workspace/underwater/mast3r"

# 既存のセッションがあるか確認
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "既存のセッション '$SESSION_NAME' が見つかりました。"
    echo "アタッチする場合は: tmux attach -t $SESSION_NAME"
    echo "削除する場合は: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# 新しいセッションを作成してコマンドを実行
cd "$SCRIPT_DIR"
tmux new-session -d -s "$SESSION_NAME" "
    cd $SCRIPT_DIR && \
    torchrun --nproc_per_node=1 train.py \
    --model \"AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R', predict_underwater_params=True, two_confs=True, head_type='catmlp+dpt', img_size=(512, 512), dec_depth=12, depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_num_heads=12)\" \
    --pretrained naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
    --auto_download_pretrained \
    --freeze encoder_decoder \
    --freeze_local_features_head \
    --train_dataset \"BlenderProcessed(ROOT='/workspace/underwater/dust3r/data/blender_processed', split='train', resolution=(512, 384), n_corres=8192, nneg=0.5)\" \
    --test_dataset \"BlenderProcessed(ROOT='/workspace/underwater/dust3r/data/blender_processed', split='val', resolution=(512,384), n_corres=1024, seed=777)\" \
    --train_criterion \"ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + 0.1*LambertBeerLossOnly(LambertBeerColorLoss(similarity_threshold=0.99, use_predicted_params=True, max_matches=20, initial_samples=500, max_iterations=10, reduction='mean'), lambert_beer_weight=1.0, spatial_reg=SpatialSmoothnessRegularizer(weight=0.01), spatial_reg_weight=0.01)\" \
    --test_criterion \"Regr3D_ScaleShiftInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)\" \
    --lr 0.0001 \
    --min_lr 1e-06 \
    --warmup_epochs 1 \
    --epochs 10 \
    --batch_size 1 \
    --accum_iter 8 \
    --save_freq 1 \
    --keep_freq 5 \
    --eval_freq 1 \
    --disable_cudnn_benchmark \
    --output_dir \"output_mast3r_ver4\" \
    ; bash
"

echo "tmuxセッション '$SESSION_NAME' を開始しました。"
echo ""
echo "セッションにアタッチ: tmux attach -t $SESSION_NAME"
echo "セッションからデタッチ: Ctrl+B を押してから D を押す"
echo "セッションの状態を確認: tmux ls"
echo "セッションを削除: tmux kill-session -t $SESSION_NAME"


