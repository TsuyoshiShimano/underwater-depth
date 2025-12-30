#!/usr/bin/env python3
"""
MASt3R改良版の適用と可視化スクリプト

2つのチェックポイントファイル（ver3とver4）を用いて、
テスト画像に対してMASt3Rを適用し、深度マップと特徴点対応を可視化します。
モデルとパラメータが一致しない場合はエラーを出力します。
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
import cv2

# MASt3R関連のインポート
from mast3r.model import load_model
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.utils.image import load_images


def compare_model_structures(model1, model2, verbose=True):
    """
    2つのモデルの構造を比較し、不一致があればエラーを出力
    
    Args:
        model1: 最初のモデル
        model2: 2番目のモデル
        verbose: 詳細情報を出力するか
        
    Returns:
        bool: 構造が一致する場合True、不一致の場合False
    """
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    
    # キーの違いをチェック
    missing_in_2 = keys1 - keys2
    missing_in_1 = keys2 - keys1
    
    if missing_in_2:
        print(f"ERROR: Keys in model1 but not in model2: {missing_in_2}")
        return False
    
    if missing_in_1:
        print(f"ERROR: Keys in model2 but not in model1: {missing_in_1}")
        return False
    
    # 各パラメータの形状をチェック
    mismatches = []
    for key in keys1:
        shape1 = state1[key].shape
        shape2 = state2[key].shape
        if shape1 != shape2:
            mismatches.append((key, shape1, shape2))
            if verbose:
                print(f"ERROR: Shape mismatch for '{key}': model1={shape1}, model2={shape2}")
    
    if mismatches:
        return False
    
    # パラメータの値の違いもチェック（オプション）
    if verbose:
        print("Model structures match. Checking parameter values...")
        max_diff = 0.0
        for key in keys1:
            if state1[key].dtype.is_floating_point:
                diff = torch.abs(state1[key] - state2[key]).max().item()
                max_diff = max(max_diff, diff)
        print(f"Maximum parameter difference: {max_diff:.6f}")
    
    return True


def load_checkpoint_with_validation(checkpoint_path, device, verbose=True):
    """
    チェックポイントを読み込み、モデル構造を検証
    
    Args:
        checkpoint_path: チェックポイントファイルのパス
        device: デバイス
        verbose: 詳細情報を出力するか
        
    Returns:
        model: 読み込まれたモデル
        checkpoint_info: チェックポイント情報
    """
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # チェックポイントを読み込んで構造を確認
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'args' not in ckpt:
        raise ValueError(f"Checkpoint missing 'args' key: {checkpoint_path}")
    if 'model' not in ckpt:
        raise ValueError(f"Checkpoint missing 'model' key: {checkpoint_path}")
    
    # モデルを読み込む（load_modelはstrict=Falseを使用するため、別途検証が必要）
    model = load_model(checkpoint_path, device, verbose=verbose)
    model.eval()
    
    # モデルにパラメータが正しく読み込まれたか確認（strict=Trueで試す）
    try:
        # 一時的にstrict=Trueで読み込みを試みる
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        if 'landscape_only' not in args:
            args = args[:-1] + ', landscape_only=False)'
        else:
            args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
        
        # head_type='catmlp+dpt'の場合、output_modeが指定されていなければ'pts3d+desc24'を追加
        if "head_type='catmlp+dpt'" in args or 'head_type="catmlp+dpt"' in args:
            if 'output_mode' not in args:
                args = args[:-1] + ", output_mode='pts3d+desc24')"
        
        # eval()で必要なクラスと定数をインポート
        from mast3r.model import AsymmetricMASt3R  # noqa
        import math
        inf = float('inf')  # eval()で使用するため
        test_model = eval(args)
        missing_keys, unexpected_keys = test_model.load_state_dict(ckpt['model'], strict=True)
        
        if missing_keys:
            raise RuntimeError(
                f"ERROR: Model structure mismatch! Missing keys in checkpoint: {missing_keys}\n"
                f"This means the checkpoint parameters don't match the model architecture."
            )
        if unexpected_keys:
            raise RuntimeError(
                f"ERROR: Model structure mismatch! Unexpected keys in checkpoint: {unexpected_keys}\n"
                f"This means the checkpoint contains parameters not in the model architecture."
            )
    except RuntimeError as e:
        # strict=Trueでの読み込み失敗は意図的な検証なので、エラーメッセージをそのまま伝播
        raise
    
    checkpoint_info = {
        'path': checkpoint_path,
        'model_str': args,
        'keys': list(ckpt['model'].keys()),
    }
    
    if verbose:
        print(f"✅ Successfully loaded checkpoint: {checkpoint_path}")
        print(f"   Model parameters: {len(ckpt['model'])} keys")
    
    return model, checkpoint_info


def find_test_images(test_dir):
    """
    テストディレクトリ内の画像を探索
    
    Args:
        test_dir: テストディレクトリのパス
        
    Returns:
        scenes: シーンごとの画像パスの辞書
    """
    test_dir = Path(test_dir)
    scenes = defaultdict(lambda: defaultdict(list))
    
    # 各シーンディレクトリを探索
    for scene_dir in sorted(test_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        
        # 条件（down_clear, down_caustics, up_clear, up_caustics）ごとに探索
        for condition_dir in sorted(scene_dir.iterdir()):
            if not condition_dir.is_dir():
                continue
            
            condition = condition_dir.name
            
            # view_XXXX ディレクトリを探索
            for view_dir in sorted(condition_dir.iterdir()):
                if not view_dir.is_dir() or not view_dir.name.startswith('view_'):
                    continue
                
                view_id = view_dir.name
                
                # colorディレクトリ内の画像を探す
                color_dir = view_dir / 'color'
                if color_dir.exists():
                    for img_file in sorted(color_dir.glob('*.png')):
                        scenes[scene_id][condition].append({
                            'path': img_file,
                            'view_id': view_id,
                            'scene_id': scene_id,
                            'condition': condition,
                        })
    
    return scenes


def fuse_depth_maps(depth_maps, conf_maps=None, tau=0.3):
    """
    複数の深度マップを信頼度で重み付け融合
    evaluate_scenes.pyのfuse_depth_mapsを参考
    
    Args:
        depth_maps: 深度マップのリスト [(H,W), ...]
        conf_maps: 信頼度マップのリスト [(H,W), ...] or None
        tau: ソフトマックス温度（小さいほど高信頼に集中）
    
    Returns:
        fused_depth: 融合された深度マップ (H, W)
        fused_conf: 融合された信頼度マップ (H, W) or None
    """
    if not depth_maps:
        return None, None
    
    import cv2
    
    D = np.stack([d.astype(np.float32) for d in depth_maps], axis=0)  # (N,H,W)
    valid = np.isfinite(D) & (D > 0)
    
    # デバッグ: 個々の深度マップの統計情報を出力（最初の1回のみ）
    if not hasattr(fuse_depth_maps, '_debug_printed'):
        print('\n=== Debug: Individual Depth Maps Before Fusion ===')
        for i, depth_map in enumerate(depth_maps):
            nan_count = np.sum(np.isnan(depth_map))
            inf_count = np.sum(np.isinf(depth_map))
            valid_count = np.sum(np.isfinite(depth_map))
            negative_count = np.sum((depth_map < 0) & np.isfinite(depth_map))
            zero_count = np.sum((depth_map == 0) & np.isfinite(depth_map))
            positive_count = np.sum((depth_map > 0) & np.isfinite(depth_map))
            
            print(f'Depth map {i}:')
            print(f'  Shape: {depth_map.shape}')
            print(f'  NaN: {nan_count} ({nan_count/depth_map.size*100:.1f}%)')
            print(f'  Inf: {inf_count} ({inf_count/depth_map.size*100:.1f}%)')
            print(f'  Valid: {valid_count} ({valid_count/depth_map.size*100:.1f}%)')
            print(f'    - Negative: {negative_count}')
            print(f'    - Zero: {zero_count}')
            print(f'    - Positive: {positive_count}')
            if valid_count > 0:
                valid_depth = depth_map[np.isfinite(depth_map)]
                print(f'  Range: {np.min(valid_depth):.6f} ~ {np.max(valid_depth):.6f}')
        print()
        fuse_depth_maps._debug_printed = True
    
    fused_conf = None
    
    if conf_maps is not None and len(conf_maps) == len(depth_maps):
        H, W = depth_maps[0].shape
        C = []
        for c in conf_maps:
            if c is None:
                C.append(np.ones((H, W), np.float32))
            else:
                if c.shape != (H, W):
                    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
                C.append(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        C = np.stack(C, axis=0)  # (N,H,W)
        
        # ピクセルごとのソフトマックス正規化
        W_exp = np.exp(C / tau) * valid  # 無効画素は重み0
        W_sum = W_exp.sum(axis=0, keepdims=True) + 1e-9
        W_norm = W_exp / W_sum  # ピクセルごとに合計1
        
        # 無効な値を0に置き換えてから計算（NaNの伝播を防ぐ）
        D_valid = D.copy()
        D_valid[~valid] = 0.0
        
        # 加重平均で融合
        out = (D_valid * W_norm).sum(axis=0)
        
        # 対応する信頼度マップも同じ重みで加重平均して融合
        fused_conf = (C * W_norm).sum(axis=0)
    else:
        # 信頼度がない場合はロバストにメディアン融合
        D[~valid] = np.nan
        out = np.nanmedian(D, axis=0)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    return out.astype(np.float32), (fused_conf.astype(np.float32) if fused_conf is not None else None)


def extract_depth_from_pts3d(pts3d):
    """
    pts3dから深度（z座標）を抽出
    evaluate_scenes.pyのextract_depth_from_predictionを参考
    
    Args:
        pts3d: 3D points (H, W, 3) or (B, H, W, 3) or (H, W, 3)など様々な形式に対応
    
    Returns:
        depth: 深度マップ (H, W)
    """
    if isinstance(pts3d, torch.Tensor):
        pts3d = pts3d.cpu().numpy()
    
    # 軸順序に対応してz座標を取得
    if pts3d.ndim == 4:
        if pts3d.shape[3] == 3:
            # (B, H, W, 3) format
            depth = pts3d[0, :, :, 2]  # z-coordinate
        elif pts3d.shape[1] == 3:
            # (B, 3, H, W) format
            depth = pts3d[0, 2, :, :]  # z-coordinate
        else:
            raise ValueError(f"Unsupported pts3d shape: {pts3d.shape}")
    elif pts3d.ndim == 3:
        if pts3d.shape[2] == 3:
            # (H, W, 3) format
            depth = pts3d[:, :, 2]  # z-coordinate
        elif pts3d.shape[0] == 3:
            # (3, H, W) format
            depth = pts3d[2, :, :]  # z-coordinate
        else:
            raise ValueError(f"Unsupported pts3d shape: {pts3d.shape}")
    else:
        raise ValueError(f"Unsupported pts3d shape: {pts3d.shape}")
    
    return depth


def visualize_depth_map(depth_map, original_img, save_path, title="Depth Map"):
    """
    深度マップを可視化
    evaluate_scenes.pyと3d_reconstruction.pyの方法を参考
    
    Args:
        depth_map: 深度マップ (H, W)
        original_img: 元の画像 (H, W, 3) numpy array
        save_path: 保存パス
        title: タイトル
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 元の画像を表示
    if isinstance(original_img, Image.Image):
        axes[0].imshow(original_img)
    else:
        # numpy配列の場合、値の範囲を確認
        if original_img.max() > 1.0:
            original_img = original_img / 255.0
        axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 深度マップの可視化
    # 有効な深度値のみを使用（NaNとInfを除外、負の値も含める）
    valid_mask = np.isfinite(depth_map)
    if valid_mask.any():
        valid_depth = depth_map[valid_mask]
        # パーセンタイルベースで正規化（外れ値の影響を減らす）
        # 0.5-99.9パーセンタイルを使用して、より多くの深度値をカバー
        vmin_percentile = np.percentile(valid_depth, 0.5)
        vmax_percentile = np.percentile(valid_depth, 99.9)
        # vmaxを実際の最大値に設定して、白くならないようにする
        vmin = vmin_percentile
        vmax = np.max(valid_depth)  # 実際の最大値を使用
        # 範囲外の値は最も近い値（vminまたはvmax）にクリップ
        # 無効な値（NaNやInf）はマスクのまま残す
        depth_map_clipped = depth_map.copy()
        depth_map_clipped[valid_mask] = np.clip(depth_map[valid_mask], vmin, vmax)
    else:
        vmin, vmax = 0, 1
        depth_map_clipped = depth_map
        print(f"Warning: No valid depth values found for {title}")
    
    # viridisカラーマップを使用（evaluate_scenes.pyと3d_reconstruction.pyの方法）
    # 無効な値のみをマスクして表示（範囲外の値はクリップ済み）
    # マスクされた部分を黒で表示するため、特別な値を設定
    depth_display = depth_map_clipped.copy()
    depth_display[~valid_mask] = vmin - (vmax - vmin) * 0.1  # vminより小さい値に設定
    depth_display = np.ma.masked_array(depth_display, mask=~valid_mask)
    
    # カラーマップを拡張して、マスクされた部分を黒で表示
    # viridisカラーマップの最初の色（紫/黒）を背景色として使用
    from matplotlib.colors import ListedColormap
    viridis = plt.cm.viridis
    colors = viridis(np.linspace(0, 1, 256))
    # マスクされた部分用に黒を追加
    colors_with_bg = np.vstack([np.array([0, 0, 0, 1]), colors])  # 黒を最初に追加
    cmap_with_bg = ListedColormap(colors_with_bg)
    
    im = axes[1].imshow(depth_display, cmap='viridis', vmin=vmin, vmax=vmax, 
                        interpolation='nearest')
    # マスクされた部分の背景色を黒に設定
    axes[1].set_facecolor('black')
    axes[1].set_title(title)
    axes[1].axis('off')
    
    # カラーバーを追加
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth', rotation=270, labelpad=15)
    
    # 統計情報をタイトルに追加
    if valid_mask.any():
        depth_mean = valid_depth.mean()
        depth_std = valid_depth.std()
        axes[1].set_title(f'{title}\n(mean={depth_mean:.3f}, std={depth_std:.3f}, range=[{vmin:.3f}, {vmax:.3f}])', 
                         fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def visualize_matches(img1, img2, pts1, pts2, confidences, save_path, max_matches=50):
    """
    特徴点対応を可視化
    
    Args:
        img1: 画像1 (H, W, 3) or PIL Image
        img2: 画像2 (H, W, 3) or PIL Image
        pts1: 画像1の特徴点 (N, 2) [x, y]
        pts2: 画像2の特徴点 (N, 2) [x, y]
        confidences: 対応の信頼度 (N,)
        save_path: 保存パス
        max_matches: 可視化する最大マッチ数
    """
    # PIL Imageをnumpy配列に変換
    if isinstance(img1, Image.Image):
        img1_np = np.array(img1)
    else:
        img1_np = img1
    
    if isinstance(img2, Image.Image):
        img2_np = np.array(img2)
    else:
        img2_np = img2
    
    H1, W1 = img1_np.shape[:2]
    H2, W2 = img2_np.shape[:2]
    
    # マッチ数を制限
    if len(pts1) > max_matches:
        indices = np.linspace(0, len(pts1) - 1, max_matches, dtype=int)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
        confidences = confidences[indices]
    
    # 画像を横に並べる
    H = max(H1, H2)
    combined = np.ones((H, W1 + W2, 3), dtype=np.float32)
    combined[:H1, :W1] = img1_np.astype(np.float32) / 255.0 if img1_np.max() > 1.0 else img1_np
    combined[:H2, W1:] = img2_np.astype(np.float32) / 255.0 if img2_np.max() > 1.0 else img2_np
    
    # 可視化
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(combined)
    ax.axis('off')
    
    # マッチを描画
    cmap = plt.cm.RdYlGn
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        conf = confidences[i] if i < len(confidences) else 1.0
        
        # 信頼度に基づいて色付け（緑=高信頼度、赤=低信頼度）
        color = cmap(np.clip(conf, 0, 1))
        
        # 線を描画
        ax.plot([x1, x2 + W1], [y1, y2], color=color, linewidth=1.5, alpha=0.7)
        
        # 点を描画
        ax.plot(x1, y1, 'o', color=color, markersize=5)
        ax.plot(x2 + W1, y2, 'o', color=color, markersize=5)
    
    ax.set_title(f'Feature Correspondences (N={len(pts1)} matches, '
                f'Conf: min={confidences.min():.3f}, mean={confidences.mean():.3f}, max={confidences.max():.3f})', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def predict_depth_with_supporting_images(model, ref_img_path, supporting_img_paths, device, image_size=512):
    """
    参照画像と複数の補助画像で深度を予測
    evaluate_scenes.pyのpredict_with_dust3rを参考
    
    Args:
        model: MASt3Rモデル
        ref_img_path: 参照画像のパス
        supporting_img_paths: 補助画像のパスのリスト
        device: デバイス
        image_size: 画像サイズ
        
    Returns:
        results: 各補助画像とのペアで予測した深度と信頼度のリスト
    """
    results = []
    
    for supporting_img_path in supporting_img_paths:
        try:
            # 画像を読み込む
            imgs = load_images([str(ref_img_path), str(supporting_img_path)], size=image_size, verbose=False)
            
            if not imgs or len(imgs) < 2:
                continue
            
            # デバイスに移動
            for img_dict in imgs:
                img_dict['img'] = img_dict['img'].to(device)
            
            # 推論を実行
            model.eval()
            with torch.no_grad():
                output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
                pred1 = output['pred1']
            
            # 深度と信頼度を抽出
            pts3d = pred1['pts3d'][0]
            depth = extract_depth_from_pts3d(pts3d)
            
            conf = None
            if 'conf' in pred1:
                conf_tensor = pred1['conf'][0]
                if isinstance(conf_tensor, torch.Tensor):
                    conf = conf_tensor.cpu().numpy()
                else:
                    conf = conf_tensor
                if conf.ndim == 3:
                    conf = conf[0]
            
            results.append({
                'depth': depth,
                'conf': conf,
                'supporting_image': supporting_img_path
            })
        except Exception as e:
            print(f"  Warning: Error processing {ref_img_path} + {supporting_img_path}: {e}")
            continue
    
    return results


def process_reference_image(model, ref_img_path, supporting_img_paths, device, output_dir, ref_id):
    """
    参照画像を処理して、複数の補助画像との深度マップを融合して可視化
    
    Args:
        model: MASt3Rモデル
        ref_img_path: 参照画像のパス
        supporting_img_paths: 補助画像のパスのリスト
        device: デバイス
        output_dir: 出力ディレクトリ
        ref_id: 参照画像ID（ファイル名に使用）
    """
    if not supporting_img_paths:
        return 0
    
    # 各補助画像とのペアで深度を予測
    results = predict_depth_with_supporting_images(
        model, ref_img_path, supporting_img_paths, device, image_size=512
    )
    
    if not results:
        return 0
    
    # 深度マップと信頼度マップを収集
    depth_maps = [r['depth'] for r in results if r.get('depth') is not None]
    conf_maps = [r.get('conf') for r in results if r.get('depth') is not None]
    
    if not depth_maps:
        return 0
    
    # デバッグ: 個々の深度マップを確認
    if ref_id.endswith('_018'):  # 特定のIDのみ確認
        print(f'\n=== Debug: Individual Depth Maps for {ref_id} ===')
        for i, depth_map in enumerate(depth_maps):
            nan_count = np.sum(np.isnan(depth_map))
            inf_count = np.sum(np.isinf(depth_map))
            valid_count = np.sum(np.isfinite(depth_map))
            negative_count = np.sum((depth_map < 0) & np.isfinite(depth_map))
            zero_count = np.sum((depth_map == 0) & np.isfinite(depth_map))
            positive_count = np.sum((depth_map > 0) & np.isfinite(depth_map))
            
            print(f'Depth map {i}:')
            print(f'  Shape: {depth_map.shape}')
            print(f'  NaN: {nan_count} ({nan_count/depth_map.size*100:.1f}%)')
            print(f'  Inf: {inf_count} ({inf_count/depth_map.size*100:.1f}%)')
            print(f'  Valid: {valid_count} ({valid_count/depth_map.size*100:.1f}%)')
            print(f'    - Negative: {negative_count}')
            print(f'    - Zero: {zero_count}')
            print(f'    - Positive: {positive_count}')
            if valid_count > 0:
                valid_depth = depth_map[np.isfinite(depth_map)]
                print(f'  Range: {np.min(valid_depth):.6f} ~ {np.max(valid_depth):.6f}')
        print()
    
    # 深度マップを融合
    fused_depth, fused_conf = fuse_depth_maps(depth_maps, conf_maps)
    
    if fused_depth is None:
        return 0
    
    # 参照画像を読み込んで可視化用に準備
    ref_img = Image.open(ref_img_path).convert('RGB')
    ref_img_tensor = load_images([str(ref_img_path)], size=512, verbose=False)[0]['img'][0].cpu()
    ref_img_vis = ref_img_tensor.permute(1, 2, 0).numpy()
    
    # 正規化を元に戻す
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    ref_img_vis = ref_img_vis * std + mean
    ref_img_vis = np.clip(ref_img_vis, 0, 1)
    
    # 融合された深度マップを可視化
    depth_path = output_dir / f'fused_depth_{ref_id}.png'
    visualize_depth_map(fused_depth, ref_img_vis, str(depth_path), 
                       title=f'Fused Depth Map - {ref_id} (from {len(depth_maps)} pairs)')
    
    # 融合された深度マップをnumpy形式で保存
    depth_npy_path = output_dir / f'fused_depth_{ref_id}.npy'
    np.save(depth_npy_path, fused_depth)
    
    # 融合された信頼度マップも保存（存在する場合）
    if fused_conf is not None:
        conf_path = output_dir / f'fused_conf_{ref_id}.png'
        visualize_depth_map(fused_conf, ref_img_vis, str(conf_path),
                           title=f'Fused Confidence Map - {ref_id}')
        conf_npy_path = output_dir / f'fused_conf_{ref_id}.npy'
        np.save(conf_npy_path, fused_conf)
    
    # 最初の補助画像との特徴点対応も可視化（オプション）
    if len(results) > 0 and supporting_img_paths:
        first_supporting = supporting_img_paths[0]
        try:
            imgs = load_images([str(ref_img_path), str(first_supporting)], size=512, verbose=False)
            for img_dict in imgs:
                img_dict['img'] = img_dict['img'].to(device)
            
            with torch.no_grad():
                output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
                pred1 = output['pred1']
                pred2 = output['pred2']
            
            desc1 = pred1['desc'][0].detach().cpu().numpy()
            desc2 = pred2['desc'][0].detach().cpu().numpy()
            
            desc1_tensor = torch.from_numpy(desc1).to(device)
            desc2_tensor = torch.from_numpy(desc2).to(device)
            
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                desc1_tensor, desc2_tensor,
                subsample_or_initxy1=8,
                device=device,
                dist='dot',
                block_size=2**13
            )
            
            # ボーダーを無視
            H0, W0 = desc1.shape[:2]
            H1, W1 = desc2.shape[:2]
            valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & \
                                (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
            valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & \
                                (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
            valid_matches = valid_matches_im0 & valid_matches_im1
            
            matches_im0 = matches_im0[valid_matches]
            matches_im1 = matches_im1[valid_matches]
            
            if isinstance(matches_im0, torch.Tensor):
                matches_im0 = matches_im0.cpu().numpy()
            if isinstance(matches_im1, torch.Tensor):
                matches_im1 = matches_im1.cpu().numpy()
            
            if len(matches_im0) > 0:
                # 信頼度を計算
                desc1_matched = desc1[matches_im0[:, 1], matches_im0[:, 0]]
                desc2_matched = desc2[matches_im1[:, 1], matches_im1[:, 0]]
                desc1_norm = desc1_matched / (np.linalg.norm(desc1_matched, axis=-1, keepdims=True) + 1e-8)
                desc2_norm = desc2_matched / (np.linalg.norm(desc2_matched, axis=-1, keepdims=True) + 1e-8)
                similarities = np.sum(desc1_norm * desc2_norm, axis=-1)
                
                # 補助画像も可視化用に準備
                supp_img_tensor = imgs[1]['img'][0].cpu()
                supp_img_vis = supp_img_tensor.permute(1, 2, 0).numpy()
                supp_img_vis = supp_img_vis * std + mean
                supp_img_vis = np.clip(supp_img_vis, 0, 1)
                
                matches_path = output_dir / f'matches_{ref_id}_with_first_supporting.png'
                visualize_matches(ref_img_vis, supp_img_vis, matches_im0, matches_im1,
                                 similarities, str(matches_path))
        except Exception as e:
            print(f"  Warning: Could not visualize matches: {e}")
    
    return len(depth_maps)
    
    # 特徴記述子を取得
    desc1 = pred1['desc'][0].detach().cpu().numpy()  # (H, W, D)
    desc2 = pred2['desc'][0].detach().cpu().numpy()  # (H, W, D)
    
    # 推論に使用した画像を取得（リサイズ済み）
    # load_imagesが返す形式: dict(img=tensor[1, 3, H, W], true_shape=...)
    img1_tensor = imgs[0]['img'][0].cpu()  # (3, H, W)
    img2_tensor = imgs[1]['img'][0].cpu()  # (3, H, W)
    
    # 可視化用に画像をnumpy配列に変換 [H, W, 3]
    img1_vis = img1_tensor.permute(1, 2, 0).numpy()
    img2_vis = img2_tensor.permute(1, 2, 0).numpy()
    
    # 正規化を元に戻す（ImgNormが適用されているので、逆変換が必要）
    # ImgNormは mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] で正規化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img1_vis = img1_vis * std + mean
    img2_vis = img2_vis * std + mean
    img1_vis = np.clip(img1_vis, 0, 1)
    img2_vis = np.clip(img2_vis, 0, 1)
    
    # 特徴点対応を計算
    desc1_tensor = torch.from_numpy(desc1).to(device)
    desc2_tensor = torch.from_numpy(desc2).to(device)
    
    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1_tensor, desc2_tensor,
        subsample_or_initxy1=8,
        device=device,
        dist='dot',
        block_size=2**13
    )
    
    # ボーダーを無視
    H0, W0 = desc1.shape[:2]
    H1, W1 = desc2.shape[:2]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & \
                        (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & \
                        (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
    valid_matches = valid_matches_im0 & valid_matches_im1
    
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]
    
    # numpy配列に変換
    if isinstance(matches_im0, torch.Tensor):
        matches_im0 = matches_im0.cpu().numpy()
    if isinstance(matches_im1, torch.Tensor):
        matches_im1 = matches_im1.cpu().numpy()
    
    # 信頼度を計算（簡易版：記述子の類似度）
    if len(matches_im0) > 0:
        # マッチした位置の記述子を取得
        desc1_matched = desc1[matches_im0[:, 1], matches_im0[:, 0]]  # (N, D)
        desc2_matched = desc2[matches_im1[:, 1], matches_im1[:, 0]]  # (N, D)
        
        # コサイン類似度
        desc1_norm = desc1_matched / (np.linalg.norm(desc1_matched, axis=-1, keepdims=True) + 1e-8)
        desc2_norm = desc2_matched / (np.linalg.norm(desc2_matched, axis=-1, keepdims=True) + 1e-8)
        similarities = np.sum(desc1_norm * desc2_norm, axis=-1)
    else:
        similarities = np.array([])
    
    # 深度マップを可視化
    depth1_path = output_dir / f'depth_map_img1_{pair_id}.png'
    depth2_path = output_dir / f'depth_map_img2_{pair_id}.png'
    visualize_depth_map(depth1, img1_vis, str(depth1_path), title=f'Depth Map - Image 1 ({pair_id})')
    visualize_depth_map(depth2, img2_vis, str(depth2_path), title=f'Depth Map - Image 2 ({pair_id})')
    
    # 特徴点対応を可視化
    if len(matches_im0) > 0:
        matches_path = output_dir / f'matches_{pair_id}.png'
        visualize_matches(
            img1_vis, img2_vis,
            matches_im0, matches_im1,
            similarities,
            str(matches_path)
        )
    
    return len(matches_im0)


def main():
    parser = argparse.ArgumentParser(
        description='Apply MASt3R models to test images and visualize depth maps and feature correspondences',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--checkpoint_ver3', type=str, required=True,
                       help='Path to checkpoint-best.pth for version 3')
    parser.add_argument('--checkpoint_ver4', type=str, required=True,
                       help='Path to checkpoint-best.pth for version 4')
    parser.add_argument('--test_dir', type=str, default='/workspace/underwater/dust3r/test',
                       help='Path to test directory')
    parser.add_argument('--output_dir', type=str, default='output_mast3r_test',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--use_ver3', action='store_true',
                       help='Use version 3 model for inference (default: use ver4)')
    parser.add_argument('--max_pairs_per_scene', type=int, default=5,
                       help='Maximum number of reference images to process per scene/condition (0 = process all)')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip model structure validation (not recommended)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("MASt3R Test Application and Visualization")
    print("="*80)
    
    # チェックポイントを読み込む
    print("\n[1/5] Loading checkpoints...")
    model_ver3, info_ver3 = load_checkpoint_with_validation(
        args.checkpoint_ver3, device, verbose=True
    )
    model_ver4, info_ver4 = load_checkpoint_with_validation(
        args.checkpoint_ver4, device, verbose=True
    )
    
    # モデル構造を比較
    if not args.skip_validation:
        print("\n[2/5] Validating model structures...")
        if not compare_model_structures(model_ver3, model_ver4, verbose=True):
            print("\nERROR: Model structures do not match!")
            print("This indicates that the two checkpoints are incompatible.")
            sys.exit(1)
        print("✅ Model structures match")
    else:
        print("\n[2/5] Skipping model structure validation (--skip_validation)")
    
    # 使用するモデルを選択
    if args.use_ver3:
        model = model_ver3
        model_name = "ver3"
        print(f"\nUsing model: version 3")
    else:
        model = model_ver4
        model_name = "ver4"
        print(f"\nUsing model: version 4")
    
    # テスト画像を探索
    print(f"\n[3/5] Finding test images in: {args.test_dir}")
    scenes = find_test_images(args.test_dir)
    
    if not scenes:
        print("ERROR: No test images found!")
        sys.exit(1)
    
    print(f"Found {len(scenes)} scenes:")
    for scene_id, conditions in scenes.items():
        print(f"  {scene_id}: {len(conditions)} conditions")
        for condition, images in conditions.items():
            print(f"    {condition}: {len(images)} images")
    
    # 各画像を参照画像として処理（他の全画像を補助画像として使用）
    print(f"\n[4/5] Processing reference images with all supporting images...")
    total_ref_images = 0
    total_depth_predictions = 0
    
    for scene_id, conditions in tqdm(scenes.items(), desc="Processing scenes"):
        scene_output_dir = output_dir / scene_id / model_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        for condition, images in conditions.items():
            if len(images) < 2:
                print(f"  Skipping {condition}: not enough images ({len(images)})")
                continue
            
            condition_output_dir = scene_output_dir / condition
            condition_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 最大処理数を制限
            num_ref_images = min(len(images), args.max_pairs_per_scene) if args.max_pairs_per_scene > 0 else len(images)
            
            # 各画像を参照画像として処理
            for ref_idx in range(num_ref_images):
                ref_img_info = images[ref_idx]
                ref_img_path = ref_img_info['path']
                
                # 他の全ての画像を補助画像として使用
                supporting_img_paths = [img_info['path'] for idx, img_info in enumerate(images) if idx != ref_idx]
                
                # 参照画像IDを作成
                ref_id = f"{scene_id}_{condition}_{ref_idx:03d}"
                
                try:
                    num_depth_maps = process_reference_image(
                        model, ref_img_path, supporting_img_paths,
                        device, condition_output_dir, ref_id
                    )
                    total_ref_images += 1
                    total_depth_predictions += num_depth_maps
                except Exception as e:
                    print(f"\nERROR processing reference image {ref_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n[5/5] Processing complete!")
    print(f"  Total reference images processed: {total_ref_images}")
    print(f"  Total depth predictions (pairs): {total_depth_predictions}")
    print(f"  Average supporting images per reference: {total_depth_predictions / total_ref_images if total_ref_images > 0 else 0:.1f}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

