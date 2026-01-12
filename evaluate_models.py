#!/usr/bin/env python3
"""
4つのモデル（DUSt3R事前学習、MASt3R事前学習、ver5、ver6）を比較評価するスクリプト

test splitのデータを使用して、各モデルの深度マップ予測精度を評価します。

注意: Pythonのhash()関数は実行環境によって異なる結果を返す可能性があります。
学習時と同じ結果を得るには、PYTHONHASHSEED環境変数を固定値に設定してください。
例: PYTHONHASHSEED=0 python3 evaluate_models.py ...
"""
import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
import cv2
from datetime import datetime
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, leave=True, **kwargs):
        if desc:
            print(desc)
        return iterable

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available. Only .npy files can be loaded.")

# MASt3R/Dust3R関連のインポート
from mast3r.model import load_model, AsymmetricMASt3R
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images, imread_cv2
from dust3r.datasets.blender_processed import BlenderProcessed


def load_depth(path):
    """深度ファイルを読み込む（.exrと.npy対応）
    
    .exrファイルの読み込みにはdust3r.utils.image.imread_cv2を使用します。
    これは適切なチャンネル（Z > V > Y > R）を自動的に選択します。
    """
    try:
        if str(path).endswith('.npy'):
            depth = np.load(str(path)).astype(np.float32)
            return depth
        
        # .exrファイルの場合はimread_cv2を使用（既存のDUSt3R/MASt3Rコードと同じ方法）
        # imread_cv2はOpenEXRが利用可能な場合は適切なチャンネル（Z > V > Y > R）を選択し、
        # 利用できない場合はOpenCVにフォールバックします
        depth = imread_cv2(str(path), cv2.IMREAD_ANYDEPTH)
        
        # 3次元配列の場合は最初のチャンネルを使用（RGB形式の場合）
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        
        # float32に変換
        depth = depth.astype(np.float32)
        
        return depth
    except Exception as e:
        print(f"ERROR: Could not read depth from {path}. Error: {e}")
        return None


def resize_crop_depth_like_image(gt_depth, target_shape, image_size=512):
    """
    GT深度マップに画像と同じ幾何変換を適用
    evaluate_mast3r_output.pyと同じ処理
    """
    H, W = gt_depth.shape
    Ht, Wt = target_shape
    target_width, target_height = image_size, image_size
    
    # Step 1: Principal point中心でクロップ（画像中央と仮定）
    cx, cy = W // 2, H // 2
    min_margin_x = min(cx, W - cx)
    min_margin_y = min(cy, H - cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    
    # クロップ
    depth_cropped = gt_depth[t:b, l:r]
    H_crop, W_crop = depth_cropped.shape
    
    # Step 2: 縦横比を保持してリサイズ（長辺 >= target）
    scale = max(target_width / W_crop, target_height / H_crop)
    new_W = int(np.floor(W_crop * scale))
    new_H = int(np.floor(H_crop * scale))
    
    # NEAREST補間で深度値を保持
    depth_scaled = cv2.resize(depth_cropped.astype(np.float32), (new_W, new_H), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Step 3: 中央クロップで正確なtargetサイズに
    cx, cy = new_W // 2, new_H // 2
    l = cx - target_width // 2
    t = cy - target_height // 2
    r = l + target_width
    b = t + target_height
    
    # 範囲チェックとクロップ
    if l >= 0 and t >= 0 and r <= new_W and b <= new_H:
        depth_final = depth_scaled[t:b, l:r]
        mask_final = np.ones((target_height, target_width), dtype=bool)
    else:
        # 範囲外の場合はパディング
        depth_final = np.zeros((target_height, target_width), dtype=np.float32)
        mask_final = np.zeros((target_height, target_width), dtype=bool)
        
        src_l = max(0, l)
        src_t = max(0, t)
        src_r = min(new_W, r)
        src_b = min(new_H, b)
        
        dst_l = max(0, -l)
        dst_t = max(0, -t)
        dst_r = dst_l + (src_r - src_l)
        dst_b = dst_t + (src_b - src_t)
        
        depth_final[dst_t:dst_b, dst_l:dst_r] = depth_scaled[src_t:src_b, src_l:src_r]
        mask_final[dst_t:dst_b, dst_l:dst_r] = True
    
    # Step 4: 予測深度のサイズに合わせる
    if (target_width, target_height) != (Wt, Ht):
        depth_final = cv2.resize(depth_final, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        mask_final = cv2.resize(mask_final.astype(np.uint8), (Wt, Ht), 
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    
    return depth_final, mask_final


def compute_depth_metrics(pred_depth, gt_depth, mask=None):
    """
    深度マップの評価指標を計算
    evaluate_mast3r_output.pyと同じメトリクス
    """
    # 堅牢なマスク処理
    # 1e10以上は無効なピクセル（背景など）を表す
    outlier_threshold = 1e5
    if mask is not None:
        eval_mask = mask & (pred_depth > 0) & (gt_depth > 0) & \
                    (pred_depth < outlier_threshold) & \
                    (gt_depth < outlier_threshold) & \
                    np.isfinite(pred_depth) & np.isfinite(gt_depth)
    else:
        eval_mask = (pred_depth > 0) & (gt_depth > 0) & \
                    (pred_depth < outlier_threshold) & \
                    (gt_depth < outlier_threshold) & \
                    np.isfinite(pred_depth) & np.isfinite(gt_depth)
    
    pred_valid = pred_depth[eval_mask]
    gt_valid = gt_depth[eval_mask]
    
    if len(pred_valid) == 0 or len(gt_valid) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'abs_rel': np.nan,
            'si_log': np.nan,
            'delta_1': np.nan,
            'delta_2': np.nan,
            'delta_3': np.nan,
            'inlier_ratio': np.nan
        }
    
    # メディアン比でスケール補正
    scale = np.median(gt_valid / np.maximum(pred_valid, 1e-9))
    pred_scaled = pred_valid * scale
    
    # 基本指標
    rmse = np.sqrt(np.mean((pred_scaled - gt_valid) ** 2))
    mae = np.mean(np.abs(pred_scaled - gt_valid))
    abs_rel = np.mean(np.abs(pred_scaled - gt_valid) / gt_valid)
    
    # si-log RMSE（スケール不変対数誤差）
    log_diff = np.log(pred_scaled + 1e-9) - np.log(gt_valid + 1e-9)
    si_log = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
    
    # δ < 1.25^k 指標
    ratio = np.maximum(pred_scaled / gt_valid, gt_valid / pred_scaled)
    delta_1 = np.mean(ratio < 1.25)
    delta_2 = np.mean(ratio < 1.25 ** 2)
    delta_3 = np.mean(ratio < 1.25 ** 3)
    
    # inlier_ratio（δ < 1.03）
    inlier_ratio = np.mean(ratio < 1.03)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'abs_rel': float(abs_rel),
        'si_log': float(si_log),
        'delta_1': float(delta_1),
        'delta_2': float(delta_2),
        'delta_3': float(delta_3),
        'inlier_ratio': float(inlier_ratio)
    }


def extract_depth_from_pts3d(pts3d):
    """
    pts3dから深度（z座標）を抽出
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


def load_model_wrapper(model_name_or_path, device, verbose=True):
    """
    モデルを読み込む（事前学習モデルまたはチェックポイント）
    
    Args:
        model_name_or_path: モデル名（'naver/DUSt3R_...'または'naver/MASt3R_...'）またはチェックポイントパス
        device: デバイス
        verbose: 詳細情報を出力するか
    
    Returns:
        model: 読み込まれたモデル
        model_type: モデルタイプ（'dust3r_pretrained', 'mast3r_pretrained', 'mast3r_checkpoint'）
    """
    if verbose:
        print(f"Loading model: {model_name_or_path}")
    
    # 事前学習モデルの場合
    if model_name_or_path.startswith('naver/DUSt3R'):
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name_or_path).to(device)
        model.eval()
        if verbose:
            print(f"✓ Loaded DUSt3R pretrained model: {model_name_or_path}")
        return model, 'dust3r_pretrained'
    
    elif model_name_or_path.startswith('naver/MASt3R'):
        model = AsymmetricMASt3R.from_pretrained(model_name_or_path).to(device)
        model.eval()
        if verbose:
            print(f"✓ Loaded MASt3R pretrained model: {model_name_or_path}")
        return model, 'mast3r_pretrained'
    
    # チェックポイントファイルの場合
    else:
        model = load_model(model_name_or_path, device, verbose=verbose)
        model.eval()
        if verbose:
            print(f"✓ Loaded checkpoint: {model_name_or_path}")
        return model, 'mast3r_checkpoint'


def predict_depth_with_model(model, img1_path, img2_path, device, image_size=512):
    """
    モデルを使用して深度を予測
    
    Args:
        model: モデル
        img1_path: 参照画像のパス
        img2_path: 補助画像のパス
        device: デバイス
        image_size: 画像サイズ
        
    Returns:
        depth: 深度マップ (H, W) or None if error
    """
    try:
        # 画像を読み込む
        imgs = load_images([str(img1_path), str(img2_path)], size=image_size, verbose=False)
        
        if not imgs or len(imgs) < 2:
            return None
        
        # デバイスに移動
        for img_dict in imgs:
            img_dict['img'] = img_dict['img'].to(device)
        
        # 推論を実行
        model.eval()
        with torch.no_grad():
            output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
            pred1 = output['pred1']
        
        # 深度を抽出
        pts3d = pred1['pts3d'][0]
        depth = extract_depth_from_pts3d(pts3d)
        
        return depth
    except Exception as e:
        print(f"  Warning: Error predicting depth: {e}")
        return None


def get_test_pairs(dataset_root, max_pairs=None):
    """
    test splitのペアを取得
    
    BlenderProcessedクラスを使用して、学習時と同じ方法でtest splitを取得します。
    これにより、hash()関数の実行ごとの違いによる問題を回避します。
    
    Args:
        dataset_root: データセットのルートディレクトリ
        max_pairs: 最大ペア数（Noneの場合は全て）
        
    Returns:
        pairs: [(scene_id1, view1, img1, scene_id2, view2, img2, score), ...]のリスト
    """
    # BlenderProcessedクラスを使用してtest splitを取得
    # これにより、学習時と同じsplit方法を保証します
    dataset = BlenderProcessed(
        ROOT=dataset_root,
        split='test',
        resolution=(512, 384)  # 解像度は評価時には使用されないが、必須パラメータ
    )
    
    # dataset.pairsは既にtest splitにフィルタリングされている
    test_pairs = dataset.pairs
    
    if max_pairs is not None and max_pairs > 0:
        test_pairs = test_pairs[:max_pairs]
    
    return test_pairs


def get_image_and_depth_paths(pair, dataset_root):
    """
    ペアから画像と深度ファイルのパスを取得
    
    Args:
        pair: [scene_id1, view1, img1, scene_id2, view2, img2, score]
        dataset_root: データセットのルートディレクトリ
        
    Returns:
        (img1_path, depth1_path, img2_path, depth2_path) or None if files not found
    """
    scene_id1, view1, img1, scene_id2, view2, img2, score = pair
    
    dataset_root = Path(dataset_root)
    
    # 画像と深度ファイルのパスを構築
    img1_path = dataset_root / scene_id1 / view1 / f"{int(img1):08d}.jpg"
    depth1_path = dataset_root / scene_id1 / view1 / f"{int(img1):08d}.exr"
    img2_path = dataset_root / scene_id2 / view2 / f"{int(img2):08d}.jpg"
    depth2_path = dataset_root / scene_id2 / view2 / f"{int(img2):08d}.exr"
    
    # ファイルの存在確認
    if not all(p.exists() for p in [img1_path, depth1_path, img2_path, depth2_path]):
        return None
    
    return img1_path, depth1_path, img2_path, depth2_path


def evaluate_models(models, dataset_root, device, image_size=512, max_pairs=None, output_dir=None):
    """
    複数のモデルを評価
    
    Args:
        models: {model_name: (model, model_type)} の辞書
        dataset_root: データセットのルートディレクトリ
        device: デバイス
        image_size: 画像サイズ
        max_pairs: 最大評価ペア数
        output_dir: 出力ディレクトリ
        
    Returns:
        all_results: 評価結果のリスト
    """
    print("Getting test pairs...")
    test_pairs = get_test_pairs(dataset_root, max_pairs=max_pairs)
    print(f"Found {len(test_pairs)} test pairs")
    
    all_results = []
    
    # 各モデルで評価
    for model_name, (model, model_type) in models.items():
        print(f"\n{'='*80}")
        print(f"Evaluating model: {model_name} ({model_type})")
        print(f"{'='*80}")
        
        model_results = []
        
        for pair_idx, pair in enumerate(tqdm(test_pairs, desc=f"  {model_name}")):
            # 画像と深度ファイルのパスを取得
            paths = get_image_and_depth_paths(pair, dataset_root)
            if paths is None:
                continue
            
            img1_path, depth1_path, img2_path, depth2_path = paths
            
            # 深度を予測（画像1を参照画像として、画像2とペアで予測）
            pred_depth = predict_depth_with_model(model, img1_path, img2_path, device, image_size)
            
            if pred_depth is None:
                continue
            
            # GT深度を読み込む
            gt_depth_raw = load_depth(depth1_path)
            if gt_depth_raw is None:
                continue
            
            # GT深度を予測深度のサイズに合わせる
            gt_depth, mask = resize_crop_depth_like_image(gt_depth_raw, pred_depth.shape, image_size)
            
            # 評価指標を計算
            metrics = compute_depth_metrics(pred_depth, gt_depth, mask=mask)
            
            scene_id1, view1, img1, scene_id2, view2, img2, score = pair
            
            result = {
                'model_name': model_name,
                'model_type': model_type,
                'scene_id1': scene_id1,
                'view1': view1,
                'img1': str(img1),
                'scene_id2': scene_id2,
                'view2': view2,
                'img2': str(img2),
                'score': float(score),
                'img1_path': str(img1_path),
                'depth1_path': str(depth1_path),
                'metrics': metrics
            }
            
            model_results.append(result)
            all_results.append(result)
        
        print(f"\n  Processed {len(model_results)} pairs for {model_name}")
    
    # 結果を保存
    if output_dir is None:
        output_dir = Path(dataset_root).parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存
    results_file = output_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # 統計情報を計算・表示
    print_statistics(all_results, output_dir)
    
    return all_results


def print_statistics(results, output_dir):
    """統計情報を計算して表示・保存"""
    if not results:
        print("No results to analyze")
        return
    
    # モデルごとに結果をグループ化
    models = {}
    for result in results:
        model_name = result['model_name']
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(result['metrics'])
    
    print("\n" + "="*80)
    print("Evaluation Statistics")
    print("="*80)
    
    # 各モデルの統計を計算
    model_stats = {}
    for model_name, metrics_list in models.items():
        stats = {}
        # 各メトリクスについて平均と標準偏差を計算
        for metric_name in ['rmse', 'mae', 'abs_rel', 'si_log', 'delta_1', 'delta_2', 'delta_3', 'inlier_ratio']:
            values = [m[metric_name] for m in metrics_list if not np.isnan(m[metric_name])]
            if values:
                stats[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                stats[metric_name] = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0}
        model_stats[model_name] = stats
    
    # 結果を表示
    error_metrics = ['rmse', 'mae', 'abs_rel', 'si_log']
    accuracy_metrics = ['delta_1', 'delta_2', 'delta_3', 'inlier_ratio']
    
    print("\nError Metrics (lower is better):")
    print("-" * 80)
    for metric in error_metrics:
        print(f"\n{metric.upper()}:")
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name][metric]
            if not np.isnan(stats['mean']):
                print(f"  {model_name:30s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                      f"min={stats['min']:.4f}, max={stats['max']:.4f} (n={stats['count']})")
            else:
                print(f"  {model_name:30s}: N/A")
    
    print("\nAccuracy Metrics (higher is better):")
    print("-" * 80)
    for metric in accuracy_metrics:
        print(f"\n{metric.upper()}:")
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name][metric]
            if not np.isnan(stats['mean']):
                print(f"  {model_name:30s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                      f"min={stats['min']:.4f}, max={stats['max']:.4f} (n={stats['count']})")
            else:
                print(f"  {model_name:30s}: N/A")
    
    # 統計情報をJSONで保存
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(model_stats, f, indent=2, default=str)
    
    print(f"\n✓ Statistics saved to {stats_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate 4 models (DUSt3R pretrained, MASt3R pretrained, ver5, ver6) on test split',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset_root', type=str, 
                       default='/workspace/underwater/dust3r/data/blender_processed',
                       help='Root directory of the dataset')
    parser.add_argument('--dust3r_pretrained', type=str,
                       default='naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt',
                       help='DUSt3R pretrained model name')
    parser.add_argument('--mast3r_pretrained', type=str,
                       default='naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric',
                       help='MASt3R pretrained model name')
    parser.add_argument('--ver5_checkpoint', type=str,
                       default='/workspace/underwater/mast3r/output_mast3r_ver5/checkpoint-best.pth',
                       help='Path to ver5 checkpoint')
    parser.add_argument('--ver6_checkpoint', type=str,
                       default='/workspace/underwater/mast3r/output_mast3r_ver6/checkpoint-best.pth',
                       help='Path to ver6 checkpoint')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for inference')
    parser.add_argument('--max_pairs', type=int, default=None,
                       help='Maximum number of pairs to evaluate (None for all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: dataset_root parent/evaluation_results_TIMESTAMP)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # PYTHONHASHSEEDの確認と警告
    hash_seed = os.environ.get('PYTHONHASHSEED')
    if hash_seed is None:
        print("WARNING: PYTHONHASHSEED is not set. Hash values may differ between runs.")
        print("         For consistent results with training, set PYTHONHASHSEED to a fixed value.")
        print("         Example: PYTHONHASHSEED=0 python3 evaluate_models.py ...")
        print()
    else:
        print(f"PYTHONHASHSEED is set to: {hash_seed}")
        print()
    
    # デバイスの設定
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # モデルを読み込む
    print("\nLoading models...")
    models = {}
    
    # DUSt3R事前学習モデル
    try:
        model, model_type = load_model_wrapper(args.dust3r_pretrained, device, verbose=True)
        models['DUSt3R_pretrained'] = (model, model_type)
    except Exception as e:
        print(f"ERROR: Failed to load DUSt3R pretrained model: {e}")
        sys.exit(1)
    
    # MASt3R事前学習モデル
    try:
        model, model_type = load_model_wrapper(args.mast3r_pretrained, device, verbose=True)
        models['MASt3R_pretrained'] = (model, model_type)
    except Exception as e:
        print(f"ERROR: Failed to load MASt3R pretrained model: {e}")
        sys.exit(1)
    
    # ver5チェックポイント
    if not os.path.exists(args.ver5_checkpoint):
        print(f"ERROR: ver5 checkpoint not found: {args.ver5_checkpoint}")
        sys.exit(1)
    try:
        model, model_type = load_model_wrapper(args.ver5_checkpoint, device, verbose=True)
        models['mast3r_ver5'] = (model, model_type)
    except Exception as e:
        print(f"ERROR: Failed to load ver5 checkpoint: {e}")
        sys.exit(1)
    
    # ver6チェックポイント
    if not os.path.exists(args.ver6_checkpoint):
        print(f"ERROR: ver6 checkpoint not found: {args.ver6_checkpoint}")
        sys.exit(1)
    try:
        model, model_type = load_model_wrapper(args.ver6_checkpoint, device, verbose=True)
        models['mast3r_ver6'] = (model, model_type)
    except Exception as e:
        print(f"ERROR: Failed to load ver6 checkpoint: {e}")
        sys.exit(1)
    
    if not models:
        print("ERROR: No models loaded")
        sys.exit(1)
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # 評価を実行
    results = evaluate_models(
        models=models,
        dataset_root=args.dataset_root,
        device=device,
        image_size=args.image_size,
        max_pairs=args.max_pairs,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Evaluation complete! Processed {len(results)} results.")


if __name__ == '__main__':
    main()

