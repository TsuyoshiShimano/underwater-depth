#!/usr/bin/env python3
"""
MASt3R出力深度マップの評価スクリプト

出力された融合深度マップをGT深度と比較して定量的に評価します。
複数のモデル（ver3、ver4など）の結果を比較できます。
"""
import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
import cv2
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # tqdmの代替
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


def load_depth(path):
    """深度ファイルを読み込む（.exrと.npy対応）"""
    try:
        if str(path).endswith('.npy'):
            depth = np.load(str(path)).astype(np.float32)
            return depth
        
        if not HAS_OPENEXR:
            raise ValueError("OpenEXR not available, cannot load .exr files")
        
        # OpenEXR形式
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # 深度チャンネルを安全に読み込む（Z優先、次にV）
        depth_str = None
        for channel_name in ['Z', 'V', 'Y']:
            try:
                depth_str = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
                break
            except:
                continue
        
        if depth_str is None:
            available_channels = list(header['channels'].keys())
            raise ValueError(f"No depth channel found in {path}. Available channels: {available_channels}")
        
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape((height, width))
        exr_file.close()
        return depth
    except Exception as e:
        print(f"ERROR: Could not read depth from {path}. Error: {e}")
        return None


def resize_crop_depth_like_image(gt_depth, target_shape, image_size=512):
    """
    GT深度マップに画像と同じ幾何変換を適用
    evaluate_scenes.pyと同じ処理
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
    evaluate_scenes.pyと同じメトリクス
    """
    # 堅牢なマスク処理
    gt_outlier_threshold = 1e10
    if mask is not None:
        eval_mask = mask & (pred_depth > 0) & (gt_depth > 0) & \
                    (gt_depth < gt_outlier_threshold) & \
                    np.isfinite(pred_depth) & np.isfinite(gt_depth)
    else:
        eval_mask = (pred_depth > 0) & (gt_depth > 0) & \
                    (gt_depth < gt_outlier_threshold) & \
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


def find_predicted_depth_files(output_dir):
    """
    出力ディレクトリ内の予測深度ファイルを探索
    
    Returns:
        files: {scene_id: {condition: {model_name: [file_paths]}}}
    """
    output_dir = Path(output_dir)
    files = {}
    
    # 各シーンディレクトリを探索
    for scene_dir in sorted(output_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_id = scene_dir.name
        files[scene_id] = {}
        
        # モデルバージョンディレクトリ（ver3, ver4など）を探索
        for model_dir in sorted(scene_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            # 条件ディレクトリ（down_caustics, up_causticsなど）を探索
            for condition_dir in sorted(model_dir.iterdir()):
                if not condition_dir.is_dir():
                    continue
                
                condition = condition_dir.name
                
                if condition not in files[scene_id]:
                    files[scene_id][condition] = {}
                
                # fused_depth_*.npyファイルを探す
                depth_files = sorted(condition_dir.glob('fused_depth_*.npy'))
                if depth_files:
                    files[scene_id][condition][model_name] = depth_files
    
    return files


def find_gt_depth_file(scene_id, condition, view_idx, test_dir):
    """
    対応するGT深度ファイルを探す
    .npyファイルを優先的に使用（OpenEXRがなくても読み込めるため）
    """
    test_dir = Path(test_dir)
    scene_dir = test_dir / scene_id / condition
    
    # view_XXXX ディレクトリを探す
    view_dirs = sorted([d for d in scene_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('view_')])
    
    if view_idx >= len(view_dirs):
        return None
    
    view_dir = view_dirs[view_idx]
    depth_dir = view_dir / 'depth'
    
    if not depth_dir.exists():
        return None
    
    # .npyファイルを優先的に探す
    npy_files = sorted(depth_dir.glob('*.npy'))
    if npy_files:
        return npy_files[0]
    
    # .npyがなければ.exrファイルを探す（OpenEXRが利用可能な場合のみ）
    if HAS_OPENEXR:
        exr_files = sorted(depth_dir.glob('*.exr'))
        if exr_files:
            return exr_files[0]
    
    return None


def evaluate_depth_maps(output_dir, test_dir, output_results_dir=None):
    """
    予測深度マップを評価
    
    Args:
        output_dir: 予測深度マップが保存されているディレクトリ
        test_dir: GT深度が保存されているテストディレクトリ
        output_results_dir: 評価結果を保存するディレクトリ
    """
    output_dir = Path(output_dir)
    test_dir = Path(test_dir)
    
    # 予測深度ファイルを探索
    print("Searching for predicted depth files...")
    predicted_files = find_predicted_depth_files(output_dir)
    
    print(f"Found {len(predicted_files)} scenes")
    for scene_id, conditions in predicted_files.items():
        print(f"  {scene_id}: {len(conditions)} conditions")
        for condition, models in conditions.items():
            print(f"    {condition}: {len(models)} models")
            for model_name, files in models.items():
                print(f"      {model_name}: {len(files)} depth maps")
    
    # 評価結果を格納
    all_results = []
    
    # 各シーン、条件、モデルを処理
    for scene_id, conditions in tqdm(predicted_files.items(), desc="Evaluating scenes"):
        for condition, models in conditions.items():
            for model_name, pred_files in models.items():
                print(f"\nProcessing {scene_id}/{condition}/{model_name}...")
                
                for pred_file in tqdm(pred_files, desc=f"  {model_name}/{condition}", leave=False):
                    # ファイル名から情報を抽出
                    # fused_depth_{scene_id}_{condition}_{idx}.npy の形式
                    stem = pred_file.stem  # 'fused_depth_5bea87f4abd34c35e1860ab5_down_caustics_000'
                    # 'fused_depth_' を削除
                    name_without_prefix = stem.replace('fused_depth_', '')
                    # scene_idは既に分かっているので、残りの部分からidxを抽出
                    # 最後の3桁の数字がidx
                    parts = name_without_prefix.split('_')
                    try:
                        # 最後の部分から数字を抽出（例: '000' -> 0）
                        idx_str = parts[-1]
                        idx = int(idx_str)
                    except (ValueError, IndexError):
                        print(f"  Warning: Could not parse index from {pred_file.name}")
                        continue
                    
                    # GT深度ファイルを探す
                    gt_file = find_gt_depth_file(scene_id, condition, idx, test_dir)
                    
                    if gt_file is None:
                        print(f"  Warning: GT depth not found for {scene_id}/{condition}/view_{idx:04d}")
                        continue
                    
                    # 深度マップを読み込む
                    pred_depth = load_depth(pred_file)
                    gt_depth_raw = load_depth(gt_file)
                    
                    if pred_depth is None or gt_depth_raw is None:
                        print(f"  Warning: Could not load depth from {pred_file} or {gt_file}")
                        continue
                    
                    # GT深度を予測深度のサイズに合わせる
                    gt_depth, mask = resize_crop_depth_like_image(gt_depth_raw, pred_depth.shape, image_size=512)
                    
                    # 評価指標を計算
                    metrics = compute_depth_metrics(pred_depth, gt_depth, mask=mask)
                    
                    result = {
                        'scene_id': scene_id,
                        'condition': condition,
                        'model_name': model_name,
                        'view_idx': idx,
                        'pred_file': str(pred_file),
                        'gt_file': str(gt_file),
                        'metrics': metrics
                    }
                    
                    all_results.append(result)
    
    # 結果を保存
    if output_results_dir is None:
        output_results_dir = output_dir.parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_results_dir = Path(output_results_dir)
    output_results_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON形式で保存
    results_file = output_results_dir / 'all_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # 統計情報を計算・表示
    print_statistics(all_results, output_results_dir)
    
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
                print(f"  {model_name:15s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                      f"min={stats['min']:.4f}, max={stats['max']:.4f} (n={stats['count']})")
            else:
                print(f"  {model_name:15s}: N/A")
    
    print("\nAccuracy Metrics (higher is better):")
    print("-" * 80)
    for metric in accuracy_metrics:
        print(f"\n{metric.upper()}:")
        for model_name in sorted(model_stats.keys()):
            stats = model_stats[model_name][metric]
            if not np.isnan(stats['mean']):
                print(f"  {model_name:15s}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                      f"min={stats['min']:.4f}, max={stats['max']:.4f} (n={stats['count']})")
            else:
                print(f"  {model_name:15s}: N/A")
    
    # 改善率を計算（複数モデルがある場合）
    if len(model_stats) > 1:
        print("\n" + "="*80)
        print("Model Comparison (Improvement %)")
        print("="*80)
        
        model_names = sorted(model_stats.keys())
        baseline = model_names[0]  # 最初のモデルをベースラインとする
        
        for metric in error_metrics + accuracy_metrics:
            if metric in model_stats[baseline] and not np.isnan(model_stats[baseline][metric]['mean']):
                baseline_mean = model_stats[baseline][metric]['mean']
                print(f"\n{metric.upper()} (baseline: {baseline}):")
                
                for model_name in model_names[1:]:
                    if metric in model_stats[model_name] and not np.isnan(model_stats[model_name][metric]['mean']):
                        model_mean = model_stats[model_name][metric]['mean']
                        
                        if metric in error_metrics:
                            # 誤差指標：減少が改善
                            improvement = ((baseline_mean - model_mean) / baseline_mean) * 100
                        else:
                            # 精度指標：増加が改善
                            improvement = ((model_mean - baseline_mean) / baseline_mean) * 100
                        
                        print(f"  {model_name:15s}: {improvement:+.2f}% vs {baseline}")
    
    # 統計情報をJSONで保存
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(model_stats, f, indent=2, default=str)
    
    print(f"\n✓ Statistics saved to {stats_file}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predicted depth maps from MASt3R output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing predicted depth maps (output_mast3r_test)')
    parser.add_argument('--test_dir', type=str, default='/workspace/underwater/dust3r/test',
                       help='Directory containing GT depth maps')
    parser.add_argument('--output_results_dir', type=str, default=None,
                       help='Directory to save evaluation results (default: output_dir parent/evaluation_results_TIMESTAMP)')
    
    args = parser.parse_args()
    
    # 評価を実行
    results = evaluate_depth_maps(
        args.output_dir,
        args.test_dir,
        args.output_results_dir
    )
    
    print(f"\n✓ Evaluation complete! Processed {len(results)} depth maps.")


if __name__ == '__main__':
    main()

