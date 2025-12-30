#!/usr/bin/env python3
"""
MASt3R出力結果とDUSt3R/MASt3Rの評価結果を比較するスクリプト

既存のevaluate_scenes.pyの結果（DUSt3R標準モデルと比較結果）を読み込んで、
MASt3Rの出力結果と比較します。
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_evaluation_results(json_path):
    """評価結果のJSONファイルを読み込む"""
    with open(json_path, 'r') as f:
        results = json.load(f)
    return results


def extract_metrics_from_evaluate_scenes_results(results):
    """
    evaluate_scenes.pyの結果から標準モデル（DUSt3R）のメトリクスを抽出
    
    Args:
        results: evaluate_scenes.pyのall_results.jsonの内容
    
    Returns:
        metrics_by_scene: {scene_id: {condition: {image: metrics_dict}}}
    """
    metrics_by_scene = defaultdict(lambda: defaultdict(lambda: {}))
    
    for result in results:
        scene_id = result.get('scene')
        condition = result.get('category')
        image_path = result.get('image')
        
        if scene_id and condition and image_path:
            # 標準モデル（DUSt3R）のメトリクスを取得
            std_metrics = result.get('standard_model', {})
            if std_metrics:
                # 画像ファイル名から識別子を作成
                image_name = Path(image_path).stem
                metrics_by_scene[scene_id][condition][image_name] = std_metrics
    
    return metrics_by_scene


def extract_metrics_from_mast3r_results(results):
    """
    evaluate_mast3r_output.pyの結果からメトリクスを抽出
    
    Args:
        results: evaluate_mast3r_output.pyのall_results.jsonの内容
    
    Returns:
        metrics_by_scene: {scene_id: {condition: {model_name: {view_idx: metrics_dict}}}}
    """
    metrics_by_scene = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    
    for result in results:
        scene_id = result.get('scene_id')
        condition = result.get('condition')
        model_name = result.get('model_name')
        view_idx = result.get('view_idx')
        metrics = result.get('metrics', {})
        
        if scene_id and condition and model_name is not None and view_idx is not None:
            metrics_by_scene[scene_id][condition][model_name][view_idx] = metrics
    
    return metrics_by_scene


def compute_comparison_statistics(dust3r_metrics_by_scene, mast3r_metrics_by_scene):
    """
    比較統計を計算
    
    Args:
        dust3r_metrics_by_scene: DUSt3Rのメトリクス
        mast3r_metrics_by_scene: MASt3Rのメトリクス
    
    Returns:
        comparison_stats: 比較統計情報
    """
    comparison_stats = defaultdict(lambda: defaultdict(lambda: {}))
    
    # 各シーン・条件について比較
    for scene_id in mast3r_metrics_by_scene:
        if scene_id not in dust3r_metrics_by_scene:
            continue
        
        for condition in mast3r_metrics_by_scene[scene_id]:
            if condition not in dust3r_metrics_by_scene[scene_id]:
                continue
            
            dust3r_images = dust3r_metrics_by_scene[scene_id][condition]
            mast3r_models = mast3r_metrics_by_scene[scene_id][condition]
            
            # 各MASt3RモデルについてDUSt3Rと比較
            for model_name in mast3r_models:
                mast3r_views = mast3r_models[model_name]
                
                # 対応する画像を見つけて比較
                # 注意: 画像名とview_idxの対応は完全ではない可能性がある
                # ここでは、view_idxをキーとして比較を試みる
                paired_metrics = []
                
                for view_idx, mast3r_metrics in mast3r_views.items():
                    # DUSt3Rの結果から対応するメトリクスを探す
                    # 画像名にview_idxが含まれている可能性を考慮
                    found_match = False
                    for img_name, dust3r_metrics in dust3r_images.items():
                        # 画像名にview_idxが含まれているかチェック
                        if str(view_idx) in img_name or f"{view_idx:04d}" in img_name:
                            paired_metrics.append({
                                'view_idx': view_idx,
                                'dust3r': dust3r_metrics,
                                'mast3r': mast3r_metrics
                            })
                            found_match = True
                            break
                    
                    # 完全なマッチが見つからない場合、最初の画像を使用（暫定）
                    if not found_match and dust3r_images:
                        # view_idx順にソートして対応付けを試みる
                        sorted_dust3r = sorted(dust3r_images.items())
                        if len(paired_metrics) < len(sorted_dust3r):
                            img_name, dust3r_metrics = sorted_dust3r[len(paired_metrics)]
                            paired_metrics.append({
                                'view_idx': view_idx,
                                'dust3r': dust3r_metrics,
                                'mast3r': mast3r_metrics
                            })
                
                if paired_metrics:
                    comparison_stats[scene_id][condition][model_name] = paired_metrics
    
    return comparison_stats


def aggregate_comparison_statistics(comparison_stats):
    """
    比較統計を集約
    
    Returns:
        aggregated: モデルごとの集約統計
    """
    aggregated = defaultdict(lambda: {
        'rmse': [],
        'mae': [],
        'abs_rel': [],
        'si_log': [],
        'delta_1': [],
        'delta_2': [],
        'delta_3': [],
        'inlier_ratio': []
    })
    
    for scene_id, conditions in comparison_stats.items():
        for condition, models in conditions.items():
            for model_name, paired_metrics in models.items():
                for pair in paired_metrics:
                    dust3r = pair['dust3r']
                    mast3r = pair['mast3r']
                    
                    for metric_name in aggregated[model_name].keys():
                        if metric_name in dust3r and metric_name in mast3r:
                            dust3r_val = dust3r[metric_name]
                            mast3r_val = mast3r[metric_name]
                            
                            if not (np.isnan(dust3r_val) or np.isnan(mast3r_val)):
                                aggregated[model_name][metric_name].append({
                                    'dust3r': dust3r_val,
                                    'mast3r': mast3r_val,
                                    'scene': scene_id,
                                    'condition': condition
                                })
    
    # 統計を計算
    stats = {}
    for model_name, metrics_dict in aggregated.items():
        stats[model_name] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                dust3r_vals = [v['dust3r'] for v in values]
                mast3r_vals = [v['mast3r'] for v in values]
                
                stats[model_name][metric_name] = {
                    'dust3r_mean': np.mean(dust3r_vals),
                    'dust3r_std': np.std(dust3r_vals),
                    'mast3r_mean': np.mean(mast3r_vals),
                    'mast3r_std': np.std(mast3r_vals),
                    'improvement': ((np.mean(dust3r_vals) - np.mean(mast3r_vals)) / np.mean(dust3r_vals) * 100) if metric_name in ['rmse', 'mae', 'abs_rel', 'si_log'] else ((np.mean(mast3r_vals) - np.mean(dust3r_vals)) / np.mean(dust3r_vals) * 100),
                    'count': len(values)
                }
            else:
                stats[model_name][metric_name] = {
                    'dust3r_mean': np.nan,
                    'dust3r_std': np.nan,
                    'mast3r_mean': np.nan,
                    'mast3r_std': np.nan,
                    'improvement': np.nan,
                    'count': 0
                }
    
    return stats


def print_comparison_results(stats):
    """比較結果を表示"""
    print("\n" + "="*80)
    print("MASt3R vs DUSt3R Comparison")
    print("="*80)
    
    error_metrics = ['rmse', 'mae', 'abs_rel', 'si_log']
    accuracy_metrics = ['delta_1', 'delta_2', 'delta_3', 'inlier_ratio']
    
    print("\nError Metrics (lower is better):")
    print("-" * 80)
    for metric in error_metrics:
        print(f"\n{metric.upper()}:")
        for model_name in sorted(stats.keys()):
            if metric in stats[model_name]:
                s = stats[model_name][metric]
                if not np.isnan(s['dust3r_mean']):
                    print(f"  {model_name:15s}:")
                    print(f"    DUSt3R:  mean={s['dust3r_mean']:.4f} ± {s['dust3r_std']:.4f}")
                    print(f"    MASt3R:  mean={s['mast3r_mean']:.4f} ± {s['mast3r_std']:.4f}")
                    print(f"    Improvement: {s['improvement']:+.2f}% (n={s['count']})")
    
    print("\nAccuracy Metrics (higher is better):")
    print("-" * 80)
    for metric in accuracy_metrics:
        print(f"\n{metric.upper()}:")
        for model_name in sorted(stats.keys()):
            if metric in stats[model_name]:
                s = stats[model_name][metric]
                if not np.isnan(s['dust3r_mean']):
                    print(f"  {model_name:15s}:")
                    print(f"    DUSt3R:  mean={s['dust3r_mean']:.4f} ± {s['dust3r_std']:.4f}")
                    print(f"    MASt3R:  mean={s['mast3r_mean']:.4f} ± {s['mast3r_std']:.4f}")
                    print(f"    Improvement: {s['improvement']:+.2f}% (n={s['count']})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare MASt3R results with DUSt3R evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--mast3r_results', type=str, required=True,
                       help='Path to MASt3R evaluation results JSON (from evaluate_mast3r_output.py)')
    parser.add_argument('--dust3r_results', type=str,
                       default='/workspace/underwater/dust3r/evaluation_results_underwater_1.0/20251116_215724/all_results.json',
                       help='Path to DUSt3R evaluation results JSON (from evaluate_scenes.py)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    # 結果を読み込む
    print("Loading MASt3R results...")
    mast3r_results = load_evaluation_results(args.mast3r_results)
    print(f"  Loaded {len(mast3r_results)} MASt3R results")
    
    print("\nLoading DUSt3R results...")
    dust3r_results = load_evaluation_results(args.dust3r_results)
    print(f"  Loaded {len(dust3r_results)} DUSt3R results")
    
    # メトリクスを抽出
    print("\nExtracting metrics...")
    dust3r_metrics = extract_metrics_from_evaluate_scenes_results(dust3r_results)
    mast3r_metrics = extract_metrics_from_mast3r_results(mast3r_results)
    
    print(f"  DUSt3R: {len(dust3r_metrics)} scenes")
    print(f"  MASt3R: {len(mast3r_metrics)} scenes")
    
    # 比較統計を計算
    print("\nComputing comparison statistics...")
    comparison_stats = compute_comparison_statistics(dust3r_metrics, mast3r_metrics)
    aggregated_stats = aggregate_comparison_statistics(comparison_stats)
    
    # 結果を表示
    print_comparison_results(aggregated_stats)
    
    # 結果を保存
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.mast3r_results).parent
    
    comparison_file = output_dir / 'comparison_with_dust3r.json'
    with open(comparison_file, 'w') as f:
        json.dump({
            'aggregated_stats': aggregated_stats,
            'detailed_comparison': comparison_stats
        }, f, indent=2, default=str)
    
    print(f"\n✓ Comparison results saved to {comparison_file}")


if __name__ == '__main__':
    main()

