#!/usr/bin/env python3
"""
個々の深度マップを確認するためのスクリプト
"""
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/underwater/mast3r')

from mast3r.model import load_model
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.utils.image import load_images
from apply_mast3r_test import extract_depth_from_pts3d

def check_individual_depth_maps():
    """個々の深度マップにNaNが含まれているか確認"""
    
    # テスト用の画像ペアを読み込む
    test_dir = Path('/workspace/underwater/dust3r/test')
    scene_id = '58cf4771d0f5fb221defe6da'
    condition = 'down_caustics'
    
    scene_dir = test_dir / scene_id / condition
    
    # view_XXXXディレクトリを探す
    view_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('view_')])
    
    if len(view_dirs) < 2:
        print(f"Not enough views found: {len(view_dirs)}")
        return
    
    # 最初の2つの画像を使用
    ref_view = view_dirs[0]
    supp_view = view_dirs[1]
    
    ref_img_path = list((ref_view / 'color').glob('*.png'))[0]
    supp_img_path = list((supp_view / 'color').glob('*.png'))[0]
    
    print(f'Reference image: {ref_img_path}')
    print(f'Supporting image: {supp_img_path}')
    print()
    
    # モデルを読み込む
    checkpoint_path = 'output_mast3r_ver4/checkpoint-best.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f'Loading model from: {checkpoint_path}')
    model = load_model(checkpoint_path, device, verbose=True)
    model.eval()
    print()
    
    # 推論を実行
    print('Running inference...')
    imgs = load_images([str(ref_img_path), str(supp_img_path)], size=512, verbose=False)
    
    for img_dict in imgs:
        img_dict['img'] = img_dict['img'].to(device)
    
    with torch.no_grad():
        output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
        pred1 = output['pred1']
    
    # 深度を抽出
    pts3d = pred1['pts3d'][0]
    depth = extract_depth_from_pts3d(pts3d)
    
    # 結果を確認
    print('=== Individual Depth Map Analysis ===\n')
    print(f'Shape: {depth.shape}')
    print(f'Total pixels: {depth.size}')
    print()
    
    # NaN/Infの確認
    nan_count = np.sum(np.isnan(depth))
    inf_count = np.sum(np.isinf(depth))
    valid_count = np.sum(np.isfinite(depth))
    negative_count = np.sum((depth < 0) & np.isfinite(depth))
    zero_count = np.sum((depth == 0) & np.isfinite(depth))
    positive_count = np.sum((depth > 0) & np.isfinite(depth))
    
    print('Value classification:')
    print(f'  NaN: {nan_count} ({nan_count/depth.size*100:.1f}%)')
    print(f'  Inf: {inf_count} ({inf_count/depth.size*100:.1f}%)')
    print(f'  Valid (finite): {valid_count} ({valid_count/depth.size*100:.1f}%)')
    print(f'    - Negative: {negative_count} ({negative_count/depth.size*100:.1f}%)')
    print(f'    - Zero: {zero_count} ({zero_count/depth.size*100:.1f}%)')
    print(f'    - Positive: {positive_count} ({positive_count/depth.size*100:.1f}%)')
    print()
    
    if valid_count > 0:
        valid_depth = depth[np.isfinite(depth)]
        print(f'Valid depth range: {np.min(valid_depth):.6f} ~ {np.max(valid_depth):.6f}')
        print(f'Valid depth mean: {np.mean(valid_depth):.6f}')
        print(f'Valid depth median: {np.median(valid_depth):.6f}')
    
    # pts3dの確認
    print()
    print('=== pts3d Analysis ===')
    if isinstance(pts3d, torch.Tensor):
        pts3d_np = pts3d.cpu().numpy()
    else:
        pts3d_np = pts3d
    
    print(f'pts3d shape: {pts3d_np.shape}')
    print(f'pts3d NaN count: {np.sum(np.isnan(pts3d_np))}')
    print(f'pts3d Inf count: {np.sum(np.isinf(pts3d_np))}')
    
    # 深度マップを保存（デバッグ用）
    output_path = Path('debug_individual_depth.npy')
    np.save(output_path, depth)
    print(f'\nSaved individual depth map to: {output_path}')

if __name__ == '__main__':
    check_individual_depth_maps()

