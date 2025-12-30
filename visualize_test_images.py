#!/usr/bin/env python3
"""
MASt3R Visualization: Test images with checkpoint-best.pth
Load two adjacent images from /workspace/underwater/dust3r/test and visualize
depth maps, confidence maps, and correspondences.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from pathlib import Path

from mast3r.model import load_model
import mast3r.utils.path_to_dust3r

def load_images(img_paths, size=512):
    """Load and resize images"""
    images = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        # Resize while maintaining aspect ratio
        w, h = img.size
        scale = size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Convert to tensor [3, H, W] and normalize to [0, 1]
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        images.append(img_tensor)
    
    return images

def visualize_matches(img1_np, img2_np, pts1, pts2, confidences, save_path):
    """Visualize matching points between two images"""
    H1, W1 = img1_np.shape[:2]
    H2, W2 = img2_np.shape[:2]
    
    # Combine images side by side
    H = max(H1, H2)
    combined = np.ones((H, W1 + W2, 3), dtype=np.float32)
    combined[:H1, :W1] = img1_np
    combined[:H2, W1:] = img2_np
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.imshow(combined)
    ax.axis('off')
    
    # Check if there are any matches
    if len(pts1) == 0 or len(confidences) == 0:
        ax.set_title(f'MASt3R Matching Results (N=0 matches - No matches found)', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"⚠️  No matches found, saved empty visualization: {save_path}")
        return
    
    # Draw matches
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        conf = confidences[i]
        
        # Color based on confidence (green=high, red=low)
        color = plt.cm.RdYlGn(np.clip(conf, 0, 1))
        
        # Draw line
        ax.plot([x1, x2 + W1], [y1, y2], 
               color=color, linewidth=1.5, alpha=0.7)
        
        # Draw points
        ax.plot(x1, y1, 'o', color=color, markersize=5)
        ax.plot(x2 + W1, y2, 'o', color=color, markersize=5)
    
    # Calculate statistics safely
    conf_min = confidences.min() if len(confidences) > 0 else 0.0
    conf_mean = confidences.mean() if len(confidences) > 0 else 0.0
    conf_max = confidences.max() if len(confidences) > 0 else 0.0
    
    ax.set_title(f'MASt3R Matching Results (N={len(pts1)} matches, '
                f'Conf: min={conf_min:.3f}, mean={conf_mean:.3f}, max={conf_max:.3f})', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved matching visualization: {save_path}")

def visualize_depth_and_confidence(depth_map, conf_map, img, save_path, title_prefix="Image"):
    """Visualize depth map and confidence map side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f'{title_prefix} - Original')
    axes[0].axis('off')
    
    # Depth map
    im1 = axes[1].imshow(depth_map, cmap='turbo')
    axes[1].set_title(f'{title_prefix} - Depth Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Confidence map - use actual data range
    conf_min = conf_map.min()
    conf_max = conf_map.max()
    im2 = axes[2].imshow(conf_map, cmap='viridis', vmin=conf_min, vmax=conf_max)
    axes[2].set_title(f'{title_prefix} - Confidence Map (range: [{conf_min:.3f}, {conf_max:.3f}])')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved depth and confidence visualization: {save_path}")

def visualize_background_color(b_inf_map, save_path, title_prefix="Image"):
    """Visualize background color (B_inf) as RGB map"""
    # b_inf_map: [H, W, 3] RGB values in [0, 1] range
    # Clip to valid range
    b_inf_map = np.clip(b_inf_map, 0, 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(b_inf_map)
    ax.set_title(f'{title_prefix} - Background Color (B_inf) RGB Map')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved background color RGB map: {save_path}")

def save_beta_values(beta_1, beta_2, save_path, title_prefix="Image"):
    """Save beta_1 and beta_2 values as text file"""
    # beta_1, beta_2: [H, W, 3] or [3] - RGB channels
    # Calculate mean values across spatial dimensions for each channel
    if beta_1.ndim == 3:
        beta_1_mean = beta_1.mean(axis=(0, 1))  # [3] - mean across H, W for each channel
        beta_1_std = beta_1.std(axis=(0, 1))  # [3]
        beta_1_min = beta_1.min(axis=(0, 1))  # [3]
        beta_1_max = beta_1.max(axis=(0, 1))  # [3]
    else:
        beta_1_mean = beta_1
        beta_1_std = np.zeros_like(beta_1)
        beta_1_min = beta_1
        beta_1_max = beta_1
    
    if beta_2.ndim == 3:
        beta_2_mean = beta_2.mean(axis=(0, 1))  # [3]
        beta_2_std = beta_2.std(axis=(0, 1))  # [3]
        beta_2_min = beta_2.min(axis=(0, 1))  # [3]
        beta_2_max = beta_2.max(axis=(0, 1))  # [3]
    else:
        beta_2_mean = beta_2
        beta_2_std = np.zeros_like(beta_2)
        beta_2_min = beta_2
        beta_2_max = beta_2
    
    with open(save_path, 'w') as f:
        f.write(f"{title_prefix} - Attenuation Coefficients\n")
        f.write("="*60 + "\n\n")
        f.write("β_1 (absorption coefficient for background light term):\n")
        f.write(f"  R channel: mean={beta_1_mean[0]:.6f}, std={beta_1_std[0]:.6f}, min={beta_1_min[0]:.6f}, max={beta_1_max[0]:.6f}\n")
        f.write(f"  G channel: mean={beta_1_mean[1]:.6f}, std={beta_1_std[1]:.6f}, min={beta_1_min[1]:.6f}, max={beta_1_max[1]:.6f}\n")
        f.write(f"  B channel: mean={beta_1_mean[2]:.6f}, std={beta_1_std[2]:.6f}, min={beta_1_min[2]:.6f}, max={beta_1_max[2]:.6f}\n")
        f.write(f"  Overall mean: {beta_1_mean.mean():.6f}\n\n")
        f.write("β_2 (absorption coefficient for restoration term):\n")
        f.write(f"  R channel: mean={beta_2_mean[0]:.6f}, std={beta_2_std[0]:.6f}, min={beta_2_min[0]:.6f}, max={beta_2_max[0]:.6f}\n")
        f.write(f"  G channel: mean={beta_2_mean[1]:.6f}, std={beta_2_std[1]:.6f}, min={beta_2_min[1]:.6f}, max={beta_2_max[1]:.6f}\n")
        f.write(f"  B channel: mean={beta_2_mean[2]:.6f}, std={beta_2_std[2]:.6f}, min={beta_2_min[2]:.6f}, max={beta_2_max[2]:.6f}\n")
        f.write(f"  Overall mean: {beta_2_mean.mean():.6f}\n")
    
    print(f"✅ Saved beta values: {save_path}")

def find_adjacent_images_from_scene(scene_dir):
    """Find two adjacent images from a single scene directory"""
    # Find down_caustics directory
    down_caustics_dir = scene_dir / 'down_caustics'
    if not down_caustics_dir.exists():
        return None
    
    # Find all view directories
    view_dirs = sorted([d for d in down_caustics_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('view_')])
    
    if len(view_dirs) < 2:
        return None
    
    # Get first two adjacent views
    view1_dir = view_dirs[0]
    view2_dir = view_dirs[1]
    
    # Find color images
    color1_dir = view1_dir / 'color'
    color2_dir = view2_dir / 'color'
    
    if not color1_dir.exists() or not color2_dir.exists():
        return None
    
    # Get first image from each directory
    img1_files = sorted(list(color1_dir.glob('*.png')) + list(color1_dir.glob('*.jpg')))
    img2_files = sorted(list(color2_dir.glob('*.png')) + list(color2_dir.glob('*.jpg')))
    
    if not img1_files or not img2_files:
        return None
    
    img1_path = str(img1_files[0])
    img2_path = str(img2_files[0])
    
    return {
        'scene_name': scene_dir.name,
        'img1_path': img1_path,
        'img2_path': img2_path,
        'view1_name': view1_dir.name,
        'view2_name': view2_dir.name
    }

def find_all_adjacent_images(test_dir):
    """Find two adjacent images from each scene directory in test folder"""
    test_path = Path(test_dir)
    
    # Find all scene directories
    scene_dirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {test_dir}")
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    # Get image pairs from each scene
    image_pairs = []
    for scene_dir in scene_dirs:
        pair_info = find_adjacent_images_from_scene(scene_dir)
        if pair_info is not None:
            image_pairs.append(pair_info)
            print(f"  ✅ {scene_dir.name}: {pair_info['view1_name']} and {pair_info['view2_name']}")
        else:
            print(f"  ⚠️  {scene_dir.name}: Skipped (insufficient views or images)")
    
    if not image_pairs:
        raise ValueError(f"No valid image pairs found in any scene directory")
    
    print(f"\nTotal {len(image_pairs)} image pairs found")
    return image_pairs

def main():
    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/workspace/underwater/mast3r/output_mast3r_ver5/checkpoint-best.pth'
    test_dir = '/workspace/underwater/dust3r/test'
    output_dir = '/workspace/underwater/mast3r/output_mast3r_ver5/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MASt3R Visualization: Test Images with Checkpoint")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test directory: {test_dir}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Load model from checkpoint
    print("Loading MASt3R model from checkpoint...")
    model = load_model(checkpoint_path, device=device, verbose=True)
    model.eval()
    print("✅ Model loaded successfully\n")
    
    # Find all adjacent image pairs from each scene
    print("Finding adjacent images from test directory...")
    image_pairs = find_all_adjacent_images(test_dir)
    print()
    
    # Process each image pair
    total_pairs = len(image_pairs)
    for pair_idx, pair_info in enumerate(image_pairs, 1):
        scene_name = pair_info['scene_name']
        img1_path = pair_info['img1_path']
        img2_path = pair_info['img2_path']
        
        print("="*80)
        print(f"Processing pair {pair_idx}/{total_pairs}: {scene_name}")
        print(f"  Views: {pair_info['view1_name']} and {pair_info['view2_name']}")
        print("="*80)
        
        # Create output subdirectory for this scene
        scene_output_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_output_dir, exist_ok=True)
        
        # Load images
        print("Loading images...")
        images = load_images([img1_path, img2_path], size=512)
        img1_tensor = images[0].unsqueeze(0).to(device)  # [1, 3, H, W]
        img2_tensor = images[1].unsqueeze(0).to(device)
        print(f"  Image 1 shape: {img1_tensor.shape}")
        print(f"  Image 2 shape: {img2_tensor.shape}\n")
        
        # Run inference
        print("Running MASt3R inference...")
        with torch.no_grad():
            # Prepare input dictionaries
            view1 = {'img': img1_tensor, 'idx': 0, 'instance': '0'}
            view2 = {'img': img2_tensor, 'idx': 1, 'instance': '1'}
            
            # Forward pass
            output = model(view1, view2)
            pred1, pred2 = output
        
        print("✅ Inference completed\n")
        
        # Debug: check output structure
        print(f"Debug - pred1 keys: {list(pred1.keys())}")
        print(f"Debug - pred2 keys: {list(pred2.keys())}\n")
        
        # Extract results
        pts3d_1 = pred1['pts3d'][0].cpu().numpy()  # [H, W, 3]
        pts3d_2 = pred2.get('pts3d_in_other_view', pred2.get('pts3d'))[0].cpu().numpy()  # [H, W, 3]
        
        # Get confidence maps
        if 'conf' in pred1:
            conf1 = pred1['conf'][0].cpu().numpy()  # [H, W]
            conf2 = pred2['conf'][0].cpu().numpy()
        else:
            conf1 = np.ones((pts3d_1.shape[0], pts3d_1.shape[1]))
            conf2 = np.ones((pts3d_2.shape[0], pts3d_2.shape[1]))
        
        # Get descriptors if available
        if 'desc' in pred1:
            desc1 = pred1['desc'][0].cpu().numpy()  # [H, W, D]
            desc2 = pred2['desc'][0].cpu().numpy()
        else:
            desc1 = None
            desc2 = None
        
        # Get underwater parameters if available (beta_1, beta_2, b_inf)
        beta_1_1 = None
        beta_2_1 = None
        b_inf_1 = None
        beta_1_2 = None
        beta_2_2 = None
        b_inf_2 = None
        
        if 'beta_1' in pred1:
            beta_1_1 = pred1['beta_1'][0].cpu().numpy()  # [H, W, 3]
            beta_2_1 = pred1['beta_2'][0].cpu().numpy()  # [H, W, 3]
            b_inf_1 = pred1['b_inf'][0].cpu().numpy()  # [H, W, 3]
            print("✅ Found underwater parameters in pred1")
        
        if 'beta_1' in pred2:
            beta_1_2 = pred2['beta_1'][0].cpu().numpy()  # [H, W, 3]
            beta_2_2 = pred2['beta_2'][0].cpu().numpy()  # [H, W, 3]
            b_inf_2 = pred2['b_inf'][0].cpu().numpy()  # [H, W, 3]
            print("✅ Found underwater parameters in pred2")
        
        H1, W1 = pts3d_1.shape[:2]
        H2, W2 = pts3d_2.shape[:2]
        
        # Compute depth from 3D points
        depth1 = np.linalg.norm(pts3d_1, axis=-1)  # [H, W]
        depth2 = np.linalg.norm(pts3d_2, axis=-1)
        
        # Prepare images for visualization
        img1_np = img1_tensor[0].permute(1, 2, 0).cpu().numpy()
        img2_np = img2_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        # Save visualizations
        print("Saving visualizations...")
        
        # 1. Depth and confidence maps for image 1
        visualize_depth_and_confidence(
            depth1, conf1, img1_np,
            os.path.join(scene_output_dir, 'depth_conf_img1.png'),
            title_prefix=f"{scene_name} - Image 1"
        )
        
        # 2. Depth and confidence maps for image 2
        visualize_depth_and_confidence(
            depth2, conf2, img2_np,
            os.path.join(scene_output_dir, 'depth_conf_img2.png'),
            title_prefix=f"{scene_name} - Image 2"
        )
        
        # 2.5. Background color RGB maps (if available)
        if b_inf_1 is not None:
            visualize_background_color(
                b_inf_1,
                os.path.join(scene_output_dir, 'background_color_img1.png'),
                title_prefix=f"{scene_name} - Image 1"
            )
        
        if b_inf_2 is not None:
            visualize_background_color(
                b_inf_2,
                os.path.join(scene_output_dir, 'background_color_img2.png'),
                title_prefix=f"{scene_name} - Image 2"
            )
        
        # 2.6. Beta values (if available)
        if beta_1_1 is not None and beta_2_1 is not None:
            save_beta_values(
                beta_1_1, beta_2_1,
                os.path.join(scene_output_dir, 'beta_values_img1.txt'),
                title_prefix=f"{scene_name} - Image 1"
            )
        
        if beta_1_2 is not None and beta_2_2 is not None:
            save_beta_values(
                beta_1_2, beta_2_2,
                os.path.join(scene_output_dir, 'beta_values_img2.txt'),
                title_prefix=f"{scene_name} - Image 2"
            )
        
        # 3. Matching visualization (if descriptors are available)
        if desc1 is not None and desc2 is not None:
            print("Computing matches...")
            D = desc1.shape[-1]
            desc1_flat = desc1.reshape(-1, D)  # [H1*W1, D]
            desc2_flat = desc2.reshape(-1, D)  # [H2*W2, D]
            
            # Normalize descriptors (L2 norm)
            desc1_flat = desc1_flat / (np.linalg.norm(desc1_flat, axis=-1, keepdims=True) + 1e-8)
            desc2_flat = desc2_flat / (np.linalg.norm(desc2_flat, axis=-1, keepdims=True) + 1e-8)
            
            # Sample 500 points from image 1
            num_samples = 500
            indices1 = np.random.choice(H1 * W1, size=min(num_samples, H1 * W1), replace=False)
            
            # Compute similarities (dot product)
            desc1_sampled = desc1_flat[indices1]  # [500, D]
            sim_matrix = desc1_sampled @ desc2_flat.T  # [500, H2*W2]
            
            # Find mutual nearest neighbors
            best_idx2 = sim_matrix.argmax(axis=1)  # [500]
            best_sim = sim_matrix.max(axis=1)
            
            # Check mutual matches
            sim_matrix_reverse = desc2_flat[best_idx2] @ desc1_flat.T  # [500, H1*W1]
            best_idx1_reverse = sim_matrix_reverse.argmax(axis=1)
            
            # Keep only mutual matches
            mutual_mask = (best_idx1_reverse == indices1) & (best_sim > 0.7)
            
            # Extract matches
            matched_idx1 = indices1[mutual_mask]
            matched_idx2 = best_idx2[mutual_mask]
            matched_sim = best_sim[mutual_mask]
            
            # Select top 20 by similarity
            if len(matched_sim) > 20:
                top_indices = np.argsort(matched_sim)[-20:]
                matched_idx1 = matched_idx1[top_indices]
                matched_idx2 = matched_idx2[top_indices]
                matched_sim = matched_sim[top_indices]
            
            # Convert to x, y coordinates
            y1 = matched_idx1 // W1
            x1 = matched_idx1 % W1
            y2 = matched_idx2 // W2
            x2 = matched_idx2 % W2
            
            if len(matched_sim) > 0:
                pts1 = np.stack([x1, y1], axis=-1)
                pts2 = np.stack([x2, y2], axis=-1)
                
                print(f"✅ Found {len(pts1)} mutual nearest neighbor matches\n")
                
                # Save matching visualization
                visualize_matches(
                    img1_np, img2_np, pts1, pts2, matched_sim,
                    os.path.join(scene_output_dir, 'matching_visualization.png')
                )
            else:
                print("⚠️  No matches found (mutual nearest neighbors with similarity > 0.7)\n")
                # Still save an empty visualization
                visualize_matches(
                    img1_np, img2_np, np.array([]), np.array([]), np.array([]),
                    os.path.join(scene_output_dir, 'matching_visualization.png')
                )
        
        # 4. Save original images side by side
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img1_np)
        plt.title(f'{scene_name} - Image 1')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img2_np)
        plt.title(f'{scene_name} - Image 2')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(scene_output_dir, 'input_images.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved input images: {scene_output_dir}/input_images.png")
        
        # Print statistics for this pair
        print(f"\n📊 Statistics for {scene_name}:")
        print(f"  Depth 1 range: [{depth1.min():.3f}, {depth1.max():.3f}], mean: {depth1.mean():.3f}")
        print(f"  Depth 2 range: [{depth2.min():.3f}, {depth2.max():.3f}], mean: {depth2.mean():.3f}")
        print(f"  Confidence 1 range: [{conf1.min():.3f}, {conf1.max():.3f}], mean: {conf1.mean():.3f}")
        print(f"  Confidence 2 range: [{conf2.min():.3f}, {conf2.max():.3f}], mean: {conf2.mean():.3f}")
        if desc1 is not None:
            num_matches = 0
            if 'pts1' in locals() and len(pts1) > 0:
                num_matches = len(pts1)
            print(f"  Matches: {num_matches}")
        print()
    
    print("\n" + "="*80)
    print("✅ All visualizations completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Processed {total_pairs} image pairs")
    print("="*80)

if __name__ == '__main__':
    main()

