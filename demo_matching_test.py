#!/usr/bin/env python3
"""
MASt3R Demo: Test descriptor matching with pretrained model
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from mast3r.model import AsymmetricMASt3R
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
    
    ax.set_title(f'MASt3R Matching Results (N={len(pts1)} matches, '
                f'Conf: min={confidences.min():.3f}, mean={confidences.mean():.3f}, max={confidences.max():.3f})', 
                fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved matching visualization: {save_path}")

def visualize_depth(depth_map, img, save_path, title="Depth Map"):
    """Visualize depth map"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Depth map
    im = ax2.imshow(depth_map, cmap='turbo')
    ax2.set_title(title)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved depth visualization: {save_path}")

def main():
    # Settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric'
    
    # Image paths
    img1_path = '/workspace/underwater/dust3r/data/blender_processed/5b2c67b5e0878c381608b8d8/down/00000018.jpg'
    img2_path = '/workspace/underwater/dust3r/data/blender_processed/5b2c67b5e0878c381608b8d8/down/00000020.jpg'
    
    output_dir = '/workspace/underwater/mast3r/demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("MASt3R Demo: Descriptor Matching Test")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")
    
    # Load model
    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    print("✅ Model loaded successfully\n")
    
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
        
        # Debug: check output structure
        print(f"\nDebug - pred1 keys: {list(pred1.keys())}")
        print(f"Debug - pred2 keys: {list(pred2.keys())}")
    
    print("✅ Inference completed\n")
    
    # Extract results (handle asymmetric output structure)
    # pred1 has 'pts3d' (points in view1 coordinate)
    # pred2 has 'pts3d_in_other_view' (points from view2 in view1 coordinate)
    pts3d_1 = pred1['pts3d'][0].cpu().numpy()  # [H, W, 3]
    pts3d_2 = pred2['pts3d_in_other_view'][0].cpu().numpy() if 'pts3d_in_other_view' in pred2 else pred2['pts3d'][0].cpu().numpy()
    
    if 'conf' in pred1:
        conf1 = pred1['conf'][0].cpu().numpy()  # [H, W]
        conf2 = pred2['conf'][0].cpu().numpy()
    else:
        conf1 = np.ones((pts3d_1.shape[0], pts3d_1.shape[1]))
        conf2 = np.ones((pts3d_2.shape[0], pts3d_2.shape[1]))
    
    if 'desc' in pred1:
        desc1 = pred1['desc'][0].cpu().numpy()  # [H, W, D]
        desc2 = pred2['desc'][0].cpu().numpy()
    else:
        raise ValueError(f"Cannot find desc in output. Available keys: {list(pred1.keys())}")
    
    H1, W1, D = desc1.shape
    H2, W2, _ = desc2.shape
    
    print(f"Results:")
    print(f"  Descriptors 1 shape: {desc1.shape}")
    print(f"  Descriptors 2 shape: {desc2.shape}")
    print(f"  3D points 1 shape: {pts3d_1.shape}")
    print(f"  3D points 2 shape: {pts3d_2.shape}\n")
    
    # Compute depth from 3D points
    depth1 = np.linalg.norm(pts3d_1, axis=-1)  # [H, W]
    depth2 = np.linalg.norm(pts3d_2, axis=-1)
    
    # Find matches using Fast Reciprocal Matching (simplified version)
    print("Computing matches...")
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
    # 1. For each point in img1, find best match in img2
    best_idx2 = sim_matrix.argmax(axis=1)  # [500]
    best_sim = sim_matrix.max(axis=1)
    
    # 2. For each matched point in img2, check if it points back to img1
    sim_matrix_reverse = desc2_flat[best_idx2] @ desc1_flat.T  # [500, H1*W1]
    best_idx1_reverse = sim_matrix_reverse.argmax(axis=1)
    
    # 3. Keep only mutual matches
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
    
    pts1 = np.stack([x1, y1], axis=-1)
    pts2 = np.stack([x2, y2], axis=-1)
    
    print(f"✅ Found {len(pts1)} mutual nearest neighbor matches\n")
    
    # Prepare images for visualization
    img1_np = img1_tensor[0].permute(1, 2, 0).cpu().numpy()
    img2_np = img2_tensor[0].permute(1, 2, 0).cpu().numpy()
    
    # Save visualizations
    print("Saving visualizations...")
    
    # 1. Matching visualization
    visualize_matches(
        img1_np, img2_np, pts1, pts2, matched_sim,
        os.path.join(output_dir, 'matching_visualization.png')
    )
    
    # 2. Depth map 1
    visualize_depth(
        depth1, img1_np,
        os.path.join(output_dir, 'depth_map_img1.png'),
        title='Depth Map - Image 1'
    )
    
    # 3. Depth map 2
    visualize_depth(
        depth2, img2_np,
        os.path.join(output_dir, 'depth_map_img2.png'),
        title='Depth Map - Image 2'
    )
    
    # 4. Save original images
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_np)
    plt.title('Image 1')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img2_np)
    plt.title('Image 2')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'input_images.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved input images: {output_dir}/input_images.png")
    
    print("\n" + "="*80)
    print("✅ Demo completed successfully!")
    print(f"📁 All outputs saved to: {output_dir}")
    print("="*80)
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"  Matches: {len(pts1)}")
    print(f"  Similarity range: [{matched_sim.min():.3f}, {matched_sim.max():.3f}]")
    print(f"  Depth 1 range: [{depth1.min():.3f}, {depth1.max():.3f}]")
    print(f"  Depth 2 range: [{depth2.min():.3f}, {depth2.max():.3f}]")

if __name__ == '__main__':
    main()

