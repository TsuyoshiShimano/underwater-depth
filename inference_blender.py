#!/usr/bin/env python3
"""
MASt3R inference script for Blender processed data
Processes images from /workspace/dust3r/test/5a489fb1c7dab83a7d7b1070/down_caustics
"""

import os
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs


def load_exr_depth(exr_path):
    """Load depth from EXR file and filter invalid values (1e10)"""
    try:
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(str(exr_path))
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Try to read depth channel
        depth_channel = None
        for channel_name in ['V', 'Z', 'Y', 'R']:
            if channel_name in header['channels']:
                depth_channel = channel_name
                break
        
        if depth_channel is None:
            raise ValueError(f"No depth channel found in {exr_path}")
        
        depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape((height, width))
        exr_file.close()
        
        # Filter out invalid depth values (1e10)
        INVALID_DEPTH_THRESHOLD = 1e9
        depth[depth >= INVALID_DEPTH_THRESHOLD] = np.nan
        
        return depth
    except Exception as e:
        print(f"Error loading EXR file {exr_path}: {e}")
        return None


def load_images_from_directory(test_dir):
    """Load image paths from test directory structure"""
    test_path = Path(test_dir)
    image_paths = []
    
    # Find all view directories
    view_dirs = sorted([d for d in test_path.iterdir() if d.is_dir() and d.name.startswith('view_')])
    
    for view_dir in view_dirs:
        color_dir = view_dir / 'color'
        if not color_dir.exists():
            continue
        
        # Find color images
        color_files = sorted([f for f in color_dir.iterdir() if f.suffix == '.png'])
        
        for color_file in color_files:
            image_paths.append(str(color_file))
    
    return image_paths


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load pretrained MASt3R model
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    print(f"Loading model: {model_name}")
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    model.eval()
    
    # Load images from test directory
    test_dir = "/workspace/dust3r/test/5a489fb1c7dab83a7d7b1070/down_caustics"
    print(f"Loading images from: {test_dir}")
    image_paths = load_images_from_directory(test_dir)
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load images using dust3r's load_images function
    loaded_images = load_images(image_paths, size=512)
    
    # Create pairs (complete graph for all images)
    pairs = make_pairs(loaded_images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"Created {len(pairs)} image pairs")
    
    # Run inference
    print("Running inference...")
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    
    # Extract results
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    
    print("\nInference completed!")
    print(f"View1 keys: {list(pred1.keys())}")
    print(f"View2 keys: {list(pred2.keys())}")
    
    # Save depth maps if available
    if 'depth' in pred1:
        depth1 = pred1['depth']
        print(f"Depth1 shape: {depth1.shape}")
        # Save depth map
        depth_np = depth1[0].cpu().numpy() if isinstance(depth1, torch.Tensor) else depth1[0]
        depth_np = (depth_np / depth_np.max() * 255).astype(np.uint8)
        cv2.imwrite('output_depth1.png', depth_np)
        print("Saved depth1 to output_depth1.png")
    
    if 'depth' in pred2:
        depth2 = pred2['depth']
        print(f"Depth2 shape: {depth2.shape}")
        # Save depth map
        depth_np = depth2[0].cpu().numpy() if isinstance(depth2, torch.Tensor) else depth2[0]
        depth_np = (depth_np / depth_np.max() * 255).astype(np.uint8)
        cv2.imwrite('output_depth2.png', depth_np)
        print("Saved depth2 to output_depth2.png")
    
    # Save confidence maps if available
    if 'conf' in pred1:
        conf1 = pred1['conf']
        print(f"Conf1 shape: {conf1.shape}")
        conf_np = conf1[0].cpu().numpy() if isinstance(conf1, torch.Tensor) else conf1[0]
        if conf_np.ndim == 3:
            conf_np = conf_np[0]  # Take first channel if multi-channel
        conf_np = (conf_np / conf_np.max() * 255).astype(np.uint8)
        cv2.imwrite('output_conf1.png', conf_np)
        print("Saved conf1 to output_conf1.png")
    
    print("\nInference results saved!")


if __name__ == '__main__':
    main()

