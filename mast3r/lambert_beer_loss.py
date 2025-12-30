# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Lambert-Beer law based color restoration loss for underwater matching
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.losses import BaseCriterion, MultiLoss
from mast3r.losses import get_similarities, MatchingLoss


class LambertBeerColorLoss(BaseCriterion):
    """
    Lambert-Beer law based color restoration loss.
    
    For matched pixels with similarity >= threshold, compute the color restoration
    using Lambert-Beer law and compare with ground truth restored colors.
    
    Modified Lambert-Beer law for underwater imaging (with different β for two terms):
    I(x,y) = I_0(x,y) * exp(-β_2 * d(x,y)) + B_inf * (1 - exp(-β_1 * d(x,y)))
    
    Where:
    - I(x,y): observed image color
    - I_0(x,y): restored color (true color)
    - d(x,y): depth
    - β_1: absorption coefficient for background light term (RGB channels)
    - β_2: absorption coefficient for restoration term (RGB channels)
    - B_inf: background light (color at infinity)
    
    Restored color:
    I_0 = (I - B_inf * (1 - exp(-β_1*d))) / exp(-β_2*d)
    
    Parameters β_1, β_2, and B_inf can be:
    - Fixed values (when use_predicted_params=False)
    - Predicted per-pixel by the model head (when use_predicted_params=True)
    """
    
    # クラス変数でoutput_dirを共有（トレーニングループから設定可能）
    _global_output_dir = None
    
    def __init__(self, similarity_threshold=0.7, beta_1=0.1, beta_2=0.1, b_inf=None,
                 reduction='mean', fp=torch.float32, use_predicted_params=True, max_matches=100,
                 visualize_matches=True, output_dir=None, initial_samples=500, max_iterations=10):
        super().__init__(reduction)
        self.similarity_threshold = similarity_threshold
        self.beta_1 = beta_1 if isinstance(beta_1, torch.Tensor) else torch.tensor(beta_1)
        self.beta_2 = beta_2 if isinstance(beta_2, torch.Tensor) else torch.tensor(beta_2)
        self.b_inf = b_inf if b_inf is not None else torch.tensor([0.5, 0.5, 0.5])
        self.fp = fp
        self.use_predicted_params = use_predicted_params  # If True, use predicted params from head
        self.max_matches = max_matches  # Maximum number of matches to use (top-K by similarity)
        self.enable_visualization = visualize_matches  # If True, visualize matching results
        self.initial_samples = initial_samples  # Initial number of sparse samples for Fast Reciprocal Matching
        self.max_iterations = max_iterations  # Maximum iterations for Fast Reciprocal Matching
        
        # output_dirを自動検出（クラス変数、環境変数、グローバル変数から取得）
        if output_dir is None:
            # クラス変数から取得
            output_dir = LambertBeerColorLoss._global_output_dir
            
            # それでもNoneなら、argsから取得を試みる
            if output_dir is None:
                import sys
                if hasattr(sys.modules.get('__main__'), 'args'):
                    main_args = getattr(sys.modules.get('__main__'), 'args', None)
                    if main_args and hasattr(main_args, 'output_dir'):
                        output_dir = main_args.output_dir
        
        self.output_dir = output_dir  # Directory to save visualizations
        self.vis_counter = 0  # Counter for visualization files
        
        if self.enable_visualization and self.output_dir:
            print(f"[LambertBeerColorLoss] マッチング可視化が有効です。保存先: {self.output_dir}/matching_visualizations/")
        
    def visualize_matches(self, img1, img2, x1, y1, x2, y2, similarities, save_path):
        """
        マッチング結果を可視化して保存
        
        Args:
            img1: [H, W, 3] 画像1 (numpy array, 0-1 range)
            img2: [H, W, 3] 画像2 (numpy array, 0-1 range)
            x1, y1: [N] 画像1のマッチング点座標
            x2, y2: [N] 画像2のマッチング点座標
            similarities: [N] マッチングの類似度
            save_path: 保存先パス
        """
        H1, W1 = img1.shape[:2]
        H2, W2 = img2.shape[:2]
        
        # 画像を横に並べる
        H = max(H1, H2)
        combined = np.ones((H, W1 + W2, 3), dtype=np.float32)
        combined[:H1, :W1] = img1
        combined[:H2, W1:] = img2
        
        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        ax.imshow(combined)
        ax.axis('off')
        
        # マッチング点を線で結ぶ
        # 類似度に応じて色を変える（高い類似度ほど緑、低いほど赤）
        for i in range(len(x1)):
            sim = similarities[i]
            # 類似度を0-1にクリップ
            sim_normalized = np.clip(sim, 0, 1)
            # 緑(高類似度)から赤(低類似度)へのグラデーション
            color = plt.cm.RdYlGn(sim_normalized)
            
            # 画像1の点
            pt1_x = x1[i]
            pt1_y = y1[i]
            # 画像2の点（横方向にW1オフセット）
            pt2_x = x2[i] + W1
            pt2_y = y2[i]
            
            # 線を引く
            ax.plot([pt1_x, pt2_x], [pt1_y, pt2_y], 
                   color=color, linewidth=1, alpha=0.6)
            
            # 点を描画
            ax.plot(pt1_x, pt1_y, 'o', color=color, markersize=3)
            ax.plot(pt2_x, pt2_y, 'o', color=color, markersize=3)
        
        # タイトルに統計情報を追加
        ax.set_title(f'Matching Results (N={len(x1)}, Sim: min={similarities.min():.3f}, '
                    f'mean={similarities.mean():.3f}, max={similarities.max():.3f})', 
                    fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[Visualization] マッチング可視化を保存: {save_path}")
    
    def restore_color(self, observed_color, depth, beta_1, beta_2, b_inf):
        """
        Restore color using Lambert-Beer law with different β for two terms.
        
        Args:
            observed_color: [B, N, 3] observed RGB colors
            depth: [B, N] or [B, N, 1] depth values
            beta_1: [3] or [B, N, 3] absorption coefficient for background term
            beta_2: [3] or [B, N, 3] absorption coefficient for restoration term
            b_inf: [3] or [B, N, 3] background light for RGB
            
        Returns:
            restored_color: [B, N, 3] restored RGB colors
        """
        # Ensure proper dimensions
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)  # [B, N, 1]
        if beta_1.dim() == 1:
            beta_1 = beta_1.unsqueeze(0).unsqueeze(0).expand_as(observed_color)  # [B, N, 3]
        if beta_2.dim() == 1:
            beta_2 = beta_2.unsqueeze(0).unsqueeze(0).expand_as(observed_color)  # [B, N, 3]
        if b_inf.dim() == 1:
            b_inf = b_inf.unsqueeze(0).unsqueeze(0).expand_as(observed_color)  # [B, N, 3]
            
        # Compute exp(-β_1*d) and exp(-β_2*d)
        exp_term_1 = torch.exp(-beta_1 * depth)  # [B, N, 3]
        exp_term_2 = torch.exp(-beta_2 * depth)  # [B, N, 3]
        
        # Avoid division by zero
        exp_term_2 = torch.clamp(exp_term_2, min=1e-8)
        
        # Restored color: I_0 = (I - B_inf * (1 - exp(-β_1*d))) / exp(-β_2*d)
        restored_color = (observed_color - b_inf * (1 - exp_term_1)) / exp_term_2
        
        return restored_color
        
    def forward(self, desc1, desc2, img1_colors, img2_colors, depth1, depth2,
                valid_matches=None, beta_1=None, beta_2=None, b_inf=None, 
                pred1=None, pred2=None, x1=None, y1=None, x2=None, y2=None, batchid=None, **kwargs):
        """
        Compute Lambert-Beer color restoration loss for high-confidence matches.
        
        Args:
            desc1: [B, N, D] descriptors from image 1
            desc2: [B, N, D] descriptors from image 2 (matched with desc1)
            img1_colors: [B, N, 3] RGB colors at matched pixels in image 1
            img2_colors: [B, N, 3] RGB colors at matched pixels in image 2
            depth1: [B, N] depth values at matched pixels in image 1
            depth2: [B, N] depth values at matched pixels in image 2
            valid_matches: [B, N] boolean mask for valid matches
            beta_1: [3] or None, absorption coefficient 1 (uses self.beta_1 if None)
            beta_2: [3] or None, absorption coefficient 2 (uses self.beta_2 if None)
            b_inf: [3] or None, background light (uses self.b_inf if None)
            pred1, pred2: prediction dicts from model (if using predicted params)
            x1, y1, x2, y2: pixel coordinates for extracting predicted params
            batchid: batch indices for extracting predicted params
            
        Returns:
            loss: scalar loss value
        """
        desc1 = desc1.to(self.fp)
        desc2 = desc2.to(self.fp)
        
        # Compute similarities (dot product)
        sim = get_similarities(desc1, desc2, euc=False)  # [B, N, M]
        
        B = sim.shape[0]
        N = sim.shape[1]  # desc1の点数
        M = sim.shape[2]  # desc2の点数
        
        # Mutual nearest neighbor matching
        # 1. desc1の各点から見たdesc2での最近傍
        best_sim_1to2, best_idx_1to2 = sim.max(dim=2)  # [B, N]
        
        # 2. desc2の各点から見たdesc1での最近傍
        best_sim_2to1, best_idx_2to1 = sim.max(dim=1)  # [B, M]
        
        # 3. 相互最近傍条件をチェック（効率的な実装）
        # desc1の各点に対応するdesc2の点が、逆にdesc1のどこを指すか
        reverse_indices = torch.gather(best_idx_2to1, 1, best_idx_1to2)  # [B, N]
        
        # 相互最近傍マスク（自分を指し返している）
        arange = torch.arange(N, device=desc1.device).unsqueeze(0).expand(B, -1)
        mutual_mask = (reverse_indices == arange)
        
        # 4. 閾値条件と相互最近傍条件を組み合わせ
        match_sim = best_sim_1to2  # [B, N]
        high_conf_mask = (match_sim >= self.similarity_threshold) & mutual_mask  # [B, N]
        
        # Apply valid_matches mask if provided
        if valid_matches is not None:
            high_conf_mask = high_conf_mask & valid_matches
        
        # 閾値以上の点の統計情報を出力
        num_high_conf = high_conf_mask.sum().item()
        num_mutual = mutual_mask.sum().item()
        num_total = high_conf_mask.numel()
        print(f"[LambertBeerColorLoss] 総点数: {num_total}, "
              f"相互最近傍: {num_mutual}, "
              f"閾値{self.similarity_threshold}以上+相互最近傍: {num_high_conf} "
              f"({100.0 * num_high_conf / num_total:.2f}%)")
        
        # バッチごとの詳細統計
        for b in range(B):
            batch_high_conf = high_conf_mask[b].sum().item()
            batch_total = high_conf_mask[b].numel()
            batch_max_sim = match_sim[b].max().item() if match_sim[b].numel() > 0 else 0.0
            batch_min_sim = match_sim[b].min().item() if match_sim[b].numel() > 0 else 0.0
            batch_mean_sim = match_sim[b].mean().item() if match_sim[b].numel() > 0 else 0.0
            print(f"  Batch {b}: 閾値以上 {batch_high_conf}/{batch_total} "
                  f"(類似度: min={batch_min_sim:.4f}, mean={batch_mean_sim:.4f}, max={batch_max_sim:.4f})")
            
        if not high_conf_mask.any():
            print(f"[LambertBeerColorLoss] 警告: 閾値{self.similarity_threshold}以上の点が見つかりませんでした。Loss=0を返します。")
            return torch.tensor(0.0, device=desc1.device, dtype=self.fp, requires_grad=True)
        
        # Select top-K matches by similarity (to limit number of matches)
        # Process per batch to select top-K matches within each batch
        B, N = match_sim.shape
        final_mask = torch.zeros_like(high_conf_mask, dtype=torch.bool)
        
        for b in range(B):
            batch_high_conf_mask = high_conf_mask[b]  # [N]
            batch_match_sim = match_sim[b]  # [N]
            
            if not batch_high_conf_mask.any():
                continue
            
            # Get similarities of high-confidence matches in this batch
            batch_high_conf_sim = batch_match_sim[batch_high_conf_mask]  # [M_b]
            
            if len(batch_high_conf_sim) > self.max_matches:
                # Select top-K by similarity within this batch
                _, topk_indices = torch.topk(batch_high_conf_sim, k=self.max_matches, dim=0)
                # Get global indices within batch
                batch_high_conf_indices = torch.where(batch_high_conf_mask)[0]  # Indices of high-conf matches
                topk_batch_indices = batch_high_conf_indices[topk_indices]  # Top-K indices within batch
                
                # Create mask for top-K matches
                batch_final_mask = torch.zeros(N, dtype=torch.bool, device=high_conf_mask.device)
                batch_final_mask[topk_batch_indices] = True
                final_mask[b] = batch_final_mask
            else:
                # Use all high-confidence matches if less than max_matches
                final_mask[b] = batch_high_conf_mask
        
        # top-K選択後の統計情報を出力
        num_final_matches = final_mask.sum().item()
        print(f"[LambertBeerColorLoss] top-K選択後の最終マッチ数: {num_final_matches} "
              f"(max_matches={self.max_matches} per batch)")
        
        if not final_mask.any():
            print(f"[LambertBeerColorLoss] 警告: top-K選択後にマッチが残りませんでした。Loss=0を返します。")
            return torch.tensor(0.0, device=desc1.device, dtype=self.fp, requires_grad=True)
        
        # マッチング可視化（閾値を超える点がある場合）
        # 頻度を減らすため、10000回に1回のみ可視化
        if self.enable_visualization and self.output_dir is not None and num_final_matches > 0 and self.vis_counter % 10000 == 0:
            # バッチごとに可視化
            for b in range(B):
                batch_mask = final_mask[b]
                if not batch_mask.any():
                    continue
                
                # このバッチのマッチング点を取得
                batch_indices = torch.where(batch_mask)[0]
                batch_x1 = x1[b, batch_indices].detach().cpu().numpy()
                batch_y1 = y1[b, batch_indices].detach().cpu().numpy()
                batch_x2 = x2[b, batch_indices].detach().cpu().numpy()
                batch_y2 = y2[b, batch_indices].detach().cpu().numpy()
                batch_sim = match_sim[b, batch_indices].detach().cpu().numpy()
                
                # 画像を取得（kwargsから）
                if 'gt1' in kwargs and 'gt2' in kwargs:
                    gt1 = kwargs['gt1']
                    gt2 = kwargs['gt2']
                    
                    if 'img' in gt1 and 'img' in gt2:
                        img1_full = gt1['img'][b]  # [3, H, W] or [H, W, 3]
                        img2_full = gt2['img'][b]
                        
                        # [H, W, 3]形式に変換
                        if img1_full.shape[0] == 3:
                            img1_full = img1_full.permute(1, 2, 0)
                            img2_full = img2_full.permute(1, 2, 0)
                        
                        img1_np = img1_full.cpu().numpy()
                        img2_np = img2_full.cpu().numpy()
                        
                        # [0, 1]範囲にクリップ
                        img1_np = np.clip(img1_np, 0, 1)
                        img2_np = np.clip(img2_np, 0, 1)
                        
                        # 保存先ディレクトリを作成
                        vis_dir = os.path.join(self.output_dir, 'matching_visualizations')
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # ファイル名
                        save_path = os.path.join(vis_dir, f'match_{self.vis_counter:06d}_batch{b}.png')
                        
                        # 可視化
                        self.visualize_matches(img1_np, img2_np, 
                                             batch_x1, batch_y1, batch_x2, batch_y2,
                                             batch_sim, save_path)
        
        # カウンターは常にインクリメント（可視化の有無に関わらず）
        self.vis_counter += 1
        
        # Select high-confidence matched pixels (now limited to top-K per batch)
        img1_colors_matched = img1_colors[final_mask]  # [M, 3] where M <= max_matches * B
        img2_colors_matched = img2_colors[final_mask]  # [M, 3]
        depth1_matched = depth1[final_mask]  # [M]
        depth2_matched = depth2[final_mask]  # [M]
        
        # Get parameters (predicted or fixed)
        # Initialize variables for logging
        b_inf_is_fixed = False
        b_inf_img1_value = None
        b_inf_img2_value = None
        
        if self.use_predicted_params and pred1 is not None and pred2 is not None and 'beta_1' in pred1:
            # Extract predicted parameters at matched pixels
            # Handle both [B, H, W, 3] and [B, 3, H, W] formats
            beta_1_map1 = pred1['beta_1']
            beta_1_map2 = pred2['beta_1']
            
            # Convert to [B, H, W, 3] if needed
            if beta_1_map1.dim() == 4 and beta_1_map1.shape[1] == 3:  # [B, 3, H, W]
                beta_1_map1 = beta_1_map1.permute(0, 2, 3, 1)
                beta_1_map2 = beta_1_map2.permute(0, 2, 3, 1)
                beta_2_map1 = pred1['beta_2'].permute(0, 2, 3, 1)
                beta_2_map2 = pred2['beta_2'].permute(0, 2, 3, 1)
                # b_inf may not exist if fix_background_color_from_depth is enabled
                if 'b_inf' in pred1 and 'b_inf' in pred2:
                    b_inf_map1 = pred1['b_inf'].permute(0, 2, 3, 1)
                    b_inf_map2 = pred2['b_inf'].permute(0, 2, 3, 1)
                else:
                    b_inf_map1 = None
                    b_inf_map2 = None
            else:  # Already [B, H, W, 3]
                beta_2_map1 = pred1['beta_2']
                beta_2_map2 = pred2['beta_2']
                # b_inf may not exist if fix_background_color_from_depth is enabled
                if 'b_inf' in pred1 and 'b_inf' in pred2:
                    b_inf_map1 = pred1['b_inf']
                    b_inf_map2 = pred2['b_inf']
                else:
                    b_inf_map1 = None
                    b_inf_map2 = None
            
            # Extract at matched pixels
            beta_1_1 = beta_1_map1[batchid, y1, x1][high_conf_mask]  # [M, 3]
            beta_2_1 = beta_2_map1[batchid, y1, x1][high_conf_mask]  # [M, 3]
            
            beta_1_2 = beta_1_map2[batchid, y2, x2][high_conf_mask]  # [M, 3]
            beta_2_2 = beta_2_map2[batchid, y2, x2][high_conf_mask]  # [M, 3]
            
            # Average parameters from both images
            beta_1_use = (beta_1_1 + beta_1_2) / 2
            beta_2_use = (beta_2_1 + beta_2_2) / 2
            
            # Handle b_inf (may be fixed from depth or predicted)
            # Check if b_inf is fixed from depth (when fix_background_color_from_depth is enabled)
            # In that case, b_inf_map exists but contains fixed values, not predicted ones
            if b_inf_map1 is not None and b_inf_map2 is not None:
                # Check if b_inf values are constant (fixed from depth) or vary (predicted)
                # Fixed from depth: values should be constant across spatial dimensions
                b_inf_1_sample = b_inf_map1[0, 0, 0, :]  # Sample one pixel
                b_inf_1_all = b_inf_map1[0, :, :, :].reshape(-1, 3)  # All pixels in first batch
                # If all pixels have the same value (within tolerance), it's fixed from depth
                if torch.allclose(b_inf_1_all, b_inf_1_sample.unsqueeze(0).expand_as(b_inf_1_all), atol=1e-5):
                    b_inf_is_fixed = True
                    b_inf_img1_value = b_inf_1_sample.cpu().numpy()  # Store fixed value for image 1
                
                b_inf_2_sample = b_inf_map2[0, 0, 0, :]  # Sample one pixel
                b_inf_2_all = b_inf_map2[0, :, :, :].reshape(-1, 3)  # All pixels in first batch
                if torch.allclose(b_inf_2_all, b_inf_2_sample.unsqueeze(0).expand_as(b_inf_2_all), atol=1e-5):
                    b_inf_img2_value = b_inf_2_sample.cpu().numpy()  # Store fixed value for image 2
                
                b_inf_1 = b_inf_map1[batchid, y1, x1][high_conf_mask]  # [M, 3]
                b_inf_2 = b_inf_map2[batchid, y2, x2][high_conf_mask]  # [M, 3]
                # Average of fixed background colors from both images
                # Note: If images have different background colors, this will vary across matches
                b_inf_use = (b_inf_1 + b_inf_2) / 2
            else:
                # b_inf is fixed from depth, use fixed parameters
                b_inf_is_fixed = True
                b_inf_use = b_inf if b_inf is not None else self.b_inf
                if isinstance(b_inf_use, torch.Tensor):
                    b_inf_use = b_inf_use.to(desc1.device)
                else:
                    b_inf_use = torch.tensor(b_inf_use, device=desc1.device, dtype=self.fp)
                if b_inf_use.dim() == 0:
                    b_inf_use = b_inf_use.expand_as(img1_colors_matched)
        else:
            # Use fixed parameters
            beta_1_use = beta_1 if beta_1 is not None else self.beta_1
            beta_2_use = beta_2 if beta_2 is not None else self.beta_2
            b_inf_use = b_inf if b_inf is not None else self.b_inf
            
            # Move to device and expand
            if isinstance(beta_1_use, torch.Tensor):
                beta_1_use = beta_1_use.to(desc1.device)
            else:
                beta_1_use = torch.tensor(beta_1_use, device=desc1.device, dtype=self.fp)
                
            if isinstance(beta_2_use, torch.Tensor):
                beta_2_use = beta_2_use.to(desc1.device)
            else:
                beta_2_use = torch.tensor(beta_2_use, device=desc1.device, dtype=self.fp)
                
            if isinstance(b_inf_use, torch.Tensor):
                b_inf_use = b_inf_use.to(desc1.device)
            else:
                b_inf_use = torch.tensor(b_inf_use, device=desc1.device, dtype=self.fp)
            
            # Expand to match shape [M, 3]
            if beta_1_use.dim() == 0:
                beta_1_use = beta_1_use.expand_as(img1_colors_matched)
            if beta_2_use.dim() == 0:
                beta_2_use = beta_2_use.expand_as(img1_colors_matched)
            if b_inf_use.dim() == 0:
                b_inf_use = b_inf_use.expand_as(img1_colors_matched)
        
        # Restore colors for both images
        restored_color1 = self.restore_color(img1_colors_matched, depth1_matched, beta_1_use, beta_2_use, b_inf_use)
        restored_color2 = self.restore_color(img2_colors_matched, depth2_matched, beta_1_use, beta_2_use, b_inf_use)
        
        # Compute L2 loss between restored colors
        color_diff = restored_color1 - restored_color2  # [M, 3]
        loss = torch.mean(color_diff ** 2)  # Mean squared error
        
        # 色復元とLoss計算の詳細情報を出力
        print(f"[LambertBeerColorLoss] 色復元統計:")
        print(f"  [入力データ]")
        print(f"    観測色1: min={img1_colors_matched.min().item():.4f}, max={img1_colors_matched.max().item():.4f}, mean={img1_colors_matched.mean().item():.4f}")
        print(f"    観測色2: min={img2_colors_matched.min().item():.4f}, max={img2_colors_matched.max().item():.4f}, mean={img2_colors_matched.mean().item():.4f}")
        print(f"    深度1: min={depth1_matched.min().item():.4f}, max={depth1_matched.max().item():.4f}, mean={depth1_matched.mean().item():.4f}")
        print(f"    深度2: min={depth2_matched.min().item():.4f}, max={depth2_matched.max().item():.4f}, mean={depth2_matched.mean().item():.4f}")
        print(f"  [Lambert-Beerパラメータ]")
        if self.use_predicted_params and pred1 is not None and 'beta_1' in pred1:
            if beta_1_use.numel() > 0:
                print(f"    β1 (予測値): min={beta_1_use.min().item():.4f}, max={beta_1_use.max().item():.4f}, mean={beta_1_use.mean().item():.4f}")
                print(f"    β2 (予測値): min={beta_2_use.min().item():.4f}, max={beta_2_use.max().item():.4f}, mean={beta_2_use.mean().item():.4f}")
                # Check if b_inf is fixed from depth
                if 'b_inf_is_fixed' in locals() and b_inf_is_fixed:
                    # Fixed from depth: show individual image values and averaged value
                    print(f"    B_inf (固定背景色 - 深度から計算):")
                    if 'b_inf_img1_value' in locals() and b_inf_img1_value is not None:
                        print(f"      画像1の固定背景色: RGB=({b_inf_img1_value[0]:.4f}, {b_inf_img1_value[1]:.4f}, {b_inf_img1_value[2]:.4f})")
                    if 'b_inf_img2_value' in locals() and b_inf_img2_value is not None:
                        print(f"      画像2の固定背景色: RGB=({b_inf_img2_value[0]:.4f}, {b_inf_img2_value[1]:.4f}, {b_inf_img2_value[2]:.4f})")
                    # Calculate the actual average value
                    if 'b_inf_img1_value' in locals() and b_inf_img1_value is not None and 'b_inf_img2_value' in locals() and b_inf_img2_value is not None:
                        avg_rgb = (b_inf_img1_value + b_inf_img2_value) / 2
                        print(f"      使用値（画像1と2の平均）: RGB=({avg_rgb[0]:.4f}, {avg_rgb[1]:.4f}, {avg_rgb[2]:.4f})")
                        print(f"      ※全マッチで同じ固定値を使用。min/max/meanはRGB 3チャンネル間の違いを示します")
                    else:
                        # Fallback: show statistics
                        print(f"      使用値: min={b_inf_use.min().item():.4f}, max={b_inf_use.max().item():.4f}, mean={b_inf_use.mean().item():.4f}")
                        print(f"      ※全マッチで同じ固定値を使用。min/max/meanはRGB 3チャンネル間の違いを示します")
                else:
                    print(f"    B_inf (予測値): min={b_inf_use.min().item():.4f}, max={b_inf_use.max().item():.4f}, mean={b_inf_use.mean().item():.4f}")
        print(f"  [色復元結果]")
        print(f"  復元色1: min={restored_color1.min().item():.4f}, max={restored_color1.max().item():.4f}, mean={restored_color1.mean().item():.4f}")
        print(f"  復元色2: min={restored_color2.min().item():.4f}, max={restored_color2.max().item():.4f}, mean={restored_color2.mean().item():.4f}")
        print(f"  色差: min={color_diff.min().item():.4f}, max={color_diff.max().item():.4f}, mean={color_diff.mean().item():.4f}")
        print(f"  LambertBeerColorLoss: {loss.item():.6f}")
        
        return loss


class SpatialSmoothnessRegularizer(nn.Module):
    """
    Spatial smoothness regularizer for predicted parameters.
    Encourages spatially homogeneous predictions.
    """
    
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
        
    def compute_spatial_gradient(self, param_map):
        """
        Compute spatial gradients (differences with neighbors).
        
        Args:
            param_map: [B, H, W, C] parameter map
            
        Returns:
            gradient_loss: scalar loss
        """
        # Horizontal gradients
        grad_h = param_map[:, 1:, :, :] - param_map[:, :-1, :, :]  # [B, H-1, W, C]
        # Vertical gradients
        grad_v = param_map[:, :, 1:, :] - param_map[:, :, :-1, :]  # [B, H, W-1, C]
        
        # L2 norm of gradients
        loss = torch.mean(grad_h ** 2) + torch.mean(grad_v ** 2)
        return loss * self.weight
    
    def forward(self, beta_1_map, beta_2_map, b_inf_map):
        """
        Compute spatial smoothness loss for all parameter maps.
        
        Args:
            beta_1_map: [B, H, W, 3] β_1 parameter map
            beta_2_map: [B, H, W, 3] β_2 parameter map
            b_inf_map: [B, H, W, 3] B_inf parameter map
            
        Returns:
            total_reg_loss: scalar regularization loss
        """
        reg_loss = (
            self.compute_spatial_gradient(beta_1_map) +
            self.compute_spatial_gradient(beta_2_map) +
            self.compute_spatial_gradient(b_inf_map)
        )
        return reg_loss


# Example usage:
# 
# from mast3r.losses import InfoNCE, MatchingLoss
# from mast3r.lambert_beer_loss import (
#     LambertBeerColorLoss, 
#     MatchingWithLambertBeerLoss,
#     SpatialSmoothnessRegularizer
# )
# from mast3r.model import AsymmetricMASt3R
# 
# # 1. Create model with underwater parameter prediction
# model = AsymmetricMASt3R(
#     predict_underwater_params=True,  # Enable β_1, β_2, B_inf prediction
#     desc_mode=('norm'),
#     two_confs=True,
#     # ... other parameters
# )
# 
# # 2. Create base matching loss
# base_criterion = InfoNCE(temperature=0.05, mode='proper')
# base_matching_loss = MatchingLoss(base_criterion, negatives_padding=0, blocksize=8192)
# 
# # 3. Create Lambert-Beer color restoration loss (with predicted params)
# lb_loss = LambertBeerColorLoss(
#     similarity_threshold=0.95,  # Only use matches with similarity >= 0.95
#     use_predicted_params=True,  # Use parameters predicted by head
#     reduction='mean'
# )
# 
# # 4. Create spatial smoothness regularizer
# spatial_reg = SpatialSmoothnessRegularizer(weight=0.01)
# 
# # 5. Combine losses
# combined_loss = MatchingWithLambertBeerLoss(
#     base_matching_loss,
#     lb_loss,
#     lambert_beer_weight=0.1,  # Weight for Lambert-Beer loss
#     spatial_reg=spatial_reg,  # Spatial regularization
#     spatial_reg_weight=0.01   # Weight for spatial regularization
# )
# 
# # 6. Use in training (same interface as MatchingLoss)
# loss, details = combined_loss(gt1, gt2, pred1, pred2)
# # details contains: 'lambert_beer_loss', 'spatial_reg_loss', 'matching_loss', 'total_loss'


class MatchingWithLambertBeerLoss(MatchingLoss):
    """
    Extended MatchingLoss that includes Lambert-Beer color restoration loss.
    
    Usage:
        base_matching_loss = MatchingLoss(InfoNCE(...), ...)
        lb_loss = LambertBeerColorLoss(similarity_threshold=0.95, ...)
        spatial_reg = SpatialSmoothnessRegularizer(weight=0.01)
        combined_loss = MatchingWithLambertBeerLoss(
            base_matching_loss, lb_loss, lambert_beer_weight=0.1,
            spatial_reg=spatial_reg, spatial_reg_weight=0.01
        )
    """
    
    def __init__(self, criterion, lambert_beer_loss, lambert_beer_weight=1.0,
                 spatial_reg=None, spatial_reg_weight=0.01, **kwargs):
        super().__init__(criterion, **kwargs)
        self.lambert_beer_loss = lambert_beer_loss
        self.lambert_beer_weight = lambert_beer_weight
        self.spatial_reg = spatial_reg
        self.spatial_reg_weight = spatial_reg_weight
        
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # Get original matching loss
        matching_loss, details = super().compute_loss(gt1, gt2, pred1, pred2, **kw)
        
        # Get descriptors and matching information
        desc1, desc2 = self.get_descs(pred1, pred2)
        
        # Get matched pixel coordinates
        (x1, y1), (x2, y2) = gt1['corres'].unbind(-1), gt2['corres'].unbind(-1)
        valid_matches = gt1['valid_corres']
        
        B, N = x1.shape
        batchid = torch.arange(B, device=x1.device)[:, None].repeat(1, N)
        
        # Extract colors and depths at matched pixels
        # Images should be in gt1/gt2 as 'img' key (normalized RGB)
        img1_colors = None
        img2_colors = None
        
        if 'img' in gt1 and 'img' in gt2:
            img1 = gt1['img']  # [B, 3, H, W] or [B, H, W, 3]
            img2 = gt2['img']
            
            # Convert to [B, H, W, 3] format if needed
            if img1.dim() == 4 and img1.shape[1] == 3:
                img1 = img1.permute(0, 2, 3, 1)
                img2 = img2.permute(0, 2, 3, 1)
            
            # Extract colors at matched pixels
            # Clamp to [0, 1] if normalized
            img1_colors = torch.clamp(img1[batchid, y1, x1], 0, 1)  # [B, N, 3]
            img2_colors = torch.clamp(img2[batchid, y2, x2], 0, 1)  # [B, N, 3]
        
        # Extract depths from predictions
        depth1 = None
        depth2 = None
        
        if 'pts3d' in pred1 and 'pts3d' in pred2:
            # Compute depth from 3D points (distance from camera)
            pts3d_1 = pred1['pts3d'][batchid, y1, x1]  # [B, N, 3]
            pts3d_2 = pred2['pts3d'][batchid, y2, x2]  # [B, N, 3]
            depth1 = torch.norm(pts3d_1, dim=-1)  # [B, N]
            depth2 = torch.norm(pts3d_2, dim=-1)  # [B, N]
        elif 'depth' in pred1 and 'depth' in pred2:
            # Direct depth prediction
            depth1 = pred1['depth'][batchid, y1, x1]  # [B, N]
            depth2 = pred2['depth'][batchid, y2, x2]  # [B, N]
        
        # Compute Lambert-Beer loss if we have all required information
        lb_loss = torch.tensor(0.0, device=matching_loss.device)
        spatial_reg_loss = torch.tensor(0.0, device=matching_loss.device)
        
        if img1_colors is not None and img2_colors is not None and depth1 is not None and depth2 is not None:
            # Extract descriptors at matched pixels for similarity computation
            matched_desc1 = desc1[batchid, y1, x1]  # [B, N, D]
            matched_desc2 = desc2[batchid, y2, x2]  # [B, N, D]
            
            # Compute Lambert-Beer loss (画像データも渡す)
            lb_loss = self.lambert_beer_loss(
                matched_desc1, matched_desc2,
                img1_colors, img2_colors,
                depth1, depth2,
                valid_matches=valid_matches,
                pred1=pred1, pred2=pred2,
                x1=x1, y1=y1, x2=x2, y2=y2,
                batchid=batchid,
                gt1=gt1, gt2=gt2  # 可視化用に画像データを渡す
            )
            
            # Compute spatial smoothness regularization if parameters are predicted
            if (self.lambert_beer_loss.use_predicted_params and 
                pred1 is not None and pred2 is not None and 
                'beta_1' in pred1 and self.spatial_reg is not None):
                # Get parameter maps and convert to [B, H, W, 3] format if needed
                # postprocess outputs as [B, H, W, 3], but might be [B, 3, H, W]
                def to_hwc(tensor):
                    if tensor.dim() == 4:
                        if tensor.shape[1] == 3:  # [B, 3, H, W]
                            return tensor.permute(0, 2, 3, 1)  # [B, H, W, 3]
                        elif tensor.shape[-1] == 3:  # [B, H, W, 3]
                            return tensor
                    return tensor
                
                beta_1_1_map = to_hwc(pred1['beta_1'])
                beta_2_1_map = to_hwc(pred1['beta_2'])
                b_inf_1_map = to_hwc(pred1['b_inf'])
                
                beta_1_2_map = to_hwc(pred2['beta_1'])
                beta_2_2_map = to_hwc(pred2['beta_2'])
                b_inf_2_map = to_hwc(pred2['b_inf'])
                
                # Average regularization loss from both images
                reg_loss_1 = self.spatial_reg(beta_1_1_map, beta_2_1_map, b_inf_1_map)
                reg_loss_2 = self.spatial_reg(beta_1_2_map, beta_2_2_map, b_inf_2_map)
                spatial_reg_loss = (reg_loss_1 + reg_loss_2) / 2
        
        # Combine losses
        total_loss = matching_loss + self.lambert_beer_weight * lb_loss + self.spatial_reg_weight * spatial_reg_loss
        details['lambert_beer_loss'] = float(lb_loss)
        details['spatial_reg_loss'] = float(spatial_reg_loss)
        details['matching_loss'] = float(matching_loss)
        details['total_loss'] = float(total_loss)
        
        print(f"[MatchingWithLambertBeerLoss] matching={matching_loss.item():.6f}, "
              f"lb_loss={lb_loss.item():.6f}, spatial_reg={spatial_reg_loss.item():.6f}, "
              f"total={total_loss.item():.6f}")
        
        return total_loss, details


class LambertBeerLossOnly(MultiLoss):
    """
    Lambert-Beer color restoration loss without matching loss.
    Usage:
        lb_loss = LambertBeerColorLoss(similarity_threshold=0.95, ...)
        spatial_reg = SpatialSmoothnessRegularizer(weight=0.01)
        color_loss = LambertBeerLossOnly(
            lb_loss,
            lambert_beer_weight=1.0,
            spatial_reg=spatial_reg,
            spatial_reg_weight=0.01
        )
        
        # Combine with Regr3D + ConfLoss
        from mast3r.losses import Regr3D, L21, ConfLoss
        regr3d_loss = Regr3D(L21(), norm_mode='?avg_dis')
        conf_regr3d_loss = ConfLoss(regr3d_loss, alpha=0.2)
        combined_loss = conf_regr3d_loss + 0.1 * color_loss
    """
    
    def __init__(self, lambert_beer_loss, lambert_beer_weight=1.0,
                 spatial_reg=None, spatial_reg_weight=0.01):
        super().__init__()
        self.lambert_beer_loss = lambert_beer_loss
        self.lambert_beer_weight = lambert_beer_weight
        self.spatial_reg = spatial_reg
        self.spatial_reg_weight = spatial_reg_weight
    
    def get_name(self):
        return 'LambertBeerLossOnly'
        
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        """
        Compute Lambert-Beer color restoration loss only (without matching loss).
        
        Uses MASt3R Matching Head output (descriptors) to find matching points between two images.
        Selects points with similarity >= threshold (default 0.95) up to max_matches (default 100).
        
        Args:
            gt1, gt2: ground truth dictionaries with 'img' key
            pred1, pred2: prediction dictionaries with 'desc', 'pts3d', and optionally 'beta_1', 'beta_2', 'b_inf'
            
        Returns:
            loss: scalar loss value
            details: dictionary with loss components
        """
        # Get descriptors from MASt3R Matching Head output
        desc1, desc2 = pred1['desc'], pred2['desc']  # [B, H, W, D]
        
        # Reshape descriptors to [B, H*W, D] for similarity computation
        B, H1, W1, D = desc1.shape
        _, H2, W2, _ = desc2.shape
        
        # ==============================================
        # Fast Reciprocal Matching (MASt3R paper, Section 3.3)
        # ==============================================
        # 複雑度: O(k*W*H) where k = initial_samples
        # 全探索O(W²H²)より遥かに高速で、かつ高精度
        
        desc1_flat = desc1.reshape(B, H1 * W1, D)  # [B, H1*W1, D]
        desc2_flat = desc2.reshape(B, H2 * W2, D)  # [B, H2*W2, D]
        
        # ============================================
        # MASt3R公式実装と同じL2正規化（重要！）
        # ============================================
        # pred['desc']は既にpostprocessで正規化済みだが、
        # reshapeやindex_selectで数値誤差が蓄積する可能性があるため、
        # Fast Reciprocal Matching前に再度正規化を適用
        desc1_flat = desc1_flat / desc1_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        desc2_flat = desc2_flat / desc2_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # デバッグ：正規化後のノルムを確認（全て1.0に近いはず）
        print(f"[Descriptor Normalization Check]")
        print(f"  desc1_flat norm: min={desc1_flat.norm(dim=-1).min().item():.6f}, "
              f"max={desc1_flat.norm(dim=-1).max().item():.6f}, "
              f"mean={desc1_flat.norm(dim=-1).mean().item():.6f}")
        print(f"  desc2_flat norm: min={desc2_flat.norm(dim=-1).min().item():.6f}, "
              f"max={desc2_flat.norm(dim=-1).max().item():.6f}, "
              f"mean={desc2_flat.norm(dim=-1).mean().item():.6f}")
        
        # 初期スパースサンプル（グリッド状またはランダム）
        k = min(self.lambert_beer_loss.initial_samples, H1 * W1)
        
        # 初期点U₀をランダムにサンプリング
        original_U_indices = torch.randperm(H1 * W1, device=desc1.device)[:k]  # [k]
        original_U_indices = original_U_indices.unsqueeze(0).expand(B, -1)  # [B, k]
        
        # 論文準拠の実装：
        # - 各初期点U₀[i]を個別に追跡
        # - 収束した点は除外（active_mask）
        # - 各初期点は最大1つのマッチのみ生成
        # - 保証: |M| ≤ k
        
        current_U = original_U_indices.clone()  # 現在のU値 [B, k]
        active_mask = torch.ones(B, k, dtype=torch.bool, device=desc1.device)  # アクティブな点 [B, k]
        
        # 最終的なマッチを格納（各初期点につき最大1つ）
        final_V_indices = torch.zeros(B, k, dtype=torch.long, device=desc1.device)  # [B, k]
        final_similarities = torch.zeros(B, k, device=desc1.device)  # [B, k]
        converged_mask_final = torch.zeros(B, k, dtype=torch.bool, device=desc1.device)  # [B, k]
        
        print(f"[Fast Reciprocal Matching] 初期サンプル数: {k}, 最大反復回数: {self.lambert_beer_loss.max_iterations}")
        
        for iteration in range(self.lambert_beer_loss.max_iterations):
            if not active_mask.any():
                break  # 全て収束
            
            # アクティブな点のみ処理
            num_active = active_mask.sum(dim=1).max().item()
            if num_active == 0:
                break
            
            # アクティブな点のdescriptorを取得
            # current_U[active_mask]を使うとバッチ間でサイズが異なるため、個別処理
            for b in range(B):
                active_b = active_mask[b]  # [k]
                if not active_b.any():
                    continue
                
                # このバッチのアクティブな点
                U_active = current_U[b, active_b]  # [num_active_b]
                U_desc = torch.index_select(desc1_flat[b], 0, U_active)  # [num_active_b, D]
                
                # U → V: desc2での最近傍
                # チャンク処理
                chunk_size = 500
                num_active_b = U_active.shape[0]
                V_active = torch.zeros(num_active_b, dtype=torch.long, device=desc1.device)
                U_to_V_sim = torch.zeros(num_active_b, device=desc1.device)
                
                for i in range(0, num_active_b, chunk_size):
                    end_i = min(i + chunk_size, num_active_b)
                    U_chunk = U_desc[i:end_i, :].unsqueeze(0)  # [1, chunk_len, D]
                    
                    # MASt3R公式実装と同じ類似度計算（正規化済みdescのdot product）
                    sim = torch.bmm(U_chunk, desc2_flat[b:b+1].transpose(1, 2))  # [1, chunk_len, H2*W2]
                    sim_max, idx_max = sim.max(dim=2)  # [1, chunk_len]
                    
                    # デバッグ：最初のイテレーション＆最初のチャンクで類似度分布を確認
                    if iteration == 0 and i == 0 and b == 0:
                        print(f"  [Debug Iteration {iteration}] 類似度分布（desc2全体に対する類似度）:")
                        print(f"    sim min={sim.min().item():.6f}, max={sim.max().item():.6f}, mean={sim.mean().item():.6f}")
                        # 最頻値のインデックスをチェック（全て同じ点を指しているか？）
                        unique_indices = idx_max[0].unique()
                        print(f"    マッチ先のユニーク数: {len(unique_indices)} / {idx_max.shape[1]} "
                              f"(全て異なる点なら良好、1つなら全て同じ点に集中)")
                    
                    V_active[i:end_i] = idx_max[0]
                    U_to_V_sim[i:end_i] = sim_max[0]
                
                # V → U_next: desc1での最近傍
                V_desc = torch.index_select(desc2_flat[b], 0, V_active)  # [num_active_b, D]
                # V_descも正規化されているはず（desc2_flatから取得）だが、念のため確認
                # （既に正規化済みなので、再正規化は不要だが、数値安定性のため残す）
                U_next_active = torch.zeros(num_active_b, dtype=torch.long, device=desc1.device)
                
                for i in range(0, num_active_b, chunk_size):
                    end_i = min(i + chunk_size, num_active_b)
                    V_chunk = V_desc[i:end_i, :].unsqueeze(0)  # [1, chunk_len, D]
                    
                    sim = torch.bmm(V_chunk, desc1_flat[b:b+1].transpose(1, 2))  # [1, chunk_len, H1*W1]
                    _, idx_max = sim.max(dim=2)  # [1, chunk_len]
                    
                    U_next_active[i:end_i] = idx_max[0]
                
                # 収束判定: 現在のU == U_next
                converged_b = (U_active == U_next_active)  # [num_active_b]
                threshold_b = (U_to_V_sim >= self.lambert_beer_loss.similarity_threshold)  # [num_active_b]
                valid_b = converged_b & threshold_b  # [num_active_b]
                
                # 収束した点を最終結果に保存（元の初期点のインデックスを使用）
                active_indices = torch.where(active_b)[0]  # active_maskがTrueの位置
                for idx, (is_valid, v_idx, sim_val) in enumerate(zip(valid_b, V_active, U_to_V_sim)):
                    if is_valid:
                        original_idx = active_indices[idx]
                        converged_mask_final[b, original_idx] = True
                        final_V_indices[b, original_idx] = v_idx
                        final_similarities[b, original_idx] = sim_val
                
                # 収束した点をアクティブマスクから除外
                for idx, is_converged in enumerate(converged_b):
                    if is_converged:
                        original_idx = active_indices[idx]
                        active_mask[b, original_idx] = False
                
                # 収束していない点のcurrent_Uを更新
                for idx, (is_converged, u_next) in enumerate(zip(converged_b, U_next_active)):
                    if not is_converged:
                        original_idx = active_indices[idx]
                        current_U[b, original_idx] = u_next
            
            num_remaining = active_mask.sum().item()
            num_converged_total = converged_mask_final.sum().item()
            print(f"  Iteration {iteration}: 総収束 {num_converged_total}/{k * B}, 残り {num_remaining}")
        
        # 収束したマッチを抽出（論文準拠：|M| ≤ k）
        if not converged_mask_final.any():
            print(f"[Fast Reciprocal Matching] 警告: マッチが見つかりませんでした。Loss=0を返します。")
            device = desc1.device
            details = {
                'lambert_beer_loss': 0.0,
                'spatial_reg_loss': 0.0,
                'total_loss': 0.0,
                'num_matches': 0,
                'num_high_conf': 0,
                'num_sampled': k
            }
            return torch.tensor(0.0, device=device, dtype=self.lambert_beer_loss.fp, requires_grad=True), details
        
        # converged_mask_finalから全マッチを抽出
        matched_u_list = []
        matched_v_list = []
        matched_sim_list = []
        matched_batch_list = []
        
        for b in range(B):
            converged_b = converged_mask_final[b]  # [k]
            if converged_b.any():
                matched_u_list.append(original_U_indices[b, converged_b])
                matched_v_list.append(final_V_indices[b, converged_b])
                matched_sim_list.append(final_similarities[b, converged_b])
                matched_batch_list.append(torch.full((converged_b.sum(),), b, dtype=torch.long, device=desc1.device))
        
        matched_u = torch.cat(matched_u_list, dim=0)  # [M]
        matched_v = torch.cat(matched_v_list, dim=0)  # [M]
        matched_sim = torch.cat(matched_sim_list, dim=0)  # [M]
        matched_batch = torch.cat(matched_batch_list, dim=0)  # [M]
        
        total_matches = len(matched_u)
        print(f"[Fast Reciprocal Matching] 総マッチ数: {total_matches} / {k * B} (初期サンプル: {k} × Batch: {B})")
        assert total_matches <= k * B, f"マッチ数 {total_matches} が初期サンプル数 {k * B} を超えています（論文違反）"
        
        # Top-K選択（max_matchesまで削減、類似度の高い順）
        if total_matches > self.lambert_beer_loss.max_matches * B:
            _, topk_indices = torch.topk(matched_sim, k=min(self.lambert_beer_loss.max_matches * B, total_matches), dim=0)
            matched_u = matched_u[topk_indices]
            matched_v = matched_v[topk_indices]
            matched_sim = matched_sim[topk_indices]
            matched_batch = matched_batch[topk_indices]
            print(f"[Fast Reciprocal Matching] Top-K選択後: {len(matched_u)} マッチ")
        
        # matched_uとmatched_vをx,y座標に変換
        y1 = matched_u // W1  # [M]
        x1 = matched_u % W1   # [M]
        y2 = matched_v // W2  # [M]
        x2 = matched_v % W2   # [M]
        batchid = matched_batch  # [M]
        
        # 画像2側の同じピクセルに対応している重複を除去
        # 各バッチごとに、画像2の同じピクセル(x2, y2)に複数の点が対応している場合、
        # 類似度に関係なく、全ての重複マッチを削除する
        matches_before_dedup = len(matched_u)
        print(f"[Fast Reciprocal Matching] 重複除去前: {matches_before_dedup} マッチ")
        unique_indices_list = []
        
        for b in range(B):
            batch_mask = (batchid == b)
            if not batch_mask.any():
                continue
            
            # このバッチのマッチングペアを取得
            batch_indices = torch.where(batch_mask)[0]
            batch_y2 = y2[batch_indices]  # [M_b]
            batch_x2 = x2[batch_indices]  # [M_b]
            
            # 画像2のピクセル座標をキーとして、出現回数をカウント
            # (y2, x2)の組み合わせで辞書を作成
            pixel_count = {}
            for y, x in zip(batch_y2, batch_x2):
                pixel_key = (y.item(), x.item())
                pixel_count[pixel_key] = pixel_count.get(pixel_key, 0) + 1
            
            # 重複がない（出現回数が1の）ピクセルのみを保持
            unique_mask = torch.zeros(len(batch_indices), dtype=torch.bool, device=desc1.device)
            for idx_in_batch, (y, x) in enumerate(zip(batch_y2, batch_x2)):
                pixel_key = (y.item(), x.item())
                if pixel_count[pixel_key] == 1:
                    unique_mask[idx_in_batch] = True
            
            # 重複がないマッチのインデックスを取得
            unique_batch_indices = batch_indices[unique_mask]
            if len(unique_batch_indices) > 0:
                unique_indices_list.append(unique_batch_indices)
        
        if len(unique_indices_list) > 0:
            unique_indices = torch.cat(unique_indices_list, dim=0)
            matched_u = matched_u[unique_indices]
            matched_v = matched_v[unique_indices]
            matched_sim = matched_sim[unique_indices]
            matched_batch = matched_batch[unique_indices]
            
            # 座標も再計算
            y1 = matched_u // W1  # [M']
            x1 = matched_u % W1   # [M']
            y2 = matched_v // W2  # [M']
            x2 = matched_v % W2   # [M']
            batchid = matched_batch  # [M']
            
            print(f"[Fast Reciprocal Matching] 重複除去後: {len(matched_u)} マッチ (削除: {matches_before_dedup - len(matched_u)} マッチ)")
        else:
            print(f"[Fast Reciprocal Matching] 重複除去後: 0 マッチ (全て削除されました)")
            # 空のテンソルを作成（後続の処理でエラーを防ぐため）
            matched_u = torch.tensor([], dtype=torch.long, device=desc1.device)
            matched_v = torch.tensor([], dtype=torch.long, device=desc1.device)
            matched_sim = torch.tensor([], dtype=matched_sim.dtype, device=desc1.device)
            matched_batch = torch.tensor([], dtype=torch.long, device=desc1.device)
            y1 = torch.tensor([], dtype=torch.long, device=desc1.device)
            x1 = torch.tensor([], dtype=torch.long, device=desc1.device)
            y2 = torch.tensor([], dtype=torch.long, device=desc1.device)
            x2 = torch.tensor([], dtype=torch.long, device=desc1.device)
            batchid = torch.tensor([], dtype=torch.long, device=desc1.device)
        
        # Fast Reciprocal Matchingの結果を使用
        # x1, y1, x2, y2, batchid はすでに計算済み
        num_final_matches = len(matched_u)
        
        # マッチング可視化（閾値を超える点がある場合）
        # 頻度を減らすため、10000回に1回のみ可視化
        if self.lambert_beer_loss.enable_visualization and self.lambert_beer_loss.output_dir is not None and num_final_matches > 0 and self.lambert_beer_loss.vis_counter % 10000 == 0:
            # バッチごとに可視化
            for b in range(B):
                batch_mask = (matched_batch == b)
                if not batch_mask.any():
                    continue
                
                # このバッチのマッチング点を取得
                batch_x1_vis = x1[batch_mask].detach().cpu().numpy()
                batch_y1_vis = y1[batch_mask].detach().cpu().numpy()
                batch_x2_vis = x2[batch_mask].detach().cpu().numpy()
                batch_y2_vis = y2[batch_mask].detach().cpu().numpy()
                batch_sim = matched_sim[batch_mask].detach().cpu().numpy()
                
                # 画像を取得
                if 'img' in gt1 and 'img' in gt2:
                    img1_full = gt1['img'][b]  # [3, H, W] or [H, W, 3]
                    img2_full = gt2['img'][b]
                    
                    # [H, W, 3]形式に変換
                    if img1_full.shape[0] == 3:
                        img1_full = img1_full.permute(1, 2, 0)
                        img2_full = img2_full.permute(1, 2, 0)
                    
                    img1_np = img1_full.cpu().numpy()
                    img2_np = img2_full.cpu().numpy()
                    
                    # [0, 1]範囲にクリップ
                    img1_np = np.clip(img1_np, 0, 1)
                    img2_np = np.clip(img2_np, 0, 1)
                    
                    # 保存先ディレクトリを作成
                    vis_dir = os.path.join(self.lambert_beer_loss.output_dir, 'matching_visualizations')
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # ファイル名
                    save_path = os.path.join(vis_dir, f'match_{self.lambert_beer_loss.vis_counter:06d}_batch{b}.png')
                    
                    # 可視化
                    self.lambert_beer_loss.visualize_matches(img1_np, img2_np, 
                                         batch_x1_vis, batch_y1_vis, batch_x2_vis, batch_y2_vis,
                                         batch_sim, save_path)
        
        # カウンターは常にインクリメント（可視化の有無に関わらず）
        self.lambert_beer_loss.vis_counter += 1
        
        # Extract colors and depths at matched pixels
        img1_colors = None
        img2_colors = None
        
        if 'img' in gt1 and 'img' in gt2:
            img1 = gt1['img']  # [B, 3, H, W] or [B, H, W, 3]
            img2 = gt2['img']
            
            # Convert to [B, H, W, 3] format if needed
            if img1.dim() == 4 and img1.shape[1] == 3:
                img1 = img1.permute(0, 2, 3, 1)
                img2 = img2.permute(0, 2, 3, 1)
            
            # Extract colors at matched pixels (now shape [M, 3])
            # Clamp to [0, 1] if normalized
            img1_colors = torch.clamp(img1[batchid, y1, x1], 0, 1)  # [M, 3]
            img2_colors = torch.clamp(img2[batchid, y2, x2], 0, 1)  # [M, 3]
        
        # Extract depths from predictions
        # IMPORTANT: In DUSt3R/MASt3R training, both pred1 and pred2 pts3d are normalized 
        # to view1's coordinate system. For Lambert-Beer law, we need depths from each 
        # camera's own coordinate system.
        depth1 = None
        depth2 = None
        
        # Try to get pts3d for both views
        pts3d_1_view1 = None  # pts3d in view1's camera coordinate system
        pts3d_2_view1 = None  # pts3d in view1's camera coordinate system (normalized)
        
        if 'pts3d' in pred1:
            pts3d_1_view1 = pred1['pts3d']
        if 'pts3d' in pred2:
            pts3d_2_view1 = pred2['pts3d']
        elif 'pts3d_in_other_view' in pred2:
            # Use transformed points if available
            pts3d_2_view1 = pred2['pts3d_in_other_view']
        
        if pts3d_1_view1 is not None and pts3d_2_view1 is not None:
            # Extract matched points in view1's coordinate system
            pts3d_1_matched_view1 = pts3d_1_view1[batchid, y1, x1]  # [M, 3] in view1's camera frame
            pts3d_2_matched_view1 = pts3d_2_view1[batchid, y2, x2]  # [M, 3] in view1's camera frame
            
            # Depth1: distance from view1's camera (already in view1's coordinate system)
            depth1 = torch.norm(pts3d_1_matched_view1, dim=-1)  # [M]
            
            # Depth2: need to transform pts3d_2 from view1's coordinate system to view2's coordinate system
            # Then compute distance from view2's camera
            # IMPORTANT: In training, pred2['pts3d'] is normalized to view1's coordinate system.
            # We need to transform it back to view2's camera coordinate system to get correct depth.
            if 'camera_pose' in gt1 and 'camera_pose' in gt2:
                from dust3r.utils.geometry import inv, geotrf
                camera_pose1 = gt1['camera_pose']  # view1 cam2world [B, 4, 4]
                camera_pose2 = gt2['camera_pose']  # view2 cam2world [B, 4, 4]
                
                # Transform matched points: view1_cam -> world -> view2_cam
                # Process batch by batch to handle indexing correctly
                B = camera_pose2.shape[0]
                pts3d_2_matched_view2_list = []
                for b in range(B):
                    batch_mask = (batchid == b)
                    if batch_mask.any():
                        pts3d_batch_view1 = pts3d_2_matched_view1[batch_mask]  # [M_b, 3] in view1's camera frame
                        # Transform to world coordinates using view1's camera pose
                        pts3d_batch_world = geotrf(camera_pose1[b:b+1], pts3d_batch_view1.unsqueeze(0))  # [1, M_b, 3]
                        pts3d_batch_world = pts3d_batch_world[0]  # [M_b, 3]
                        # Transform to view2's camera coordinates
                        in_camera2_b = inv(camera_pose2[b:b+1])  # [1, 4, 4]
                        pts3d_batch_view2 = geotrf(in_camera2_b, pts3d_batch_world.unsqueeze(0))  # [1, M_b, 3]
                        pts3d_batch_view2 = pts3d_batch_view2[0]  # [M_b, 3]
                        pts3d_2_matched_view2_list.append(pts3d_batch_view2)
                
                if pts3d_2_matched_view2_list:
                    pts3d_2_matched_view2 = torch.cat(pts3d_2_matched_view2_list, dim=0)  # [M, 3]
                    # Compute depth from view2's camera
                    depth2 = torch.norm(pts3d_2_matched_view2, dim=-1)  # [M]
                else:
                    depth2 = None
            else:
                # Fallback: assume pts3d_2 is already in view2's coordinate system
                # (This should not happen in normal training, but handle gracefully)
                depth2 = torch.norm(pts3d_2_matched_view1, dim=-1)  # [M]
        elif 'depth' in pred1 and 'depth' in pred2:
            # Direct depth prediction (should already be in each camera's coordinate system)
            depth1 = pred1['depth'][batchid, y1, x1]  # [M]
            depth2 = pred2['depth'][batchid, y2, x2]  # [M]
        
        # Check depth validity and filter out invalid depths
        valid_depth_mask = None
        if depth1 is not None and depth2 is not None:
            # Check for NaN, inf, zero, negative, and excessively large depths
            # Upper limit: 1e6 (1000km) - depths larger than this cause numerical issues in exp()
            # Lower limit: 1e-6 to avoid division issues
            MAX_DEPTH = 1e6  # Maximum reasonable depth (1000km)
            MIN_DEPTH = 1e-6  # Minimum depth threshold
            valid_depth1 = torch.isfinite(depth1) & (depth1 > MIN_DEPTH) & (depth1 < MAX_DEPTH)  # [M]
            valid_depth2 = torch.isfinite(depth2) & (depth2 > MIN_DEPTH) & (depth2 < MAX_DEPTH)  # [M]
            valid_depth_mask = valid_depth1 & valid_depth2  # [M]
            
            num_valid = valid_depth_mask.sum().item()
            num_total = len(depth1)
            num_invalid = num_total - num_valid
            
            print(f"[Depth Validity Check] 総マッチ数: {num_total}, "
                  f"有効深度: {num_valid}, 無効深度: {num_invalid} "
                  f"({100.0 * num_invalid / num_total:.1f}%)")
            
            if num_invalid > 0:
                # Show statistics for invalid depths
                invalid_depth1 = depth1[~valid_depth_mask]
                invalid_depth2 = depth2[~valid_depth_mask]
                print(f"  無効深度1: NaN={torch.isnan(invalid_depth1).sum().item()}, "
                      f"Inf={torch.isinf(invalid_depth1).sum().item()}, "
                      f"≤{MIN_DEPTH}={(invalid_depth1 <= MIN_DEPTH).sum().item()}, "
                      f"≥{MAX_DEPTH}={(invalid_depth1 >= MAX_DEPTH).sum().item()}")
                print(f"  無効深度2: NaN={torch.isnan(invalid_depth2).sum().item()}, "
                      f"Inf={torch.isinf(invalid_depth2).sum().item()}, "
                      f"≤{MIN_DEPTH}={(invalid_depth2 <= MIN_DEPTH).sum().item()}, "
                      f"≥{MAX_DEPTH}={(invalid_depth2 >= MAX_DEPTH).sum().item()}")
            
            if num_valid == 0:
                print(f"[Depth Validity Check] ⚠️  警告: 有効な深度がありません。Loss=0を返します。")
                depth1 = None
                depth2 = None
            else:
                # Filter to keep only valid depths
                depth1 = depth1[valid_depth_mask]
                depth2 = depth2[valid_depth_mask]
                img1_colors = img1_colors[valid_depth_mask] if img1_colors is not None else None
                img2_colors = img2_colors[valid_depth_mask] if img2_colors is not None else None
                # Also update matched indices for parameter extraction
                matched_u_valid = matched_u[valid_depth_mask]
                matched_v_valid = matched_v[valid_depth_mask]
                matched_batch_valid = matched_batch[valid_depth_mask]
                # Recompute coordinates for valid matches
                y1_valid = matched_u_valid // W1
                x1_valid = matched_u_valid % W1
                y2_valid = matched_v_valid // W2
                x2_valid = matched_v_valid % W2
                batchid_valid = matched_batch_valid
                # Update for use in parameter extraction
                y1, x1, y2, x2, batchid = y1_valid, x1_valid, y2_valid, x2_valid, batchid_valid
        else:
            print(f"[Depth Validity Check] ⚠️  警告: 深度情報が取得できませんでした。")
            if 'pts3d' not in pred1 or 'pts3d' not in pred2:
                print(f"  pts3dが存在しません: pred1に'pts3d'={'pts3d' in pred1}, pred2に'pts3d'={'pts3d' in pred2}")
            if 'depth' not in pred1 or 'depth' not in pred2:
                print(f"  depthが存在しません: pred1に'depth'={'depth' in pred1}, pred2に'depth'={'depth' in pred2}")
        
        # Compute Lambert-Beer loss if we have all required information
        lb_loss = torch.tensor(0.0, device=desc1.device)
        spatial_reg_loss = torch.tensor(0.0, device=desc1.device)
        
        if img1_colors is not None and img2_colors is not None and depth1 is not None and depth2 is not None:
            # Get parameters (predicted or fixed) for matched pixels
            if self.lambert_beer_loss.use_predicted_params and pred1 is not None and pred2 is not None and 'beta_1' in pred1:
                # Extract predicted parameters at matched pixels
                # Handle both [B, H, W, 3] and [B, 3, H, W] formats
                beta_1_map1 = pred1['beta_1']
                beta_1_map2 = pred2['beta_1']
                
                # Convert to [B, H, W, 3] format if needed
                if beta_1_map1.dim() == 4 and beta_1_map1.shape[1] == 3:  # [B, 3, H, W]
                    beta_1_map1 = beta_1_map1.permute(0, 2, 3, 1)
                    beta_1_map2 = beta_1_map2.permute(0, 2, 3, 1)
                    beta_2_map1 = pred1['beta_2'].permute(0, 2, 3, 1)
                    beta_2_map2 = pred2['beta_2'].permute(0, 2, 3, 1)
                    b_inf_map1 = pred1['b_inf'].permute(0, 2, 3, 1)
                    b_inf_map2 = pred2['b_inf'].permute(0, 2, 3, 1)
                else:  # Already [B, H, W, 3]
                    beta_2_map1 = pred1['beta_2']
                    beta_2_map2 = pred2['beta_2']
                    b_inf_map1 = pred1['b_inf']
                    b_inf_map2 = pred2['b_inf']
                
                # Extract at matched pixels
                beta_1_1 = beta_1_map1[batchid, y1, x1]  # [M, 3]
                beta_2_1 = beta_2_map1[batchid, y1, x1]  # [M, 3]
                b_inf_1 = b_inf_map1[batchid, y1, x1]  # [M, 3]
                
                beta_1_2 = beta_1_map2[batchid, y2, x2]  # [M, 3]
                beta_2_2 = beta_2_map2[batchid, y2, x2]  # [M, 3]
                b_inf_2 = b_inf_map2[batchid, y2, x2]  # [M, 3]
                
                # Average parameters from both images
                beta_1_use = (beta_1_1 + beta_1_2) / 2  # [M, 3]
                beta_2_use = (beta_2_1 + beta_2_2) / 2  # [M, 3]
                b_inf_use = (b_inf_1 + b_inf_2) / 2  # [M, 3]
            else:
                # Use fixed parameters
                beta_1_use = self.lambert_beer_loss.beta_1
                beta_2_use = self.lambert_beer_loss.beta_2
                b_inf_use = self.lambert_beer_loss.b_inf
                
                # Move to device and expand
                if isinstance(beta_1_use, torch.Tensor):
                    beta_1_use = beta_1_use.to(desc1.device)
                else:
                    beta_1_use = torch.tensor(beta_1_use, device=desc1.device, dtype=self.lambert_beer_loss.fp)
                    
                if isinstance(beta_2_use, torch.Tensor):
                    beta_2_use = beta_2_use.to(desc1.device)
                else:
                    beta_2_use = torch.tensor(beta_2_use, device=desc1.device, dtype=self.lambert_beer_loss.fp)
                    
                if isinstance(b_inf_use, torch.Tensor):
                    b_inf_use = b_inf_use.to(desc1.device)
                else:
                    b_inf_use = torch.tensor(b_inf_use, device=desc1.device, dtype=self.lambert_beer_loss.fp)
                
                # Expand to match shape [M, 3]
                if beta_1_use.dim() == 0:
                    beta_1_use = beta_1_use.expand_as(img1_colors)
                if beta_2_use.dim() == 0:
                    beta_2_use = beta_2_use.expand_as(img1_colors)
                if b_inf_use.dim() == 0:
                    b_inf_use = b_inf_use.expand_as(img1_colors)
            
            # Restore colors for both images using Lambert-Beer law
            restored_color1 = self.lambert_beer_loss.restore_color(
                img1_colors.unsqueeze(0), depth1.unsqueeze(0).unsqueeze(-1), 
                beta_1_use.unsqueeze(0), beta_2_use.unsqueeze(0), b_inf_use.unsqueeze(0)
            )  # [1, M, 3]
            restored_color2 = self.lambert_beer_loss.restore_color(
                img2_colors.unsqueeze(0), depth2.unsqueeze(0).unsqueeze(-1),
                beta_1_use.unsqueeze(0), beta_2_use.unsqueeze(0), b_inf_use.unsqueeze(0)
            )  # [1, M, 3]
            
            # Compute L2 loss between restored colors
            color_diff = restored_color1 - restored_color2  # [1, M, 3]
            lb_loss = torch.mean(color_diff ** 2)  # Mean squared error
            
            # 色復元とLoss計算の詳細情報を出力
            print(f"[Lambert-Beer Loss] 色復元統計:")
            print(f"  観測色1: min={img1_colors.min().item():.4f}, max={img1_colors.max().item():.4f}, mean={img1_colors.mean().item():.4f}")
            print(f"  観測色2: min={img2_colors.min().item():.4f}, max={img2_colors.max().item():.4f}, mean={img2_colors.mean().item():.4f}")
            print(f"  深度1: min={depth1.min().item():.4f}, max={depth1.max().item():.4f}, mean={depth1.mean().item():.4f}")
            print(f"  深度2: min={depth2.min().item():.4f}, max={depth2.max().item():.4f}, mean={depth2.mean().item():.4f}")
            if self.lambert_beer_loss.use_predicted_params and 'beta_1' in pred1:
                print(f"  β1: min={beta_1_use.min().item():.4f}, max={beta_1_use.max().item():.4f}, mean={beta_1_use.mean().item():.4f}")
                print(f"  β2: min={beta_2_use.min().item():.4f}, max={beta_2_use.max().item():.4f}, mean={beta_2_use.mean().item():.4f}")
                print(f"  B_inf: min={b_inf_use.min().item():.4f}, max={b_inf_use.max().item():.4f}, mean={b_inf_use.mean().item():.4f}")
            print(f"  復元色1: min={restored_color1.min().item():.4f}, max={restored_color1.max().item():.4f}, mean={restored_color1.mean().item():.4f}")
            print(f"  復元色2: min={restored_color2.min().item():.4f}, max={restored_color2.max().item():.4f}, mean={restored_color2.mean().item():.4f}")
            print(f"  色差: min={color_diff.min().item():.4f}, max={color_diff.max().item():.4f}, mean={color_diff.mean().item():.4f}")
            print(f"  Lambert-Beer Loss: {lb_loss.item():.6f}")
            
            # Compute spatial smoothness regularization if parameters are predicted
            if (self.lambert_beer_loss.use_predicted_params and 
                pred1 is not None and pred2 is not None and 
                'beta_1' in pred1 and self.spatial_reg is not None):
                # Get parameter maps and convert to [B, H, W, 3] format if needed
                def to_hwc(tensor):
                    if tensor.dim() == 4:
                        if tensor.shape[1] == 3:  # [B, 3, H, W]
                            return tensor.permute(0, 2, 3, 1)  # [B, H, W, 3]
                        elif tensor.shape[-1] == 3:  # [B, H, W, 3]
                            return tensor
                    return tensor
                
                beta_1_1_map = to_hwc(pred1['beta_1'])
                beta_2_1_map = to_hwc(pred1['beta_2'])
                b_inf_1_map = to_hwc(pred1['b_inf'])
                
                beta_1_2_map = to_hwc(pred2['beta_1'])
                beta_2_2_map = to_hwc(pred2['beta_2'])
                b_inf_2_map = to_hwc(pred2['b_inf'])
                
                # Average regularization loss from both images
                reg_loss_1 = self.spatial_reg(beta_1_1_map, beta_2_1_map, b_inf_1_map)
                reg_loss_2 = self.spatial_reg(beta_1_2_map, beta_2_2_map, b_inf_2_map)
                spatial_reg_loss = (reg_loss_1 + reg_loss_2) / 2
        
        # Combine losses (only Lambert-Beer and spatial regularization, no matching loss)
        total_loss = self.lambert_beer_weight * lb_loss + self.spatial_reg_weight * spatial_reg_loss
        num_matches = final_mask.sum().item() if 'final_mask' in locals() else 0
        details = {
            'lambert_beer_loss': float(lb_loss),
            'spatial_reg_loss': float(spatial_reg_loss),
            'total_loss': float(total_loss),
            'num_matches': num_matches,
            'num_high_conf': num_high_conf if 'num_high_conf' in locals() else 0,
            'num_sampled': num_total_sampled if 'num_total_sampled' in locals() else 0,
            'similarity_threshold': self.lambert_beer_loss.similarity_threshold
        }
        
        print(f"[Lambert-Beer Loss] 最終結果: total_loss={total_loss.item():.6f}, "
              f"lb_loss={lb_loss.item():.6f}, spatial_reg={spatial_reg_loss.item():.6f}")
        
        return total_loss, details

