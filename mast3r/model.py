# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os
import numpy as np

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape, is_symmetrized  # noqa

inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    
    # head_type='catmlp+dpt'の場合、output_modeが指定されていなければ'pts3d+desc24'を追加
    if "head_type='catmlp+dpt'" in args or 'head_type="catmlp+dpt"' in args:
        if 'output_mode' not in args:
            # 最後の閉じ括弧の前にoutput_modeを追加
            args = args[:-1] + ", output_mode='pts3d+desc24')"
            if verbose:
                print(f"  Added output_mode='pts3d+desc24' for head_type='catmlp+dpt'")
    
    # Check if checkpoint has underwater_params head and adjust model structure to match checkpoint
    # Note: Model structure must match checkpoint structure to load weights properly
    # The fix_background_color_from_depth flag only affects postprocessing (which channels are used)
    original_fix_bg_from_depth = None
    checkpoint_channels = None
    
    if 'predict_underwater_params' in args and 'head_underwater_params' in str(ckpt.get('model', {}).keys()):
        # Find the fc2 weight shape for head_underwater_params
        for key in ckpt['model'].keys():
            if 'head_underwater_params.fc2.weight' in key:
                actual_out_features = ckpt['model'][key].shape[0]
                # Determine number of channels: out_features = num_channels * patch_size^2
                # Assume patch_size=16 (standard), so patch_size^2 = 256
                patch_size_sq = 256  # default patch_size=16
                checkpoint_channels = actual_out_features // patch_size_sq
                
                # Save original fix_background_color_from_depth value
                if "fix_background_color_from_depth=True" in args:
                    original_fix_bg_from_depth = True
                elif "fix_background_color_from_depth=False" in args:
                    original_fix_bg_from_depth = False
                
                if checkpoint_channels == 9:
                    # Checkpoint has 9 channels - model must be built with 9 channels to match
                    # Temporarily set fix_background_color_from_depth=False to build 9-channel model
                    if "fix_background_color_from_depth=True" in args:
                        args = args.replace("fix_background_color_from_depth=True", "fix_background_color_from_depth=False")
                        if verbose:
                            print(f"  ⚠️  Checkpoint has {checkpoint_channels} channels in head_underwater_params")
                            print(f"     Temporarily setting fix_background_color_from_depth=False to match checkpoint structure")
                            print(f"     (Will restore original value after loading to control postprocessing)")
                elif checkpoint_channels == 6:
                    # Checkpoint has 6 channels - model must be built with 6 channels to match
                    if "fix_background_color_from_depth=False" in args:
                        args = args.replace("fix_background_color_from_depth=False", "fix_background_color_from_depth=True")
                        if verbose:
                            print(f"  Updated fix_background_color_from_depth=True based on checkpoint (6 channels detected)")
                break
    
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    
    # Restore original fix_background_color_from_depth value if it was changed
    # This affects postprocessing behavior (which channels are used), not model structure
    if original_fix_bg_from_depth is not None and checkpoint_channels == 9:
        if original_fix_bg_from_depth:
            # Restore to True - postprocess will use only first 6 channels (β_1, β_2), ignore B_inf
            net.fix_background_color_from_depth = True
            # Also update the head's fix_background_color_from_depth attribute
            if hasattr(net, 'downstream_head1') and hasattr(net.downstream_head1, 'fix_background_color_from_depth'):
                net.downstream_head1.fix_background_color_from_depth = True
            if hasattr(net, 'downstream_head2') and hasattr(net.downstream_head2, 'fix_background_color_from_depth'):
                net.downstream_head2.fix_background_color_from_depth = True
            if verbose:
                print(f"  ✅ Restored fix_background_color_from_depth=True")
                print(f"     Model structure: 9 channels (matches checkpoint)")
                print(f"     Postprocessing: will use only first 6 channels (β_1, β_2), B_inf fixed from depth")
        else:
            # Keep False - postprocess will use all 9 channels
            net.fix_background_color_from_depth = False
            if verbose:
                print(f"  ✅ fix_background_color_from_depth=False")
                print(f"     Model structure: 9 channels (matches checkpoint)")
                print(f"     Postprocessing: will use all 9 channels (β_1, β_2, B_inf)")
    
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, 
                 predict_underwater_params=False, fix_background_color_from_depth=False, **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.predict_underwater_params = predict_underwater_params
        self.fix_background_color_from_depth = fix_background_color_from_depth
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        # Handle both integer and tuple img_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_freeze(self, freeze, unfreeze_encoder_layers=0, unfreeze_decoder_layers=0,
                   freeze_local_features_head=False, freeze_pts3d_head=False, freeze_underwater_params_head=False):
        """
        Set freeze configuration for the MASt3R model.
        Overrides parent class method to handle MASt3R-specific head structure.
        
        Args:
            freeze: Base freeze mode ('none', 'mask', 'encoder', 'encoder_decoder')
            unfreeze_encoder_layers: Number of encoder layers to unfreeze from the end (0 to keep all frozen)
            unfreeze_decoder_layers: Number of decoder layers to unfreeze from the end (0 to keep all frozen)
            freeze_local_features_head: If True, freeze the local features head (matching head)
            freeze_pts3d_head: If True, freeze the 3D points prediction head
            freeze_underwater_params_head: If True, freeze the underwater parameters prediction part (β_1, β_2, B_inf)
                                          in head_local_features. Only works when predict_underwater_params=True.
        """
        # Handle freeze configuration manually since parent class only accepts 'freeze' parameter
        from dust3r.utils.misc import freeze_all_params
        
        if freeze == 'none':
            to_be_frozen = []
        elif freeze == 'mask':
            to_be_frozen = [self.mask_token]
        elif freeze == 'encoder':
            to_be_frozen = [self.mask_token, self.patch_embed, self.enc_blocks]
        elif freeze == 'encoder_decoder':
            to_be_frozen = [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2]
        else:
            raise ValueError(f"Unknown freeze mode: {freeze}")
        
        freeze_all_params(to_be_frozen)
        
        # Handle unfreeze_encoder_layers and unfreeze_decoder_layers
        if unfreeze_encoder_layers > 0 and hasattr(self, 'enc_blocks'):
            for i in range(-unfreeze_encoder_layers, 0):
                if abs(i) <= len(self.enc_blocks):
                    for param in self.enc_blocks[i].parameters():
                        param.requires_grad = True
        
        if unfreeze_decoder_layers > 0:
            if hasattr(self, 'dec_blocks'):
                for i in range(-unfreeze_decoder_layers, 0):
                    if abs(i) <= len(self.dec_blocks):
                        for param in self.dec_blocks[i].parameters():
                            param.requires_grad = True
            if hasattr(self, 'dec_blocks2'):
                for i in range(-unfreeze_decoder_layers, 0):
                    if abs(i) <= len(self.dec_blocks2):
                        for param in self.dec_blocks2[i].parameters():
                            param.requires_grad = True
        
        # Then handle MASt3R-specific head freezing
        self.freeze_local_features_head = freeze_local_features_head
        self.freeze_pts3d_head = freeze_pts3d_head
        self.freeze_underwater_params_head = freeze_underwater_params_head
        
        if freeze_local_features_head or freeze_pts3d_head or freeze_underwater_params_head:
            self._freeze_mast3r_heads(freeze_local_features_head, freeze_pts3d_head, freeze_underwater_params_head)
    
    def _freeze_mast3r_heads(self, freeze_local_features_head=False, freeze_pts3d_head=False, freeze_underwater_params_head=False):
        """
        Freeze specific heads in MASt3R downstream_head1 and downstream_head2.
        
        Args:
            freeze_local_features_head: If True, freeze the local features head (matching head)
            freeze_pts3d_head: If True, freeze the 3D points prediction head
            freeze_underwater_params_head: If True, freeze only the underwater parameters prediction part
                                          (last 9 channels: β_1, β_2, B_inf) in head_local_features
        """
        for head_name in ['downstream_head1', 'downstream_head2']:
            if not hasattr(self, head_name):
                continue
                
            head = getattr(self, head_name)
            
            # Freeze local features head (matching head)
            # This is present in both Cat_MLP_LocalFeatures_DPT_Pts3d and Cat_MLP_LocalFeatures_MiniConv_Pts3d
            if freeze_local_features_head:
                if hasattr(head, 'head_local_features'):
                    print(f"Freezing {head_name}.head_local_features (MASt3R matching head)")
                    for param in head.head_local_features.parameters():
                        param.requires_grad = False
            
            # Freeze underwater parameters prediction part (last 9 channels: β_1, β_2, B_inf)
            # This only works when predict_underwater_params=True
            if freeze_underwater_params_head and hasattr(head, 'head_local_features'):
                if hasattr(head, 'predict_underwater_params') and head.predict_underwater_params:
                    # Freeze the last 9 channels of the output layer (fc2) in head_local_features
                    # Output shape: local_feat_dim * patch_size**2, where last 9 * patch_size**2 are underwater params
                    mlp = head.head_local_features
                    if hasattr(mlp, 'fc2'):
                        patch_size = getattr(head, 'patch_size', 16)
                        underwater_params_channels = 9 * patch_size**2
                        total_channels = mlp.fc2.out_features
                        
                        if total_channels >= underwater_params_channels:
                            # Freeze weights: last 9 * patch_size**2 rows of fc2.weight
                            print(f"Freezing {head_name}.head_local_features.fc2 (last {underwater_params_channels} channels for underwater params)")
                            mlp.fc2.weight.data[-underwater_params_channels:].requires_grad = False
                            # Freeze bias: last 9 * patch_size**2 elements of fc2.bias
                            if mlp.fc2.bias is not None:
                                mlp.fc2.bias.data[-underwater_params_channels:].requires_grad = False
                        else:
                            print(f"Warning: Cannot freeze underwater params in {head_name}.head_local_features: "
                                  f"output channels ({total_channels}) < required ({underwater_params_channels})")
                else:
                    print(f"Warning: Cannot freeze underwater params in {head_name}: predict_underwater_params=False")
            
            # Freeze 3D points head
            if freeze_pts3d_head:
                # For DPT head (Cat_MLP_LocalFeatures_DPT_Pts3d)
                if hasattr(head, 'dpt'):
                    print(f"Freezing {head_name}.dpt (MASt3R 3D points head)")
                    for param in head.dpt.parameters():
                        param.requires_grad = False
                # For MiniConv head (Cat_MLP_LocalFeatures_MiniConv_Pts3d)
                elif hasattr(head, 'head_pts3d'):
                    print(f"Freezing {head_name}.head_pts3d (MASt3R 3D points head)")
                    for param in head.head_pts3d.parameters():
                        param.requires_grad = False


def load_dune_mast3r_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    net = AsymmetricMASt3RWithDUNEBackbone(ckpt['dune_backbone_name'], ckpt['mast3r_model_str'], landscape_only=False)
    missing_keys, unexpected_keys = net.load_state_dict(ckpt['model'], strict=False)
    assert all(k.startswith('dune_backbone') or k.startswith('imagenet') for k in missing_keys), missing_keys
    assert len(unexpected_keys) == 0, unexpected_keys
    return net.to(device)


class AsymmetricMASt3RWithDUNEBackbone(torch.nn.Module):

    def __init__(self, dune_backbone_name, mast3r_model_str, landscape_only=True):  # "dune_vitbase_14_448_paper_encoder"
        super().__init__()
        self.dune_backbone = torch.hub.load("naver/dune", dune_backbone_name)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1))
        self.norm = torch.nn.LayerNorm(self.dune_backbone.num_features)
        assert 'landscape_only' not in mast3r_model_str
        self.landscape_only = landscape_only
        if not self.landscape_only:
            mast3r_model_str = mast3r_model_str[:-1] + f', landscape_only=False)'
        patch_size = self.dune_backbone.patch_size
        self.patch_size = patch_size
        self.square_ok = True
        if patch_size != 16:
            assert not "patch_size" in mast3r_model_str, mast3r_model_str
            assert mast3r_model_str.endswith(')'), mast3r_model_str
            mast3r_model_str = mast3r_model_str[:-1] + f', patch_size={patch_size})'
        self.mast3r = eval(mast3r_model_str)
        if patch_size != 16:  # might depend on mast3r stuff
            # we need to hack the head a bit ...
            for head in [self.mast3r.downstream_head1, self.mast3r.downstream_head2]:
                head.dpt.patch_size = (patch_size, patch_size)
                head.dpt.P_H = max(1, patch_size // head.dpt.stride_level)
                head.dpt.P_W = max(1, patch_size // head.dpt.stride_level)
                head.dpt.head[1].scale_factor *= 14 / 16
        # freeze dune
        self.dune_backbone.eval()
        for p in self.dune_backbone.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_dune_mast3r_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def train(self, mode=True):
        self = super().train(mode=mode)
        self.dune_backbone.eval()
        return self

    @torch.no_grad()
    def _encode_image(self, image, true_shape):
        # a bit tricky due to mixing aspect ratios at training ...
        B, C, H, W = image.shape
        patch_size = self.dune_backbone.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        image = image * 0.5 + 0.5  # undo normalization from mast3r
        image = (image - self.imagenet_mean) / self.imagenet_std  # and do the one from imagenet
        W //= patch_size[0]
        H //= patch_size[1]
        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape
        if self.landscape_only:
            assert W >= H, f'image should be in landscape mode, but got {W=} {H=}'
            assert H % patch_size[0] == 0, f"Input image height ({H}) is not a multiple of patch size ({patch_size[0]})."
            assert W % patch_size[1] == 0, f"Input image width ({W}) is not a multiple of patch size ({patch_size[1]})."
            assert true_shape.shape == (B, 2), f"true_shape has the wrong shape={true_shape.shape}"
            if torch.any(is_portrait):
                x1 = self.dune_backbone.prepare_tokens_with_masks(image[is_landscape])
                x2 = self.dune_backbone.prepare_tokens_with_masks(image[is_portrait].permute(0, 1, 3, 2))
                x = torch.empty(B, x1.size(1), x1.size(2), device=image.device, dtype=x1.dtype)
                x[is_landscape] = x1
                x[is_portrait] = x2
                pos = image.new_zeros(
                    (B, x.shape[1] - self.dune_backbone.num_register_tokens - 1, 2), dtype=torch.int64)
                pos[is_landscape] = self.mast3r.patch_embed.position_getter(1, H, W, pos.device)
                pos[is_portrait] = self.mast3r.patch_embed.position_getter(1, W, H, pos.device)
            else:
                x = self.dune_backbone.prepare_tokens_with_masks(image)
                pos = self.mast3r.patch_embed.position_getter(B, H, W, image.device)
        else:
            assert torch.all(is_landscape) or torch.all(is_portrait)
            x = self.dune_backbone.prepare_tokens_with_masks(image)
            pos = self.mast3r.patch_embed.position_getter(B, H, W, image.device)

        for blk in self.dune_backbone.blocks[0]:
            x = blk(x)
        features = x[:, self.dune_backbone.num_register_tokens + 1:, :]
        return features.detach(), pos

    @torch.no_grad()
    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos = self._encode_image(torch.cat((img1, img2), dim=0),
                                          torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # If fix_background_color_from_depth is enabled, add background color from views
        if self.fix_background_color_from_depth and self.predict_underwater_params:
            # Get background colors from views
            bg_color1 = view1.get('background_color', None)
            bg_color2 = view2.get('background_color', None)
            
            if bg_color1 is not None:
                # Convert numpy array to torch tensor if needed
                if isinstance(bg_color1, np.ndarray):
                    bg_color1 = torch.from_numpy(bg_color1).float().to(res1['beta_1'].device)
                # Expand to [B, H, W, 3] shape
                B, H, W = res1['beta_1'].shape[:3]
                if bg_color1.dim() == 1:
                    bg_color1 = bg_color1.view(1, 1, 1, 3).expand(B, H, W, 3)
                elif bg_color1.dim() == 2 and bg_color1.shape[0] == B:
                    # [B, 3] -> [B, 1, 1, 3]
                    bg_color1 = bg_color1.view(B, 1, 1, 3).expand(B, H, W, 3)
                # Convert to [B, H, W, 3] if currently [B, 3, H, W]
                if bg_color1.dim() == 4 and bg_color1.shape[1] == 3:
                    bg_color1 = bg_color1.permute(0, 2, 3, 1)
                res1['b_inf'] = bg_color1
            
            if bg_color2 is not None:
                # Convert numpy array to torch tensor if needed
                if isinstance(bg_color2, np.ndarray):
                    bg_color2 = torch.from_numpy(bg_color2).float().to(res2['beta_1'].device)
                # Expand to [B, H, W, 3] shape
                B, H, W = res2['beta_1'].shape[:3]
                if bg_color2.dim() == 1:
                    bg_color2 = bg_color2.view(1, 1, 1, 3).expand(B, H, W, 3)
                elif bg_color2.dim() == 2 and bg_color2.shape[0] == B:
                    # [B, 3] -> [B, 1, 1, 3]
                    bg_color2 = bg_color2.view(B, 1, 1, 3).expand(B, H, W, 3)
                # Convert to [B, H, W, 3] if currently [B, 3, H, W]
                if bg_color2.dim() == 4 and bg_color2.shape[1] == 3:
                    bg_color2 = bg_color2.permute(0, 2, 3, 1)
                res2['b_inf'] = bg_color2

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
