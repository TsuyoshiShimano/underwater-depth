# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf  # noqa
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT  # noqa
import dust3r.utils.path_to_croco  # noqa
from models.blocks import Mlp  # noqa
from models.dpt_block import Interpolate  # noqa


def reg_desc(desc, mode):
    if 'norm' in mode:
        desc = desc / desc.norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown desc mode {mode}")
    return desc


def postprocess(out, depth_mode, conf_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None, 
                predict_underwater_params=False, fix_background_color_from_depth=False):
    """
    Postprocess model output.
    
    Args:
        predict_underwater_params: If True, extract β_1, β_2, B_inf from output
        fix_background_color_from_depth: If True, only predict β_1 and β_2 (6 channels), B_inf is fixed
    """
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    start_idx = 3
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., start_idx], mode=conf_mode)
        start_idx += 1
    if desc_dim is not None:
        res['desc'] = reg_desc(fmap[..., start_idx:start_idx + desc_dim], mode=desc_mode)
        start_idx += desc_dim
        if two_confs:
            res['desc_conf'] = reg_dense_conf(fmap[..., start_idx], mode=desc_conf_mode)
            start_idx += 1
        else:
            res['desc_conf'] = res['conf'].clone()
    
    # Extract underwater parameters: β_1, β_2 (RGB each), B_inf (RGB)
    # If fix_background_color_from_depth=True: only β_1, β_2 (6 channels)
    # Otherwise: β_1, β_2, B_inf (9 channels)
    if predict_underwater_params:
        remaining_dims = fmap.shape[-1] - start_idx
        if fix_background_color_from_depth:
            # Only predict β_1 and β_2 (6 channels)
            if remaining_dims >= 6:
                beta_1_raw = fmap[..., start_idx:start_idx + 3]  # [B, H, W, 3]
                beta_2_raw = fmap[..., start_idx + 3:start_idx + 6]  # [B, H, W, 3]
                
                # Apply activation functions to constrain values
                res['beta_1'] = torch.sigmoid(beta_1_raw) * 0.5  # [0, 0.5] range
                res['beta_2'] = torch.sigmoid(beta_2_raw) * 0.5  # [0, 0.5] range
                # B_inf will be set from view data in forward pass
        else:
            # Predict all 9 channels (β_1, β_2, B_inf)
            if remaining_dims >= 9:
                beta_1_raw = fmap[..., start_idx:start_idx + 3]  # [B, H, W, 3]
                beta_2_raw = fmap[..., start_idx + 3:start_idx + 6]  # [B, H, W, 3]
                b_inf_raw = fmap[..., start_idx + 6:start_idx + 9]  # [B, H, W, 3]
                
                # Apply activation functions to constrain values
                res['beta_1'] = torch.sigmoid(beta_1_raw) * 0.5  # [0, 0.5] range
                res['beta_2'] = torch.sigmoid(beta_2_raw) * 0.5  # [0, 0.5] range
                res['b_inf'] = torch.sigmoid(b_inf_raw)  # [0, 1] range
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, head_type="regression", **kwargs):
        super().__init__(num_channels=num_channels, feature_dim=feature_dim, last_dim=last_dim, hooks_idx=hooks_idx,
                         dim_tokens=dim_tokens, depth_mode=depth_mode, postprocess=postprocess, conf_mode=conf_mode, head_type=head_type)
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        self.predict_underwater_params = getattr(net, 'predict_underwater_params', False)
        self.fix_background_color_from_depth = getattr(net, 'fix_background_color_from_depth', False)
        idim = net.enc_embed_dim + net.dec_embed_dim

        # Local features head (descriptors + confidence)
        # Keep original output dimension to load pretrained weights
        local_feat_dim = self.local_feat_dim + self.two_confs
        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=local_feat_dim * self.patch_size**2)
        
        # Separate underwater parameters head
        # If fix_background_color_from_depth=True: only β_1, β_2 (6 channels)
        # Otherwise: β_1, β_2, B_inf (9 channels)
        # This head is randomly initialized and trained from scratch
        if self.predict_underwater_params:
            num_underwater_channels = 6 if self.fix_background_color_from_depth else 9
            self.head_underwater_params = Mlp(in_features=idim,
                                             hidden_features=int(hidden_dim_factor * idim),
                                             out_features=num_underwater_channels * self.patch_size**2)

    def forward(self, decout, img_shape):
        # pass through the heads
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # recover encoder and decoder outputs
        enc_output, dec_output = decout[0], decout[-1]
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features (descriptors + confidence) using pretrained weights
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W
        
        # extract underwater parameters (β_1, β_2, B_inf) using separate head
        if self.predict_underwater_params:
            underwater_params = self.head_underwater_params(cat_output)  # B,S,9*patch_size^2
            underwater_params = underwater_params.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            underwater_params = F.pixel_shuffle(underwater_params, self.patch_size)  # B,9,H,W
            # concatenate local_features and underwater_params
            local_features = torch.cat([local_features, underwater_params], dim=1)

        # post process 3D pts, descriptors and confidences
        out = torch.cat([pts3d, local_features], dim=1)
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode,
                                   predict_underwater_params=self.predict_underwater_params,
                                   fix_background_color_from_depth=self.fix_background_color_from_depth)
        return out


class MLP_MiniConv_Head(nn.Module):
    """
    A special Convolutional head inspired by DPT architecture
    A MLP predicts pixelwise feats in lower resolution. Prediction is upsampled to target res and goes through a mini convolutional head

    Input : [B, S, D]  # S = (H//p) * (W//p)

    MLP: 
        D -> (mlp_hidden_dim) -> out_mlp_dim * (p/2)*2 
        reshape to [out_mlp_dim, H/2, W/2] (MLP predicts in half-res)

    MiniConv head from DPT: 
        Upsample x2 -> [out_mlp_dim,H,W]
        Conv 3x3 -> [conv_inner_dim,H,W]
        ReLU
        Conv 1x1 -> [odim,H,W]

    """

    def __init__(self, idim, mlp_hidden_dim, mlp_odim, conv_inner_dim, odim, patch_size, subpatch=2, **kw):
        super().__init__()
        self.patch_size = patch_size
        self.subpatch = subpatch
        self.sub_patch_size = patch_size // subpatch
        self.mlp = Mlp(idim, mlp_hidden_dim, mlp_odim * self.sub_patch_size**2, **kw)  # D -> mlp_odim*sub_patch_size**2

        # DPT conv head
        self.head = nn.Sequential(Interpolate(scale_factor=self.subpatch, mode="bilinear", align_corners=True) if self.subpatch != 1 else nn.Identity(),
                                  nn.Conv2d(mlp_odim, conv_inner_dim, kernel_size=3, stride=1, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(conv_inner_dim, odim, kernel_size=1, stride=1, padding=0)
                                  )

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape
        # extract features
        feat = self.mlp(tokens)  # [B, S, mlp_odim*sub_patch_size**2]
        feat = feat.transpose(-1, -2).reshape(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.sub_patch_size)  # B,mlp_odim,H/sub,W/sub

        return self.head(feat)  # B, odim, H, W


class Cat_MLP_LocalFeatures_MiniConv_Pts3d(nn.Module):
    """ Mixture between MLP and MLP-Convolutional head that outputs 3d points (with miniconv) and local features (with MLP).
    simply contains two MLP_MiniConv_Head: one for 3D points and one for features.
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., mlp_odim=24, conv_inner_dim=100, subpatch=2, **kw):
        super().__init__()

        self.local_feat_dim = local_feat_dim
        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.desc_mode = net.desc_mode
        self.desc_conf_mode = net.desc_conf_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.predict_underwater_params = getattr(net, 'predict_underwater_params', False)
        self.fix_background_color_from_depth = getattr(net, 'fix_background_color_from_depth', False)
        idim = net.enc_embed_dim + net.dec_embed_dim
        self.head_pts3d = MLP_MiniConv_Head(idim=idim,
                                            mlp_hidden_dim=int(hidden_dim_factor * idim),
                                            mlp_odim=mlp_odim + self.has_conf,
                                            conv_inner_dim=conv_inner_dim,
                                            odim=3 + self.has_conf,
                                            subpatch=subpatch,
                                            patch_size=self.patch_size,
                                            **kw)

        # Local features head (descriptors + confidence)
        # Keep original output dimension to load pretrained weights
        local_feat_dim = self.local_feat_dim + self.two_confs
        self.head_local_features = Mlp(in_features=idim,
                                       hidden_features=int(hidden_dim_factor * idim),
                                       out_features=local_feat_dim * self.patch_size**2)
        
        # Separate underwater parameters head
        # If fix_background_color_from_depth=True: only β_1, β_2 (6 channels)
        # Otherwise: β_1, β_2, B_inf (9 channels)
        # This head is randomly initialized and trained from scratch
        if self.predict_underwater_params:
            num_underwater_channels = 6 if self.fix_background_color_from_depth else 9
            self.head_underwater_params = Mlp(in_features=idim,
                                             hidden_features=int(hidden_dim_factor * idim),
                                             out_features=num_underwater_channels * self.patch_size**2)

    def forward(self, decout, img_shape):
        enc_output, dec_output = decout[0], decout[-1]  # recover encoder and decoder outputs
        cat_output = torch.cat([enc_output, dec_output], dim=-1)  # concatenate
        # pass through the heads
        pts3d = self.head_pts3d([cat_output], img_shape)

        H, W = img_shape
        B, S, D = cat_output.shape

        # extract local_features (descriptors + confidence) using pretrained weights
        local_features = self.head_local_features(cat_output)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W
        
        # extract underwater parameters (β_1, β_2, B_inf) using separate head
        if self.predict_underwater_params:
            num_channels = 6 if self.fix_background_color_from_depth else 9
            underwater_params = self.head_underwater_params(cat_output)  # B,S,num_channels*patch_size^2
            underwater_params = underwater_params.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
            underwater_params = F.pixel_shuffle(underwater_params, self.patch_size)  # B,num_channels,H,W
            # concatenate local_features and underwater_params
            local_features = torch.cat([local_features, underwater_params], dim=1)

        # post process 3D pts, descriptors and confidences
        out = postprocess(torch.cat([pts3d, local_features], dim=1),
                          depth_mode=self.depth_mode,
                          conf_mode=self.conf_mode,
                          desc_dim=self.local_feat_dim,
                          desc_mode=self.desc_mode,
                          two_confs=self.two_confs,
                          desc_conf_mode=self.desc_conf_mode,
                          predict_underwater_params=self.predict_underwater_params,
                          fix_background_color_from_depth=self.fix_background_color_from_depth)
        return out


def mast3r_head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               head_type='regression')
    elif head_type == 'catconv' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        # more params (anounced by a ':' and comma separated)
        kw = {}
        if ':' in head_type:
            kw = eval("dict(" + head_type[8:] + ")")
        return Cat_MLP_LocalFeatures_MiniConv_Pts3d(net, local_feat_dim=local_feat_dim, has_conf=has_conf, **kw)
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
