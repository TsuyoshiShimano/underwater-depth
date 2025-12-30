# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import numpy as np
from .base.mast3r_base_stereo_view_dataset import MASt3RBaseStereoViewDataset

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.datasets.arkitscenes import ARKitScenes as DUSt3R_ARKitScenes  # noqa
from dust3r.datasets.blendedmvs import BlendedMVS as DUSt3R_BlendedMVS  # noqa
from dust3r.datasets.co3d import Co3d as DUSt3R_Co3d  # noqa
from dust3r.datasets.megadepth import MegaDepth as DUSt3R_MegaDepth  # noqa
from dust3r.datasets.scannetpp import ScanNetpp as DUSt3R_ScanNetpp  # noqa
from dust3r.datasets.staticthings3d import StaticThings3D as DUSt3R_StaticThings3D  # noqa
from dust3r.datasets.waymo import Waymo as DUSt3R_Waymo  # noqa
from dust3r.datasets.wildrgbd import WildRGBD as DUSt3R_WildRGBD  # noqa
from dust3r.datasets.blender_processed import BlenderProcessed as DUSt3R_BlenderProcessed  # noqa


class ARKitScenes(DUSt3R_ARKitScenes, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class BlendedMVS(DUSt3R_BlendedMVS, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, split=None, **kwargs):
        super().__init__(*args, ROOT=ROOT, split=split, **kwargs)
        self.is_metric_scale = False


class Co3d(DUSt3R_Co3d, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class MegaDepth(DUSt3R_MegaDepth, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        super().__init__(*args, split=split, ROOT=ROOT, **kwargs)
        self.is_metric_scale = False


class ScanNetpp(DUSt3R_ScanNetpp, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class StaticThings3D(DUSt3R_StaticThings3D, MASt3RBaseStereoViewDataset):
    def __init__(self, ROOT, *args, mask_bg='rand', **kwargs):
        super().__init__(ROOT, *args, mask_bg=mask_bg, **kwargs)
        self.is_metric_scale = False


class Waymo(DUSt3R_Waymo, MASt3RBaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class WildRGBD(DUSt3R_WildRGBD, MASt3RBaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.is_metric_scale = True


class BlenderProcessed(DUSt3R_BlenderProcessed, MASt3RBaseStereoViewDataset):
    """ Dataset of Blender rendered scenes with down/up view pairs for MASt3R
    Filters out invalid depth points (1e10) from .exr files
    """

    def _get_views(self, pair_idx, resolution, rng):
        views = super()._get_views(pair_idx, resolution, rng)
        
        # Filter out invalid depth points (1e10) from .exr files
        INVALID_DEPTH_THRESHOLD = 1e9  # Threshold for invalid depth (1e10)
        
        for view in views:
            depthmap = view['depthmap']
            # Mask out invalid depth values (>= 1e9, which includes 1e10)
            if isinstance(depthmap, np.ndarray):
                invalid_mask = depthmap >= INVALID_DEPTH_THRESHOLD
                # Set invalid depths to NaN or 0 (will be filtered by valid_mask later)
                depthmap[invalid_mask] = np.nan
                view['depthmap'] = depthmap
        
        return views
