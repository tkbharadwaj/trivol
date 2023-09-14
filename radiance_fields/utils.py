"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import random
from typing import Optional, Sequence

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch

from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator 
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from .variable_channel_rendering import (
    accumulate_along_rays_,
    render_weight_from_density,
    variable_rendering,
)
import gc
import collections

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    voxels_xyz: torch.Tensor,
    aabb: torch.Tensor,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
    #convolutional MLP
    conv_mlp: torch.nn.Module = None,
    render_channels: int = 16,
    patch_size: int = 10,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape
    
    def voxel_query(voxels, xyz, aabb):
        """
        Computes RGB\sigma values from a tri-plane representation + MLP

        voxels_x: [B, C, P, S, S]
        voxels_y: [B, C, S, P, S]
        voxels_z: [B, C, S, S, P]
        xyz: [B, N, 3]
        aabb: [B, 2, 3]
        """
        voxels_x, voxels_y, voxels_z = voxels
        coords = (xyz - aabb[:, 0:1]) / (aabb[:, 1:2] - aabb[:, 0:1]) * 2.0 - 1.0  # [B, N, 3]
        coords_xyz = coords[..., [2, 1, 0]].unsqueeze(1).unsqueeze(1) # (B, 1, 1, N, 3)
        Fyz = torch.nn.functional.grid_sample(voxels_x, coords_xyz).squeeze(2).squeeze(2) # (B, C, N)
        Fxz = torch.nn.functional.grid_sample(voxels_y, coords_xyz).squeeze(2).squeeze(2) # (B, C, N)
        Fxy = torch.nn.functional.grid_sample(voxels_z, coords_xyz).squeeze(2).squeeze(2) # (B, C, N)

        # out = (Fyz + Fxz + Fxy).permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        out = torch.cat([Fyz, Fxz, Fxy], dim=1).permute(0, 2, 1)  # (B, C, N) -> (B, N, 3*C)
        return out

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        feats = voxel_query(voxels_xyz, positions[None], aabb)[0] # planes: [B, 3 * F, R, S] -> [B, N, C]

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            sigmas = radiance_field.query_density(positions, feats, t)
        else:
            sigmas = radiance_field.query_density(positions, feats)
        return sigmas.squeeze(-1)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        feats = voxel_query(voxels_xyz, positions[None], aabb)[0] # planes: [B, 3 * F, R, S] -> [B, N, C]

        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            rgbs, sigmas = radiance_field(positions, t, t_dirs)
        else:
            rgbs, sigmas = radiance_field(positions, feats, t_dirs)
        return rgbs, sigmas.squeeze(-1)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    
    for i in range(0, num_rays, chunk):
        #print(i*chunk)
        
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        # print("Chunk rays origins shape:",chunk_rays.origins.shape)
        # print("Chunk rays viewdirs shape:", chunk_rays.viewdirs.shape)
        
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=None,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth, extras = variable_rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
            num_channels=render_channels
        )
        if conv_mlp is not None:
            rgb = rgb.reshape(int(rgb.shape[0]/patch_size**2), patch_size, patch_size, rgb.shape[1])
            rgb = rgb.permute(0, 3, 1, 2)
            rgb = conv_mlp(rgb)
        
        chunk_results = [rgb, opacity, depth, extras['alphas'], extras['rgbs'], ray_indices+i, len(t_starts)]
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
       
    rgbs, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        rgbs.reshape((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )
