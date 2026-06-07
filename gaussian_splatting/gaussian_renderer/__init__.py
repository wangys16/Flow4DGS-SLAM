#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh

import numpy as np


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)
    
    return depth_silhouette

def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
    dynamic=False,
    dx=None,
    ds=None,
    dr=None,
    do=None,
    dc=None,
    novel=0,
    render_mask=False,
    uid=None,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    
    #if viewpoint_camera.uid == 0:
    #    dx, ds, dr = None, None, None  # ??
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    
        

    means3D = pc.get_xyz
    means2D = screenspace_points

    if pc.use_gmm:
        if uid is None:
            opacity = pc.get_gmm_opacity(viewpoint_camera.uid)
        else:
            opacity = pc.get_gmm_opacity(uid)
    else:
        opacity = pc.get_opacity
    
    time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        
    else:
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        
        if not (pc.use_gmm_rot and pc.use_gmm):
            rotations = pc.get_rotation
        else:
            rotations = pc.get_gmm_rotation(uid if uid is not None else viewpoint_camera.uid)



    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if render_mask:
        shs = torch.ones_like(pc.get_features, device=pc.get_features.device, dtype=pc.get_features.dtype)
    else:
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                    pc.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color
    
    if dynamic:
        if pc.get_scaling.shape[-1] == 1: 
            means3D, scales_final, rotations_final, _, _, _ = pc._deformation(means3D, pc._scaling.repeat(1, 3), 
                                                                                         pc._rotation, pc._opacity, shs, time)
        else:
            means3D, scales_final, rotations_final, _, _, _ = pc._deformation(means3D, pc._scaling, 
                                                                                         pc._rotation, pc._opacity, shs, time)
        scales = pc.scaling_activation(scales_final)
        rotations = pc.rotation_activation(rotations_final)
    if dx is not None and ds is not None and dr is not None:
        dxyz=torch.zeros_like(means3D)
        dxyz[pc.dygs] = dx
        means3D = pc.get_xyz + dxyz
        del dxyz
        dscale = torch.zeros_like(scales)
        dscale[pc.dygs] = ds
        scales = scales + dscale
        del dscale
        drot=torch.zeros_like(rotations)
        drot[pc.dygs] = dr
        if not (pc.use_gmm_rot and pc.use_gmm):
            rotations = pc.get_rotation
        else:
            rotations = pc.get_gmm_rotation(uid if uid is not None else viewpoint_camera.uid)
        rotations =  rotations + drot
        del drot
    
    if dc is not None:
        dshs = torch.zeros_like(shs)
        dshs[pc.dygs] = dx
        shs = pc.get_features + dshs


    if mask is not None:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )
        
    else:
        rendered_image, radii, depth, opacity, n_touched = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
            theta=viewpoint_camera.cam_rot_delta,
            rho=viewpoint_camera.cam_trans_delta,
        )

    d = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
    }

    return d

    
    

def render_flow(
    pc: GaussianModel,
    viewpoint_camera1,
    viewpoint_camera2,
    d_xyz1, d_xyz2,
    d_rotation1, d_scaling1,
    scaling_modifier=1.0,
    compute_cov3D_python=False,
    scale_const=None,
    d_rot_as_res=True,
    ds=1,
    **kwargs
):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device="cuda",
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera1.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera1.FoVy * 0.5)

    # About Motion
    carnonical_xyz = pc.get_xyz.clone()
    xyz_at_t1 = xyz_at_t2 = carnonical_xyz.detach()  # Detach coordinates of Gaussians here

    dxyz1=torch.zeros_like(xyz_at_t1)
    if d_xyz1 is not None:
        dxyz1[pc.dygs] = d_xyz1
    xyz_at_t1 = xyz_at_t1 + dxyz1
    #xyz_at_t1 = xyz_at_t1 + d_xyz1
    dxyz2=torch.zeros_like(xyz_at_t1)
    if d_xyz2 is not None:
        dxyz2[pc.dygs] = d_xyz2
    xyz_at_t2 = xyz_at_t2 + dxyz2

    
    
    gaussians_homogeneous_coor_t2 = torch.cat([xyz_at_t2, torch.ones_like(xyz_at_t2[..., :1])], dim=-1)
    full_proj_transform = viewpoint_camera2.full_proj_transform if viewpoint_camera2 is not None else viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_homogeneous_coor_t2 @ full_proj_transform
    gaussians_uvz_coor_at_cam2 = gaussians_uvz_coor_at_cam2[..., :3] / (gaussians_uvz_coor_at_cam2[..., -1:] + 1e-7)

    gaussians_homogeneous_coor_t1 = torch.cat([xyz_at_t1, torch.ones_like(xyz_at_t1[..., :1])], dim=-1)
    gaussians_uvz_coor_at_cam1 = gaussians_homogeneous_coor_t1 @ viewpoint_camera1.full_proj_transform
    gaussians_uvz_coor_at_cam1 = gaussians_uvz_coor_at_cam1[..., :3] / (gaussians_uvz_coor_at_cam1[..., -1:] + 1e-7)

    flow_uvz_1to2 = gaussians_uvz_coor_at_cam2 - gaussians_uvz_coor_at_cam1
    
    # Rendering motion mask
    # if not pc.use_canonical:
    flow_uvz_1to2[..., -1:] = pc.dygs.unsqueeze(1)  #pc.motion_mask


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera1.image_height) // ds,
        image_width=int(viewpoint_camera1.image_width) // ds,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg = torch.zeros_like(flow_uvz_1to2[0]),  # Background set as 0
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera1.world_view_transform,
        projmatrix=viewpoint_camera1.full_proj_transform,
        projmatrix_raw=viewpoint_camera1.projection_matrix,
        sh_degree=0,
        campos=viewpoint_camera1.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    #means3D = pc.get_xyz + d_xyz1  # About Motion
    means3D = carnonical_xyz + dxyz1
    means2D = screenspace_points


    if pc.use_gmm:
        opacity = pc.get_gmm_opacity(viewpoint_camera1.uid).clone().detach()
    else:
        opacity = pc.get_opacity.clone().detach()

    if scale_const is not None:
        # If providing scale_const, directly use scale_const
        scales = torch.ones_like(pc.get_scaling) * scale_const
        if d_rot_as_res:
            rotations = pc.get_rotation + d_rotation1
        else:
            rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1, pc.get_rotation)
        cov3D_precomp = None
    else:
        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier, d_rotation=None if type(d_rotation1) is float else d_rotation1)
        else:
            dscale = torch.zeros_like(pc.get_scaling)
            dscale[pc.dygs] = d_scaling1
            scales = pc.get_scaling.clone().detach() + dscale
            del dscale
            #scales = pc.get_scaling + d_scaling1
            if d_rot_as_res:
                drot=torch.zeros_like(pc.get_rotation)
                drot[pc.dygs] = d_rotation1
                rotations = pc.get_rotation.clone().detach() + drot
                del drot
                #rotations = pc.get_rotation + d_rotation1
            else:
                rotations = pc.get_rotation if type(d_rotation1) is float else quaternion_multiply(d_rotation1, pc.get_rotation)

    if pc.use_gmm and pc.use_gmm_rot:
        rotations = pc.get_gmm_rotation(viewpoint_camera1.uid).clone().detach()

    
        
    rendered_image, radii, rendered_depth, rendered_alpha, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=flow_uvz_1to2,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    


def get_dynamic_mask(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    override_color=None,
    dynamic=True,
):
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    means3D = pc.get_xyz.clone().detach()
    
    time = torch.tensor(viewpoint_camera.time - 1).to(means3D.device).repeat(means3D.shape[0],1)
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    if dynamic:
        if pc.get_scaling.shape[-1] == 1: 
            _, _, _, dx, ds, dr = pc._deformation(means3D, pc._scaling.repeat(1, 3).clone().detach(), 
                                                  pc._rotation.clone().detach(), pc._opacity.clone().detach(), shs.clone().detach(), time)
        else:
            _, _, _, dx, ds, dr = pc._deformation(means3D, pc._scaling, 
                                                  pc._rotation, pc._opacity, shs, time)
        print(torch.norm(dx, dim=1).mean(), torch.norm(ds, dim=1).mean(), torch.norm(dr, dim=1).mean())
        print(torch.norm(dx, dim=1).max(), torch.norm(ds, dim=1).max(), torch.norm(dr, dim=1).max())
        position_mask = (torch.norm(dx, dim=1) < 1)
        scale_mask = (torch.norm(ds, dim=1) < 2) 
        direction_mask = (torch.norm(dr, dim=1) < 1)
        
        static_mask = position_mask & scale_mask & direction_mask
        return static_mask   

    