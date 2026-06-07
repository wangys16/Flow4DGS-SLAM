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

import os

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn
import time
import torch.nn.functional as F


from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
    get_linear_noise_func,
    find_orthonormal_vectors_batch,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.deformation import deform_network
from gaussian_splatting.scene.deform_model import DeformModel

def _normalize_quat(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def _align_to_first(Q, k_axis=1):
    """
    Flip quaternions so all components share the hemisphere of the k=0 reference.
    Q: (..., K, ..., 4) with quaternion in last dim.
    k_axis: axis corresponding to K (mixture components). Default 1 for (N,K,4).
    """
    q0 = Q.select(dim=k_axis, index=0).unsqueeze(k_axis)   # (...,1,...,4)
    dot = (Q * q0).sum(dim=-1, keepdim=True)               # (...,K,...,1)
    sign = torch.where(dot >= 0, 1.0, -1.0)
    return Q * sign


class GaussianModel:
    def __init__(self, sh_degree: int, config=None, args=None, fea_dim=0, init_deform='mlp', nframes=1):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        
        self.dygs = torch.empty(0, device="cuda").bool()
        
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log


        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.use_gmm = config['model_params'].get('use_gmm', False)
        self.use_gmm_rot = config['model_params'].get('use_gmm_rot', False)
        self.use_tsh = config['model_params'].get('use_tsh', False)
        self.ply_input = None

        self.covariance_activation = (
            self.build_covariance_from_scaling_rotation
        )

        self.isotropic = False
        self.with_motion_mask = False
        self.time_interval = 0
        self.nframes = nframes

        self.kf_list = []
        self.window_size = self.config["Training"]["window_size"]
        
        self.deform_init = False
        self.init_deform = init_deform
        if init_deform == 'mlp':
            self.deform = DeformModel(K=args.K, deform_type=args.deform_type, is_blender=args.is_blender,
                                      skinning=args.skinning, hyper_dim=args.hyper_dim, node_num=args.node_num,
                                      pred_opacity=args.pred_opacity, pred_color=args.pred_color, use_hash=args.use_hash,
                                      hash_time=args.hash_time, d_rot_as_res=args.d_rot_as_res and not args.d_rot_as_rotmat,
                                      local_frame=args.local_frame, progressive_brand_time=args.progressive_brand_time,
                                      with_arap_loss=not args.no_arap_loss, max_d_scale=args.max_d_scale,
                                      enable_densify_prune=args.node_enable_densify_prune, is_scene_static=args.is_scene_static)
        
        elif init_deform == 'offset':
            

            self._delta_xyz = nn.ParameterList([torch.empty(0, device="cuda")])


        if self.use_gmm:
            self.gmm_K = config['model_params'].get('gmm_K', 3)
            self._gmm_logits = torch.empty(0, self.gmm_K, device="cuda")
            self._gmm_means = torch.empty(0, self.gmm_K, device="cuda")
            self._gmm_scales = torch.empty(0, self.gmm_K, device="cuda")
            self._gmm_amp = torch.empty(0, device="cuda")
            self._tstart = torch.empty(0, device="cuda")

            if self.use_gmm_rot:
                self._rot_logits = torch.empty(0, self.gmm_K, device="cuda")
                self._rot_quat = torch.empty(0, self.gmm_K, 4, device="cuda")

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_dygs_xyz(self):
        return self._xyz[self.dygs]
        
    @property
    def motion_dy_mask(self):
        return torch.ones_like(self._xyz[self.dygs].unsqueeze(1))
        
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    

    def get_delta_mask(self, idx):
        return self.unique_kfIDs.cuda()[self.dygs] <= self.kf_list[idx]
    

    def get_delta_xyz(self, idx):
        return self._delta_xyz[idx]

    
    @property
    def get_tcenter(self):
        return self._tcenter
    
    @property
    def get_tscale(self):
        return self._tscale
    
    @property
    def get_tscale_left(self):
        return self._tscale_left
    
    @property
    def get_tscale_right(self):
        return self._tscale_right

    @property
    def get_tstart(self):
        return self._tstart

    

    def _logit(self, x: torch.Tensor, eps: float = 1e-6):
        x = x.clamp(eps, 1 - eps)
        return torch.log(x) - torch.log1p(-x)

    def _softplus_inv(self, y: torch.Tensor, eps: float = 1e-8):
        y = y.clamp(min=eps)
        return y + torch.log(-torch.expm1(-y))
    
    def get_gmm_opacity(self, t):
        """
        t: (...,) real timestamps in [0, T]
        base_opacity: (N,) in [0,1]
        returns: (N, ...): time-varying opacity = coef * base_opacity
        """
        if torch.sum(self.dygs) == 0:
            return self.get_opacity
        # normalize time
        t_hat = (torch.tensor(t, device=self._gmm_logits.device) / self.nframes).clamp(0, 1)

        # params
        w = F.softmax(self._gmm_logits, dim=1)         # (N,K)
        mu = torch.sigmoid(self._gmm_means)            # (N,K)
        sigma = F.softplus(self._gmm_scales) + 1e-4    # (N,K)
        A = F.softplus(self._gmm_amp) + 1e-4           # (N,)

        # broadcasting: (1,1,...)
        t_ = t_hat.reshape((1,)*2 + t_hat.shape)
        z = (t_ - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) + 1e-8)
        gauss = torch.exp(-0.5 * z*z) / (sigma.unsqueeze(-1) * 2.5066282746310002)  # 1/sqrt(2π)
        mix = torch.sum(w.unsqueeze(-1) * gauss, dim=1)                              # (N,...)
        activation = 1.0 - torch.exp(-A.unsqueeze(-1) * mix)                  # (N,...)

        multiplier = torch.ones_like(self._opacity)
        if self.dygs.sum() > 0:
            multiplier[self.dygs] = activation * (t_hat>=self._tstart).float()

        return self.get_opacity * multiplier


    def get_gmm_rotation(self, t):
        """
        Return per-Gaussian quaternion (N, ..., 4) at time t.
        Uses shared GMM means/scales to make a smooth Gaussian-weighted blend
        of K control quaternions per primitive.
        """
        if torch.sum(self.dygs) == 0:
            return self.get_rotation
        device = self._gmm_logits.device
        dtype  = self._gmm_logits.dtype

        t_hat = torch.as_tensor(t, device=device, dtype=dtype) / self.nframes
        t_hat = t_hat.clamp(0, 1)

        mu    = torch.sigmoid(self._gmm_means)               
        sigma = F.softplus(self._gmm_scales) + 1e-4          

        t_ = t_hat.reshape((1, 1) + tuple(t_hat.shape))      
        z  = (t_ - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) + 1e-8)
        gauss = torch.exp(-0.5 * z*z) / (sigma.unsqueeze(-1) * 2.5066282746310002)  

        base_w = F.softmax(self._rot_logits, dim=1)         
        weights = base_w.unsqueeze(-1) * gauss             
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  

        q_aligned = _align_to_first(self._rot_quat, k_axis=1)  

        w_exp = weights.unsqueeze(-1)                         
        q_sum = (w_exp * q_aligned.unsqueeze(2)).sum(dim=1)   

        q_t = _normalize_quat(q_sum)                       
        
        all_rotations = self.get_rotation
        all_rotations[self.dygs] = q_t.squeeze()

        return all_rotations



    def get_mapped_means3D(self, frameidx, frozen_offset=False):
        xyz = self.get_xyz.clone()
        if frozen_offset:
            delta_xyz = self.get_delta_xyz(self.kf_list.index(frameidx)).detach().clone()
            
        else:
            delta_xyz = self.get_delta_xyz(self.kf_list.index(frameidx)).clone()
        xyz[self.dygs] += delta_xyz
        return xyz
    
    def save_offsets(self, save_dir, iteration):
        out_weights_path = os.path.join(save_dir, "offsets/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        to_save = {'delta_xyz': self._delta_xyz, 'kf_list': self.kf_list}
        if self.use_gmm:
            to_save['gmm_logits'] = self._gmm_logits
            to_save['gmm_means'] = self._gmm_means
            to_save['gmm_scales'] = self._gmm_scales
            to_save['gmm_amp'] = self._gmm_amp
            to_save['tstart'] = self._tstart
            if self.use_gmm_rot:
                to_save['rot_logits'] = self._rot_logits
                to_save['rot_quat'] = self._rot_quat
        torch.save(to_save, os.path.join(out_weights_path, 'offsets.pth'))
    
    @property
    def motion_mask(self):
        return torch.ones_like(self.get_dygs_xyz[..., :1])
        return self.dygs.unsqueeze(1)
        if self.with_motion_mask:
            return torch.sigmoid(self.feature[..., -1:])
        else:
            return torch.ones_like(self._xyz[..., :1])
    
    def get_rotation_bias(self, rotation_bias=None):
        rotation_bias = rotation_bias if rotation_bias is not None else 0.
        return self.rotation_activation(self._rotation + rotation_bias)
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None, add_dygs=False, new_mask=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
                
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        if add_dygs:
            depth = np.copy(cam_info.depth)
            depth[cam.motion_mask.cpu().numpy()] = 0
            depth = o3d.geometry.Image(depth.astype(np.float32))
        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init, new_mask=new_mask)


    def create_pcd_from_image_and_depth(
        self,
        cam,
        rgb,            # open3d.geometry.Image (RGB uint8)
        depth,          # open3d.geometry.Image (float32 or uint16 per your pipeline)
        init=False,
        new_mask=None,  # [H, W] bool/uint8/float (prob) motion mask in image space
        visualize=False, # if True, return an overlay image (RGB) with projected points
        vis_point_radius=2,
        vis_alpha=0.7,
        vis_out_overlay=None,     # optional: path to save overlay image
    ):
        # ------------------------ config ------------------------
        downsample_factor = self.config["Dataset"]["pcd_downsample_init"] if init else self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if self.config["Dataset"].get("adaptive_pointsize", False):
            depth_np_for_median = getattr(cam, "depth", None)
            if depth_np_for_median is None:
                depth_np_for_median = np.asarray(depth)
                if depth_np_for_median.dtype == np.uint16:
                    depth_np_for_median = depth_np_for_median.astype(np.float32) / 1000.0
                else:
                    depth_np_for_median = depth_np_for_median.astype(np.float32)
            valid_median = depth_np_for_median > 0.1
            if np.any(valid_median):
                point_size = min(0.05, point_size * np.median(depth_np_for_median[valid_median]))

        # ------------------------ Open3D RGBD ------------------------
        # Convert depth to meters for stable behavior
        depth_np = np.asarray(depth)
        if depth_np.dtype == np.uint16:  # common for RealSense: millimeters
            depth_m = depth_np.astype(np.float32) / 1000.0
        else:
            depth_m = depth_np.astype(np.float32)
        

        # if new_mask exists, fill in depth in new_mask using nearest depth in cam.motion_mask
        if new_mask is not None and hasattr(cam, "motion_mask") and cam.motion_mask is not None:
            depth_m_filled = depth_m.copy()
            motion_mask_np = cam.motion_mask.cpu().numpy().astype(bool)
            # print('motion_mask_np: ', motion_mask_np)
            # print('new_mask: ', new_mask)
            if np.any(motion_mask_np) and torch.any(new_mask):
                # Get coordinates of valid depth pixels (cam.motion_mask)
                valid_y, valid_x = np.nonzero(motion_mask_np & (depth_m > 0) & np.isfinite(depth_m))
                valid_points = np.stack([valid_x, valid_y], axis=-1).astype(np.float32)

                # Get coordinates of pixels to fill (new_mask)
                fill_y, fill_x = np.nonzero(new_mask.cpu().numpy() & ~motion_mask_np)
                fill_points = np.stack([fill_x, fill_y], axis=-1).astype(np.float32)

                if valid_points.shape[0] > 0 and fill_points.shape[0] > 0:
                    # Use FRNN to find nearest valid depth pixel for each pixel in new_mask
                    valid_tensor = torch.from_numpy(valid_points).unsqueeze(0).cuda()

        # Keep a copy of the validity mask from the ORIGINAL depth (no inpainting),
        # valid if finite and > 0
        valid_depth_mask = np.isfinite(depth_m) & (depth_m > 0.0)

        rgb_np = np.asarray(rgb)  # HxWx3 uint8 RGB
        Hc, Wc = rgb_np.shape[:2]
        Hd, Wd = depth_m.shape[:2]
        if (Hc, Wc) != (Hd, Wd):
            # If sizes differ, resize depth validity to color size
            depth_m = cv2.resize(depth_m, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
            valid_depth_mask = cv2.resize(valid_depth_mask.astype(np.uint8), (Wc, Hc), interpolation=cv2.INTER_NEAREST).astype(bool)

        depth_o3d_m = o3d.geometry.Image(depth_m.astype(np.float32))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth_o3d_m,
            depth_scale=1.0,           # already meters
            depth_trunc=float(self.config["Dataset"].get("depth_trunc", 1e6)),
            convert_rgb_to_intensity=False,
        )

        # world -> camera
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()

        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                int(cam.image_width),
                int(cam.image_height),
                float(cam.fx),
                float(cam.fy),
                float(cam.cx),
                float(cam.cy),
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )

        # Downsample
        if new_mask is None:
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / max(1.0, float(downsample_factor)))

            # N x 3 arrays
            new_xyz = np.asarray(pcd_tmp.points)
            new_rgb = np.asarray(pcd_tmp.colors)
        
        else:
            # ================= STRATIFIED DOWNSAMPLE =================
            # Extract all points/colors first
            all_xyz = np.asarray(pcd_tmp.points)   # (N,3) world
            all_rgb = np.asarray(pcd_tmp.colors)   # (N,3) [0..1]
            N = all_xyz.shape[0]

            # Default: everything is non-motion
            in_motion_full = np.zeros(N, dtype=bool)

            # If we have new_mask, determine per-point motion (by projection)
            if new_mask is not None and N > 0:
                # normalize new_mask -> bool numpy
                if isinstance(new_mask, torch.Tensor):
                    nm = new_mask.detach().cpu().numpy()
                else:
                    nm = np.asarray(new_mask)
                if nm.dtype.kind == "f":
                    nm = nm > 0.5
                else:
                    nm = nm.astype(bool)

                H_img, W_img = nm.shape

                # project world->camera->pixel
                Rcw = W2C[:3, :3]
                tcw = W2C[:3, 3]
                pts_c = (all_xyz @ Rcw.T) + tcw                 # (N,3)
                z = pts_c[:, 2]
                valid_z = z > 1e-6
                u = cam.fx * (pts_c[:, 0] / z) + cam.cx
                v = cam.fy * (pts_c[:, 1] / z) + cam.cy
                ui = np.rint(u).astype(np.int32)
                vi = np.rint(v).astype(np.int32)
                in_bounds = (ui >= 0) & (ui < W_img) & (vi >= 0) & (vi < H_img)
                proj_ok = valid_z & in_bounds

                idx = np.nonzero(proj_ok)[0]
                if idx.size > 0:
                    in_motion_full[idx] = nm[vi[idx], ui[idx]]

            # Compute per-point keep probability:
            D = max(1.0, float(downsample_factor))        # safeguard
            keep_prob_non = 1.0 / D
            keep_prob_mot = min(1.0, 1.0 / self.config["Dataset"]["pcd_downsample_init"])  # motion points less downsampled

            keep_prob = np.where(in_motion_full, keep_prob_mot, keep_prob_non)

            # Sample
            rng = np.random.default_rng()                 # or np.random for legacy
            keep_mask = rng.random(N) < keep_prob

            # If everything got dropped (rare), keep at least something
            if not np.any(keep_mask) and N > 0:
                keep_mask[rng.integers(0, N)] = True

            # Apply selection
            new_xyz = all_xyz[keep_mask]
            new_rgb = all_rgb[keep_mask]
            
            motion_mask_points = torch.from_numpy(in_motion_full[keep_mask]).to(torch.bool).cuda()

        # Keep on self
        pcd = BasicPointCloud(points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3)))
        self.ply_input = pcd

        # ------------------------ motion mask: image -> points ------------------------
        motion_mask_points = torch.zeros(new_xyz.shape[0], dtype=torch.bool, device="cuda")
        proj_ok = None
        ui = vi = None
        if new_mask is not None and new_xyz.shape[0] > 0:
            pts_w = torch.from_numpy(new_xyz).float().cuda()       # (N,3)
            W2C_t = torch.from_numpy(W2C).float().cuda()           # (4,4)
            fx, fy = float(cam.fx), float(cam.fy)
            cx, cy = float(cam.cx), float(cam.cy)

            H, W = (rgb_np.shape[0], rgb_np.shape[1])  # image size
            Rcw = W2C_t[:3, :3]
            tcw = W2C_t[:3, 3]
            pts_c = (pts_w @ Rcw.T) + tcw                           # (N,3)
            z = pts_c[:, 2]
            valid_z = z > 1e-6

            u = fx * (pts_c[:, 0] / z) + cx
            v = fy * (pts_c[:, 1] / z) + cy
            ui_t = torch.round(u).long()
            vi_t = torch.round(v).long()

            in_bounds = (ui_t >= 0) & (ui_t < W) & (vi_t >= 0) & (vi_t < H)
            proj_ok_t = valid_z & in_bounds

            # mask on CUDA
            if isinstance(new_mask, torch.Tensor):
                mask_img = new_mask
                if mask_img.device.type != "cuda":
                    mask_img = mask_img.cuda()
            else:
                mask_img = torch.from_numpy(new_mask).cuda()

            if mask_img.dtype.is_floating_point:
                positive = mask_img > 0.5
            else:
                positive = mask_img != 0

            idx = torch.nonzero(proj_ok_t, as_tuple=False).squeeze(1)
            if idx.numel() > 0:
                sampled = positive[vi_t[idx], ui_t[idx]]
                motion_mask_points[idx] = sampled

            proj_ok = proj_ok_t.detach().cpu().numpy()
            ui = ui_t.detach().cpu().numpy()
            vi = vi_t.detach().cpu().numpy()
        else:
            # If no new_mask, we still need projection for visualization
            if new_xyz.shape[0] > 0:
                Rcw = W2C[:3, :3]
                tcw = W2C[:3, 3]
                pts_c_np = (new_xyz @ Rcw.T) + tcw
                z_np = pts_c_np[:, 2]
                valid_z_np = z_np > 1e-6
                u_np = cam.fx * (pts_c_np[:, 0] / z_np) + cam.cx
                v_np = cam.fy * (pts_c_np[:, 1] / z_np) + cam.cy
                ui = np.rint(u_np).astype(np.int32)
                vi = np.rint(v_np).astype(np.int32)
                proj_ok = valid_z_np & (ui >= 0) & (ui < rgb_np.shape[1]) & (vi >= 0) & (vi < rgb_np.shape[0])

        # ------------------------ features for your pipeline ------------------------
        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                1e-7,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )


        # ------------------------ visualization (optional) ------------------------
        if visualize and new_xyz.shape[0] > 0 and (new_mask is not None):
            # --- overlay points over RGB: red(motion)/blue(static) ---
            base_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            overlay = base_bgr.copy()

            motion_np = motion_mask_points.detach().cpu().numpy() if isinstance(motion_mask_points, torch.Tensor) else np.array(motion_mask_points, dtype=bool)
            idx = np.nonzero(proj_ok)[0] if proj_ok is not None else np.array([], dtype=np.int32)
            if idx.size > 0:
                motion_idx = idx[motion_np[idx]]
                static_idx = idx[~motion_np[idx]]

                for i in motion_idx:
                    cv2.circle(overlay, (int(ui[i]), int(vi[i])), vis_point_radius, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)


            vis_bgr = cv2.addWeighted(overlay, vis_alpha, base_bgr, 1 - vis_alpha, 0)
            overlay_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            if vis_out_overlay is not None:
                cv2.imwrite(vis_out_overlay, vis_bgr)

            # --- valid-depth mask view: green(valid)/red(invalid), blended over grayscale RGB ---
            gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
            gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            mask = valid_depth_mask.astype(bool)
            mask_color = np.zeros_like(gray_3, dtype=np.uint8)
            mask_color[~mask] = (0, 0, 255)   # invalid -> red
            mask_color[ mask] = (0, 255, 0)   # valid   -> green
            depth_vis_bgr = cv2.addWeighted(mask_color, 0.5, gray_3, 0.5, 0)

            # --- side-by-side panel ---
            # Make sure same size (they are, but safeguard)
            if depth_vis_bgr.shape[:2] != vis_bgr.shape[:2]:
                depth_vis_bgr = cv2.resize(depth_vis_bgr, (vis_bgr.shape[1], vis_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            panel_bgr = cv2.hconcat([vis_bgr, depth_vis_bgr])

            
            panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
            cv2.imwrite('unprojection.png', panel_bgr)

        return fused_point_cloud, features, scales, rots, opacities, motion_mask_points


    def create_node_from_depth(self, cam, opt_params, sc_params, remove_outlier=False):
        if cam.motion_mask is not None and torch.all(cam.motion_mask):
            print("no dynamic object")
            return False
        elif torch.sum(~cam.motion_mask) < self.config["Dataset"]["pcd_downsample"]*2:
            print("False count is too low.")
            return False
        if not self.deform_init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]*2
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        depth_raw = np.copy(cam.depth)
        depth_raw[cam.motion_mask.cpu().numpy()] = 0
        if remove_outlier:
            depth_raw[~cam.get_mask_outlier()] = 0

        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depth_raw.astype(np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )
        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
        if self.deform_init:
            print(len(fused_point_cloud))
            self.deform.extend_node_from_point(init_pcl=fused_point_cloud, keep_all=True,force_init=True, reset_bbox=False)
        else:
            self.deform.deform.init(opt=opt_params, init_pcl=fused_point_cloud, keep_all=True,
                                    force_init=True, reset_bbox=False)

            fused_point_cloud[:, 2] += 0.2
            self.deform.train_setting(sc_params)
            self.deform.extend_node_from_point(init_pcl=fused_point_cloud, keep_all=True,force_init=True, reset_bbox=False)
            self.deform_init = True
            return True
        return False
        
    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
    

    def _init_means_centered(self, t_center, N, mean_noise=0.3):
        """
        Initialize pre-sigmoid means so that sigmoid(means) clusters around t_center/T.
        If t_center is None, fall back to a broad uniform around 0.5 (old behavior style).
        """
        if t_center is None:
            # center at 0.5 with mild noise
            center = 0.5
        else:
            center = max(1e-4, min(1 - 1e-4, float(t_center / self.nframes)))
        center_logit = self._logit(torch.tensor(center))
        return center_logit + mean_noise * torch.randn(N, self.gmm_K)

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id, add_dygs, motion_pts_mask,
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()

        if add_dygs:
            new_dygs = torch.ones(new_xyz.shape[0], device="cuda").bool()
        else:
            new_dygs = motion_pts_mask.to(new_xyz.device)
        
        new_delta_xyz = None
        # new_delta_features = None
        # new_delta_scaling = None

        new_delta_rotation = None
        
        if self.init_deform == 'offset':
            if torch.any(new_dygs):
                new_delta_xyz = nn.ParameterList([nn.Parameter(
                    torch.zeros_like(new_xyz[new_dygs]).requires_grad_(True)
                ) for _ in range(len(self.kf_list))])
            else:
                new_delta_xyz = nn.ParameterList([torch.empty((0, 3), device=new_opacity.device).requires_grad_(True) for _ in range(len(self.kf_list))])


        else:
            new_delta_xyz = None
        
        new_tstart = None
        new_tcenter = None
        new_tscale = None
        new_tscale_left = None
        new_tscale_right = None
        new_gmm_logits = None
        new_gmm_means = None
        new_gmm_scales = None
        new_gmm_amp = None
        new_rot_logits = None
        new_rot_quat = None

        if self.use_gmm:
            if torch.any(new_dygs):
                N = new_opacity[new_dygs].shape[0]
                time = float(kf_id) / float(self.nframes) if not add_dygs else 0.0
                new_gmm_logits = torch.zeros((N, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                new_gmm_means  = self._init_means_centered(kf_id, N).to(new_opacity.device).requires_grad_(True)       # -> sigmoid * T
                new_gmm_scales = torch.log(torch.exp(torch.tensor(0.3, device=new_opacity.device)) - 1.).expand(N, self.gmm_K).requires_grad_(True) # softplus^-1
                new_gmm_amp    = torch.zeros(N, device=new_opacity.device).requires_grad_(True)
                new_tstart = time * torch.ones_like(new_opacity[new_dygs], device=new_opacity.device).requires_grad_(False)
                if self.use_gmm_rot:
                    new_rot_logits = torch.zeros((N, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                    q0 = torch.zeros((N, self.gmm_K, 4), device=new_opacity.device); q0[..., 0] = 1.0
                    new_rot_quat = _normalize_quat(q0 + 0.02*torch.randn_like(q0)).requires_grad_(True)
            else:
                new_gmm_logits = torch.empty((0, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                new_gmm_means  = torch.empty((0, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                new_gmm_scales = torch.empty((0, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                new_gmm_amp    = torch.empty((0,), device=new_opacity.device).requires_grad_(True)
                new_tstart = torch.empty((0,), device=new_opacity.device).requires_grad_(False)

                if self.use_gmm_rot:
                    new_rot_logits = torch.empty((0, self.gmm_K), device=new_opacity.device).requires_grad_(True)
                    new_rot_quat = torch.empty((0, self.gmm_K, 4), device=new_opacity.device).requires_grad_(True)



        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_dygs,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_delta_xyz=new_delta_xyz,
            new_delta_rotation=new_delta_rotation,
            new_tcenter=new_tcenter,
            new_tscale=new_tscale,
            new_tscale_left=new_tscale_left,
            new_tscale_right=new_tscale_right,
            new_tstart=new_tstart,
            new_gmm_logits=new_gmm_logits,
            new_gmm_means=new_gmm_means,
            new_gmm_scales=new_gmm_scales,
            new_gmm_amp=new_gmm_amp,
            new_rot_logits=new_rot_logits,
            new_rot_quat=new_rot_quat,
            reset=False,
        )


        closest_cache_idx = self.kf_list[(self.kf_list.index(kf_id) // self.window_size) * self.window_size]


    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None, add_dygs=False, new_mask=None, 
    ):
        fused_point_cloud, features, scales, rots, opacities, motion_pts_mask = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap, add_dygs=add_dygs, new_mask=new_mask)
        )
        if fused_point_cloud.shape[0] > 0:
            self.extend_from_pcd(
                fused_point_cloud, features, scales, rots, opacities, kf_id, add_dygs, motion_pts_mask=motion_pts_mask,
            )

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float=5., print_info=True, max_point_num=150_000):
        self.spatial_lr_scale = 5
        if type(pcd.points) == np.ndarray:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        else:
            fused_point_cloud = pcd.points
        if type(pcd.colors) == np.ndarray:
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = pcd.colors
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if print_info:
            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.with_motion_mask:
            self.feature.data[..., -1] = torch.zeros_like(self.feature[..., -1])
    

    def fix_parameters(self):
        self._xyz.requires_grad = False
        self._features_dc.requires_grad = False
        self._features_rest.requires_grad = False
        self._scaling.requires_grad = False
        self._rotation.requires_grad = False
        self._opacity.requires_grad = False
    
    def training_setup(self, training_args):
        self.training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        if self.init_deform == 'offset':
            for i in range(len(self._delta_xyz)):
                l.append({'params': [self._delta_xyz[i]], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "delta_xyz_%s" % i})
            
        
        if self.use_gmm:
            l.append({'params': [self._gmm_logits], 'lr': 0.02, "name": "gmm_logits"})
            l.append({'params': [self._gmm_means], 'lr': 1e-3, "name": "gmm_means"})
            l.append({'params': [self._gmm_scales], 'lr': 1e-3, "name": "gmm_scales"})
            l.append({'params': [self._gmm_amp], 'lr': 1e-3, "name": "gmm_amp"})

            l.append({'params': [self._tstart], 'lr': training_args.trbfs_lr, "name": "tstart"})

            if self.use_gmm_rot:
                l.append({'params': [self._rot_logits], 'lr': 0.02, "name": "rot_logits"})
                l.append({'params': [self._rot_quat], 'lr': 1e-3, "name": "rot_quat"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        
        
        self.max_steps = training_args.position_lr_max_steps
    
    def training_network_setup(self, training_args):

        l = [
            {
                "params": list(self._deformation.get_mlp_parameters()),
                "lr": training_args.deformation_lr_init * self.spatial_lr_scale, 
                "name": "deformation",
            },
            {
                "params": list(self._deformation.get_grid_parameters()), 
                'lr': training_args.grid_lr_init * self.spatial_lr_scale, 
                "name": "grid",
            },
         ]
        self.network_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=training_args.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)    
        self.grid_scheduler_args = get_expon_lr_func(lr_init=training_args.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.deformation_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.deformation_lr_init = training_args.deformation_lr_init*self.spatial_lr_scale
        self.deformation_lr_final = training_args.deformation_lr_final*self.spatial_lr_scale
        self.deformation_lr_delay_mult = training_args.deformation_lr_delay_mult
        
        self.grid_lr_init = training_args.grid_lr_init*self.spatial_lr_scale
        self.grid_lr_final = training_args.grid_lr_final*self.spatial_lr_scale
        
    
    
    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                
        return lr
        
    def update_learning_rate_deformation(self, iteration):
        for param_group in self.network_optimizer.param_groups:
            if param_group["name"] == "deformation":
                lr = helper(
                     iteration,
                     lr_init=self.deformation_lr_init,
                     lr_final=self.deformation_lr_final,
                     lr_delay_mult=self.deformation_lr_delay_mult,
                     max_steps=self.max_steps,
                 )
            elif param_group["name"] == "grid":
                lr = helper(
                    iteration,
                    lr_init=self.grid_lr_init,
                    lr_final=self.grid_lr_final,
                    lr_delay_mult=self.deformation_lr_delay_mult,
                    max_steps=self.max_steps,
                )
        return lr
        
    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        l.append("dygs")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        #print(dygs.shape, opacities.shape)
        dygs = self.dygs.detach().cpu().numpy().reshape(-1, 1)
        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation, dygs), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        
        if self.init_deform == 'offset':
            opacities_new[self.dygs] = self.get_opacity[self.dygs]

        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path) 

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        
        if "dygs" in plydata.elements[0]:
            dygs = np.asarray(plydata.elements[0]["dygs"])[..., np.newaxis]
            self.dygs = torch.tensor(dygs, dtype=torch.bool, device="cuda")
            self.dygs = self.dygs.squeeze()
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
    

    def load_offsets(self, path):
        # Load delta_xyz, delta_features, delta_scaling, delta_rotation, tcenter, tscale from .pth files
        checkpoint = torch.load(path)
        self._delta_xyz = nn.ParameterList()
        self.kf_list = checkpoint['kf_list']
        for i in range(len(checkpoint['delta_xyz'])):
            self._delta_xyz.append(nn.Parameter(checkpoint['delta_xyz'][i].to('cuda').requires_grad_(True)))
        if self.use_gmm:
            self._gmm_logits = nn.Parameter(checkpoint['gmm_logits'].to('cuda').requires_grad_(True))
            self._gmm_means = nn.Parameter(checkpoint['gmm_means'].to('cuda').requires_grad_(True))
            self._gmm_scales = nn.Parameter(checkpoint['gmm_scales'].to('cuda').requires_grad_(True))
            self._gmm_amp = nn.Parameter(checkpoint['gmm_amp'].to('cuda').requires_grad_(True))
            self._tstart = nn.Parameter(checkpoint['tstart'].to('cuda').requires_grad_(False))
            if self.use_gmm_rot:
                self._rot_logits = nn.Parameter(checkpoint['rot_logits'].to('cuda').requires_grad_(True))
                self._rot_quat = nn.Parameter(checkpoint['rot_quat'].to('cuda').requires_grad_(True))
        print(f'Loaded offsets from {path}, total {len(self._delta_xyz)} sets of offsets.')
        
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, dygs_mask=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:  
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if 'delta' in group["name"] or (group["name"] in ['tcenter', 'tscale', 'tstart', 'tscale_left', 'tscale_right', 
                            'gmm_logits', 'gmm_means', 'gmm_scales', 'gmm_amp', 'rot_logits', 'rot_quat']):
                selected_mask = dygs_mask
                if selected_mask is None or not torch.any(selected_mask):
                    continue
            else:
                selected_mask = mask
            
            if selected_mask is not None:
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][selected_mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][selected_mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        (group["params"][0][selected_mask].requires_grad_(True))
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        group["params"][0][selected_mask].requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_points(self, mask, dygs_mask=None):
        valid_points_mask = ~mask
        if dygs_mask is not None:
            valid_dygs_mask = ~dygs_mask
        else:
            valid_dygs_mask = None
        
        optimizable_tensors = self._prune_optimizer(valid_points_mask, valid_dygs_mask)


        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.init_deform == 'offset' and dygs_mask is not None and torch.any(valid_dygs_mask):
            self._delta_xyz = nn.ParameterList([optimizable_tensors[f"delta_xyz_{idx}"] for idx in range(len(self.kf_list))])

        if self.use_gmm and dygs_mask is not None and torch.any(valid_dygs_mask):
            self._gmm_logits = optimizable_tensors["gmm_logits"]
            self._gmm_means = optimizable_tensors["gmm_means"]
            self._gmm_scales = optimizable_tensors["gmm_scales"]
            self._gmm_amp = optimizable_tensors["gmm_amp"]
            self._tstart = optimizable_tensors["tstart"]

            if self.use_gmm_rot:
                self._rot_logits = optimizable_tensors["rot_logits"]
                self._rot_quat = optimizable_tensors["rot_quat"]
        

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.dygs = self.dygs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
    
    
            
    
    def update_delta_xyz(self, cur_idx, latest_kf, values=None, idx=None, max_motion=0.1):

        new_delta_xyz = torch.zeros_like(self._xyz[self.dygs]).requires_grad_(True)
        self._delta_xyz = nn.ParameterList(list(self._delta_xyz) + [new_delta_xyz])
        self.optimizer.add_param_group({'params': self._delta_xyz[-1], 'lr': self.training_args.position_lr_init * self.spatial_lr_scale, 'name': f'delta_xyz_{len(self.kf_list)-1}'})
        



        with torch.no_grad():
            # latest_kf = self.kf_list[-1]
            if values is None:
                print(f'Copying delta_xyz from time {latest_kf} to {cur_idx}')
                self._delta_xyz[self.kf_list.index(cur_idx)].copy_(self._delta_xyz[self.kf_list.index(latest_kf)])
            else:
                print(f"Updating delta_xyz from time {latest_kf} to {cur_idx} with flow offsets")
                
                if idx is not None:
                    target_idx = idx
                    max_motion_mask = torch.norm(values, dim=1) < max_motion
                    target_idx = target_idx[max_motion_mask]
                    target_delta_xyz = self._delta_xyz[self.kf_list.index(latest_kf)].clone().detach()
                    

                    # Compute the rest idx other than target_idx
                    rest_idx = torch.arange(self._delta_xyz[self.kf_list.index(cur_idx)].shape[0], device='cuda')
                    rest_idx = rest_idx[~torch.isin(rest_idx, target_idx)]

                    target_delta_xyz[target_idx] += values[max_motion_mask]
                    
                    self._delta_xyz[self.kf_list.index(cur_idx)].copy_(target_delta_xyz.clone().detach())


                else:
                    self._delta_xyz[self.kf_list.index(cur_idx)][self.dygs].copy_(self._delta_xyz[self.kf_list.index(latest_kf)].clone().detach() + values.clone().detach())
                    


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"])>1: 
                continue
            assert len(group["params"]) == 1
            if 'delta' in group["name"]:
                idx = int(group["name"].split('_')[-1])
                param_name = f'delta_'+group["name"].split('_')[1]
                if param_name not in tensors_dict:
                    continue
                if tensors_dict[param_name][idx] is None:
                    optimizable_tensors[group["name"]] = group["params"][0]
                    continue
                extension_tensor = tensors_dict[param_name][idx]
            else:
                extension_tensor = tensors_dict.get(group["name"], None)
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if extension_tensor is not None:
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat(
                        (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                    )
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                        dim=0,
                    )

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    self.optimizer.state[group["params"][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat(
                            (group["params"][0], extension_tensor), dim=0
                        ).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_dygs,
        new_kf_ids=None,
        new_n_obs=None,
        new_delta_xyz=None,
        new_delta_features=None,
        new_delta_scaling=None,
        new_delta_rotation=None,
        new_tcenter=None,
        new_tscale=None,
        new_tscale_left=None,
        new_tscale_right=None,
        new_tstart=None,
        new_gmm_logits=None,
        new_gmm_means=None,
        new_gmm_scales=None,
        new_gmm_amp=None,
        new_rot_logits=None,
        new_rot_quat=None,
        reset=True,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        
        if new_delta_xyz is not None:
            d["delta_xyz"] = new_delta_xyz
        
        
        if new_delta_rotation is not None:
            d["delta_rotation"] = new_delta_rotation
        
        if new_tcenter is not None:
            d["tcenter"] = new_tcenter
        
        if new_tscale is not None:
            d["tscale"] = new_tscale

        if new_tscale_left is not None:
            d["tscale_left"] = new_tscale_left

        if new_tscale_right is not None:
            d["tscale_right"] = new_tscale_right

        if new_tstart is not None:
            d["tstart"] = new_tstart
        
        if new_gmm_logits is not None:
            d["gmm_logits"] = new_gmm_logits
        
        if new_gmm_means is not None:
            d["gmm_means"] = new_gmm_means
        
        if new_gmm_scales is not None:
            d["gmm_scales"] = new_gmm_scales
        
        if new_gmm_amp is not None:
            d["gmm_amp"] = new_gmm_amp
        
        if new_rot_logits is not None:
            d["rot_logits"] = new_rot_logits
        
        if new_rot_quat is not None:
            d["rot_quat"] = new_rot_quat

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


        if new_delta_xyz is not None:
            self._delta_xyz = nn.ParameterList([optimizable_tensors[f"delta_xyz_{idx}"] for idx in range(len(self.kf_list))])
        
        if new_delta_rotation is not None:
            self._delta_rotation = nn.ParameterList([optimizable_tensors[f"delta_rotation_{idx}"] for idx in range(len(self.kf_list))])
        
        if new_tcenter is not None:
            self._tcenter = optimizable_tensors["tcenter"]
        
        if new_tscale is not None:
            self._tscale = optimizable_tensors["tscale"]
        
        if new_tscale_left is not None:
            self._tscale_left = optimizable_tensors["tscale_left"]
        
        if new_tscale_right is not None:
            self._tscale_right = optimizable_tensors["tscale_right"]

        if new_tstart is not None:
            self._tstart = optimizable_tensors["tstart"]
        
        if new_gmm_logits is not None:
            self._gmm_logits = optimizable_tensors["gmm_logits"]
        
        if new_gmm_means is not None:
            self._gmm_means = optimizable_tensors["gmm_means"]
        
        if new_gmm_scales is not None:
            self._gmm_scales = optimizable_tensors["gmm_scales"]
        
        if new_gmm_amp is not None:
            self._gmm_amp = optimizable_tensors["gmm_amp"]
        
        if new_rot_logits is not None:
            self._rot_logits = optimizable_tensors["rot_logits"]
        
        if new_rot_quat is not None:
            self._rot_quat = optimizable_tensors["rot_quat"]

        self.dygs = torch.cat((self.dygs, new_dygs))

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        
    

    def select_masked_fast(self, param_list: nn.ParameterList, mask: torch.Tensor, kf_id=None):
        if mask.dtype == torch.bool:
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        else:
            idx = mask
            

        device = param_list[0].device
        idx = idx.to(device)

        stacked = torch.stack([p for p in param_list], dim=0)   # [N, M, 3]
        sliced  = stacked.index_select(dim=1, index=idx)        # [N, K, 3]
        return list(sliced.unbind(dim=0))                       # list of N tensors [K,3]


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )


        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")

        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
            
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)
        
        new_dygs = self.dygs[selected_pts_mask].repeat(N)


        dygs_mask = torch.gather(selected_pts_mask, 0, torch.nonzero(self.dygs, as_tuple=False).squeeze(1))
        prune_filter_dygs = torch.cat(
            (
                dygs_mask,
                torch.zeros(N * dygs_mask.sum(), device="cuda", dtype=bool),
            )
        )

        new_delta_xyz = None
        new_delta_rotation = None

        if self.init_deform == 'offset':
            if torch.any(dygs_mask):
                new_delta_xyz = self.select_masked_fast(self._delta_xyz, dygs_mask)
                new_delta_xyz_list = []
                for p in new_delta_xyz:
                    if p is not None:
                        new_delta_xyz_list.append(p.repeat(N, 1))
                    else:
                        new_delta_xyz_list.append(None)

                new_delta_xyz = nn.ParameterList(new_delta_xyz_list)
            else:
                new_delta_xyz = nn.ParameterList([torch.empty((0, 3), device=self._delta_xyz[0].device) for _ in range(len(self._delta_xyz))])
            
            
        else:
            new_delta_xyz = None
        
        new_tstart = None
        new_tcenter = None
        new_tscale = None
        new_tscale_left = None
        new_tscale_right = None
        new_gmm_logits = None
        new_gmm_means = None
        new_gmm_scales = None
        new_gmm_amp = None
        new_rot_logits = None
        new_rot_quat = None

        if self.use_gmm:
            if torch.any(dygs_mask):
                new_gmm_logits = self._gmm_logits[dygs_mask].repeat(N, 1)
                new_gmm_means = self._gmm_means[dygs_mask].repeat(N, 1)
                new_gmm_scales = self._gmm_scales[dygs_mask].repeat(N, 1)
                new_gmm_amp = self._gmm_amp[dygs_mask].repeat(N)
                if self.use_gmm_rot:
                    new_rot_logits = self._rot_logits[dygs_mask].repeat(N, 1)
                    new_rot_quat = self._rot_quat[dygs_mask].repeat(N, 1, 1)
                new_tstart = self._tstart[dygs_mask].repeat(N, 1)
            else:
                new_gmm_logits = torch.empty((0, self._gmm_logits.shape[1]), device=self._gmm_logits.device)
                new_gmm_means = torch.empty((0, self._gmm_means.shape[1]), device=self._gmm_means.device)
                new_gmm_scales = torch.empty((0, self._gmm_scales.shape[1]), device=self._gmm_scales.device)
                new_gmm_amp = torch.empty((0), device=self._gmm_amp.device)
                new_tstart = torch.empty((0), device=self._tstart.device)
                if self.use_gmm_rot:
                    new_rot_logits = torch.empty((0, self._rot_logits.shape[1]), device=self._rot_logits.device)
                    new_rot_quat = torch.empty((0, self._rot_quat.shape[1], self._rot_quat.shape[2]), device=self._rot_quat.device)
        
        

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_dygs,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_delta_xyz=new_delta_xyz,
            new_delta_rotation=new_delta_rotation,
            new_tcenter=new_tcenter,
            new_tscale=new_tscale,
            new_tscale_left=new_tscale_left,
            new_tscale_right=new_tscale_right,
            new_tstart=new_tstart,
            new_gmm_logits=new_gmm_logits,
            new_gmm_means=new_gmm_means,
            new_gmm_scales=new_gmm_scales,
            new_gmm_amp=new_gmm_amp,
            new_rot_logits=new_rot_logits,
            new_rot_quat=new_rot_quat,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )


        self.prune_points(prune_filter, prune_filter_dygs)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )


        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]


        new_delta_xyz = None
        new_delta_rotation = None
        

        dygs_mask = torch.gather(selected_pts_mask, 0, torch.nonzero(self.dygs, as_tuple=False).squeeze(1))
        if self.init_deform == 'offset':
            if torch.any(dygs_mask):
                new_delta_xyz = self.select_masked_fast(self._delta_xyz, dygs_mask)
                new_delta_xyz = nn.ParameterList(new_delta_xyz)
            else:
                new_delta_xyz = nn.ParameterList([torch.empty((0, 3), device=self._delta_xyz[0].device) for _ in range(len(self.kf_list))])


        else:
            new_delta_xyz = None
        
        new_tcenter = None
        new_tscale = None
        new_tscale_left = None
        new_tscale_right = None
        new_tstart = None
        new_gmm_logits = None
        new_gmm_means = None
        new_gmm_scales = None
        new_gmm_amp = None  
        new_rot_logits = None
        new_rot_quat = None


        if self.use_gmm and torch.any(dygs_mask):
            new_gmm_logits = self._gmm_logits[dygs_mask]
            new_gmm_means = self._gmm_means[dygs_mask]
            new_gmm_scales = self._gmm_scales[dygs_mask]
            new_gmm_amp = self._gmm_amp[dygs_mask]
            if self.use_gmm_rot:
                new_rot_logits = self._rot_logits[dygs_mask]
                new_rot_quat = self._rot_quat[dygs_mask]
            new_tstart = self._tstart[dygs_mask]


        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]


        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        new_dygs = self.dygs[selected_pts_mask]

        


        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_dygs,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_delta_xyz=new_delta_xyz,
            new_delta_rotation=new_delta_rotation,
            new_tcenter=new_tcenter,
            new_tscale=new_tscale,
            new_tscale_left=new_tscale_left,
            new_tscale_right=new_tscale_right,
            new_tstart=new_tstart,
            new_gmm_logits=new_gmm_logits if self.use_gmm else None,
            new_gmm_means=new_gmm_means if self.use_gmm else None,
            new_gmm_scales=new_gmm_scales if self.use_gmm else None,
            new_gmm_amp=new_gmm_amp if self.use_gmm else None,
            new_rot_logits=new_rot_logits if self.use_gmm_rot else None,
            new_rot_quat=new_rot_quat if self.use_gmm_rot else None,
        )


    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        if self.init_deform == 'offset':
            prune_mask = (self.get_opacity < min(0.4, min_opacity)).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        
        prune_dygs_mask = torch.gather(prune_mask, 0, torch.nonzero(self.dygs, as_tuple=False).squeeze(1))

        self.prune_points(prune_mask, prune_dygs_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
        
        
    def compute_plane_smoothness(self, t):
        batch_size, c, h, w = t.shape
        # Convolve with a second derivative filter, in the time dimension which is dimension 2
        first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
        second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
        # Take the L2 norm of the result
        return torch.square(second_difference).mean()
        
    def _plane_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =  [0,1,3]
            for grid_id in time_grids:
                total += self.compute_plane_smoothness(grids[grid_id])
        return total
        
    def _time_regulation(self):
        multi_res_grids = self._deformation.deformation_net.grid.grids
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids =[2, 4, 5]
            for grid_id in time_grids:
                total += self.compute_plane_smoothness(grids[grid_id])
        return total
        
    def _l1_regulation(self):
                # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids = self._deformation.deformation_net.grid.grids

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return total
        
    def compute_regulation(self, time_smoothness_weight=0.01, l1_time_planes_weight=0.0001, plane_tv_weight=0.0001):
        return plane_tv_weight * self._plane_regulation() + time_smoothness_weight * self._time_regulation() + l1_time_planes_weight * self._l1_regulation()
        
class StandardGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, fea_dim=0, with_motion_mask=True, all_the_same=False):
        super().__init__(sh_degree, fea_dim)
        self.all_the_same = all_the_same
        self.with_motion_mask = with_motion_mask
