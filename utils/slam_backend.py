import random
import time

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render, render_flow
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, gradient_loss_flow
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_network, pearson_loss
import os
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.colors import Normalize
# import imageio.v2 as imageio  # or imageio.v3 if needed

from matplotlib.colors import hsv_to_rgb
import torch.nn.functional as F
from matplotlib.patches import Patch
from scipy import ndimage as ndi
import seaborn as sns
from pytorch3d.ops import knn_points

def print_lrs(optimizer, step, every=50):
    if step % every != 0:
        return

    print(f"\n[Iter {step}] Learning rates:")
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{i}")
        lr = group["lr"]
        print(f"  {name:20s}: {lr:.6e}")


def flow_to_rgb(flow: torch.Tensor, max_magnitude: float = None, percentile: float = 95.0):
    """
    Convert optical flow [H, W, 2] to an RGB image using HSV:
    - Hue encodes direction (angle)
    - Value encodes normalized magnitude (speed)
    - Saturation = 1
    If max_magnitude is None, uses the given percentile of magnitudes for robust scaling.
    Returns uint8 RGB array [H, W, 3].
    """
    # Ensure CPU float32
    flow = flow.detach().to(torch.float32).cpu()
    assert flow.ndim == 3 and flow.shape[-1] == 2, "flow must be [H, W, 2]"
    u, v = flow[..., 0], flow[..., 1]

    # Handle NaNs/Infs
    valid = torch.isfinite(u) & torch.isfinite(v)
    u = torch.where(valid, u, torch.zeros_like(u))
    v = torch.where(valid, v, torch.zeros_like(v))

    # Magnitude & angle
    mag = torch.sqrt(u * u + v * v)                # [H, W]
    ang = torch.atan2(v, u)                        # [-pi, pi]
    H = (ang % (2 * torch.pi)) / (2 * torch.pi)    # [0, 1]
    S = torch.ones_like(H)

    if max_magnitude is None:
        vm = mag[valid]
        if vm.numel() == 0:
            max_magnitude = 1.0
        else:
            max_magnitude = torch.quantile(vm, percentile / 100.0).item()
            if max_magnitude <= 1e-8:
                max_magnitude = 1.0

    V = torch.clamp(mag / max_magnitude, 0, 1)

    hsv = torch.stack([H, S, V], dim=-1).numpy()    # float in [0,1]
    rgb = hsv_to_rgb(hsv)                           # float in [0,1]
    # Set invalid pixels to black
    invalid = ~valid.numpy()
    rgb[invalid] = 0.0

    return (rgb * 255.0).astype(np.uint8)

def show_flow(flow: torch.Tensor, with_quiver: bool = True, step: int = 16, figsize=(7, 5), save_path: str = None):
    """
    Visualize the flow as a color image. Optionally overlay a quiver (arrows) subsampled by `step`.
    """
    rgb = flow_to_rgb(flow)  # [H, W, 3], uint8
    H, W = rgb.shape[:2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, origin='upper')
    ax.set_axis_off()

    if with_quiver:
        # Subsample grid
        yy = np.arange(0, H, step)
        xx = np.arange(0, W, step)
        X, Y = np.meshgrid(xx, yy)

        flow_np = flow.detach().cpu().numpy()
        U = flow_np[yy[:, None], xx[None, :], 0]
        V = flow_np[yy[:, None], xx[None, :], 1]

        # Draw arrows; invert Y so arrows point the same way as image coordinates
        ax.quiver(X, Y, U, -V, angles='xy', scale_units='xy', scale=1.0, width=0.002, alpha=0.7)
        ax.set_ylim(H, 0)  # keep origin at top-left for consistency with the image

    plt.tight_layout()
    if save_path:
        import imageio
        imageio.imwrite(save_path, rgb)  # saves the color visualization (without quiver)
    plt.show()
    return rgb

def vis_render_process(gaussians, pipeline_params, background, viewpoint, cur_frame_idx, save_dir, out_dir="map", mask=None, dynamic=False, dynamic_model='mlp'):
    with torch.no_grad():
        if dynamic:
            if dynamic_model == 'mlp':
                time_input = gaussians.deform.deform.expand_time(viewpoint.fid)
                d_values = gaussians.deform.step(gaussians.get_dygs_xyz.detach(), time_input, 
                                                iteration=0, feature=None, 
                                                motion_mask=gaussians.motion_mask, 
                                                camera_center=viewpoint.camera_center, 
                                                time_interval=gaussians.time_interval)
                dxyz = d_values['d_xyz']
                d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                d_shs = None
            elif dynamic_model == 'offset':
                dxyz = gaussians.get_delta_xyz(gaussians.kf_list.index(viewpoint.uid))
                d_rot = gaussians.get_delta_rotation[gaussians.kf_list.index(viewpoint.uid)]
                d_scale, d_shs = 0, None
        else:
            dxyz, d_rot, d_scale, d_shs = 0, 0, 0, None
        render_pkg = render(
            viewpoint, gaussians, pipeline_params, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, mask=mask, dc=d_shs)
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        
        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_im)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}.png")
        plt.savefig(save_path)
        plt.close()
        return
    

def flow_to_pixels(flow, H, W, mode='grid'):
    """
    flow: [2,H,W]
    mode: 'grid' | 'ratio' | 'pixel'
    returns: flow_px [2,H,W] in *pixels*
    """
    if mode == 'pixel':
        return flow
    if mode == 'grid':
        sx, sy = (W - 1) / 2.0, (H - 1) / 2.0
    elif mode == 'ratio':
        sx, sy = float(W), float(H)
    else:
        raise ValueError("flow_mode must be 'grid' | 'ratio' | 'pixel'")
    scale = torch.tensor([sx, sy], device=flow.device, dtype=flow.dtype)[:, None, None]
    return flow * scale

def pixels_to_flow_units(flow_px, H, W, mode='grid'):
    """
    Inverse of flow_to_pixels for output consistency.
    """
    if mode == 'pixel':
        return flow_px
    if mode == 'grid':
        sx, sy = (W - 1) / 2.0, (H - 1) / 2.0
    elif mode == 'ratio':
        sx, sy = float(W), float(H)
    else:
        raise ValueError("flow_mode must be 'grid' | 'ratio' | 'pixel'")
    inv_scale = torch.tensor([1.0/sx, 1.0/sy], device=flow_px.device, dtype=flow_px.dtype)[:, None, None]
    return flow_px * inv_scale

def scale_intrinsics(K, sx, sy):
    """
    Scale intrinsics when resizing image by (sx, sy) on width/height.
    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]
    """
    Kds = K.clone()
    Kds[0,0] = K[0,0] * sx  # fx
    Kds[1,1] = K[1,1] * sy  # fy
    Kds[0,2] = (K[0,2] + 0.0) * sx  # cx
    Kds[1,2] = (K[1,2] + 0.0) * sy  # cy
    return Kds

# ----------------------- core model -----------------------

def _mesh(H, W, device, dtype):
    y, x = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                          torch.arange(W, device=device, dtype=dtype),
                          indexing='ij')
    return x, y  # pixel coords

def build_interaction_matrix(depth, K):
    """
    depth: [H,W] (meters), K: [3,3]
    Returns L: [2*H*W, 6] for all pixels (no masking applied here)
    NOTE: caller must mask out invalid rows.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x, y = _mesh(H, W, depth.device, depth.dtype)
    u, v = x, y
    Z = depth.clamp_min(1e-6)

    # translation terms (∝ 1/Z)
    du_dt = torch.stack([ -fx/Z, torch.zeros_like(Z), (u - cx)/Z ], dim=-1)  # [H,W,3]
    dv_dt = torch.stack([ torch.zeros_like(Z), -fy/Z, (v - cy)/Z ], dim=-1)

    # rotation terms (depth-independent)
    du_dw = torch.stack([
        (u - cx)*(v - cy)/fy,                 # wx
        -(fx + (u - cx)*(u - cx)/fx),         # wy
        (v - cy)                              # wz
    ], dim=-1)
    dv_dw = torch.stack([
        (fy + (v - cy)*(v - cy)/fy),
        - (u - cx)*(v - cy)/fx,
        - (u - cx)
    ], dim=-1)

    L_top = torch.cat([du_dt, du_dw], dim=-1)      # [H,W,6]
    L_bot = torch.cat([dv_dt, dv_dw], dim=-1)      # [H,W,6]
    L = torch.stack([L_top, L_bot], dim=2).reshape(-1, 6)  # [2HW,6]
    return L

def fit_twist_weighted(depth, flow_px, K, depth_valid, robust=True, iters=2, plain=False):
    """
    depth: [H,W], flow_px: [2,H,W] in pixels, K: [3,3]
    depth_valid: [H,W] boolean (depth>0)
    Returns xi: [6]
    """
    H, W = depth.shape
    L = build_interaction_matrix(depth, K)                 # [2HW,6]
    f = flow_px.permute(1,2,0).reshape(-1,2).reshape(-1)  # [2HW]

    # validity: depth valid + finite f + finite L rows
    Lfinite = torch.isfinite(L).all(dim=1)
    ffinite = torch.isfinite(f)
    valid = (depth_valid.reshape(-1).repeat_interleave(2)) & Lfinite & ffinite

    Lv = L[valid]             # [M,6]
    fv = f[valid]             # [M]
    if Lv.shape[0] < 1000 or plain:
        # very small support: fall back to unweighted least squares on whatever we have
        A = Lv.T @ Lv
        b = Lv.T @ fv
        return torch.linalg.lstsq(A, b).solution if hasattr(torch.linalg, "lstsq") else torch.linalg.solve(A, b)

    # robust IRLS on residuals r = fv - Lv @ xi
    w = torch.ones_like(fv)
    xi = torch.zeros(6, device=depth.device, dtype=depth.dtype)

    for _ in range(iters):
        # (L^T W L) xi = L^T W f
        Wv = w
        A = (Lv.T * Wv) @ Lv
        b = (Lv.T * Wv) @ fv
        # Solve (prefer solve if well-conditioned; fall back to lstsq)
        try:
            xi = torch.linalg.solve(A, b)
        except RuntimeError:
            xi = torch.linalg.lstsq(A, b).solution

        if not robust:
            break

        r = fv - (Lv @ xi)
        s = 1.4826 * torch.median(r.abs()) + 1e-6   # robust scale (MAD)
        z = r / (2.0 * s)
        w = 1.0 / (1.0 + z*z)                       # Cauchy weights

    return xi

def predict_rigid_flow_px(depth, K, xi):
    """
    Predicts rigid flow (pixels) at full resolution given depth, K, xi.
    """
    H, W = depth.shape
    L = build_interaction_matrix(depth, K)                 # [2HW,6]
    fhat = (L @ xi).reshape(H, W, 2).permute(2,0,1)        # [2,H,W]
    return fhat

# ----------------------- main entry -----------------------

@torch.no_grad()
def segment_motion_parametric(
    depth, flow, K,
    flow_mode='grid',   # 'grid' | 'ratio' | 'pixel'
    ds=2,               # downsample factor for fitting (>=1)
    robust_iters=2,
    k_mad=3.5,
    motion_mask=None,
):
    """
    Efficient motion segmentation via depth–flow parametric fit.

    depth: [H,W]  (meters), zeros invalid
    flow:  [2,H,W] (normalized in [-1,1] or pixels depending on flow_mode)
    K:     [3,3] intrinsics for *full-res* image (pixels)

    Returns dict with:
      mask_bool: [H,W]     True = dynamic
      rigid_flow_out: [2,H,W]  rigid flow in *same units as input flow*
      resid_px: [H,W]      residual magnitude (pixels)
      xi: [6]              estimated twist
    """
    assert depth.ndim == 2 and flow.ndim == 3 and flow.shape[0] == 2
    H, W = depth.shape

    flow_px = flow_to_pixels(flow, H, W, mode=flow_mode)

    depth_valid = (depth > 0) & torch.isfinite(depth)
    if motion_mask is not None:
        depth_valid = depth_valid & motion_mask

    if ds > 1:
        Hds, Wds = H // ds, W // ds
        depth_ds = F.interpolate(depth[None,None].float(), size=(Hds, Wds), mode='nearest').squeeze(0).squeeze(0)
        depth_valid_ds = (depth_ds > 0) & torch.isfinite(depth_ds)
        flow_px_ds = F.interpolate(flow_px[None].float(), size=(Hds, Wds), mode='area').squeeze(0)
        sx, sy = (Wds / W), (Hds / H)
        Kds = scale_intrinsics(K, sx, sy)
    else:
        depth_ds, depth_valid_ds, flow_px_ds, Kds = depth, depth_valid, flow_px, K

    xi = fit_twist_weighted(depth_ds, flow_px_ds, Kds, depth_valid_ds, robust=True, iters=robust_iters,)

    rigid_flow_px = predict_rigid_flow_px(depth, K, xi)

    resid = (flow_px - rigid_flow_px).pow(2).sum(0).sqrt()  
    resid_valid = resid[depth_valid]
    resid[~depth_valid] = 0.0
    if resid_valid.numel() == 0:
        mask_bool = torch.zeros_like(depth, dtype=torch.bool)
        rigid_flow_out = pixels_to_flow_units(rigid_flow_px, H, W, mode=flow_mode)
        return dict(mask_bool=mask_bool, rigid_flow_out=rigid_flow_out, resid_px=resid, xi=xi)

    med = resid_valid.median()
    mad = (resid_valid - med).abs().median().clamp_min(1e-6)
    tau = med + k_mad * 1.4826 * mad

    mask_bool = (resid > tau) & depth_valid

    rigid_flow_out = pixels_to_flow_units(rigid_flow_px, H, W, mode=flow_mode)

    rigid_flow_out[:, ~depth_valid] = 0.0

    return dict(
        mask_bool=mask_bool,
        rigid_flow_out=rigid_flow_out,
        resid_px=resid,
        xi=xi
    )

        
class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.sc_params = None
        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.dynamic_model = config["model_params"].get("dynamic_model", False)
        self.flow_offset = config["model_params"].get("flow_offset", False)
        self.insertion = config["model_params"].get("insertion", False)
        self.normal_loss = config["model_params"].get("normal_loss", 0.0)
        self.use_knn = config["model_params"].get("use_knn", False)
        self.knn_n = config["model_params"].get("knn_n", 16)
        self.knn_r = config["model_params"].get("knn_r", 0.2)
        self.depth_check = config["model_params"].get("depth_check", False)
        self.sd = config["model_params"].get("sd", False)

        self.flow_ds = config["Training"].get("flow_ds", 1)
        self.insertion_type = config["model_params"].get("insertion_type", "new")
        self.iters = 200

        w, h = 640, 480
        self.def_pix = torch.tensor(
            np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
        self.pix_ones = torch.ones(h * w, 1).cuda().float()

        self.mapping_time = []

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        
        self.save_dir = self.config["Results"]["save_dir"]
        
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.dygs_gaussian_update_every = self.config["Training"].get("dygs_gaussian_update_every", self.gaussian_update_every)
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        if self.iters < 100:
            self.gaussian_update_offset = self.iters // 4
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Training"]["single_thread"]
            if "single_thread" in self.config["Training"]
            else False
        )
    
    def fill_depth_holes_with_motion_nn_np(self, depth: np.ndarray,
                                       new_mask: np.ndarray,
                                       motion_mask: np.ndarray) -> np.ndarray:
        depth = depth.copy()
        new_mask = new_mask.astype(bool)
        motion_mask = motion_mask.astype(bool)

        invalid = (depth == 0) & new_mask
        seeds = motion_mask & (depth != 0)

        if not invalid.any() or not seeds.any():
            return depth

        # Option 1: get only indices
        idx = ndi.distance_transform_edt(~seeds, return_distances=False, return_indices=True)
        rr = idx[0][invalid]   # rows of nearest seed
        cc = idx[1][invalid]   # cols of nearest seed

        # Option 2 (equivalent): _, idx = ndi.distance_transform_edt(~seeds, return_indices=True)

        depth[invalid] = depth[rr, cc]
        return depth

    def visualize_depth_filling(self, depth, new_mask, motion_mask):
        filled = self.fill_depth_holes_with_motion_nn_np(depth, new_mask, motion_mask)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Depth Map")
        plt.imshow(depth, cmap='turbo')
        plt.colorbar(label='Depth')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Filled Depth Map")
        plt.imshow(filled, cmap='turbo')
        plt.colorbar(label='Depth')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('depth_filling.png')
        plt.close()

        return filled 

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, flow_back=None, closest_frame=None):
        print(f'START self.gaussians.dygs.shape: {self.gaussians.dygs.shape}, self.gaussians.dygs.sum(): {self.gaussians.dygs.sum()}')
        
        new_mask = None
        if self.insertion and flow_back is not None:
            new_mask = self.insertion_mask(viewpoint, flow_back, closest_frame)
            depth_map = self.fill_depth_holes_with_motion_nn_np(depth_map, new_mask.cpu().numpy(), ~viewpoint.motion_mask.cpu().numpy())

            depth_map[(~viewpoint.motion_mask.cpu().numpy()) & (~new_mask.cpu().numpy())] = 0

        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map, new_mask=new_mask
        )
        print(f'AFTER self.gaussians.dygs.shape: {self.gaussians.dygs.shape}, self.gaussians.dygs.sum(): {self.gaussians.dygs.sum()}')
        if frame_idx == self.dystart:
            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=frame_idx, init=True, scale=scale, depthmap=depth_map, add_dygs=True,
            )
            try:
                print(f'AFTER DYSTART self.gaussians.dygs.shape: {self.gaussians.dygs.shape}, self.gaussians.dygs.sum(): {self.gaussians.dygs.sum()}')
            except:
                pass
            
    def add_next_node(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        depth = np.copy(viewpoint.depth)
        depth[viewpoint.get_mask_outlier().cpu().numpy()] = 0
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth
        )
        if self.dynamic_model=='mlp' and frame_idx>0 and not self.gaussians.deform_init:
            first_init = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if first_init:
                self.initialize_network(frame_idx, viewpoint)
        elif self.gaussians.deform_init and self.config["Dataset"]["type"] == "CoFusion":
            self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
        self.initialize_network(frame_idx, viewpoint, update_gaussians=True)

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()
    
    def initialize_network(self, cur_frame_idx, viewpoint, update_gaussians=False):
        if cur_frame_idx == self.dystart:
            inited = self.gaussians.create_node_from_depth(viewpoint, self.opt_params, self.sc_params)
            if not inited:
                return
        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
        for mapping_iteration in range(100):
            if self.dynamic_model == 'mlp':
                d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                    iteration=0, feature=None, 
                                                    motion_mask=self.gaussians.motion_mask, 
                                                    camera_center=viewpoint.camera_center, 
                                                    time_interval=self.gaussians.time_interval)#, detach_node=False)
                dxyz = d_values['d_xyz']
                d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            else:
                dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid))
                d_rot, d_scale = 0, 0
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot,
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True,
            )
            
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                self.gaussians.deform.optimizer.step()
                self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                if update_gaussians:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        
        vis_render_process(self.gaussians, self.pipeline_params, self.background, viewpoint, 
                           viewpoint.uid, self.save_dir, out_dir="mapping", mask=None, dynamic=True,
                           dynamic_model=self.dynamic_model)
        Log("Initialized mlp", tag="Backend")
        
    
    def initialize_map(self, cur_frame_idx, viewpoint):

        for mapping_iteration in range(self.init_itr_num):  # self.init_itr_num
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background,
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True, rm_dynamic=not (self.dystart==cur_frame_idx)
            )
            loss_init.backward()

            
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map", tag="Backend")

        return render_pkg
    
    def find_closest_keyframe(self, uid):
        keys = [key for key in self.viewpoints if key < uid]
        if not keys:
            return None
        closest_key = max(keys)
        return closest_key

    def map(self, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=False):
        if len(current_window) == 0:
            return
        #
        key_opt = []
        if len(current_window) > 3:
            key_opt = self.viewpoints[current_window[0]].keyframe_selection_overlap(self.dataset, self.viewpoints, self.viewpoints[current_window[2]].uid)
        
        key_opt = current_window[:3] + key_opt
        
        self.viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]
        current_window_set = set(key_opt)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        flow_weights = self.config["Training"]["flow_loss"]
        delta = self.config["Training"]["delta"] if "delta" in self.config["Training"] else 5
        

        motion_start = {}
        rgb_start = {}
        first_uid = self.viewpoint_stack[0].uid


        
            
        for i in range(iters):
            if (self.dynamic_model == 'mlp' and (i>iters//2)) or (self.dynamic_model == 'offset' and iters>1):
                self.iteration_count += 1
            loss_network = 0
            self.last_sent += 1
            dygs_scaling = 0
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            flag = i < iters/2
            static_stage = self.sd and flag


            start_time = time.time()
            
            if i < iters/2:
                dynamic = True
                flow_weights = self.config["Training"]["flow_loss"]
            else:
                dynamic = False
                flow_weights = self.config["Training"]["flow_loss_fine"] if "flow_loss_fine" in self.config["Training"] else self.config["Training"]["flow_loss"]  
                
            for cam_idx in range(len(current_window)):
                viewpoint = self.viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                if self.dynamic_model == 'mlp':
                    if dynamic_network and self.gaussians.deform_init:
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                        iteration=0, feature=None, 
                                                        motion_mask=self.gaussians.motion_mask, 
                                                        camera_center=viewpoint.camera_center, 
                                                        time_interval=self.gaussians.time_interval)
                        dxyz = d_values['d_xyz']
                        d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                        d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                    elif dynamic_render and self.gaussians.deform_init:
                        with torch.no_grad():
                            time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                            N = time_input.shape[0]
                            ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count) 
                            d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise, 
                                                            iteration=0, feature=None, 
                                                            motion_mask=self.gaussians.motion_mask, 
                                                            camera_center=viewpoint.camera_center, 
                                                            time_interval=self.gaussians.time_interval)
                            dxyz = d_values['d_xyz'].detach()
                            d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
                            if d_values['d_opacity'] is not None: 
                                d_opac=d_values['d_opacity'].detach()
                            else:
                                d_opac =None
                            if d_values["d_color"] is not None: 
                                d_color = d_values["d_color"].detach()
                            else:
                                d_color=None
                    else:
                        dxyz = 0
                        d_rot, d_scale, d_opac, d_color = None, 0, None, None
                    dygs_scaling += d_scale
                
                elif self.dynamic_model == 'offset':
                    dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid))
                    d_rot = 0
                    d_scale, d_opac, d_color = 0, None, None
                

                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    dynamic=False, 
                    dx=dxyz, 
                    ds=d_scale, 
                    dr=d_rot, 
                    do=d_opac, 
                    dc=d_color,
                )
                
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                use_mask_loss = False

                if (not static_stage) and 'mask_loss' in self.config['Training'] and self.config["Training"]["mask_loss"] > 0:
                    use_mask_loss = True
                    render_pkg_dygs = render(
                        viewpoint, 
                        self.gaussians, 
                        self.pipeline_params, 
                        torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), 
                        dynamic=False, 
                        dx=dxyz, 
                        ds=d_scale, 
                        dr=d_rot, 
                        do=d_opac, 
                        dc=d_color,
                        mask=(self.gaussians.dygs==True),
                        render_mask=True,
                    )

                    opacity_np_dygs = render_pkg_dygs["render"][0, :, :].detach().cpu().squeeze(0).numpy()
                    mask_loss = F.binary_cross_entropy(render_pkg_dygs["render"][0:1, :, :], 
                                                       (~viewpoint.motion_mask.unsqueeze(0)).type(torch.float32))
                    

                    loss_mapping = loss_mapping + self.config["Training"]["mask_loss"] * mask_loss


                # print(f"Mapping {viewpoint.uid} with {viewpoint.original_image.shape} image, {viewspace_point_tensor.shape} points, {visibility_filter.sum()} visible points, depth: {depth.mean().item():.3f}")
                # Extract from render_pkg
                image = render_pkg["render"]  # torch.Size([3, H, W]), assumed in [0, 1]
                depth = render_pkg["depth"]   # torch.Size([1, H, W]) or [H, W]
                viewpoint_id = viewpoint.uid  # int or str

                if i==0 and self.save_results:
                    rgb_start[cam_idx] = image.detach().cpu()

                # Setup output directory
                output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
                os.makedirs(output_dir, exist_ok=True)
                
                if iters-1 == i and self.save_results and iters>1:
                    #---------- Save RGB ----------
                    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
                    image_np = image_np.clip(0.0, 1.0)
                    # Scale to [0, 255] and convert to uint8
                    image_np = (image_np * 255.0).astype("uint8")

                    image_start_np = rgb_start[cam_idx].permute(1, 2, 0).numpy()
                    image_start_np = image_start_np.clip(0.0, 1.0)
                    # Scale to [0, 255] and convert to uint8
                    image_start_np = (image_start_np * 255.0).astype("uint8")

                    image_gt_np = viewpoint.original_image.cpu().permute(1, 2, 0).numpy()
                    image_gt_np = (image_gt_np * 255.0).astype("uint8")
                    
                    # ---------- Save Depth ----------
                    depth_np = depth.detach().cpu().squeeze().numpy()
                    depth_viz = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                    # Setup figure
                    fig, axes = plt.subplots(1, 3+use_mask_loss, figsize=(20, 6))

                    axes[0].imshow(image_start_np)
                    # axes[0].imshow(opacity_np_dygs, cmap='viridis', vmin=0, vmax=1, origin='upper')
                    axes[0].set_title("Rendered RGB Start")
                    axes[0].axis("off")

                    axes[1].imshow(image_np)
                    axes[1].set_title("Rendered RGB")
                    axes[1].axis("off")

                    axes[2].imshow(image_gt_np, cmap='plasma')
                    axes[2].set_title("GT RGB")
                    axes[2].axis("off")

                    if use_mask_loss:

                        axes[3].imshow(opacity_np_dygs, cmap='viridis', vmin=0, vmax=1, origin='upper')
                        axes[3].set_title("Rendered Dynamic Mask")
                        axes[3].axis("off")


                    plt.tight_layout()

                    if cam_idx == 0:
                        post = '_start'
                    else:
                        post = f'_{max(current_window)}'

                    viewpoint_id = viewpoint.uid
                    save_path = os.path.join(output_dir, f"mapping_{viewpoint_id}_loss{post}.png")
                    plt.savefig(save_path)
                    plt.close()

                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
                    
                if (flow_weights > 0) and ((dynamic_network == 'mlp' and self.gaussians.deform_init) or (dynamic_network == 'offset' and (not static_stage))):
                    flow_start_time = time.time()
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                    if closest_keyframe is not None:
                        flow, flow_back = viewpoint.generate_flow(viewpoint.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda(), ds=self.flow_ds)
                        if self.dynamic_model == 'mlp':
                            time_input = self.gaussians.deform.deform.expand_time(self.viewpoints[closest_keyframe].fid)
                            N = time_input.shape[0]
                            d_value2 = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                            iteration=0, feature=None, 
                                                            motion_mask=self.gaussians.motion_mask, 
                                                            camera_center=self.viewpoints[closest_keyframe].camera_center, 
                                                            time_interval=self.gaussians.time_interval)
                            d_xyz2 = d_value2["d_xyz"]
                        elif self.dynamic_model == 'offset':
                            d_xyz2 = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(self.viewpoints[closest_keyframe].uid))
                            d_rot2 = 0
                            d_value2 = {"d_rotation": d_rot2, "d_scaling": 0, "d_opacity": None, "d_color": None}

                        ## backward flow
                        render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint, viewpoint_camera2=self.viewpoints[closest_keyframe], d_xyz1=dxyz, d_xyz2=d_xyz2, d_rotation1=d_rot, d_scaling1=d_scale, scale_const=None,
                                                  ds=self.flow_ds)
                        coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)

                        
                        # using motion_mask
                        dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()

                        if self.flow_ds > 1:
                            dynamic_mask = F.interpolate(dynamic_mask.permute(2,0,1).unsqueeze(0).float(), scale_factor=1/self.flow_ds, mode='nearest').squeeze(0).permute(1,2,0).bool()
                        if self.dynamic_model == 'mlp':
                            loss_network += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)
                        else:
                            loss_mapping += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)

                        if "flow_grad_loss" in self.config["Training"]:
                            flow_grad_loss = gradient_loss_flow(flow_back, coor1to2_motion, mask=dynamic_mask)
                            flow_grad_weight = self.config["Training"]["flow_grad_loss"]
                            if self.dynamic_model == 'mlp':
                                loss_network += flow_grad_weight*flow_grad_loss
                            else:
                                loss_mapping += flow_grad_weight*flow_grad_loss
                        
                        
                        ## forward flow
                        render_pkg_back = render_flow(pc=self.gaussians, viewpoint_camera1=self.viewpoints[closest_keyframe], viewpoint_camera2=viewpoint, d_xyz1=d_xyz2, d_xyz2=dxyz, d_rotation1=d_value2["d_rotation"], d_scaling1=d_value2["d_scaling"], scale_const=None,
                                                      ds=self.flow_ds)
                        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                        # using motion_mask
                        dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()

                        if self.flow_ds > 1:
                            dynamic_mask = F.interpolate(dynamic_mask.permute(2,0,1).unsqueeze(0).float(), scale_factor=1/self.flow_ds, mode='nearest').squeeze(0).permute(1,2,0).bool()

                        
                        if self.dynamic_model == 'mlp':
                            loss_network += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask) 
                        else:
                            loss_mapping += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)

                        if "flow_grad_loss" in self.config["Training"]:
                            flow_grad_loss = gradient_loss_flow(flow, coor2to1_motion, mask=dynamic_mask)
                            if self.dynamic_model == 'mlp':
                                loss_network += flow_grad_weight*flow_grad_loss
                            else:
                                loss_mapping += flow_grad_weight*flow_grad_loss
                        

                        if i==(0+self.sd*iters//2) and self.save_results:
                            motion_start[cam_idx] = coor2to1_motion.clone().detach()
                        if i==iters-1 and iters>1 and self.save_results:
                            if cam_idx == 0:
                                viewpoint.save_flow(motion_start[cam_idx], coor2to1_motion, gt=flow, save_path=os.path.join(output_dir, f"mapping_{viewpoint.uid}_flow_start.png"))
                            else:
                                viewpoint.save_flow(motion_start[cam_idx], coor2to1_motion, gt=flow, save_path=os.path.join(output_dir, f"mapping_{viewpoint.uid}_flow_{max(current_window)}.png"))
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render), dynamic=dynamic
                    )
                        
                else:
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render),
                    )
                if dynamic_network and self.gaussians.deform_init and (self.dynamic_model == 'mlp'):
                    loss_network += 1e-3 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid, delta_t=delta*self.gaussians.time_interval, t_samp_num=4)
                    loss_network += 1e-3 * self.gaussians.deform.deform.elastic_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

                
            random_views = torch.randperm(len(random_viewpoint_stack))[:2]
            random_uids = [random_viewpoint_stack[rv].uid for rv in random_views]
            
            for cam_idx in random_views:
                viewpoint = random_viewpoint_stack[cam_idx]
                
                if self.dynamic_model == 'mlp':
                    if dynamic_network and self.gaussians.deform_init:
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                        N = time_input.shape[0]
                        d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                        iteration=0, feature=None, 
                                                        motion_mask=self.gaussians.motion_mask, 
                                                        camera_center=viewpoint.camera_center, 
                                                        time_interval=self.gaussians.time_interval)
                        dxyz = d_values['d_xyz']
                        d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                        d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                    elif dynamic_render and self.gaussians.deform_init:
                        with torch.no_grad():
                            time_input = self.gaussians.deform.deform.expand_time(viewpoint.fid)
                            N = time_input.shape[0]
                            ast_noise = torch.randn(1, 1, device=time_input.device).expand(N, -1) * self.gaussians.time_interval * self.gaussians.smooth_term(self.iteration_count) 
                            d_values = self.gaussians.deform.step(self.gaussians.get_xyz.detach(), time_input+ast_noise, 
                                                            iteration=0, feature=None, 
                                                            motion_mask=self.gaussians.motion_mask, 
                                                            camera_center=viewpoint.camera_center, 
                                                            time_interval=self.gaussians.time_interval)
                            dxyz = d_values['d_xyz'].detach()
                            d_rot, d_scale = d_values['d_rotation'].detach(), d_values['d_scaling'].detach()
                            if d_values['d_opacity'] is not None: 
                                d_opac=d_values['d_opacity'].detach()
                            else:
                                d_opac =None
                            if d_values["d_color"] is not None: 
                                d_color = d_values["d_color"].detach()
                            else:
                                d_color=None
                    else:
                        dxyz = 0
                        d_rot, d_scale, d_opac, d_color = None, 0, None, None
                    dygs_scaling += d_scale
                elif self.dynamic_model == 'offset':
                    dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid))
                    d_rot = 0
                    d_scale, d_opac, d_color = 0, None, None
                
                
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color,
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
                if (flow_weights > 0) and ((not static_stage) and ((dynamic_network and self.gaussians.deform_init) or self.dynamic_model == 'offset')):
                    if dynamic or True:
                        closest_keyframe = self.find_closest_keyframe(viewpoint.uid)
                        if closest_keyframe is not None:
                            
                            flow, flow_back = viewpoint.generate_flow(viewpoint.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda(), ds=self.flow_ds)
                            
                            if self.dynamic_model == 'mlp':
                                time_input = self.gaussians.deform.deform.expand_time(self.viewpoints[closest_keyframe].fid)
                                N = time_input.shape[0]
                                d_value2 = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, 
                                                                iteration=0, feature=None, 
                                                                motion_mask=self.gaussians.motion_mask, 
                                                                camera_center=self.viewpoints[closest_keyframe].camera_center, 
                                                                time_interval=self.gaussians.time_interval)
                                d_xyz2 = d_value2["d_xyz"]
                            elif self.dynamic_model == 'offset':
                                d_xyz2 = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(self.viewpoints[closest_keyframe].uid))
                                d_rot2 = 0
                                d_value2 = {"d_rotation": d_rot2, "d_scaling": 0, "d_opacity": None, "d_color": None}
                            ## backward flow
                            render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint, viewpoint_camera2=self.viewpoints[closest_keyframe], d_xyz1=dxyz, d_xyz2=d_xyz2, d_rotation1=d_rot, d_scaling1=d_scale, scale_const=None,
                                                      ds=self.flow_ds)
                            coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)
                            # using motion_mask
                            dynamic_mask = (~viewpoint.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                            if self.flow_ds > 1:
                                dynamic_mask = F.interpolate(dynamic_mask.permute(2,0,1).unsqueeze(0).float(), scale_factor=1/self.flow_ds, mode='nearest').squeeze(0).permute(1,2,0).bool()
                            if self.dynamic_model == 'mlp':
                                loss_network += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)
                            else:
                                loss_mapping += flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)

                            ## forward flow
                            render_pkg_back = render_flow(pc=self.gaussians, viewpoint_camera1=self.viewpoints[closest_keyframe], viewpoint_camera2=viewpoint, d_xyz1=d_xyz2, d_xyz2=dxyz, d_rotation1=d_value2["d_rotation"], d_scaling1=d_value2["d_scaling"], scale_const=None,
                                                          ds=self.flow_ds)
                            coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                            # using motion_mask
                            dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                            if self.flow_ds > 1:
                                dynamic_mask = F.interpolate(dynamic_mask.permute(2,0,1).unsqueeze(0).float(), scale_factor=1/self.flow_ds, mode='nearest').squeeze(0).permute(1,2,0).bool()
                            if self.dynamic_model == 'mlp':
                                loss_network += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)
                            else:
                                loss_mapping += flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)


                        loss_mapping += get_loss_mapping(
                            self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render), dynamic=dynamic,
                        )
                    
                else:
                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, rm_dynamic=not (dynamic_network or dynamic_render), mask=mask,
                    )
                
                if dynamic_network and self.gaussians.deform_init and self.dynamic_model == 'mlp':
                    loss_network += 1e-4 * self.gaussians.deform.deform.elastic_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                    loss_network += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint.fid, delta_t=5*self.gaussians.time_interval)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                
            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            
            loss_mapping.backward(retain_graph=True)
            
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized 4DGS-SLAM", tag="Backend")

                    
                    
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )
                
                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset and (i>(iters//2) or (self.dynamic_model=='offset'))
                ) 

                if rm_initdy:
                    update_gaussian = (iters - i-10 == 0)  # 
                

                # TODO
                if update_gaussian:
                    print(f'update gaussians in iteration {self.iteration_count}')
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ) and ((i>(iters//2)) or (self.dynamic_model=='offset')):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                
                

                self.keyframe_optimizers.step()
                self.keyframe_scheduler.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = self.viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
                
                if self.dynamic_model == 'mlp':
                    if (dynamic_network) and self.gaussians.deform_init:
                        loss_network.backward()
                        self.gaussians.deform.optimizer.step()
                        self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
                        self.keyframe_optimizers.zero_grad(set_to_none=True)
                    
                    if i>(iters//2):
                        self.gaussians.optimizer.step()
                        self.gaussians.optimizer.zero_grad(set_to_none=True)
                        self.gaussians.update_learning_rate(self.iteration_count)
                    else:
                        self.gaussians.optimizer.zero_grad(set_to_none=True)

                else:
                
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
            


        return gaussian_split
    
    
    def color_refinement(self, dynamic_network=False):
        Log("Starting color refinement")

            
        flow_weights = self.config["Training"]["flow_loss"]

        iteration_total = 1500
        for iteration in tqdm(range(1, iteration_total + 1)):
            loss = 0
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_list = []
            for _ in range(min(10, len(viewpoint_idx_stack))):
                scaling = 0
                viewpoint_cam_idx = viewpoint_idx_stack.pop(
                    random.randint(0, len(viewpoint_idx_stack) - 1)
                )
                viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
                viewpoint_list.append(viewpoint_cam_idx)
                
                if self.dynamic_model == 'mlp':
                    if dynamic_network and self.gaussians.deform_init:
                        time_input = self.gaussians.deform.deform.expand_time(viewpoint_cam.fid)
                        N = time_input.shape[0]
                        d_values = self.gaussians.deform.step(self.gaussians.get_dygs_xyz.detach(), time_input, #+ast_noise, 
                                                        iteration=0, feature=None, 
                                                        motion_mask=self.gaussians.motion_mask, 
                                                        camera_center=viewpoint_cam.camera_center, 
                                                        time_interval=self.gaussians.time_interval)
                        dxyz = d_values['d_xyz']
                        d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
                        d_opac, d_color=d_values['d_opacity'], d_values["d_color"]
                    else:
                        dxyz, d_rot, d_scale, d_opac, d_color = 0, 0, 0, None, None
                elif self.dynamic_model == 'offset':
                    dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint_cam.uid))
                    d_rot = 0
                    d_scale, d_opac, d_color = 0, None, None
                    
                render_pkg = render(
                        viewpoint_cam, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, do=d_opac, dc=d_color,
                )
                
                image, depth, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["depth"], 
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )


                if self.dynamic_model == 'mlp':
                    image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b

                gt_image = viewpoint_cam.original_image.cuda()
                gt_depth = torch.from_numpy(viewpoint_cam.depth).to(
                    dtype=torch.float32, device=image.device
                )[None]
                depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
                if dynamic_network:
                    Ll1 = l1_loss(image, gt_image)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
                    if self.dynamic_model == 'mlp':
                        loss += 1e-4 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=5*self.gaussians.time_interval, t_samp_num=8)  #1e-1 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=20*self.gaussians.time_interval)

                else:
                    Ll1 = l1_loss(image, gt_image, mask=viewpoint_cam.motion_mask)
                    loss += (1.0 - self.opt_params.lambda_dssim) * (
                        Ll1
                    ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image, mask=viewpoint_cam.motion_mask))
                    depth_pixel_mask = viewpoint_cam.motion_mask.view(*gt_depth.shape) * depth_pixel_mask
                    
                l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
                loss += 0.1*l1_depth.mean()  

                if 'mask_loss' in self.config['Training'] and self.config["Training"]["mask_loss"] > 0:
                    use_mask_loss = True
                    render_pkg_dygs = render(
                        viewpoint_cam, 
                        self.gaussians, 
                        self.pipeline_params, 
                        torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), 
                        dynamic=False, 
                        dx=dxyz, 
                        ds=d_scale, 
                        dr=d_rot, 
                        do=d_opac, 
                        dc=d_color,
                        mask=(self.gaussians.dygs==True),
                        render_mask=True,
                    )

                    opacity_np_dygs = render_pkg_dygs["render"][0, :, :].detach().cpu().squeeze(0).numpy()
                    mask_loss = F.binary_cross_entropy(render_pkg_dygs["render"][0:1, :, :], 
                                                       (~viewpoint_cam.motion_mask.unsqueeze(0)).type(torch.float32),
                                                       weight=viewpoint_cam.motion_mask.unsqueeze(0).type(torch.float32)+1e-6)
                    

                    loss += 0.1 * self.config["Training"]["mask_loss"] * mask_loss
                

                if self.dynamic_model == 'offset' and flow_weights > 0:
                    closest_keyframe = self.find_closest_keyframe(viewpoint_cam.uid)
                    if closest_keyframe is not None:
                        flow, flow_back = viewpoint_cam.generate_flow(viewpoint_cam.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda(), ds=self.flow_ds)
                        
                        d_xyz2 = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(self.viewpoints[closest_keyframe].uid))
                        d_rot2 = 0
                        d_value2 = {"d_rotation": d_rot2, "d_scaling": 0, "d_opacity": None, "d_color": None}

                        ## backward flow
                        render_pkg2 = render_flow(pc=self.gaussians, viewpoint_camera1=viewpoint_cam, viewpoint_camera2=self.viewpoints[closest_keyframe], d_xyz1=dxyz, d_xyz2=d_xyz2, d_rotation1=d_rot, d_scaling1=d_scale, scale_const=None,
                                                  ds=self.flow_ds)
                        coor1to2_motion = render_pkg2["render"][:2].permute(1, 2, 0)

                        
                        # using motion_mask
                        dynamic_mask = (~viewpoint_cam.motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()
                        loss += 0.5 * flow_weights*l1_loss(flow_back*dynamic_mask, coor1to2_motion*dynamic_mask)

                        if "flow_grad_loss" in self.config["Training"]:
                            flow_grad_loss = gradient_loss_flow(flow_back, coor1to2_motion, mask=dynamic_mask)
                            flow_grad_weight = self.config["Training"]["flow_grad_loss"]
                            loss += 0.1 * flow_grad_weight*flow_grad_loss
                        
                        
                        ## forward flow
                        render_pkg_back = render_flow(pc=self.gaussians, viewpoint_camera1=self.viewpoints[closest_keyframe], viewpoint_camera2=viewpoint_cam, d_xyz1=d_xyz2, d_xyz2=dxyz, d_rotation1=d_value2["d_rotation"], d_scaling1=d_value2["d_scaling"], scale_const=None,
                                                      ds=self.flow_ds)
                        coor2to1_motion = render_pkg_back["render"][:2].permute(1, 2, 0)
                        # using motion_mask
                        dynamic_mask = (~self.viewpoints[closest_keyframe].motion_mask).unsqueeze(0).permute(1, 2, 0).repeat(1,1,2).detach()

                        
                        loss += 0.5 * flow_weights*l1_loss(flow*dynamic_mask, coor2to1_motion*dynamic_mask)

                        if "flow_grad_loss" in self.config["Training"]:
                            flow_grad_loss = gradient_loss_flow(flow, coor2to1_motion, mask=dynamic_mask)
                            loss += 0.1 * flow_grad_weight*flow_grad_loss


                
            scaling = self.gaussians.get_scaling
            # print('scaling.shape:', scaling.shape)
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss += 10 * isotropic_loss.mean()

            # print('viewpoint_list:', viewpoint_list)


            

            
            #if dynamic_network:
            #    loss += 1e-2 * self.gaussians.deform.deform.arap_loss(t=viewpoint_cam.fid, delta_t=20*self.gaussians.time_interval)
            loss.backward()
            
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
                if dynamic_network and self.gaussians.deform_init:
                    self.gaussians.deform.optimizer.step()
                    self.gaussians.deform.optimizer.zero_grad(set_to_none=True)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"
        if self.dynamic_model == 'mlp':
            self.gaussians.deform.deform.reg_loss = 0.  # Prevent deepcopy errors

        cloned_gaussians = clone_obj(self.gaussians)
        msg = [tag, cloned_gaussians, self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def push_to_frontend_final(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend_final"
        if self.dynamic_model == 'mlp':
            self.gaussians.deform.deform.reg_loss = 0.  # Prevent deepcopy errors
        
        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes, np.mean(self.mapping_time)]
        self.frontend_queue.put(msg)
    
    

    def rgbd2pcd(self, depth, w2c, k, project_to_cam_w_scale=None):
        """
        depth: [H,W] z-depth in camera coords (meters)
        w2c:   [4,4] world->camera
        k:     [3,3] intrinsics
        self.def_pix: [N,3] homogeneous pixel coords (u, v, 1) at pixel centers
        """
        device = depth.device
        dtype  = torch.float32

        invk = torch.linalg.inv(torch.as_tensor(k,   device=device, dtype=dtype))
        c2w  = torch.linalg.inv(torch.as_tensor(w2c, device=device, dtype=dtype))

        # Unnormalized camera rays; rays[...,2] == 1
        def_pix = self.def_pix.to(device=device, dtype=dtype)          # [N,3] = (u, v, 1)
        rays    = (invk @ def_pix.T).T                                 # [N,3]

        z = depth.reshape(-1).to(dtype)                                # [N]
        pts_cam = rays * z[:, None]                                    # [N,3]; Z_cam == z

        if project_to_cam_w_scale is not None:
            s = project_to_cam_w_scale / pts_cam[:, 2].clamp_min(1e-12)
            pts_cam = pts_cam * s[:, None]

        ones  = torch.ones((pts_cam.shape[0], 1), device=device, dtype=dtype)
        pts4  = torch.cat([pts_cam, ones], dim=-1)                     # [N,4]
        pts_w = (c2w @ pts4.T).T[:, :3]                                # [N,3]
        return pts_w, z > 0.01

    
    def get_uv_coordinates(self, uvz, H, W):
        u = uvz[:, 0]  # x in [-1, 1]
        v = uvz[:, 1]  # y in [-1, 1]

        # Validity mask for finite coords within [-1, 1]
        finite   = torch.isfinite(u) & torch.isfinite(v)
        # in_range = (u >= -1) & (u <= 1) & (v >= -1) & (v <= 1)
        # mask_valid = finite & in_range
        mask_valid = finite

        idx_valid = torch.nonzero(mask_valid, as_tuple=False).squeeze(1)
        if idx_valid.numel() == 0:
            return 

        u_valid = u[idx_valid]
        v_valid = v[idx_valid]

        # Map NDC [-1,1] -> pixel indices [0..W-1]/[0..H-1], nearest neighbor
        # x = ((u+1)/2)*(W-1), y = ((v+1)/2)*(H-1)
        x = (((u_valid + 1) * 0.5) * (W - 1)).round().long()
        y = (((v_valid + 1) * 0.5) * (H - 1)).round().long()

        # (Paranoia clamp to be safe against tiny numeric drift)
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
        
        lin_idx   = y * W + x                    # [K]

        return lin_idx, idx_valid
    
    def insertion_mask(self, viewpoint, flow, closest_viewpoint):
        image = viewpoint.original_image
        closest_image = closest_viewpoint.original_image
        motion_mask = viewpoint.motion_mask
        
        
        alpha = 0.5
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).contiguous()  # CHW -> HWC

        if closest_image.ndim == 3 and closest_image.shape[0] == 3:
            closest_image = closest_image.permute(1, 2, 0).contiguous()  # CHW -> HWC

        H, W, _ = image.shape
        device = image.device
        dtype = image.dtype
        assert motion_mask.shape == (H, W) and flow.shape == (H, W, 2)

        # --- source pixels to warp ---
        src_mask = (motion_mask == 0)

        if self.depth_check:
            dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid))
            d_rot = 0
            d_scale, d_opac, d_color = 0, None, None

                
            render_pkg = render(
                viewpoint, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                dynamic=False, 
                dx=dxyz, 
                ds=d_scale, 
                dr=d_rot, 
                do=d_opac, 
                dc=d_color,
            )
            
            rendered_depth = render_pkg["depth"].squeeze()  # [H,W]
            depth_map = torch.from_numpy(viewpoint.depth).to(device=device, dtype=dtype)
            # print('render_depth.shape:', rendered_depth.shape, 'depth_map.shape:', depth_map.shape)
            depth_diff = rendered_depth - depth_map
            depth_threshold = 0.05  # meters
            depth_mask = (depth_diff.abs() > depth_threshold)
            src_mask = src_mask & depth_mask

        # --- absolute destination coords ---
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )

        flow_px = torch.zeros_like(flow, dtype=dtype, device=device)
        flow_px[..., 0] = flow[..., 0] * (W - 1) / 2.0  # NDC to pixels
        flow_px[..., 1] = flow[..., 1] * (H - 1) / 2.0
        xw = xx + flow_px[..., 0]
        yw = yy + flow_px[..., 1]

        # --- out-of-bounds ---
        in_bounds = (xw >= 0) & (xw <= (W - 1)) & (yw >= 0) & (yw <= (H - 1))


        if self.insertion_type == 'new':
            # Additional check that the warped pixel lands on a motion region

            x_norm = (xw / (W - 1)) * 2 - 1
            y_norm = (yw / (H - 1)) * 2 - 1

            # Stack into a grid of shape [1, H, W, 2]
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)

            # Prepare motion_mask for sampling: [1, 1, H, W]
            enlarged_closest_motion_mask, _ = self.enlarge_false(closest_viewpoint.motion_mask, pixels=5, outside_is_false=True)
        
            mask = enlarged_closest_motion_mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

            # Bilinear sampling
            sampled_mask = F.grid_sample(mask, grid, align_corners=True)

            # Back to [H, W]
            sampled_mask = sampled_mask.squeeze().to(torch.bool)

            out_mask = src_mask & ((~in_bounds) | sampled_mask)
        else:
            out_mask = src_mask & (~in_bounds)

        # --- discrete warped mask ---
        xi = torch.round(xw).to(torch.int64)
        yi = torch.round(yw).to(torch.int64)
        valid = src_mask & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        warped_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
        warped_mask[yi[valid], xi[valid]] = True

        # --- helper to overlay mask on image ---
        def overlay_mask(img, mask, color):
            blended = img.clone()
            overlay = torch.zeros_like(img)
            overlay[..., 0] = color[0]
            overlay[..., 1] = color[1]
            overlay[..., 2] = color[2]
            mask3 = mask.unsqueeze(-1).expand_as(img)
            blended[mask3] = (1 - alpha) * img[mask3] + alpha * overlay[mask3]
            return blended

        # --- apply three colors ---
        img_src    = overlay_mask(image, src_mask,    (1.0, 1.0, 0.0))  # yellow
        img_warped = overlay_mask(closest_image, warped_mask, (0.0, 1.0, 1.0))  # cyan
        img_out    = overlay_mask(image, out_mask,    (1.0, 0.0, 1.0))  # magenta

        # --- visualize side by side ---
        if self.save_results:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, img, title in zip(
                axes,
                [img_src, img_warped, img_out],
                ["motion_mask == 0 (source)", "warped (in-bounds)", "warped out-of-image"]
            ):
                ax.imshow(img.cpu().numpy())
                ax.set_title(title)
                ax.axis("off")

            output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
            os.makedirs(output_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'mapping_{viewpoint.uid}_insertion.png'), dpi=150)
            plt.close()

            plt.imshow(closest_viewpoint.motion_mask.cpu().numpy(), cmap='gray')
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f'mapping_{viewpoint.uid}_insertion_prev.png'), dpi=150)
            plt.close()

            plt.imshow(out_mask.cpu().numpy(), cmap='gray')
            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f'mapping_{viewpoint.uid}_insertion_out.png'), dpi=150)
            plt.close()

        return out_mask
    
    def smooth_deformation_knn_radius(
        self,
        points,        # [N, 3]
        deform,        # [N, 3]
        k = 16,                 # max neighbors to consider
        r_max = None,  # maximum neighbor distance (same units as points). None = no cutoff
        weighting = "inv_dist", # "inv_dist" | "gaussian" | "uniform"
        sigma = 0.05,         # for gaussian weighting
        include_self = True,   # keep the point itself (distance 0) as a neighbor
        eps = 1e-8,
    ):
        """
        Smooth a per-point deformation field using KNN with an optional radius cutoff.

        The weight for neighbor j of point i depends on ||x_i - x_j|| (and optionally on r_max).
        If r_max is given, neighbors farther than r_max are ignored (weight = 0).
        If only the self-neighbor remains, the deformation is unchanged for that point.

        Returns:
            deform_smooth: [N, 3]
        """
        assert points.ndim == 2 and points.shape[1] == 3
        assert deform.shape == points.shape

        device = points.device
        dtype = points.dtype
        N = points.shape[0]

        P = points[None, ...]  # [1, N, 3]
        Q = points[None, ...]  # [1, N, 3]

        K_eff = k if include_self else (k + 1)

        knn = knn_points(P, Q, K=K_eff, return_nn=False)
        dists2 = knn.dists.squeeze(0)  
        idx    = knn.idx.squeeze(0)    

        if not include_self:
            dists2 = dists2[:, 1:]
            idx    = idx[:, 1:]

        deform_neighbors = deform[idx, :]  

        if weighting == "uniform":
            w = torch.ones_like(dists2, dtype=dtype, device=device)
        elif weighting == "inv_dist":
            w = 1.0 / (torch.sqrt(dists2 + eps) + eps)
        elif weighting == "gaussian":
            w = torch.exp(-dists2 / (2.0 * (sigma ** 2) + eps))
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        if r_max is not None:
            r2 = (r_max ** 2)
            mask_in = (dists2 <= r2)  
            w = w * mask_in.to(dtype)

        if include_self:
            w[:, 0] = torch.clamp(w[:, 0], min=eps)

        w_sum = w.sum(dim=-1, keepdim=True) + eps   
        w_norm = w / w_sum                           

        
        deform_smooth = (w_norm[..., None] * deform_neighbors).sum(dim=1)  
        return deform_smooth



    def enlarge_false(self, mask: torch.Tensor, pixels: int = 2, outside_is_false: bool = True):
        """
        Enlarge the False regions of a boolean mask by `pixels` and
        also return a mask marking the newly enlarged pixels.

        Args:
            mask: Bool tensor [H, W]
            pixels: Number of pixels to enlarge the False regions
            outside_is_false: If True, treat out-of-bounds as False

        Returns:
            enlarged_mask: Bool tensor [H, W] after enlarging False regions
            enlarged_region: Bool tensor [H, W], True where newly turned False
        """
        assert mask.dtype == torch.bool and mask.ndim == 2

        k = 2 * pixels + 1
        inv = (~mask).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if outside_is_false:
            inv = F.pad(inv, (pixels, pixels, pixels, pixels), value=1.0)
            dilated = F.max_pool2d(inv, kernel_size=k, stride=1, padding=0)
        else:
            dilated = F.max_pool2d(inv, kernel_size=k, stride=1, padding=pixels)

        enlarged_mask = ~(dilated.squeeze(0).squeeze(0) > 0.0)

        # Newly enlarged region = previously True but now turned False
        enlarged_region = mask & (~enlarged_mask)

        return enlarged_mask, enlarged_region


    def apply_flow_offset(self, flow, viewpoint, closest_viewpoint, depth_consistent=True):
        """
        Apply flow offset to the gaussians based on the flow between the current viewpoint and the closest keyframe.
        """
        carnonical_xyz = self.gaussians.get_xyz.clone()[self.gaussians.dygs]
        xyz_at_t1 = carnonical_xyz.detach()  # Detach coordinates of Gaussians here

        d_xyz1 = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(closest_viewpoint.uid))
        xyz_at_t1 = xyz_at_t1 + d_xyz1

        gaussians_homogeneous_coor_t1 = torch.cat([xyz_at_t1, torch.ones_like(xyz_at_t1[..., :1])], dim=-1)
        full_proj_transform = closest_viewpoint.full_proj_transform
        gaussians_uvz_coor_at_cam1 = gaussians_homogeneous_coor_t1 @ full_proj_transform
        z = gaussians_uvz_coor_at_cam1[..., -1]
        gaussians_uvz_coor_at_cam1 = gaussians_uvz_coor_at_cam1[..., :3] / (gaussians_uvz_coor_at_cam1[..., -1:] + 1e-7)

        H, W, _ = flow.shape
        device = flow.device

        uvz = gaussians_uvz_coor_at_cam1.to(device)

        

        # Gather
        flow_flat = flow.reshape(-1, 2)          # [H*W, 2]
        lin_idx, idx_valid = self.get_uv_coordinates(uvz, H, W)  # [K]


        enlarged_closest_motion_mask, fillin_closest_mask = self.enlarge_false(closest_viewpoint.motion_mask, pixels=5, outside_is_false=True)
        _, fillin_mask = self.enlarge_false(torch.from_numpy(viewpoint.depth).to(device)==0, pixels=5, outside_is_false=True)


        if depth_consistent:
            closest_depth_map = closest_viewpoint.depth.copy()
            closest_depth_map_filled = self.fill_depth_holes_with_motion_nn_np(closest_depth_map, \
                            ((~closest_viewpoint.motion_mask.cpu().numpy())*(closest_depth_map==0)) | \
                                fillin_closest_mask.cpu().numpy(), \
                                ~closest_viewpoint.motion_mask.cpu().numpy()
                                # closest_viewpoint.depth>0
                                )
            uvz_depth = torch.gather(torch.from_numpy(closest_depth_map_filled.reshape(-1)).to(device), 0, lin_idx)  # [K]
            diff = torch.abs(uvz_depth - z[idx_valid])
            depth_mask = (diff < 0.3).squeeze(-1)  # [K]
            idx_valid = idx_valid[depth_mask]
            lin_idx = lin_idx[depth_mask]
        
        lin_idx = lin_idx.view(-1)
        idx_2d = lin_idx.unsqueeze(1).expand(-1, 2)  # [K,2]


        flow_at_uv = torch.gather(flow_flat, 0, idx_2d) # [K,2]
        gaussians_uvz_coor_at_cam2 = gaussians_uvz_coor_at_cam1.clone()
        gaussians_uvz_coor_at_cam2[idx_valid] += torch.cat([flow_at_uv, torch.zeros_like(flow_at_uv[:, :1], device=flow_at_uv.device)], dim=-1)


        
        lin_idx2, idx2_valid = self.get_uv_coordinates(gaussians_uvz_coor_at_cam2, H, W)  # [K]
        lin_idx2 = lin_idx2[idx_valid].view(-1)
        idx2_valid = idx2_valid[idx_valid]


        depth_map = viewpoint.depth.copy()
        depth_map_filled = self.fill_depth_holes_with_motion_nn_np(depth_map, 
                    ((~viewpoint.motion_mask.cpu().numpy())*(depth_map==0)) | \
                        fillin_mask.cpu().numpy(), \
                    ~viewpoint.motion_mask.cpu().numpy()
                        # viewpoint.depth>0
                    )


        

        deformed_gaussians_xyz, valid_mask = self.rgbd2pcd(depth=torch.from_numpy(depth_map_filled).to(device), 
                                               w2c=viewpoint.w2c,
                                               k=viewpoint.intrinsic,)
        
        valid_mask = valid_mask.to(device)
                                               
        sampled = torch.gather(valid_mask, 0, lin_idx2)
        sampled_valid_idx = torch.nonzero(sampled, as_tuple=False).squeeze(1)
        idx2_valid = idx2_valid[sampled_valid_idx]
        deformed_gaussians_xyz = deformed_gaussians_xyz[lin_idx2][sampled_valid_idx]

        deformation = deformed_gaussians_xyz - xyz_at_t1[idx2_valid]

        max_motion_mask = torch.norm(deformation, dim=-1) < 1.0
        if max_motion_mask.shape[0] > 0:
            idx2_valid = idx2_valid[max_motion_mask]
            deformation = deformation[max_motion_mask]


            if self.use_knn:
                deformation = self.smooth_deformation_knn_radius(
                    points=xyz_at_t1[idx2_valid],
                    deform=deformation,
                    k=self.knn_n,
                    r_max=self.knn_r,
                    weighting="gaussian",
                    sigma=0.05,
                    include_self=True,
                    eps=1e-8,
                )
            
            self.gaussians.update_delta_xyz(viewpoint.uid, closest_viewpoint.uid, deformation, idx2_valid, 
                                            max_motion=0.3)

        else:
            self.gaussians.update_delta_xyz(viewpoint.uid, closest_viewpoint.uid, None, idx2_valid, 
                                            max_motion=0.3)


        # --------------------- Visualization ---------------------

        if self.save_results:

            output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            depth_np = depth_map
            depth_viz = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)

            depth_filled_np = depth_map_filled
            depth_filled_viz = (depth_filled_np - depth_filled_np.min()) / (depth_filled_np.max() - depth_filled_np.min() + 1e-8)
            # Setup figure
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            axes[0].imshow(depth_viz, cmap='plasma')
            axes[0].set_title("Raw Depth")
            axes[0].axis("off")

            axes[1].imshow(depth_filled_viz, cmap='plasma')
            axes[1].set_title("Filled Depth")
            axes[1].axis("off")

            plt.tight_layout()

            viewpoint_id = viewpoint.uid
            save_path = os.path.join(output_dir, f"mapping_{viewpoint_id}_filldepth.png")
            plt.savefig(save_path)
            plt.close()

            os.makedirs(output_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))

            x1 = (((gaussians_uvz_coor_at_cam1[:, 0] + 1) * 0.5) * (W - 1)).round().long()
            y1 = (((gaussians_uvz_coor_at_cam1[:, 1] + 1) * 0.5) * (H - 1)).round().long()

            # (Paranoia clamp to be safe against tiny numeric drift)
            x1 = x1.clamp(0, W - 1)
            y1 = y1.clamp(0, H - 1)



            x2 = (((gaussians_uvz_coor_at_cam2[:, 0] + 1) * 0.5) * (W - 1)).round().long()
            y2 = (((gaussians_uvz_coor_at_cam2[:, 1] + 1) * 0.5) * (H - 1)).round().long()

            # (Paranoia clamp to be safe against tiny numeric drift)
            x2 = x2.clamp(0, W - 1)
            y2 = y2.clamp(0, H - 1)

            mask_viz = torch.zeros(gaussians_uvz_coor_at_cam2.shape[0], dtype=torch.bool, device=device)
            mask_viz[idx2_valid] = True

            axes[0].imshow(closest_viewpoint.original_image.cpu().numpy().transpose(1, 2, 0))
            axes[0].scatter(x1[mask_viz].cpu().numpy(), y1[mask_viz].cpu().numpy(), c='g', s=10, marker='o', alpha=0.2)
            axes[0].scatter(x1[~mask_viz].cpu().numpy(), y1[~mask_viz].cpu().numpy(), c='r', s=10, marker='o', alpha=0.2)
            axes[0].axis("off")

            axes[1].imshow(viewpoint.original_image.cpu().numpy().transpose(1, 2, 0))
            axes[1].scatter(x2[mask_viz].cpu().numpy(), y2[mask_viz].cpu().numpy(), c='g', s=10, marker='o', alpha=0.2)
            axes[1].scatter(x2[~mask_viz].cpu().numpy(), y2[~mask_viz].cpu().numpy(), c='r', s=10, marker='o', alpha=0.2)
            axes[1].axis("off")


            plt.tight_layout()

            plt.axis("off")
            plt.savefig(os.path.join(output_dir, f"mapping_{viewpoint.uid}_debug.png"))
            plt.close()


    
    ## backend thread
    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else: # get info from frondend
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "sync_backend_final":
                    self.push_to_frontend_final()
                elif data[0] == "color_refinement":
                    self.color_refinement(dynamic_network=self.dynamic_model)
                    self.push_to_frontend_final()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system",tag="Backend")
                    self.reset()

                    self.gaussians.kf_list.append(viewpoint.uid)

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    if self.dynamic_model == 'mlp' and self.dystart==0:
                        self.initialize_network(cur_frame_idx, viewpoint)
                    
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    add_new_gaussian = data[5]
                    dynamic_render = data[6]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    flow_back = None
                    closest_keyframe = self.find_closest_keyframe(viewpoint.uid)


                    
                    for cam_idx in range(len(current_window)):
                        if self.viewpoints[current_window[cam_idx]].uid not in self.gaussians.kf_list:
                            self.gaussians.kf_list.append(self.viewpoints[current_window[cam_idx]].uid)
                            self.gaussians.kf_list.sort()
                        

                    if self.dynamic_model == 'offset':

                        flow, flow_back = viewpoint.generate_flow(viewpoint.original_image.cuda(), self.viewpoints[closest_keyframe].original_image.cuda(), 
                                                                      ds=self.flow_ds, return_full=True)
                        
                        if (not self.flow_offset) or (closest_keyframe is None) or (self.dystart >= cur_frame_idx) or (self.gaussians.dygs.sum()==0):
                            self.gaussians.update_delta_xyz(cur_frame_idx, self.viewpoints[closest_keyframe].uid)
                        else:
                            
                            
                            self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])

                            dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid)-1)
                            d_rot, d_scale, d_opac, d_color = 0, 0, None, None

                            render_pkg_dygs = render(
                                viewpoint, 
                                self.gaussians, 
                                self.pipeline_params, 
                                torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"), 
                                dynamic=False, 
                                dx=dxyz, 
                                ds=d_scale, 
                                dr=d_rot, 
                                do=d_opac, 
                                dc=d_color,
                                mask=(self.gaussians.dygs==True),
                                render_mask=True,
                            )

                            render_alpha = render_pkg_dygs["render"][0, :, :].detach().cpu().numpy()
                            depth_map[render_alpha>0.5] = 0.0

                            if self.save_results:
                                dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid)-1)
                                d_rot, d_scale, d_opac, d_color = 0, 0, None, None


                                
                                render_pkg = render(
                                    viewpoint, 
                                    self.gaussians, 
                                    self.pipeline_params, 
                                    self.background, 
                                    dynamic=False, 
                                    dx=dxyz, 
                                    ds=d_scale, 
                                    dr=d_rot, 
                                    do=d_opac, 
                                    dc=d_color,
                                    uid=self.viewpoints[closest_keyframe].uid,
                                )
                                image = render_pkg["render"]  # torch.Size([3, H, W]), assumed in [0, 1]

                                # Setup output directory
                                output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
                                os.makedirs(output_dir, exist_ok=True)
                                #---------- Save RGB ----------
                                image_start_np = image.detach().cpu().permute(1, 2, 0).numpy()
                                image_start_np = image_start_np.clip(0.0, 1.0)
                                # Scale to [0, 255] and convert to uint8
                                image_start_np = (image_start_np * 255.0).astype("uint8")

                                dxyz = self.gaussians.get_delta_xyz(self.gaussians.kf_list.index(viewpoint.uid))
                                d_rot, d_scale, d_opac, d_color = 0, 0, None, None

                                
                                render_pkg = render(
                                    viewpoint, 
                                    self.gaussians, 
                                    self.pipeline_params, 
                                    self.background, 
                                    dynamic=False, 
                                    dx=dxyz, 
                                    ds=d_scale, 
                                    dr=d_rot, 
                                    do=d_opac, 
                                    dc=d_color,
                                    uid=self.viewpoints[closest_keyframe].uid,
                                )
                                image = render_pkg["render"]  # torch.Size([3, H, W]), assumed in [0, 1]

                                # Setup output directory
                                output_dir = os.path.join(self.config["Results"]["save_dir"], "mapping")
                                os.makedirs(output_dir, exist_ok=True)
                                
                                #---------- Save RGB ----------
                                image_after_np = image.detach().cpu().permute(1, 2, 0).numpy()
                                image_after_np = image_after_np.clip(0.0, 1.0)
                                # Scale to [0, 255] and convert to uint8
                                image_after_np = (image_after_np * 255.0).astype("uint8")

                                # ---------- Save Depth ----------
                                
                                fig, axes = plt.subplots(1, 2, figsize=(20, 6))

                                axes[0].imshow(image_start_np)
                                axes[0].set_title("Rendered RGB Start")
                                axes[0].axis("off")

                                axes[1].imshow(image_after_np)
                                axes[1].set_title("Rendered RGB After")
                                axes[1].axis("off")

                                plt.tight_layout()
                                plt.savefig(os.path.join(output_dir, f'mapping_{viewpoint.uid}_flow_offset.png'), dpi=150)
                                plt.close()

                            
                    if add_new_gaussian:
                        self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, flow_back=flow_back, closest_frame=self.viewpoints[closest_keyframe])
                    
                    if self.dystart==cur_frame_idx:
                        self.initialize_map(cur_frame_idx, viewpoint)
                        if self.dynamic_model == 'mlp':
                            self.initialize_network(cur_frame_idx, viewpoint)
                    
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    iter_per_kf = 70
                    #print(iter_per_kf)
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization", tag="Backend")
                        else:
                            iter_per_kf = self.mapping_itr_num
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        ratio = 1.0

                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5 * ratio,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5 * ratio,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01 * ratio, # * (1+('mask_loss' in self.config['Training'])),
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01 * ratio, # * (1+('mask_loss' in self.config['Training'])),
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    target_mult = 1.0 / ratio

                    def lr_lambda(step: int) -> float:
                        # step: 0..sched_T
                        s = step / float(max(1, self.iters))
                        return (1.0 - s) + s * target_mult  # linear interpolation

                    self.keyframe_scheduler = torch.optim.lr_scheduler.LambdaLR(
                        self.keyframe_optimizers,
                        lr_lambda=lr_lambda,
                    )

                    if self.dystart > cur_frame_idx:  #
                        # print('self.gaussians._delta_xyz:', self.gaussians._delta_xyz)
                        self.map_static(self.current_window, iters=int(20))  #
                        self.map_static(self.current_window, prune=True)  #
                    elif add_new_gaussian:
                        start = time.time()
                        self.map(self.current_window, iters=self.iters, dynamic_network=self.dynamic_model)
                        mapping_time = time.time() - start
                        print("Mapping Time:", mapping_time)
                        self.mapping_time.append(mapping_time)
                        start = time.time()
                        self.map(self.current_window, prune=True, dynamic_network=self.dynamic_model)

                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def map_static(self, current_window, prune=False, iters=1, dynamic_network=False, dynamic_render=False, rm_initdy=True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        for i in range(iters):
            self.iteration_count += 1
            self.last_sent += 1
            scaling = 0
            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                dxyz = 0
                d_rot, d_scale, d_opac, d_color = None, 0, None, None

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz,
                    ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )

                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None

                
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                    rm_dynamic=True, #mask=mask
                )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)
                
            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]

                dxyz = 0
                d_rot, d_scale, d_opac, d_color = None, 0, None, None

                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background, dynamic=False, dx=dxyz,
                    ds=d_scale, dr=d_rot, do=d_opac, dc=d_color
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                if rm_initdy:
                    with torch.no_grad():
                        mask = viewpoint.reproject_mask(self.dataset, self.viewpoints[0])
                else:
                    mask = None
    
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity,
                    rm_dynamic=True, #mask=mask
                )

                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]  # ֻprune
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask  # prune
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized 4DGS-SLAM", tag="Backend")
                        # # make sure we don't split the gaussians, break here.

                    vis_render_process(self.gaussians, self.pipeline_params, self.background,
                                       self.viewpoints[current_window[0]],
                                       self.viewpoints[current_window[0]].uid, self.save_dir, out_dir="map", mask=None,
                                       dynamic=(dynamic_network and self.gaussians.deform_init))

                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                )

                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    # dygs = self.gaussians.get_dygs_xyz.detach()
                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                        not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians", tag="Backend")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                if True:
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(self.iteration_count)
                else:
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                if True:  # not (dynamic_network or dynamic_render):
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    for cam_idx in range(min(frames_to_optimize, len(current_window))):
                        viewpoint = viewpoint_stack[cam_idx]
                        if viewpoint.uid == 0:
                            continue
                        update_pose(viewpoint)
                else:
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
        return gaussian_split