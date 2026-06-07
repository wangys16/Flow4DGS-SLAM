import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


def vis_render_process(render_pkg, gt_color, gt_depth, cur_frame_idx, save_dir, out_dir="fin_render"):
    with torch.no_grad():
        fig, ax = plt.subplots(2, 2, figsize=(54, 34))
        viz_im = torch.clip(render_pkg["render"].permute(1, 2, 0).detach().cpu(), 0, 1)
        viz_depth = render_pkg['depth'][0, :, :].unsqueeze(0).detach().cpu()
        gt_im = torch.clip(gt_color.permute(1, 2, 0).detach().cpu(), 0, 1)
        gt_depth = gt_depth
        ax[0, 0].grid(False)
        ax[0, 0].imshow(gt_im)
        ax[0, 0].set_title("GT RGB", fontsize=30)
        ax[0, 1].grid(False)
        ax[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=6)
        ax[0, 1].set_title("GT Depth", fontsize=30)
        ax[1, 0].grid(False)
        ax[1, 0].imshow(viz_im)
        ax[1, 0].set_title("render color", fontsize=30)
        ax[1, 1].grid(False)
        ax[1, 1].imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        ax[1, 1].set_title("render depth", fontsize=30)
        os.makedirs(save_dir, exist_ok=True)
        process_dir = os.path.join(save_dir, out_dir)
        os.makedirs(process_dir, exist_ok=True)
        fig.suptitle(f"Frame: {cur_frame_idx}", y=0.95, fontsize=50)
        plt.close()
        
        h, w, _ = viz_im.shape
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_im)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}_color.png")
        plt.savefig(save_path)
        plt.close()
        
        
        fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100) 
        cax = ax.imshow(viz_depth[0], cmap='jet', vmin=0, vmax=6)
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path = os.path.join(process_dir, f"{cur_frame_idx}_depth.png")
        plt.savefig(save_path)
        plt.close()
        
        return
        
        
        
def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close()
    return ape_stat


def write_pose(pose, save_dir):
    plot_dir = os.path.join(save_dir, "pose.txt")
    with open(plot_dir, "a") as f:
        for row in pose:
            c2w = ' '.join(str(x) for x in row)
            f.write(str(c2w)+"\n")
        f.close()


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Args:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)

    Returns:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1).reshape((3,-1))
    data_zerocentered = data - data.mean(1).reshape((3,-1))

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,
                         column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U*S*Vh
    trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

    model_aligned = rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(
        alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error
    

def evaluate_ate(gt_traj, est_traj):
    """
    Input : 
        gt_traj: list of 4x4 matrices 
        est_traj: list of 4x4 matrices
        len(gt_traj) == len(est_traj)
    """
    gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
    est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]

    gt_traj_pts  =  np.stack(gt_traj_pts).T
    est_traj_pts =  np.stack(est_traj_pts).T

    _, _, trans_error = align(gt_traj_pts, est_traj_pts)

    avg_trans_error = trans_error.mean()

    return avg_trans_error
            

def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []
    
    trj_est_ate, trj_gt_ate = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose
        
    if final == True:
        plot_dir = os.path.join(save_dir, "pose.txt")
        with open(plot_dir, "wb") as f:
            f.truncate()
            f.close()
        for frame_id in range(0, len(frames)):
            kf = frames[frame_id]
            pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
            write_pose(pose_est, save_dir)
            pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))
            trj_est_ate.append(pose_est)
            trj_gt_ate.append(pose_gt)
        ate_rmse = evaluate_ate(trj_gt_ate, trj_est_ate)
        print("\nTacking ATE is :",ate_rmse,"\n")
            
    #for kf_id in kf_ids:
    for kf_id in range(0, len(frames)):
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)
        
    if final == True:
        with open(
            os.path.join(plot_dir, "ATE_{}.json".format(iterations)),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(ate_rmse, f, indent=4)
    
    try:
        ate = evaluate_evo(
            poses_gt=trj_gt_np,
            poses_est=trj_est_np,
            plot_dir=plot_dir,
            label=label_evo,
            monocular=monocular,
        )
    except Exception as e:
        print("running real dataset")
        ate = 0

    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def _gather_time(arr, idx):
    """
    arr: (B, T, N, D)
    idx: (B, 1, 1) long, time indices k
    returns: (B, 1, N, D) slice at time k
    """
    B, T, N, D = arr.shape
    idx4 = idx.view(B, 1, 1, 1).expand(B, 1, N, D)  # (B,1,N,D)
    return torch.gather(arr, 1, idx4)

def _hermite_blend_nonuniform(y_km1, y_k, y_k1, y_k2,
                              t_km1, t_k, t_k1, t_k2, t):
    """
    y_*: (B, 1, N, D)
    t_*: (B, 1, 1, 1)  (broadcastable)
    t:   (B, 1, 1, 1)
    Returns (B, 1, N, D)
    """
    eps = 1e-8
    hk  = (t_k1 - t_k).clamp_min(eps)
    s   = (t - t_k) / hk

    def h00(s): return  2*s**3 - 3*s**2 + 1
    def h10(s): return    s**3 - 2*s**2 + s
    def h01(s): return -2*s**3 + 3*s**2
    def h11(s): return    s**3 -   s**2

    d_km1 = (y_k  - y_km1) / (t_k   - t_km1).clamp_min(eps)
    d_k   = (y_k1 - y_k ) / (t_k1  - t_k  ).clamp_min(eps)
    d_k1  = (y_k2 - y_k1) / (t_k2  - t_k1 ).clamp_min(eps)

    mk  = ((t_k1 - t_k ) / (t_k1 - t_km1).clamp_min(eps)) * d_km1 \
        + ((t_k  - t_km1) / (t_k1 - t_km1).clamp_min(eps)) * d_k
    mk1 = ((t_k1 - t_k ) / (t_k2 - t_k ).clamp_min(eps)) * d_k   \
        + ((t_k2 - t_k1) / (t_k2 - t_k ).clamp_min(eps)) * d_k1

    return h00(s)*y_k + h10(s)*(hk*mk) + h01(s)*y_k1 + h11(s)*(hk*mk1)

def _hermite_one_sided(y_k, y_k1, t_k, t_k1, t):
    """
    One-segment Hermite with one-sided slopes at ends (reduces to linear for constant slope).
    y_k, y_k1: (B,1,N,D); t_k, t_k1, t: (B,1,1,1)
    """
    eps = 1e-8
    hk = (t_k1 - t_k).clamp_min(eps)
    s  = (t - t_k) / hk

    def h00(s): return  2*s**3 - 3*s**2 + 1
    def h10(s): return    s**3 - 2*s**2 + s
    def h01(s): return -2*s**3 + 3*s**2
    def h11(s): return    s**3 -   s**2

    d_k = (y_k1 - y_k) / hk
    m_k  = d_k
    m_k1 = d_k
    return h00(s)*y_k + h10(s)*(hk*m_k) + h01(s)*y_k1 + h11(s)*(hk*m_k1)

def interpolation_nonuniform(y, key_times, seg_idx, t_query):
    """
    y:         (B, T, N, D)   positions (use all D)
    key_times: (T,) or (B, T) monotonic increasing
    seg_idx:   (B, 1, 1) long OR int — k s.t. t in [t_k, t_{k+1}]
    t_query:   (B, 1, 1) float OR float
    Returns:   (B, N, D)
    """
    B, T, N, D = y.shape
    is_tensor = isinstance(seg_idx, torch.Tensor)

    # Prepare times to (B,T,1,1) for easy broadcasting
    if key_times.dim() == 1:
        kt = key_times.view(1, T, 1, 1).to(y.device).type_as(y)
        kt = kt.expand(B, T, 1, 1)
    else:
        kt = key_times.to(y.device).type_as(y).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)

    if not isinstance(t_query, torch.Tensor):
        t_query = torch.tensor([[t_query]], dtype=y.dtype, device=y.device)
    tQ = t_query.view(B, 1, 1, 1).type_as(y)

    if not is_tensor:
        seg_idx = torch.tensor([[[seg_idx]]], dtype=torch.long, device=y.device)
    k = seg_idx.view(B, 1, 1)

    # Handle small T explicitly
    if T == 2:
        # Linear / one-segment Hermite
        k = torch.zeros_like(k)  # only segment [0,1]
        yk  = _gather_time(y, k)       # (B,1,N,D)
        yk1 = _gather_time(y, k+1)
        tk  = _gather_time(kt, k)      # (B,1,1,1)
        tk1 = _gather_time(kt, k+1)
        out = _hermite_one_sided(yk, yk1, tk, tk1, tQ)  # (B,1,N,D)
        return out.squeeze(1)  # (B,N,D)

    if T == 3:
        # Three keyframes: interior uses non-uniform, ends use one-sided
        k = k.clamp(0, 1)  # only valid segments: 0 or 1
        yk  = _gather_time(y, k)
        yk1 = _gather_time(y, k+1)
        tk  = _gather_time(kt, k)
        tk1 = _gather_time(kt, k+1)

        # If you want, you could form mk/mk1 using the interior slope, but one-sided is robust:
        out = _hermite_one_sided(yk, yk1, tk, tk1, tQ)
        return out.squeeze(1)

    # General case T >= 4: use full non-uniform Hermite with neighbors
    k = k.clamp(1, T-3)  # ensure k-1 and k+2 exist

    ykm1 = _gather_time(y, k-1)
    yk   = _gather_time(y, k)
    yk1  = _gather_time(y, k+1)
    yk2  = _gather_time(y, k+2)

    tk_m1 = _gather_time(kt, k-1)
    tk    = _gather_time(kt, k)
    tk1   = _gather_time(kt, k+1)
    tk2   = _gather_time(kt, k+2)

    out = _hermite_blend_nonuniform(ykm1, yk, yk1, yk2, tk_m1, tk, tk1, tk2, tQ)  # (B,1,N,D)
    return out.squeeze(1)  # (B,N,D)


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
    save_interval=None,
    interp_type='linear',
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    depth_array = []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    #print(kf_indices)

    traj = []
    opacity_list = []

    for idx in range(0, end_idx):
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, gt_depth, _, motion_mask = dataset[idx]

        if gaussians.init_deform == 'mlp':
            if gaussians.deform_init:
                time_input = gaussians.deform.deform.expand_time(frame.fid)
                d_values = gaussians.deform.step(gaussians.get_dygs_xyz.detach(), time_input, 
                                                iteration=0, feature=None, 
                                                motion_mask=gaussians.motion_mask, 
                                                camera_center=frame.camera_center, 
                                                time_interval=gaussians.time_interval)
                dxyz = d_values['d_xyz']
                d_rot, d_scale = d_values['d_rotation'], d_values['d_scaling']
            else:
                dxyz, d_rot, d_scale = 0, 0, 0
        
        elif gaussians.init_deform == 'offset':
            d_rot = 0
            if interp_type == 'linear':
                if frame.uid in gaussians.kf_list:
                    dxyz = gaussians.get_delta_xyz(gaussians.kf_list.index(frame.uid))
                elif frame.uid < gaussians.kf_list[-1]: # Interpolate the non-keyframe dxyz
                    lower_kf = max([kf for kf in gaussians.kf_list if kf < frame.uid])
                    upper_kf = min([kf for kf in gaussians.kf_list if kf > frame.uid])
                    lower_idx = gaussians.kf_list.index(lower_kf)
                    upper_idx = gaussians.kf_list.index(upper_kf)
                    weight = (frame.uid - lower_kf) / (upper_kf - lower_kf)
                    dxyz = (1 - weight) * gaussians.get_delta_xyz(lower_idx) + weight * gaussians.get_delta_xyz(upper_idx)
                else:
                    dxyz = gaussians.get_delta_xyz(-1)
            
            elif interp_type == 'cube':
                if frame.uid in gaussians.kf_list:
                    dxyz = gaussians.get_delta_xyz[gaussians.kf_list.index(frame.uid)]
                elif frame.uid < gaussians.kf_list[1] or (frame.uid > gaussians.kf_list[-2] and frame.uid < gaussians.kf_list[-1]): # Interpolate the non-keyframe dxyz
                    lower_kf = max([kf for kf in gaussians.kf_list if kf < frame.uid])
                    upper_kf = min([kf for kf in gaussians.kf_list if kf > frame.uid])
                    lower_idx = gaussians.kf_list.index(lower_kf)
                    upper_idx = gaussians.kf_list.index(upper_kf)
                    weight = (frame.uid - lower_kf) / (upper_kf - lower_kf)
                    dxyz = (1 - weight) * gaussians.get_delta_xyz[lower_idx] + weight * gaussians.get_delta_xyz[upper_idx]

                elif frame.uid > gaussians.kf_list[-1]:
                    dxyz = gaussians.get_delta_xyz[-1]
                
                else:

                    y = torch.stack([p for p in gaussians.get_delta_xyz], dim=0).unsqueeze(0)  # (1,T,N,3)

                    key_times = torch.tensor(gaussians.kf_list, dtype=torch.float32, device=y.device)

                    seg_idx = torch.tensor([[max(i for i,kf in enumerate(gaussians.kf_list) if kf <= frame.uid)]],
                                        dtype=torch.long, device=y.device).unsqueeze(-1)  # (1,1,1)

                    t_query = torch.tensor([[float(frame.uid)]], dtype=torch.float32, device=y.device).unsqueeze(-1)  # (1,1,1)

                    dxyz = interpolation_nonuniform(y, key_times, seg_idx, t_query).squeeze(0)
            
            traj.append(dxyz.detach().cpu().numpy())
            d_scale, d_opac, d_color = 0, None, None

            if gaussians.use_gmm:
                opacity_list.append(gaussians.get_gmm_opacity(frame.uid).detach().cpu().numpy())
            else:
                opacity_list.append(gaussians.get_opacity.detach().cpu().numpy())

        
        
        render_pkg = render(
            frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot
        )


        if iteration == "before_opt":
            if (idx + 1) % save_interval == 0 or idx == 0:
                vis_render_process(render_pkg, gt_image, gt_depth, idx, save_dir, out_dir="before_opt")
        elif iteration == "after_opt":
            if (idx + 1) % save_interval == 0 or idx == 0:
                vis_render_process(render_pkg, gt_image, gt_depth, idx, save_dir, out_dir="after_opt")
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0
        
        valid_depth_mask = gt_depth > 0
        mask = mask * torch.from_numpy(valid_depth_mask).to(device=motion_mask.device) 
        
        l1_depth = np.abs((gt_depth[None] - render_pkg['depth'].cpu().detach().numpy()) * valid_depth_mask)

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))
        depth_score = l1_depth.sum() / (valid_depth_mask.sum()+1e-7)
        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        depth_array.append(depth_score)

        print(f'Frame.uid: {frame.uid}, PSNR: {psnr_score}, SSIM: {ssim_score}, LPIPS: {lpips_score}, L1_depth: {depth_score}')
        
        if iteration == "after_opt":
            with torch.no_grad():
                render_pkg_novel = render(
                    frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, novel=1,
                )
                vis_render_process(render_pkg_novel, gt_image, gt_depth, idx, save_dir, out_dir="novel_render")

        else:
            with torch.no_grad():
                render_pkg_novel = render(
                    frame, gaussians, pipe, background, dynamic=False, dx=dxyz, ds=d_scale, dr=d_rot, novel=1,
                )
                vis_render_process(render_pkg_novel, gt_image, gt_depth, idx, save_dir, out_dir="before_novel_render")

  
    if iteration == "after_opt":
        if end_idx < 100:
            np.save(os.path.join(save_dir, "deform_traj.npy"), traj)

            np.save(os.path.join(save_dir, "opacity_list.npy"), opacity_list)

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["l1_depth"] = float(np.mean(depth_array))
    
    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, l1_depth: {output["l1_depth"]}, \
            num static/dynamic/total gaussians: {gaussians._xyz.shape[0]-gaussians.dygs.sum().item()}/{gaussians.dygs.sum().item()}/{gaussians._xyz.shape[0]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
