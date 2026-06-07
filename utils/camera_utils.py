import torch
from torch import nn
import numpy as np
from scipy.ndimage import label
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask
from RAFT.raft import RAFT
from RAFT.utils import flow_viz
from RAFT.utils.utils import InputPadder
from GMA.network import RAFTGMA
from flow_utils import *
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from utils.normal_utils import intrins_to_intrins_inv, get_cam_coords, d2n_tblr
import time


class raft_param():
    def __init__(self):
        # self.dataset_path =  "/path/to/dataset"
        self.model = "pretrained/raft-things.pth"
        self.small = False  #
        self.mixed_precision = False  #

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        time,
        motion_mask=None,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.depth_mask = np.isfinite(depth) & (depth > 0.0)
        # self.disps = torch.where(depth>0, 1.0/depth, depth)
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width
        self.time = time
        self.fid = torch.Tensor(np.array([time])).to(device)
        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        self.motion_mask = motion_mask
        self.rendered_mask = None
        self.save_mask = False
        self.flow = None
        self.flow_back=None
        self.mask_fwd=None
        self.mask_bwd=None

        self.flow_mini = None
        self.flow_back_mini = None
        self.mask_fwd_mini = None
        self.mask_bwd_mini = None

        self.intrinsic = torch.from_numpy(np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])).to(device)
        self.intrins_inv = intrins_to_intrins_inv(self.intrinsic).float().unsqueeze(0).to(0)


        args = raft_param()
        self.model = torch.nn.DataParallel(RAFT(args))
        #model = torch.nn.DataParallel(RAFTGMA(args))
        self.model.load_state_dict(torch.load(args.model))
    
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix, model=None):
        gt_color, gt_depth, gt_pose, motion_mask = dataset[idx]
        time = (idx) / (dataset.num_imgs - 1)
        #if model is not None:   
        
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            time,
            motion_mask,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W, dataset):
        time = (uid) / (dataset.num_imgs - 1)
        gt_color, gt_depth, gt_pose, motion_mask = dataset[uid]
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W, time, None, device=dataset.device
        )
            
    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)
    

    @property
    def w2c(self):
        return getWorld2View2(self.R, self.T)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
    @property
    def novel_proj_transform(self):
        return (
            (self.world_view_transform+
            torch.tensor([[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 10.0],[0.0, 0.0, 0.0, 0.0]], 
            device=self.world_view_transform.device)).unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)
    
    def update_RT_vel(self, cameras, idx, step=1):
        if idx < step*2:
            self.R = cameras[idx-step].R.to(device=self.device)
            self.T = cameras[idx-step].T.to(device=self.device)
        else:
            R_prev = F.normalize(cameras[idx-step*2].R.to(device=self.device))
            R_curr = F.normalize(cameras[idx-step].R.to(device=self.device))
            self.R = F.normalize(R_curr + (R_curr - R_prev)).to(device=self.device)
            
            T_prev = cameras[idx-step*2].T.to(device=self.device)
            T_curr = cameras[idx-step].T.to(device=self.device)
            self.T = (T_curr + (T_curr - T_prev)).to(device=self.device)
    
    def get_mask_inlier(self):
        result_mask = torch.ones_like(self.motion_mask, dtype=torch.bool, device=self.device)
        motion_mask = ~self.motion_mask.cpu()
        labeled_mask, num_features = label(motion_mask.numpy())
        for region in range(1, num_features + 1):
            current_region = (labeled_mask == region)
            if np.any(current_region[0, :]) or np.any(current_region[-1, :]) or \
               np.any(current_region[:, 0]) or np.any(current_region[:, -1]):
                continue
            else:
                result_mask[current_region] = False
        return ~result_mask
    
    def get_mask_outlier(self):
        result_mask = torch.ones_like(self.motion_mask, dtype=torch.bool, device=self.device)
        motion_mask = ~self.motion_mask.cpu()
        labeled_mask, num_features = label(motion_mask.numpy())
        for region in range(1, num_features + 1):
            current_region = (labeled_mask == region)
            if np.any(current_region[0, :]) or np.any(current_region[-1, :]) or \
               np.any(current_region[:, 0]) or np.any(current_region[:, -1]):
                result_mask[current_region] = False
        return ~result_mask
    
    def render_mask(self, dataset, image):
        if self.uid==0:
            return None
        if self.rendered_mask is not None:
            return self.rendered_mask
        combined_mask = torch.zeros((image.shape[1], image.shape[2]), device=self.device, dtype=torch.bool)
        if dataset.yolo_model is not None:
            results = dataset.yolo_model.predict(source=image.clamp(0.0, 1.0).unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device=self.device)
            for result in results:
                masks = result.masks
                if masks is not None:
                    for mask in masks.data:
                        mask = mask.to(torch.bool)
                        combined_mask |= mask
            if dataset.seg_chair:
                results = dataset.yolo_model.predict(source=image.clamp(0.0, 1.0).unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device=self.device)
                for result in results:
                    masks = result.masks
                    if masks is not None:
                        for mask in masks.data:
                            mask = mask.to(torch.bool)
                            combined_mask |= mask
            self.rendered_mask = torch.logical_not(combined_mask)
            return self.rendered_mask
        else:
            return None
    
    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

        
        gt_image = self.original_image.cuda()
        _, h, w = self.original_image.cuda().shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = 0.05
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(
            *mask_shape
        )
        self.rgb_pixel_mask = rgb_pixel_mask * self.grad_mask

        self.rgb_pixel_mask = rgb_pixel_mask
        self.rgb_pixel_mask_mapping = rgb_pixel_mask

        if self.depth is not None:
            # print('self.depth.sum(): ', self.depth.sum())
            self.gt_depth = torch.from_numpy(self.depth).to(
                dtype=torch.float32, device=self.device
            )[None]

            depth = self.gt_depth.unsqueeze(0)
            points = get_cam_coords(self.intrins_inv, depth)
            normal, valid_mask = d2n_tblr(points, d_min=1e-3, d_max=1000.0)
            normal = normal * valid_mask
            self.normal = normal
            self.normal_raw = self.normal.squeeze(0).permute(1, 2, 0).cpu().numpy()
            self.mask = valid_mask

        if self.mask is not None:
            self.rgb_pixel_mask = self.rgb_pixel_mask * self.mask
            self.rgb_pixel_mask_mapping = self.rgb_pixel_mask_mapping
            
    def get_pointcloud(self, depth, dataset, w2c, indices):
        CX = dataset.cx
        CY = dataset.cy
        FX = dataset.fx
        FY = dataset.fy

        # Compute indices of sampled pixels
        xx = (indices[:, 1] - CX) / FX
        yy = (indices[:, 0] - CY) / FY
        depth_z = depth[0, indices[:, 0], indices[:, 1]]

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)

        #w2c = getWorld2View2(viewpoint.R, viewpoint.T)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]

        # Remove points at camera origin
        A = torch.abs(torch.round(pts, decimals=4))
        B = torch.zeros((1, 3)).cuda().float()
        _, idx, counts = torch.cat([A, B], dim=0).unique(
            dim=0, return_inverse=True, return_counts=True)
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        invalid_pt_idx = mask[:len(A)]
        valid_pt_idx = ~invalid_pt_idx
        pts = pts[valid_pt_idx]

        return pts
    
    def reproject_mask(self, dataset, cam_0):
        #if self.uid==0:  # ��0֡��ȥ����̬���巴����Ч������� 
        #    return None
        gt_depth_0 = torch.from_numpy(np.copy(cam_0.depth)).to(
            dtype=torch.float32, device=self.device
        )[None]
        W, H = dataset.width, dataset.height
        
        if torch.all(~((gt_depth_0[0] > 0) & (~cam_0.motion_mask))):
            return torch.ones((H, W), device=self.device, dtype=torch.bool)
            
        valid_depth_indices = torch.where((gt_depth_0[0] > 0) & (~cam_0.motion_mask))  # ��ȡgt_depth_0 ����0������
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        w2c = getWorld2View2(cam_0.R, cam_0.T)
        pts = self.get_pointcloud(gt_depth_0, dataset, w2c, valid_depth_indices)
        curr_w2c = getWorld2View2(self.R, self.T)
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        transformed_pts = (curr_w2c @ pts4.T).T[:, :3]

        intrinsics = torch.eye(3).to(device=self.device)
        intrinsics[0][2] = dataset.cx
        intrinsics[1][2] = dataset.cy
        intrinsics[0][0] = dataset.fx
        intrinsics[1][1] = dataset.fy
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2].long()
        valid_projected_pts = projected_pts[(projected_pts[:, 0] >= 0) & (projected_pts[:, 0] < W) & (projected_pts[:, 1] >= 0) & (projected_pts[:, 1] < H)]
        rendered_mask = torch.zeros((H, W), device=self.device, dtype=torch.bool)
        rendered_mask[valid_projected_pts[:, 1], valid_projected_pts[:, 0]] = True
        
        import torch.nn.functional as F
        kernel = torch.ones((3, 3), device=self.device, dtype=torch.bool)  # ʹ�� 3x3 �ľ�����
        for _ in range(3):
            dilated_mask = F.conv2d(rendered_mask.unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0).float(), padding=1)
            # �����ת���� bool ���ͣ���ȥ�������ά��
            rendered_mask = dilated_mask.squeeze().bool()
        
        if not self.save_mask and False:
            import matplotlib.pyplot as plt
            import os
            mask_image = (~rendered_mask.cpu().numpy().astype('uint8'))*255
            plt.imshow(mask_image)
            plt.axis("off")
            os.makedirs("results/render_mask/", exist_ok=True)
            plt.savefig(f"results/render_mask/mask_{self.uid}.png", bbox_inches="tight", pad_inches=0)
            self.save_mask = True
            
        return ~rendered_mask
        
    def keyframe_selection_overlap(self, dataset, cam, time, pixels=1600, pose_window=3):
        intrinsics = torch.eye(3).to(device=self.device)
        intrinsics[0][2] = dataset.cx
        intrinsics[1][2] = dataset.cy
        intrinsics[0][0] = dataset.fx
        intrinsics[1][1] = dataset.fy
        W, H = dataset.width, dataset.height
        gt_depth_0 = torch.from_numpy(np.copy(self.depth)).to(
            dtype=torch.float32, device=self.device
        )[None]

        valid_depth_indices = torch.where((gt_depth_0[0] > 0))
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        w2c = getWorld2View2(self.R, self.T)
        pts = self.get_pointcloud(gt_depth_0, dataset, w2c, valid_depth_indices)
        list_keyframe = []
        for cam_idx, viewpoint in cam.items():
            if cam_idx >= time:
                continue 
            est_w2c = getWorld2View2(viewpoint.R, viewpoint.T)
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < W - edge) * (projected_pts[:, 0] > edge) * \
                   (projected_pts[:, 1] < H - edge) * (projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum() / projected_pts.shape[0]
            list_keyframe.append(
                {'id': cam_idx, 'percent_inside': percent_inside})

            # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:8-pose_window])

        return selected_keyframe_list
        
    def generate_flow(self, image, image_last, tracking=False, ds=1, return_full=False):
        if not tracking:
            if self.flow is not None:
                return self.flow, self.flow_back

        # else:
        #     if self.flow_back_mini is not None:
        #         return self.flow_back_mini

        start_time = time.time()
        with torch.no_grad():
            image_copy = image.detach().clone()*255
            image_last_copy = image_last.detach().clone()*255
            image_copy, image_last_copy = image_copy[None], image_last_copy[None]
            padder = InputPadder(image_last_copy.shape)
            image_last_copy, image_copy = padder.pad(image_last_copy, image_copy)

            _, flow_bwd = self.model(image_copy, image_last_copy, iters=20, test_mode=True)  # image->image_last
            flow_bwd = padder.unpad(flow_bwd[0]).permute(1, 2, 0)
            # flow_bwd_copy = flow_bwd.clone().cpu().numpy()
            coor1to2_flow_back = flow_bwd / torch.tensor(flow_bwd.shape[:2][::-1], dtype=torch.float32).cuda() * 2

            if not tracking:
                _, flow_fwd = self.model(image_last_copy, image_copy, iters=20, test_mode=True)  # image_last->image
            
                flow_fwd = padder.unpad(flow_fwd[0]).permute(1, 2, 0)
                
                # flow_fwd_copy = flow_fwd.clone().cpu().numpy()
                
                coor1to2_flow = flow_fwd / torch.tensor(flow_fwd.shape[:2][::-1], dtype=torch.float32).cuda() * 2
                # mask_fwd, mask_bwd = self.compute_fwdbwd_mask(flow_fwd_copy, flow_bwd_copy)
                
            #flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            #flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)
        
        # print('flow estimation time:', time.time() - start_time)

        if ds != 1:
            assert ds > 1 and isinstance(ds, int)
            coor1to2_flow_final = F.interpolate(coor1to2_flow.permute(2,0,1).unsqueeze(0), scale_factor=1/ds, mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)
            coor1to2_flow_back_final = F.interpolate(coor1to2_flow_back.permute(2,0,1).unsqueeze(0), scale_factor=1/ds, mode='bilinear', align_corners=False).squeeze(0).permute(1,2,0)   
        elif not tracking:
            coor1to2_flow_final = coor1to2_flow
            coor1to2_flow_back_final = coor1to2_flow_back

        if not tracking:
            self.flow = coor1to2_flow_final
            self.flow_back = coor1to2_flow_back_final
            if not return_full:
                return self.flow, self.flow_back
            else:
                return coor1to2_flow, coor1to2_flow_back
        else:
            # self.flow_back_mini = coor1to2_flow_back
            return coor1to2_flow_back
    
    def merge_images_with_text(self, imgs, texts=None):
        # Ensure all are RGB
        imgs = [im.convert("RGB") for im in imgs]
        if texts is None:
            texts = [f"Image {i+1}" for i in range(len(imgs))]

        # Get sizes
        widths, heights = zip(*(im.size for im in imgs))
        max_h = max(heights)

        margin = 40
        text_space = 40
        out_w = sum(widths) + margin * (len(imgs) + 1)
        out_h = max_h + margin * 2 + text_space

        merged = Image.new("RGB", (out_w, out_h), "white")

        # Draw images
        x_offset = margin
        draw = ImageDraw.Draw(merged)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        for im, txt, w, h in zip(imgs, texts, widths, heights):
            merged.paste(im, (x_offset, margin))
            tx = x_offset + (w - draw.textlength(txt, font=font)) // 2
            ty = margin + h + 10
            draw.text((tx, ty), txt, fill="black", font=font)
            x_offset += w + margin

        return merged
    
    def save_flow(self, flow_start, flow, gt, save_path='fwd.png'):
        if flow_start is not None:
            flow_start_image = Image.fromarray(flow_viz.flow_to_image(flow_start.detach().cpu().numpy()))
            flow_image = Image.fromarray(flow_viz.flow_to_image(flow.detach().cpu().numpy()))
            gt_image = Image.fromarray(flow_viz.flow_to_image(gt.detach().cpu().numpy()))
            merged_image = self.merge_images_with_text([flow_start_image, flow_image, gt_image], ["Start Flow", "Optimized Flow", "Ground Truth Flow"])
        else:
            flow_image = Image.fromarray(flow_viz.flow_to_image(flow.detach().cpu().numpy()))
            gt_image = Image.fromarray(flow_viz.flow_to_image(gt.detach().cpu().numpy()))
            merged_image = self.merge_images_with_text([flow_image, gt_image], ["Optimized Flow", "Ground Truth Flow"])
        merged_image.save(save_path)
    
    def warp_flow(self, img, flow):
        h, w = flow.shape[:2]
        flow_new = flow.copy()
        flow_new[:,:,0] += np.arange(w)
        flow_new[:,:,1] += np.arange(h)[:,np.newaxis]
    
        res = cv2.remap(img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        return res

    def compute_fwdbwd_mask(self, fwd_flow, bwd_flow):
        alpha_1 = 0.5
        alpha_2 = 0.5
    
        bwd2fwd_flow = self.warp_flow(bwd_flow, fwd_flow)
        fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
        fwd_mask = fwd_lr_error < alpha_1  * (np.linalg.norm(fwd_flow, axis=-1) \
                    + np.linalg.norm(bwd2fwd_flow, axis=-1)) + alpha_2
    
        fwd2bwd_flow = self.warp_flow(fwd_flow, bwd_flow)
        bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)
    
        bwd_mask = bwd_lr_error < alpha_1  * (np.linalg.norm(bwd_flow, axis=-1) \
                    + np.linalg.norm(fwd2bwd_flow, axis=-1)) + alpha_2
    
        return fwd_mask, bwd_mask
    
    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None
        self.motion_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
    
    def clean_key(self):
        self.grad_mask = None
