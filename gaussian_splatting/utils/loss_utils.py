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

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        loss = torch.abs((network_output - gt))
        loss = torch.where(mask.unsqueeze(0), loss, 0.)
        return loss.mean()
    return torch.abs((network_output - gt)).mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if mask is not None:
        img1 = torch.where(mask.unsqueeze(0), img1, 0.)
        img2 = torch.where(mask.unsqueeze(0), img2, 0.)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gradient_loss_flow(prediction, target, mask=None, weight=None):
    """
    Gradient-difference loss for optical flow maps (u, v).
    Expects channel-last tensors with 2 channels:
      - prediction: [H, W, 2] or [B, H, W, 2]
      - target    : same shape as prediction
      - mask      : optional, [H, W], [H, W, 1], [H, W, 2], or batched equivalents
      - weight    : optional per-pixel/per-channel weight with same broadcast rules as mask

    Returns:
      scalar tensor loss
    """
    # Ensure 4D: [B, H, W, C], with C=2
    if prediction.dim() == 3:  # [H, W, 2]
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if weight is not None and weight.dim() == 2:
            weight = weight.unsqueeze(0)
        elif weight is not None and weight.dim() == 3:
            weight = weight.unsqueeze(0)

    B, H, W, C = prediction.shape
    assert C == 2, f"Expected 2 channels for flow (u,v), got C={C}"

    device = prediction.device
    dtype = prediction.dtype

    # Default mask: ones per channel to match original normalization behavior
    if mask is None:
        mask = torch.ones((B, H, W, C), dtype=dtype, device=device)
    else:
        # Broadcast mask to [B,H,W,2]
        if mask.dim() == 3:  # [B,H,W] or [H,W,1] already handled above
            mask = mask.unsqueeze(-1)
        if mask.shape[-1] == 1:
            mask = mask.expand(-1, -1, -1, C)
        mask = mask.to(dtype=dtype, device=device)

    # Optional per-pixel/channel weight
    if weight is not None:
        if weight.dim() == 3:
            weight = weight.unsqueeze(-1)
        if weight.shape[-1] == 1:
            weight = weight.expand(-1, -1, -1, C)
        mask = mask * weight.to(dtype=dtype, device=device)

    # Normalization term (keeps behavior similar to your original code)
    M = mask.float().sum()

    diff = (prediction - target) * mask

    # Spatial gradients (forward differences) on the diff field
    # x-gradient along width (dim=2), y-gradient along height (dim=1)
    grad_x = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :])
    mask_x = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    grad_x = grad_x * mask_x

    grad_y = torch.abs(diff[:, 1:, :, :] - diff[:, :-1, :, :])
    mask_y = mask[:, 1:, :, :] * mask[:, :-1, :, :]
    grad_y = grad_y * mask_y

    # Sum over H, W, C for each item, then sum over batch
    image_loss = grad_x.sum(dim=(1, 2, 3)) + grad_y.sum(dim=(1, 2, 3))

    if M == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return image_loss.sum() / M
