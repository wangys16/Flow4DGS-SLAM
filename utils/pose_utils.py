import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    if torch.isnan(W).any():
        print("Matrix W contains NAN")
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )


def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device
    if torch.isnan(tau).any():
        print("Matrix W contains NAN")
        return torch.eye(4, device=device, dtype=dtype)
    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged


def skew(w):
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    W = torch.zeros((*w.shape[:-1], 3,3), device=w.device, dtype=w.dtype)
    W[...,0,1], W[...,0,2] = -wz,  wy
    W[...,1,0], W[...,1,2] =  wz, -wx
    W[...,2,0], W[...,2,1] = -wy,  wx
    return W

def so3_exp(w):
    a = torch.norm(w)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    if a < 1e-8:
        W = skew(w)
        return I + W + 0.5*(W@W)
    W_hat = skew(w / a)
    A = torch.sin(a)/a
    B = (1 - torch.cos(a))/(a*a)
    return I + A*W_hat + B*(W_hat@W_hat)

def so3_log(R):
    tr = torch.clamp((torch.trace(R) - 1.0)/2.0, -1.0, 1.0)
    a = torch.acos(tr)
    if a < 1e-8:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    w_hat = (R - R.T) / (2.0*torch.sin(a))
    # vee:
    return a*torch.tensor([w_hat[2,1], w_hat[0,2], w_hat[1,0]], device=R.device, dtype=R.dtype)

def V_left(w):
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    a = torch.norm(w)
    if a < 1e-8:
        W = skew(w)
        return I + 0.5*W + (1.0/6.0)*(W@W)
    W_hat = skew(w / a)
    A = torch.sin(a)/a
    B = (1 - torch.cos(a))/(a*a)
    C = (1 - A)/(a*a)
    return I + B*W_hat + C*(W_hat@W_hat)

def scale_se3_step(T_rel, s: float):
    """
    Scale an SE(3) relative motion T_rel=(R,t) toward identity by factor s in (0,1],
    with translation coupled to rotation via the left Jacobian.
    """
    R = T_rel[:3,:3]
    t = T_rel[:3, 3]
    w  = so3_log(R)              # rotation vector
    R_s = so3_exp(s*w)
    Vw    = V_left(w)
    Vs_w  = V_left(s*w)
    # guard in case Vw is near-singular (very large angles):
    Vw_inv = torch.linalg.inv(Vw)
    t_s = Vs_w @ (Vw_inv @ t)

    T_s = torch.eye(4, device=T_rel.device, dtype=T_rel.dtype)
    T_s[:3,:3] = R_s
    T_s[:3, 3] = t_s
    return T_s