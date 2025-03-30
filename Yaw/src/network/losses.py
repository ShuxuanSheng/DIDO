import logging

import torch


"""
MSE loss between prediction and target, no covariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""

l1loss = torch.nn.L1Loss()
def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss

def loss_e(pred, targ):
    loss = torch.exp((pred - targ).pow(2))
    return loss

"""
Log Likelihood loss, with covariance (only support diag cov)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_cov: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_cov meaning:
pred_cov:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov)) + pred_cov
    return loss

"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""

def get_loss(pred, pred_cov, pos, vel, yaw, epoch):
    delta_t = 0.01
    # 横摆角速度损失
    true_gyr_z = (yaw[1:] - yaw[:-1]) / delta_t
    predicted_gyr_z = pred[1:]
    gyr_loss = torch.mean(torch.abs(true_gyr_z - predicted_gyr_z))

    # 横摆角损失
    true_yaw_increment = yaw[-1] - yaw[0]
    predicted_yaw_increment = torch.sum(pred * delta_t)
    yaw_increment_loss = torch.abs(true_yaw_increment - predicted_yaw_increment)

    # 位置损失
    true_position_increment = pos[1:, :2] - pos[:-1, :2]

    initial_yaw = yaw[0]
    integrated_yaw = initial_yaw + torch.cumsum(pred * delta_t, dim=0)
    pred_x_increment = vel * torch.cos(integrated_yaw) * delta_t
    pred_y_increment = vel * torch.sin(integrated_yaw) * delta_t

    pred_pos_increment = torch.stack([pred_x_increment, pred_y_increment], dim=1)
    pred_pos_increment = pred_pos_increment.squeeze()
    pos_loss = torch.norm(true_position_increment - pred_pos_increment[:-1], dim=1)

    # 计算总损失
    total_loss = gyr_loss + 0.5 * pos_loss
    return total_loss.unsqueeze(1)
