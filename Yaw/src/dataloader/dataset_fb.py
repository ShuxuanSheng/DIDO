"""
Pytorch dataloader for FB dataset
"""

import random
import os
import torch
from abc import ABC, abstractmethod
from os import path as osp

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from pyquaternion import Quaternion
from utils.lie_algebra import SO3
from utils.logging import logging

class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

def q_integrate(gyr, ts_win,q0):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    # st_t = time.time()
    feat_gyr = gyr
    dalte_w = (feat_gyr[1:] + feat_gyr[:-1]) / 2
    # dalte_w = feat_gyr[:-1]
    dalte_gyr_norm = np.linalg.norm(dalte_w, ord=2, axis=1, keepdims=True)
    dalte_intint = dalte_gyr_norm * np.expand_dims(ts_win[0:], 1) / 2 # 因为我输入的是dt，所以这边原本的ts_win[1:]改成了ts_win[0:]
    w_point = dalte_w / dalte_gyr_norm
    dalte_q_w = np.cos(dalte_intint)
    dalte_q_xyz = w_point * np.sin(dalte_intint)
    dalte_q_wxyz = np.concatenate((dalte_q_w, dalte_q_xyz), axis=1)

    # dalte_q_1 = Quaternion(np.array([1, 0, 0, 0]))
    dalte_q_1 = Quaternion(q0)
    dalte_q_2 = Quaternion(dalte_q_wxyz[0])
    dalte_q_x = dalte_q_1 * dalte_q_2
    dalte_q_winst = np.expand_dims(dalte_q_1.q, axis=0)
    dalte_q_win = np.concatenate((dalte_q_winst, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)
    for iii in range(1, len(dalte_q_wxyz[:, 0])):
        dalte_q_x = dalte_q_x * Quaternion(dalte_q_wxyz[iii])
        if dalte_q_x.w < 0:
            dalte_q_x.q = - dalte_q_x.q
        dalte_q_win = np.concatenate((dalte_q_win, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)

        # print(iii)
    dalte_q_x_xnorm = dalte_q_x.normalised
    dalte_q_diff = dalte_q_x_xnorm.q
    dalte_q_diff = np.array(dalte_q_diff)
    dalte_q_win = np.array(dalte_q_win)

    # end_t = time.time()
    # print("计算时间：", end_t - st_t)
    return dalte_q_diff, dalte_q_win

class FbSequence(CompiledSequence):
    def __init__(self, data_path, args, data_window_config, **kwargs):
        super().__init__(**kwargs)
        (self.ts, self.features, self.targets, self.orientations, self.gt_pos, self.gt_ori,) = (None, None, None, None, None, None)
        self.target_dim = args.output_dim
        self.imu_freq = args.imu_freq
        self.imu_base_freq = args.imu_base_freq
        self.interval = data_window_config["window_size"]
        self.mode = kwargs.get("mode", "train")

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        time_factor = 1e9  # ns -> s

        def interpolate(x, t, t_int, angle=False):
            """
            Interpolate ground truth with sensors
            """
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if angle:
                    x[:, i] = np.unwrap(x[:, i])
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            return x_int

        # read each sequence
        path_imu = os.path.join(data_path, "sensor_data", "xsens_imu.csv")
        path_gt = os.path.join(data_path, "global_pose.csv")
        path_wheel = os.path.join(data_path, "sensor_data", "encoder.csv")
        imu = np.genfromtxt(path_imu, delimiter=",")
        gt = np.genfromtxt(path_gt, delimiter=",")
        wheel = np.genfromtxt(path_wheel, delimiter=",")

        # imu时间戳重新采样
        t = imu[:, 0]
        start = t[0]
        end = t[-1]
        num_points = len(t)
        imu[:, 0] = np.linspace(start, end, num=num_points)

        # time synchronization between IMU and ground truth
        t0 = np.max([gt[0, 0], imu[0, 0], wheel[0, 0]])
        t_end = np.min([gt[-1, 0], imu[-1, 0], wheel[-1, 0]])

        # start index
        idx0_imu = np.searchsorted(imu[:, 0], t0)
        idx0_gt = np.searchsorted(gt[:, 0], t0)
        idx0_wheel = np.searchsorted(wheel[:, 0], t0)

        # end index
        idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
        idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')
        idx_end_wheel = np.searchsorted(wheel[:, 0], t_end, 'right')

        # subsample
        imu = imu[idx0_imu: idx_end_imu]
        gt = gt[idx0_gt: idx_end_gt]
        wheel = wheel[idx0_wheel: idx_end_wheel]

        t = imu[:, 0]
        # take ground truth position
        p_gt = gt[:, [4, 8, 12]]
        p_gt = p_gt - p_gt[0]
        # take ground matrix pose
        Rot_gt = torch.Tensor(gt.shape[0], 3, 3)
        for j in range(3):
            Rot_gt[:, j] = torch.Tensor(gt[:, 1 + 4 * j: 1 + 4 * j + 3])
        q_gt = SO3.to_quaternion(Rot_gt)
        rpys = SO3.to_rpy(Rot_gt)

        wheel_cnt = wheel[:, [1, 2]]

        # interpolate ground-truth
        t_gt = gt[:, 0]
        t_wheel = wheel[:, 0]
        p_gt = interpolate(p_gt, t_gt, t)
        rpys = interpolate(rpys.numpy(), t_gt, t, angle=True)
        wheel_cnt = interpolate(wheel_cnt, t_wheel, t)

        # convert from numpy
        ts = (t - t0)/time_factor
        p_gt = torch.Tensor(p_gt)
        rpys = torch.Tensor(rpys).float()
        q_gt = SO3.to_quaternion(SO3.from_rpy(rpys[:, 0], rpys[:, 1], rpys[:, 2]))
        imu = torch.Tensor(imu).float()
        wheel_cnt = torch.Tensor(wheel_cnt).float()

        # take IMU gyro and accelerometer and magnetometer
        gyro = imu[:, 8:11]
        acc = imu[:, 11:14]

        dt = ts[1:] - ts[:-1]

        # compute wheel speed
        v_wheel = torch.zeros(wheel_cnt.shape[0], 2)
        for j in range(2):
            wheel_cnt_j = wheel_cnt[:, j]
            v_j = (wheel_cnt_j[1:] - wheel_cnt_j[:-1]) / dt * 0.0005
            v_wheel[1:, j] = torch.Tensor(v_j)

        # compute speed ground truth (apply smoothing)
        v_gt = torch.zeros(p_gt.shape[0], 3)
        for j in range(3):
            p_gt_smooth = savgol_filter(p_gt[:, j], 11, 1)
            v_j = (p_gt_smooth[1:] - p_gt_smooth[:-1]) / dt
            v_j_smooth = savgol_filter(v_j, 11, 0)
            v_gt[1:, j] = torch.Tensor(v_j_smooth)

        yaw_np = (rpys[:, 2]).numpy()
        yaw_np = np.unwrap(yaw_np)
        Rot = SO3.from_rpy(rpys[:, 0], rpys[:, 1], torch.from_numpy(yaw_np))
        Rot_T = Rot.transpose(1, 2)
        v_imu_gt = torch.einsum('bij,bj->bi', Rot_T, v_gt)

        self.ts = ts  # ts of the beginning of each window
        self.features = np.concatenate([gyro, acc, v_wheel[:, 0].unsqueeze(1), v_wheel[:, 1].unsqueeze(1)], axis=1)
        self.targets = gyro[:, 2].unsqueeze(1).cpu().numpy()
        self.pos = p_gt.detach().cpu().numpy()
        self.vel = torch.norm(v_imu_gt[:, :2], dim=1)
        self.vel = self.vel.unsqueeze(1).cpu().numpy()
        self.yaw = torch.Tensor(rpys[:, 2])
        self.yaw = self.yaw.unsqueeze(1).cpu().numpy()

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_pos(self):
        return self.pos

    def get_vel(self):
        return self.vel

    def get_yaw(self):
        return self.yaw

class FbSequenceDataset(Dataset):
    def __init__(self, root_dir, data_list, args, data_window_config, **kwargs):
        super(FbSequenceDataset, self).__init__()

        self.window_size = data_window_config["window_size"]
        self.past_data_size = data_window_config["past_data_size"]
        self.future_data_size = data_window_config["future_data_size"]
        self.step_size = data_window_config["step_size"]

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform = False, False
        if self.mode == "train":
            self.shuffle = False
        elif self.mode == "val":
            self.shuffle = False
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts, self.orientations, self.ori_r, self.gt_v = [], [], [], []
        self.features, self.targets = [], []
        self.pos, self.vel, self.yaw = [], [], []
        self.no_yaw_q = []
        self.gt_q = []
        self.rpm = []
        self.kf = []
        self.D = []

        for i in range(len(data_list)):
            logging.info("Loading " + osp.join(root_dir, data_list[i]))
            seq = FbSequence(osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs)
            feat, targ, pos, vel, yaw = seq.get_feature(), seq.get_target(), seq.get_pos(), seq.get_vel(), seq.get_yaw()
            self.features.append(feat)
            self.targets.append(targ)
            self.pos.append(pos)
            self.vel.append(vel)
            self.yaw.append(yaw)
            self.index_map += [
                [i, j]
                for j in range(0 + self.past_data_size, self.targets[i].shape[0] - self.future_data_size, self.step_size)
            ] #self.features中存放了多个seq的数据，通过map建立seq及其对应数据起始～终止索引的映射关系。一个i对应多个j，例如[i=0,j=2][i=0,j=4][i=0,j=6][i=1,j=2]……

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # 返回一个样本所包含的数据，item不是显示传递的，是自动确定的
        # item = 0 时，self.index_map[0] 是 [seq_id=0, frame_id=2]，
        # item = 1 时，self.index_map[1] 是 [seq_id=0, frame_id=4]，
        # item = 2 时，self.index_map[2] 是 [seq_id=0, frame_id=6]，
        # item = 3 时，self.index_map[3] 是 [seq_id=1, frame_id=2]……
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        # in the world frame
        # 提取features中指定范围的数据,目前是20帧(imu 400hz)
        feat = self.features[seq_id][frame_id - self.past_data_size: frame_id + self.window_size + self.future_data_size]
        targ = self.targets[seq_id][frame_id]  # the beginning of the sequence
        pos = self.pos[seq_id][frame_id]
        vel = self.vel[seq_id][frame_id]
        yaw = self.yaw[seq_id][frame_id]
        return feat.astype(np.float32).T, targ.astype(np.float32), pos.astype(np.float32), vel.astype(np.float32), yaw.astype(np.float32), seq_id, frame_id
    def __len__(self):
        return len(self.index_map)
