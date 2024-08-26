import time
import torch
import tqdm

from network.model_factory import get_model
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from gravity_align_ekf import gravity_align_EKF
import argparse
import os
from os import path as osp
from generate_net_acc_net_gyr import euler2quat

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def q_integrate(gyr, ts_win, q0):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    feat_gyr = gyr
    dalte_w = (feat_gyr[1:] + feat_gyr[:-1]) / 2   #相邻两帧的平均值

    dalte_gyr_norm = np.linalg.norm(dalte_w, ord=2, axis=1, keepdims=True) #每一帧gyr的2范数，这是什么量？-》这里应该是把三个轴的角速度看成纯虚四元数[0,wx,wy,wz]
    # 因为我输入的是dt，所以这边原本的ts_win[1:]改成了ts_win[0:] jcx
    dalte_intint = dalte_gyr_norm * np.expand_dims(ts_win[0:], 1) / 2  #np.expand_dims是在第一个维度之后（即第二个维度的位置）插入一个新维度，维度大小是1，例如[0.1, 0.2, 0.3] -> [[0.1],[0.2],[0.3]]

    # np.savetxt('dalte_gyr_norm_bit.txt', dalte_gyr_norm, delimiter=',')
    # np.savetxt('ts_win_bit.txt', ts_win, delimiter=',')
    # np.savetxt('dalte_intint_bit.txt', dalte_intint, delimiter=',')

    w_point = dalte_w / dalte_gyr_norm  #方向
    dalte_q_w = np.cos(dalte_intint)  #dalte_q_w是实部
    dalte_q_xyz = w_point * np.sin(dalte_intint) #dalte_q_xyz是虚部
    dalte_q_wxyz = np.concatenate((dalte_q_w, dalte_q_xyz), axis=1)   #通过q = cos(theta/2) + u*sin(theta/2)构造出来的四元数是已经归一化的，因为u是单位向量模是1，q ^2 = cos^2 + 1*sin^2 = 1

    dalte_q_1 = Quaternion(q0)
    dalte_q_2 = Quaternion(dalte_q_wxyz[0])
    dalte_q_x = dalte_q_1 * dalte_q_2  #使用初始四元数 q0 和第一个时间窗口增量四元数dalte_q_2更新姿态
    dalte_q_winst = np.expand_dims(dalte_q_1.q, axis=0)  #.q返回w，x,y,z四个分量
    dalte_q_win = np.concatenate((dalte_q_winst, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)
    for iii in range(1, len(dalte_q_wxyz[:, 0])):
        dalte_q_x = dalte_q_x * Quaternion(dalte_q_wxyz[iii])  #迭代更新
        if dalte_q_x.w < 0:
            dalte_q_x.q = - dalte_q_x.q
        dalte_q_win = np.concatenate((dalte_q_win, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)

    dalte_q_x_xnorm = dalte_q_x.normalised
    dalte_q_diff = dalte_q_x_xnorm.q
    dalte_q_diff = np.array(dalte_q_diff)
    dalte_q_win = np.array(dalte_q_win)

    return dalte_q_diff, dalte_q_win

def q_integrate_new(gyr, ts_win, q0):

    return

# 避免出现出现不连续的跳跃
def change_yaw(euler):
    th = 330
    d_euler = np.diff(euler, axis=0, prepend=euler[0].reshape(1, 3))
    for i in np.where(d_euler[:, 2] < -th)[0]:
        euler[i:, 2] += 2 * 180
    for i in np.where(d_euler[:, 2] > th)[0]:
        euler[i:, 2] -= 2 * 180
    return euler

def ROTATION_EKF(args):

    start_step = 0

    imu_freq = args.imu_freq
    root_dir = args.root_dir
    out_dir = args.out_dir
    test_path = args.test_list
    data_path_s = get_datalist(test_path)
    for data_path in tqdm.tqdm(data_path_s):
        if args.data_type == "hdf5":
            """
                hdf5格式的数据
            """
            with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
                ts = np.copy(f["ts"])  # timestamp
                gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame

        if args.data_type == "txt":
            """
                BIT compus txt格式的数据
            """
            inspva_file = '/home/meister/SSX/Datasets/bit_compus_1/inspvax.txt'
            inspav_data = np.genfromtxt(inspva_file, delimiter=',', skip_header=1)
            ts = inspav_data[:, 2] * 1e-9  # ns -> s
            ts = ts[:6000] # duration = 1min
            gt_rpy = inspav_data[:, 20:23] - np.array([13, 0, 0])  # 扣除安装角
            # gt_rpy[:, 2] = ((gt_rpy[:, 2] + 180) % 360) - 180 [0,2pi]-> [-pi,pi]
            gt_rpy = gt_rpy[:6000, :]
            gt_q = Rotation.from_euler('xyz', gt_rpy, degrees=True).as_quat()  #默认顺序是[x, y, z, w]
            gt_q = gt_q[:, [3, 0, 1, 2]]  # DIDO是[w,x,y,z]

        dt = np.diff(ts)
        dt = np.append(dt[0], dt).reshape(-1, 1)  #将dt[0]append到dt中，即dt的第一个元素重复一次，然后reshape成(n,1),-1表示自动匹配维度n，例如[1,1,2,3] -> [[1],[1],[2],[3]]

        with open(osp.join(args.network_gyr_out_path + data_path + "_pred_gyr.txt"), encoding='utf-8') as f:
            pred_gyr_all = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_acc_out_path + data_path + "_pred_acc.txt"), encoding='utf-8') as f:
            pred_acc_all = np.loadtxt(f, delimiter=",")

        with open(osp.join(args.network_gyr_out_path + data_path + "_raw_gyr.txt"), encoding='utf-8') as f:
            raw_gyr_all = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_acc_out_path + data_path + "_raw_acc.txt"), encoding='utf-8') as f:
            raw_acc_all = np.loadtxt(f, delimiter=",")

        with open(osp.join(args.network_gyr_out_path + data_path + "_gt_gyr.txt"), encoding='utf-8') as f:
            gt_gyr_all = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_acc_out_path + data_path + "_gt_acc.txt"), encoding='utf-8') as f:
            gt_acc_all = np.loadtxt(f, delimiter=",")

        raw_gyr_all = raw_gyr_all[:6000, :] * np.array([1, -1, -1]) # FLU -> FRD
        raw_acc_all = raw_acc_all[:6000, :]

        pred_gyr_all = pred_gyr_all[:6000, :] * np.array([1, -1, -1]) # FLU -> FRD
        pred_acc_all = pred_acc_all[:6000, :]

        _, pred_q = q_integrate(pred_gyr_all, dt[start_step + 1:, 0],gt_q[start_step]) #dt[start_step + 1:, 0]从第1行到最后一行的第一列
        _, raw_q = q_integrate(raw_gyr_all, dt[start_step + 1:, 0],gt_q[start_step]) #dt[start_step + 1:, 0]从第1行到最后一行的第一列
        #_, gt_inte_q = q_integrate(gt_gyr_all, dt[start_step + 1:, 0],gt_q[start_step]) #dt[start_step + 1:, 0]从第1行到最后一行的第一列

        euler_gt_q = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        euler_pred = Rotation.from_quat(pred_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        euler_raw = Rotation.from_quat(raw_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        # euler_gt_inte_q = Rotation.from_quat(gt_inte_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)

        acc_cov = np.expand_dims([0.01], axis=1).repeat(pred_acc_all.shape[0], axis=0)
        # ekf_raw = gravity_align_EKF(gyr=(raw_gyr_all[1:, :] + raw_gyr_all[:-1, :]) / 2, acc=raw_acc_all,
        #                                acc_cov=acc_cov ** 2,
        #                                noises=[0.05 ** 2],  # big是0.05
        #                                frequency=imu_freq,
        #                                dt_seq=dt[start_step + 1:, 0],  # 中值积分 + 1
        #                                update_rate=10,
        #                                arch='origin',
        #                                Init_state_cov=0.0000000001,
        #                                q0=gt_q[start_step], frame='ENU')
        #
        # ekf_raw_q = ekf_raw.Q / np.linalg.norm(ekf_raw.Q, axis=1, keepdims=True)
        # ekf_raw_q[ekf_raw_q[:, 0] < 0] = (-1) * ekf_raw_q[ekf_raw_q[:, 0] < 0]
        # ekf_raw_euler = Rotation.from_quat(ekf_raw_q[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        #
        # ekf_pred = gravity_align_EKF(gyr=(pred_gyr_all[1:, :] + pred_gyr_all[:-1, :]) / 2, acc=pred_acc_all,
        #                                acc_cov=acc_cov ** 2,
        #                                noises=[0.05 ** 2],  # big是0.05
        #                                frequency=imu_freq,
        #                                dt_seq=dt[start_step + 1:, 0],  # 中值积分 + 1
        #                                update_rate=10,
        #                                arch='origin',
        #                                Init_state_cov=0.0000000001,
        #                                q0=gt_q[start_step], frame='ENU')
        #
        # ekf_pred_q = ekf_pred.Q / np.linalg.norm(ekf_pred.Q, axis=1, keepdims=True)
        # ekf_pred_q[ekf_pred_q[:, 0] < 0] = (-1) * ekf_pred_q[ekf_pred_q[:, 0] < 0]
        # ekf_pred_euler = Rotation.from_quat(ekf_pred_q[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)

        euler_raw = change_yaw(euler_raw)
        euler_pred = change_yaw(euler_pred)
        euler_gt_q = change_yaw(euler_gt_q)
        # euler_gt_inte_q = change_yaw(euler_gt_inte_q)
        # ekf_raw_euler = change_yaw(ekf_raw_euler)
        # ekf_pred_euler = change_yaw(ekf_pred_euler)

        xlb = 'ts'
        ylbs = ['roll', 'pitch', 'yaw']

        dpi = 90
        figsize = (16, 9)
        fig1 = plt.figure(dpi=dpi, figsize=figsize)
        fig1.suptitle('error euler')
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts,np.abs(euler_raw[:, i] - euler_gt_q[start_step :, i]))
            # plt.plot(ts,np.abs(euler_gt_inte_q[:, i] - euler_gt_q[start_step :, i]))
            plt.plot(ts,np.abs(euler_pred[:, i] - euler_gt_q[start_step :, i]))
            # plt.plot(ts,np.abs(ekf_raw_euler[:, i] - euler_gt_q[start_step :, i]))
            # plt.plot(ts,np.abs(ekf_pred_euler[:, i] - euler_gt_q[start_step :, i])) ,'error_ekf_raw_euler','error_ekf_pred_euler'
            plt.legend(['error_raw_euler','error_pred_euler'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        dpi = 90
        figsize = (16, 9)
        fig2 = plt.figure(dpi=dpi, figsize=figsize)
        fig2.suptitle('euler')
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts,euler_raw[:, i])
            plt.plot(ts,euler_pred[:, i])
            plt.plot(ts,euler_gt_q[:, i])
            # plt.plot(ts,ekf_raw_euler[:, i])
            # plt.plot(ts,ekf_pred_euler[:, i])  'euler_ekf_raw_euler', 'euler_ekf_pred_euler'
            plt.legend(['euler_raw','euler_pred','euler_gt_euler'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        if args.save_result:
            file_names = ['net_gyr','net_acc','ekf_q','error_euler','euler']
            for file_name in file_names:
                if not osp.isdir( out_dir + '/'+file_name+'/'):
                    os.makedirs(out_dir + '/'+file_name+'/')
                    print('create '+ out_dir + '/'+file_name+'/')
            # np.savetxt(out_dir+'/net_gyr/' + data_path + '_gyr.txt', pred_gyr_all, delimiter=',')
            # np.savetxt(out_dir+'/net_acc/' + data_path + '_acc.txt', pred_acc_all, delimiter=',')
            # np.savetxt(out_dir+'/ekf_q/'+data_path + '_ekf_q.txt', ekf_q, delimiter=',')
            fig1.savefig(out_dir+'/error_euler/'+data_path + '_end_error.png')
            fig2.savefig(out_dir+'/euler/'+data_path + '_end.png')

        if args.show_figure:
            fig1.show()
            fig2.show()

        print(data_path)

    print('a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_acc_out_path", type=str, default= "../output/net_acc/")
    parser.add_argument("--network_gyr_out_path", type=str, default= "../output/net_gyr/")
    parser.add_argument("--win_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1256)
    parser.add_argument("--imu_freq", type=int, default=100)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_list", type=str, default="../../dataset/test_bit.txt")
    parser.add_argument("--data_type", type=str, default="txt")  # hdf5
    parser.add_argument("--out_dir", type=str, default="../output")
    parser.add_argument("--root_dir", type=str, default="../../dataset", help="Path to data directory")

    args = parser.parse_args()

    ROTATION_EKF(args)
