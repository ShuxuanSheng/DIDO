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
import pymap3d as pm

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def convert_lla_to_enu(gt_lla):
    pos_enu = []
    ref_lat = gt_lla[0][0] #deg
    ref_lon = gt_lla[0][1]
    ref_alt = gt_lla[0][2]
    print("ref = ", ref_lat, ref_lon, ref_alt)
    for lla in gt_lla:
        lat, lon, alt = lla
        east, north, up = pm.geodetic2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
        pos_enu.append([east, north, up])
    return np.array(pos_enu)

def euler2quat(gt_rpy):
    q = []
    for rpy in gt_rpy:
        roll, pitch, yaw = rpy
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch)
        yaw_rad = np.deg2rad(yaw)

        quat = Quaternion(axis=[1, 0, 0], angle=roll_rad) * \
            Quaternion(axis=[0, 1, 0], angle=pitch_rad) * \
            Quaternion(axis=[0, 0, 1], angle=yaw_rad)
        # print("rpy bf= ",roll_rad, pitch_rad, yaw_rad)
        # yaw_af,pitch_af,roll_af = q.yaw_pitch_roll
        # print("rpy af= ",roll_af, pitch_af, yaw_af)
        q.append([quat.x, quat.y, quat.z, quat.w])
    return np.array(q)

def output_debias_net(args):

    network_acc_path = args.network_acc_path
    network_gyr_path = args.network_gyr_path
    device = args.device
    win_size = args.win_size
    batch_size = args.batch_size
    imu_freq = args.imu_freq
    data_type = args.data_type
    net_config = {"in_dim": int((win_size * imu_freq) // 4)}

    network_acc = get_model("resnet", net_config, int(3), int(3)).to(device)
    checkpoint = torch.load(network_acc_path, map_location="cuda")
    network_acc.load_state_dict(checkpoint["model_state_dict"])
    network_acc.eval()

    network_gyr = get_model("resnet", net_config, int(3), int(3)).to(device)
    checkpoint = torch.load(network_gyr_path, map_location="cuda")
    network_gyr.load_state_dict(checkpoint["model_state_dict"])
    network_gyr.eval()

    root_dir = args.root_dir
    out_dir = args.out_dir
    test_path = args.test_list
    data_path_s = get_datalist(test_path)

    for data_path in tqdm.tqdm(data_path_s):
        #  hdf5格式的数据
        if (data_type == "hdf5"):
            with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
                ts = np.copy(f["ts"])  # timestamp
                gt_p = np.copy(f["gt_p"])  # position in world frame
                gt_v = np.copy(f["gt_v"])  # velocity in world frame
                gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame   rpy的范围是[-pi,pi]
                gyr = np.copy(f["gyr"])  # unbiased gyr
                acc = np.copy(f["acc"])
                gt_acc = np.copy(f["gt_acc"])
                gt_gyr = np.copy(f["gt_gyr"])
                gt_rpy = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)

        # bit txt格式的数据
        if (data_type == "txt"):
            inspva_file = '/home/meister/SSX/Datasets/bit_compus_1/inspvax.txt'
            inspav_data = np.genfromtxt(inspva_file, delimiter=',', skip_header=1)
            ts = inspav_data[:, 2]

            gt_lla = inspav_data[:, 13:16] #13 lat 、14 lon、15 alt
            # lla -> enu
            gt_p = convert_lla_to_enu(gt_lla)

            gt_vel_neu = inspav_data[:, 17:20]

            gt_rpy = inspav_data[:, 20:23]
            gt_rpy = gt_rpy - np.array([13, 0, 0]) #扣除安装角
            # # rpy [0,2pi]-> [-pi,pi]
            gt_rpy[:,2] = ((gt_rpy[:,2] + 180) % 360) - 180

            gt_q = Rotation.from_euler('xyz', gt_rpy, degrees=True).as_quat()

            imu_file = '/home/meister/SSX/Datasets/bit_compus_1/imu.txt'
            imu_data = np.genfromtxt(imu_file, delimiter=',', skip_header=1)
            acc = imu_data[:, 29:32]
            gyr = imu_data[:, 17:20]
            gt_acc = imu_data[:, 29:32]  # 与DIDO不同，这里gt_acc是i系下的
            gt_gyr = imu_data[:, 17:20]


        if args.plot:
            # plot
            dpi = 90
            figsize = (16, 9)
            xlb = 'East/m'
            ylb = 'North/m'
            fig1 = plt.figure(dpi=dpi, figsize=figsize)
            fig1.suptitle('gt_p')
            plt.plot(gt_p[:, 0], gt_p[:, 1])
            plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig1.savefig(out_dir+'/net_gyr/' + data_path + '_gt_p.png')

            dpi = 90
            figsize = (16, 9)
            xlb = 'ts'
            ylb = 'deg'
            legends = ['raw', 'pitch', 'yaw']
            fig2 = plt.figure(dpi=dpi, figsize=figsize)
            fig2.suptitle('gt_rpy')
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts, gt_rpy[:, i])
                plt.legend(legends[i])
                plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig2.savefig(out_dir+'/net_gyr/' + data_path + '_gt_rpy.png')

            dpi = 90
            figsize = (16, 9)
            xlb = 'ts'
            ylb = 'm/s/s'
            legends = ['x', 'y', 'z']
            fig3 = plt.figure(dpi=dpi, figsize=figsize)
            fig3.suptitle('acc')
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts, acc[1:, i])
                plt.legend(legends[i])
                plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig3.savefig(out_dir+'/net_acc/' + data_path + 'acc.png')

            dpi = 90
            figsize = (16, 9)
            xlb = 'ts'
            ylb = 'rad/s'
            legends = ['x', 'y', 'z']
            fig4 = plt.figure(dpi=dpi, figsize=figsize)
            fig4.suptitle('pred_bg')
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts, gyr[1:, i])
                plt.legend(legends[i])
                plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig4.savefig(out_dir+'/net_gyr/' + data_path + '_gyr.png')

        # 我们没有gt_acc，imu中的acc跟inspvax中的gt_R维度不匹配
        # 在DIDO数据集中，gt_acc是b系相对于重力系的加速度，也就是说是扣除了重力加速度的，
        # print(gt_R.shape) #(24046, 3, 3)
        # print(acc.shape) #(24046, 3)
        # tpi表示第一个张量的维度，t是第一个维度，p是第二个维度，i是第三个维度。
        # tp: 表示第二个张量的维度，t 是第一个维度，p 是第二个维度。
        # ->ti: 表示输出张量的维度，其中t和i是保留的维度
        # einsum对在两个张量中共有的维度（这里是 t 和 p）进行求和，保留指定的维度（在此例中为 t 和 i），它们将成为输出张量的维度
        # gt_R = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_matrix()
        # gt_acc = np.einsum("tpi,tp->ti", gt_R, gt_acc + np.array([0, 0, 9.8]))  # 特别注意 jcx 而且是旋转的转置

        dt = np.diff(ts)
        dt = np.append(dt[0], dt).reshape(-1, 1)

        acc = acc.astype(np.float32)
        batch_acc = []
        batch_gyr = []

        start_step = 0

        for i in range(start_step, int(acc.shape[0] - (imu_freq * win_size))):
            batch_acc.append(acc[i:i + int(imu_freq * win_size)])
            batch_gyr.append(gyr[i:i + int(imu_freq * win_size)])

        batch_acc = np.array(batch_acc).astype(np.float32)
        batch_acc = torch.tensor(batch_acc).to(device)
        batch_acc = batch_acc.permute(0, 2, 1)

        batch_gyr = np.array(batch_gyr).astype(np.float32)
        batch_gyr = torch.tensor(batch_gyr).to(device)
        batch_gyr = batch_gyr.permute(0, 2, 1)

        ba_all = []
        bg_all = []
        for i in range(int(batch_acc.shape[0] // batch_size)):
            ba = network_acc(batch_acc[i * batch_size: (i + 1) * batch_size])
            ba_all.append(torch_to_numpy(ba))

            bg = network_gyr(batch_gyr[i * batch_size: (i + 1) * batch_size])
            bg_all.append(torch_to_numpy(bg))

        # 如果batch_acc的长度不是batch_size的整数倍，则需要处理最后剩余的数据。将这些数据通过network_acc进行预测，并将结果添加到ba_all列表。最后，将ba_all中的所有预测结果连接成一个大数组。
        # 这里调用训练好的network_acc进行预测
        ba = network_acc(batch_acc[(int(batch_acc.shape[0] // batch_size)) * batch_size:])
        ba_all.append(torch_to_numpy(ba))
        ba_all = np.concatenate(ba_all, axis=0)

        bg = network_gyr(batch_gyr[(int(batch_acc.shape[0] // batch_size)) * batch_size:])
        bg_all.append(torch_to_numpy(bg))
        bg_all = np.concatenate(bg_all, axis=0)

        pred_a = acc[start_step + int(win_size * imu_freq):] + ba_all  #扣除掉ba后的acc
        pred_w = gyr[start_step + int(win_size * imu_freq):] + bg_all

        pred_gyr_all = np.concatenate((gt_gyr[:int(win_size * imu_freq)], pred_w), axis=0) #将gt_gyr和经过网络预测的pred_gyr连接起来

        # print(acc.shape) #(24046, 3)
        # print(acc[start_step + int(win_size * imu_freq):].shape) #(23646, 3)
        # print(ba_all.shape) # (23646, 3)
        # print(pred_w.shape) # (23646, 3)
        # print (gt_gyr.shape) # (24046, 3)
        # print(gt_gyr[:int(win_size * imu_freq)].shape) #(400, 3)
        # print(pred_w.shape) #(23646, 3)
        # print(pred_gyr_all.shape) #(24046, 3)  pred_gyr_all的前400是gt_gyr,后23646是pred_w
        pred_acc_all = np.concatenate((gt_acc[:int(win_size * imu_freq)], pred_a), axis=0)

        if args.plot:
            dpi = 90
            figsize = (16, 9)
            xlb = 'ts'
            ylb = 'm/s/s'
            legends = ['x', 'y', 'z']
            fig5 = plt.figure(dpi=dpi, figsize=figsize)
            fig5.suptitle('pred_ba')
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts[400:], ba_all[start_step+1:, i])
                plt.legend(legends[i])
                plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig5.savefig(out_dir+'/net_acc/' + data_path + '_pred_ba.png')

            dpi = 90
            figsize = (16, 9)
            xlb = 'ts'
            ylb = 'rad/s'
            legends = ['x', 'y', 'z']
            fig6 = plt.figure(dpi=dpi, figsize=figsize)
            fig6.suptitle('pred_bg')
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts[400:], bg_all[start_step+1:, i])
                plt.legend(legends[i])
                plt.ylabel(ylb)
            plt.xlabel(xlb)
            fig6.savefig(out_dir+'/net_gyr/' + data_path + '_pred_bg.png')

        if args.save_result:
            file_names = ['net_gyr','net_acc']
            for file_name in file_names:
                if not osp.isdir( out_dir + '/'+file_name+'/'):
                    os.makedirs(out_dir + '/'+file_name+'/')
                    print('create '+ out_dir + '/'+file_name+'/')
            np.savetxt(out_dir+'/net_gyr/' + data_path + '_pred_gyr.txt', pred_gyr_all, delimiter=',')
            np.savetxt(out_dir+'/net_acc/' + data_path + '_pred_acc.txt', pred_acc_all, delimiter=',')
            np.savetxt(out_dir+'/net_gyr/' + data_path + '_raw_gyr.txt', gyr, delimiter=',')
            np.savetxt(out_dir+'/net_acc/' + data_path + '_raw_acc.txt', acc, delimiter=',')
            np.savetxt(out_dir+'/net_gyr/' + data_path + '_gt_gyr.txt', gt_gyr, delimiter=',')
            np.savetxt(out_dir+'/net_acc/' + data_path + '_gt_acc.txt', gt_acc, delimiter=',')

    print('a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_acc_path", type=str, default= "../../De_bias_acc/train_outputs/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--network_gyr_path", type=str, default="../../De_bias_gyr/train_outputs/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--win_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1256)
    parser.add_argument("--imu_freq", type=int, default=400)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--show_figure", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_list", type=str, default="../../dataset/test_bit.txt")
    parser.add_argument("--data_type", type=str, default="hdf5") #hdf5
    parser.add_argument("--out_dir", type=str, default="../output")
    parser.add_argument("--root_dir", type=str, default="../../dataset", help="Path to data directory")

    args = parser.parse_args()

    output_debias_net(args)
