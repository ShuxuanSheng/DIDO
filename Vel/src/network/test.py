"""
This file includes the main libraries in the network testing module
"""

import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.losses import get_loss
from network.model_factory import get_model
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from utils.logging import logging
from utils.math_utils import *

import numpy as np
import scipy
import pickle


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer(): # 判断 过去时间*imu_freq 是否是整数 (imu数据输入size必须为整数)
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),    # (0.05-0.0025)*400=19
            ("window_size", int(args.window_time * args.imu_freq)),     # 0.0025*400=1
            ("future_data_size", int(args.future_time * args.imu_freq)), # 0*400=0
            ("step_size", int(args.imu_freq / args.sample_freq)),       # 400/100=4
        ]
    )

    net_config = {
        "inter_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 4

    }

    return data_window_config, net_config

def jifen_v(dt,acc_inter,device,args):
    if len(acc_inter.shape) == 4:  # acc_inter [ seq，batch， ， ] 所以如果这个seq是被pad的，那么shape[2]对应的都是0
        dt_temp = dt.to(device)
        acc_inter_temp = acc_inter.to(device)
        acc_inter_temp = acc_inter_temp[:, :,:, int(args.past_time * args.imu_freq):].to(device)
        gravity = torch.tensor([[[[0, 0, args.gravity]]]]).repeat((acc_inter.shape[0],acc_inter.shape[1], acc_inter.shape[2], 1)).to(device)  # blackbird是 +9.8
        # dt = np.expand_dims(dt, 1).repeat(3, axis=1)
        acc_no_g = acc_inter_temp[:, :, :,:].to(device) + gravity
        temp = (acc_no_g.shape[0],acc_no_g.shape[1],acc_no_g.shape[2],1)
        acc_no_g = torch.where((acc_no_g[:,:,:,0] == 0).reshape(temp) * (acc_no_g[:,:,:,1] == 0).reshape(temp) * (acc_no_g[:,:,:,2] == -9.8).reshape(temp), torch.full_like(acc_no_g,0), acc_no_g)
        delta_v = acc_no_g * dt_temp.repeat(1,acc_inter.shape[1],1).unsqueeze(3)
        delta_v_integrate = torch.sum(delta_v[:, :, :,:], axis=2)
    else:
        dt_temp = dt.to(device)
        if acc_inter.shape[2] == 3:
            acc_inter_temp = acc_inter.permute(0,2,1).to(device)
        else:
            acc_inter_temp = acc_inter.to(device)
        acc_inter_temp = acc_inter_temp[:,:,int(args.past_time*args.imu_freq):].to(device)
        gravity = torch.tensor([[[0], [0], [args.gravity]]]).repeat((acc_inter.shape[0],1,1)).to(device) # blackbird是 +9.8
        # dt = np.expand_dims(dt, 1).repeat(3, axis=1)
        delta_v = ( acc_inter_temp[:, :, :].to(device) + gravity ) * dt_temp[:, :, :]
        delta_v_integrate = torch.sum(delta_v[:,:,:], axis=2)
    return delta_v_integrate

def q2r(q):
    if len(q.shape) == 3:
        r = torch.zeros(q.shape[0],q.shape[1],3,3).to(q.device)
        r[:,:,0,0] = 1-2*q[:,:,2]**2-2*q[:,:,3]**2
        r[:,:,0,1] = 2*q[:,:,1]*q[:,:,2] - 2*q[:,:,0]*q[:,:,3]
        r[:,:,0,2] = 2*q[:,:,1]*q[:,:,3] + 2*q[:,:,0]*q[:,:,2]
        r[:,:,1,0] = 2*q[:,:,1]*q[:,:,2] + 2*q[:,:,0]*q[:,:,3]
        r[:,:, 1, 1] = 1-2*q[:,:,1]**2-2*q[:,:,3]**2
        r[:,:, 1, 2] = 2 * q[:, :,2] * q[:, :,3] - 2 * q[:, :,0] * q[:, :,1]
        r[:,:, 2, 0] = 2 * q[:, :,1] * q[:, :,3] - 2 * q[:, :,0] * q[:, :,2]
        r[:,:, 2, 1] = 2 * q[:, :,2] * q[:, :,3] + 2 * q[:, :,0] * q[:, :,1]
        r[:,:, 2, 2] = 1 - 2 * q[:, :,1]**2 - 2 * q[:, :,2]**2

    elif len(q.shape) == 4:
        r = torch.zeros(q.shape[0],q.shape[1],q.shape[2],3,3).to(q.device)
        r[:,:,:,0,0] = 1-2*q[:,:,:,2]**2-2*q[:,:,:,3]**2
        r[:,:,:,0,1] = 2*q[:,:,:,1]*q[:,:,:,2] - 2*q[:,:,:,0]*q[:,:,:,3]
        r[:,:,:,0,2] = 2*q[:,:,:,1]*q[:,:,:,3] + 2*q[:,:,:,0]*q[:,:,:,2]
        r[:,:,:,1,0] = 2*q[:,:,:,1]*q[:,:,:,2] + 2*q[:,:,:,0]*q[:,:,:,3]
        r[:,:, :,1, 1] = 1-2*q[:,:,:,1]**2-2*q[:,:,:,3]**2
        r[:,:,:, 1, 2] = 2 * q[:, :,:,2] * q[:, :,:,3] - 2 * q[:, :,:,0] * q[:, :,:,1]
        r[:,:,:, 2, 0] = 2 * q[:, :,:,1] * q[:, :,:,3] - 2 * q[:, :,:,0] * q[:, :,:,2]
        r[:,:, :,2, 1] = 2 * q[:, :,:,2] * q[:, :,:,3] + 2 * q[:, :,:,0] * q[:, :,:,1]
        r[:,:, :,2, 2] = 1 - 2 * q[:, :,:,1]**2 - 2 * q[:, :,:,2]**2

    return r

def plot_result(attr_dict):
    # xlb = 'ts'
    # ylbs = ['x', 'y', 'z']
    # dpi = 90
    # figsize = (16, 9)
    plt.figure()
    plt.subplot(2, 1,1)
    plt.plot(attr_dict["preds"][:,0], label="preds_x")
    plt.plot(attr_dict["targets"][:,0], label="targets_x")
    # plt.plot(attr_dict["smooth_preds"][:,0], label="smooth_preds_x")
    plt.plot(attr_dict["raw_wheel_left"], label="raw_wheel_left")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1,2)
    plt.plot(attr_dict["preds"][:,1], label="preds_y")
    # plt.plot(attr_dict["preds"][:,2], label="preds_z")
    plt.plot(attr_dict["targets"][:,1], label="targets_y")
    # plt.plot(attr_dict["targets"][:,2], label="targets_z")
    # plt.plot(attr_dict["smooth_preds"][:,1], label="smooth_preds_y")
    # plt.plot(attr_dict["smooth_preds"][:,2], label="smooth_preds_z")

    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.subplot(2, 1,1)
    plt.plot(attr_dict["preds_cov"][:,0], label="preds_cov_x")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1,2)
    plt.plot(attr_dict["preds_cov"][:,1], label="preds_cov_y")
    plt.plot(attr_dict["preds_cov"][:,2], label="preds_cov_z")

    plt.legend()
    plt.grid()
    plt.show()
    # data_vp = np.concatenate((ts, v, p), axis=1)
    # np.savetxt(args.out_dir + '/vp/' + data_name + '_vp.txt', data_vp, delimiter=',')
def get_inference(network, data_loader, device, epoch,args):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all, smoothed_v, raw_wheel_left = [], [], [], [], [], []
    network.eval()

    for bid, (feat, targ, wheel_left, wheel_right, yaw, _, _) in enumerate(data_loader):
        net_dirs = args.x_model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(net_dirs, map_location=device)
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()

        feat, targ, wheel_left, wheel_right = feat.to(device), targ.to(device), wheel_left.to(device), wheel_right.to(device)

        pred, pred_cov = network(feat.to(device))

        zeros_col = torch.zeros((pred.shape[0], 2)).to(device)
        wheel_v_combined = torch.concatenate((wheel_left.unsqueeze(1), zeros_col), dim=1)

        pred = pred + wheel_v_combined
        """
        if torch.cuda.is_available():
            # 获取当前 GPU 的总内存（以字节为单位）
            total_memory = torch.cuda.get_device_properties(device).total_memory
            # 获取当前已分配的 GPU 内存（以字节为单位）
            allocated_memory = torch.cuda.memory_allocated(device)
            # 获取当前缓存的 GPU 内存（以字节为单位）
            cached_memory = torch.cuda.memory_reserved(device)
            # 计算可用内存（以字节为单位）
            available_memory = total_memory - allocated_memory - cached_memory

            # 将字节转换为 MiB
            total_memory_mib = total_memory / (1024 ** 2)
            allocated_memory_mib = allocated_memory / (1024 ** 2)
            cached_memory_mib = cached_memory / (1024 ** 2)
            available_memory_mib = available_memory / (1024 ** 2)

            print(f"Total GPU memory: {total_memory_mib:.2f} MiB")
            print(f"Allocated GPU memory: {allocated_memory_mib:.2f} MiB")
            print(f"Cached GPU memory: {cached_memory_mib:.2f} MiB")
            print(f"Available GPU memory: {available_memory_mib:.2f} MiB")
        else:
            print("No CUDA device available.")
        """


        loss = get_loss(pred, pred_cov, targ, epoch).to(device)

        # 将不同batch的结果合并起来
        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        preds_cov_all.append(torch_to_numpy(pred_cov))
        losses_all.append(torch_to_numpy(loss))
        raw_wheel_left.append(torch_to_numpy(wheel_left))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    raw_wheel_left = np.concatenate(raw_wheel_left, axis=0)
    logging.info(f"preds_all shape = {preds_all.shape}")

    # 低通滤波
    fs = 2000  # 采样频率（Hz）
    cutoff_freq = 5  # 截止频率（Hz）
    # 计算归一化截止频率
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    # 设计 4 阶巴特沃斯低通滤波器
    b, a = scipy.signal.butter(4, normal_cutoff, btype='low', analog=False)
    smoothed_v = np.zeros_like(preds_all)

    b = b.astype(np.double)
    a = a.astype(np.double)
    preds_all = preds_all.astype(np.double)
    for i in range(pred.shape[1]):
        smoothed_v[:, i] = scipy.signal.lfilter(b, a, preds_all[:, i])

    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
        "smooth_preds": smoothed_v,
        "raw_wheel_left":raw_wheel_left}

    return attr_dict

def net_test(args):
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.test_list is None:
            raise ValueError("test_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir + '/'+'imu_p_cov_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_cov_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_cov_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_p_error_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_error_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_error_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_p_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_cov_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_cov_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_cov_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_error_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_error_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_error_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'vp'):
                os.makedirs(args.out_dir + '/'+'vp')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'vp'}")
            if not osp.isdir(args.out_dir + '/'+'vp_cov'):
                os.makedirs(args.out_dir + '/'+'vp_cov')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'vp_cov'}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(device)

    # initialize containers
    all_metrics = {}

    test_list = get_datalist(args.test_list)
    train_loader_list = []
    for data in test_list:
        f_name_vel = os.path.join(args.results_dir, data + '_vel.p')
        logging.info(f"save net predict vel ： {f_name_vel}")
        data = [data]
        try:
            seq_dataset = FbSequenceDataset(args.root_dir, data, args, data_window_config, mode="test")
            seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        except OSError as e:
            logging.error(e)
            return
        # Obtain trajectory
        attr_dict = get_inference(network, seq_loader, device, 500000000, args)

        with open(f_name_vel, "wb") as f:
            pickle.dump(attr_dict, f)

        plot_result(attr_dict)
    return
