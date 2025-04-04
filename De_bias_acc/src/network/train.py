"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp

import numpy as np
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.losses import get_loss
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from scipy.spatial.transform import Rotation


def q2r(q):
    if len(q.shape) == 3:
        r = torch.zeros(q.shape[0], q.shape[1], 3, 3).to("cuda")
        r[:, :, 0, 0] = 1 - 2 * q[:, :, 2] ** 2 - 2 * q[:, :, 3] ** 2
        r[:, :, 0, 1] = 2 * q[:, :, 1] * q[:, :, 2] - 2 * q[:, :, 0] * q[:, :, 3]
        r[:, :, 0, 2] = 2 * q[:, :, 1] * q[:, :, 3] + 2 * q[:, :, 0] * q[:, :, 2]
        r[:, :, 1, 0] = 2 * q[:, :, 1] * q[:, :, 2] + 2 * q[:, :, 0] * q[:, :, 3]
        r[:, :, 1, 1] = 1 - 2 * q[:, :, 1] ** 2 - 2 * q[:, :, 3] ** 2
        r[:, :, 1, 2] = 2 * q[:, :, 2] * q[:, :, 3] - 2 * q[:, :, 0] * q[:, :, 1]
        r[:, :, 2, 0] = 2 * q[:, :, 1] * q[:, :, 3] - 2 * q[:, :, 0] * q[:, :, 2]
        r[:, :, 2, 1] = 2 * q[:, :, 2] * q[:, :, 3] + 2 * q[:, :, 0] * q[:, :, 1]
        r[:, :, 2, 2] = 1 - 2 * q[:, :, 1] ** 2 - 2 * q[:, :, 2] ** 2

    elif len(q.shape) == 4:
        r = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 3, 3).to("cuda")
        r[:, :, :, 0, 0] = 1 - 2 * q[:, :, :, 2] ** 2 - 2 * q[:, :, :, 3] ** 2
        r[:, :, :, 0, 1] = 2 * q[:, :, :, 1] * q[:, :, :, 2] - 2 * q[:, :, :, 0] * q[:, :, :, 3]
        r[:, :, :, 0, 2] = 2 * q[:, :, :, 1] * q[:, :, :, 3] + 2 * q[:, :, :, 0] * q[:, :, :, 2]
        r[:, :, :, 1, 0] = 2 * q[:, :, :, 1] * q[:, :, :, 2] + 2 * q[:, :, :, 0] * q[:, :, :, 3]
        r[:, :, :, 1, 1] = 1 - 2 * q[:, :, :, 1] ** 2 - 2 * q[:, :, :, 3] ** 2
        r[:, :, :, 1, 2] = 2 * q[:, :, :, 2] * q[:, :, :, 3] - 2 * q[:, :, :, 0] * q[:, :, :, 1]
        r[:, :, :, 2, 0] = 2 * q[:, :, :, 1] * q[:, :, :, 3] - 2 * q[:, :, :, 0] * q[:, :, :, 2]
        r[:, :, :, 2, 1] = 2 * q[:, :, :, 2] * q[:, :, :, 3] + 2 * q[:, :, :, 0] * q[:, :, :, 1]
        r[:, :, :, 2, 2] = 1 - 2 * q[:, :, :, 1] ** 2 - 2 * q[:, :, :, 2] ** 2

    return r

    """
        对acc积分得到速度变化
    """
def jifen_v(dt, acc_inter, device, args, k=1, r=None):
    dt_temp = dt.to(device)
    if acc_inter.shape[2] == 3:
        acc_inter_temp = acc_inter.permute(0, 2, 1).to(device)
    else:
        acc_inter_temp = acc_inter.to(device)
    acc_inter_temp = acc_inter_temp[:, :, int(args.past_time * args.imu_freq):].to(device) #根据 args.past_time 和 args.imu_freq 计算出需要的时间段，并从 acc_inter_temp 中截取该时间段的数据
    dt_temp = dt_temp[:, :, int(args.past_time * args.imu_freq):].to(device)
    gravity = torch.tensor([[[0], [0], [- 9.8]]]).repeat((acc_inter.shape[0], 1, 1)).to(device)  # blackbird是 +9.8
    # dt = np.expand_dims(dt, 1).repeat(3, axis=1)
    if r is None:
        gravity_new = gravity.clone()
        if type(k) == int:
            gravity_new[:, 2, :] = k * gravity[:, 2, :]
        else:
            gravity_new[:, 2, :] = k.unsqueeze(1) * gravity[:, 2, :]
    else:
        r = r.to(device)
        gravity_r = torch.einsum("tip,tpk->tik", r, gravity)
        gravity_new = gravity_r.clone()
        if type(k) == int:
            gravity_new[:, 2, :] = k * gravity_r[:, 2, :]
        else:
            gravity_new[:, 2, :] = k.unsqueeze(1) * gravity_r[:, 2, :]

    # delta_v = (0.5 * (acc_inter_temp[:,:,:-1] + acc_inter_temp[:,:,1:]).to(device) +  gravity_new) * dt_temp[:,:,:-1]
    delta_v = (acc_inter_temp[:, :, :].to(device) + gravity_new) * dt_temp[:, :, :]
    delta_v_integrate = torch.sum(delta_v[:, :, :], axis=2)   #一个区间预测的速度(积分加速度)
    return delta_v_integrate


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def g_rotation(roll, pitch, batch_size):
    g_r = torch.eye(3).repeat(batch_size, 1, 1)
    g_roll = g_r.clone()
    g_roll[:, 1, 1] = torch.cos(roll)
    g_roll[:, 2, 2] = torch.cos(roll)
    g_roll[:, 1, 2] = torch.sin(roll)
    g_roll[:, 2, 1] = - torch.sin(roll)

    g_pitch = g_r.clone()
    g_pitch[:, 0, 0] = torch.cos(pitch)
    g_pitch[:, 2, 2] = torch.cos(pitch)
    g_pitch[:, 2, 0] = torch.sin(pitch)
    g_pitch[:, 0, 2] = - torch.sin(pitch)

    return torch.einsum("tik,tkj -> tij", g_roll, g_pitch)


def get_inference(network, data_loader, device, epoch, args):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, (feat, targ, ts_inter, ori_r, _, _) in enumerate(data_loader):
        if args.arch == "resnet":
            feat, targ = feat.to(device), targ.to(device)
            ts_inter = ts_inter.unsqueeze(1)
            ori_r = ori_r.to(device)

            pred = network(feat[:, 3:6, :].to(device))

            a_body = (feat[:, 3:6, :].to(device) + pred.unsqueeze(2).repeat(1, 1, feat[:, 3:6, :].shape[2])).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            a_world = torch.einsum("atip,atp->ati", ori_r, a_body).permute(0, 2, 1)

            pred = jifen_v(ts_inter, a_world, device, args, r=None, k=1).to(device)

            loss = get_loss(pred, _, targ, epoch).to(device)  # targ是gt_dv

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        losses_all.append(torch_to_numpy(loss))

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "losses": losses_all,
    }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, args):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    # 遍历train_loader 中的每个批次。每个批次包含特征 feat、目标值 targ、时间间隔 ts_inter、原始旋转 ori_r等数据
    # 这些数据与在DataSet中的__getitem__中定义的返回值一致
    for bid, (feat, targ, ts_inter, ori_r, _, _) in enumerate(train_loader):
        if args.arch == "resnet":
            feat, targ = feat.to(device), targ.to(device)
            optimizer.zero_grad()
            ts_inter = ts_inter.unsqueeze(1)
            ori_r = ori_r.to(device)

            pred = network(feat[:, 3:6, :].to(device))  #把acc输入到网络中，进行前向传播，传播后的是de-bias的acc?
            # 从 feat 张量中提取通道 3 到 5 的数据；将 pred 张量扩展到与 feat 张量相同的长度；将 feat 张量和扩展后的 pred 张量逐元素相加；重新排列张量的维度顺序
            a_body = (feat[:, 3:6, :].to(device) + pred.unsqueeze(2).repeat(1, 1, feat[:, 3:6, :].shape[2])).permute(0,2,1)

            a_world = torch.einsum("atip,atp->ati", ori_r, a_body).permute(0, 2, 1) # 对 ori_r 张量中的每个元素和 a_body 张量中的对应元素进行逐元素相乘、相加

            pred = jifen_v(ts_inter, a_world, device, args, r=None, k=1).to(device)

            loss = get_loss(pred, _, targ, epoch).to(device)

        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_losses.append(torch_to_numpy(loss))

        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }
    return train_attr_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode, args):
    """ Given the attr_dict write summary and log the losses """
    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    ml_loss = np.average(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_z", mse_loss[2], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", np.mean(mse_loss), epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {mse_loss}/{np.mean(mse_loss)}"
    )


def save_model(args, epoch, network, optimizer, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
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
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
                          data_window_config["past_data_size"]
                          + data_window_config["window_size"]
                          + data_window_config["future_data_size"]
                  )
                  // 4  ## jcx
    }

    return data_window_config, net_config


def net_train(args):
    """
    Main function for network training
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.train_list is None:
            raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                    os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        if args.val_list is None:
            logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info("Size: "
                 + str(data_window_config["past_data_size"])
                 + "+"
                 + str(data_window_config["window_size"])
                 + "+"
                 + str(data_window_config["future_data_size"])
                 + ", "
                 + "Time: "
                 + str(args.past_time)
                 + "+"
                 + str(args.window_time)
                 + "+"
                 + str(args.future_time))
    logging.info("Sample frequency: %s" % args.sample_freq)

    train_loader, val_loader = None, None
    start_t = time.time()
    train_list = get_datalist(args.train_list)
    try:
        train_dataset = FbSequenceDataset(args.root_dir, train_list, args, data_window_config, mode="train")
        # DataLoader根据train_dataset中定义的__getitem__ 和 __len__ 方法来加载数据，并返回
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    except OSError as e:
        logging.error(e)
        return
    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(train_dataset)}")

    if args.val_list is not None:
        val_list = get_datalist(args.val_list)
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(val_dataset)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(device)

    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(f"Detected saved checkpoint, starting from epoch {start_epoch}")

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")

    attr_dict = get_inference(network, train_loader, device, start_epoch, args)

    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train", args)

    if val_loader is not None:
        attr_dict = get_inference(network, val_loader, device, start_epoch, args)
        write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val", args)

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer))
        signal.signal(signal.SIGTERM, partial(stop_signal_handler, args, epoch, network, optimizer), )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()

        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args)

        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train", args)
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if val_loader is not None:
            val_attr_dict = get_inference(network, val_loader, device, epoch, args)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val", args)
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer)
            # save_model(args, epoch, network, optimizer)
        else:
            save_model(args, epoch, network, optimizer)

    logging.info("Training complete.")

    return
