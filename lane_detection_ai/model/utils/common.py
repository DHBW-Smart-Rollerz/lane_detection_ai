import argparse

import torch

from lane_detection_ai.model.utils.config import Config
from lane_detection_ai.model.utils.dist_utils import is_main_process


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--data_root", default=None, type=str)
    parser.add_argument("--epoch", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--optimizer", default=None, type=str)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument("--weight_decay", default=None, type=float)
    parser.add_argument("--momentum", default=None, type=float)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--steps", default=None, type=int, nargs="+")
    parser.add_argument("--gamma", default=None, type=float)
    parser.add_argument("--warmup", default=None, type=str)
    parser.add_argument("--warmup_iters", default=None, type=int)
    parser.add_argument("--backbone", default=None, type=str)
    parser.add_argument("--griding_num", default=None, type=int)
    parser.add_argument("--use_aux", default=None, type=str2bool)
    parser.add_argument("--sim_loss_w", default=None, type=float)
    parser.add_argument("--shp_loss_w", default=None, type=float)
    parser.add_argument("--note", default=None, type=str)
    parser.add_argument("--log_path", default=None, type=str)
    parser.add_argument("--finetune", default=None, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--test_model", default=None, type=str)
    parser.add_argument("--test_work_dir", default=None, type=str)
    parser.add_argument("--num_lanes", default=None, type=int)
    parser.add_argument(
        "--auto_backup",
        action="store_false",
        help="automatically backup current code in the log path",
    )
    parser.add_argument("--var_loss_power", default=None, type=float)
    parser.add_argument("--num_row", default=None, type=int)
    parser.add_argument("--num_col", default=None, type=int)
    parser.add_argument("--train_width", default=None, type=int)
    parser.add_argument("--train_height", default=None, type=int)
    parser.add_argument("--num_cell_row", default=None, type=int)
    parser.add_argument("--num_cell_col", default=None, type=int)
    parser.add_argument("--mean_loss_w", default=None, type=float)
    parser.add_argument("--fc_norm", default=None, type=str2bool)
    parser.add_argument("--soft_loss", default=None, type=str2bool)
    parser.add_argument("--cls_loss_col_w", default=None, type=float)
    parser.add_argument("--cls_ext_col_w", default=None, type=float)
    parser.add_argument("--mean_loss_col_w", default=None, type=float)
    parser.add_argument("--eval_mode", default=None, type=str)
    parser.add_argument("--eval_during_training", default=None, type=str2bool)
    parser.add_argument("--split_channel", default=None, type=str2bool)
    parser.add_argument(
        "--match_method", default=None, type=str, choices=["fixed", "hungarian"]
    )
    parser.add_argument("--selected_lane", default=None, type=int, nargs="+")
    parser.add_argument("--cumsum", default=None, type=str2bool)
    parser.add_argument("--masked", default=None, type=str2bool)

    return parser


import numpy as np


def get_config(path: str):
    cfg = Config.fromfile(path)

    if cfg.dataset == "CULane":
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Tusimple":
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "CurveLanes":
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Smartrollerz":
        cfg.row_anchor = np.linspace(100, 1540, cfg.num_row) / 1550
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    return cfg


def save_model(net, optimizer, epoch, save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {"model": model_state_dict, "optimizer": optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        if epoch is None:
            model_path = os.path.join(save_path, "best_model.pth")
        else:
            model_path = os.path.join(save_path, f"model_epoch{epoch}.pth")
        torch.save(state, model_path)


import datetime
import os


def get_work_dir(cfg):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyper_param_str = "_lr_%1.0e_b_%d" % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)


def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print("unkonwn module", m)


import importlib


def get_model(cfg):
    return importlib.import_module(
        "lane_detection_ai.model.model.model_" + cfg.dataset.lower()
    ).get_model(cfg)


def inference(net, data_label, dataset):
    if dataset == "CurveLanes":
        return inference_curvelanes(net, data_label)
    elif dataset in ["Tusimple", "CULane", "Smartrollerz"]:
        return inference_culane_tusimple(net, data_label)
    else:
        raise NotImplementedError


def inference_culane_tusimple(net, data_label):
    pred = net(data_label["images"])
    cls_out_ext_label = (data_label["labels_row"] != -1).long()
    cls_out_col_ext_label = (data_label["labels_col"] != -1).long()
    res_dict = {
        "cls_out": pred["loc_row"],
        "cls_label": data_label["labels_row"],
        "cls_out_col": pred["loc_col"],
        "cls_label_col": data_label["labels_col"],
        "cls_out_ext": pred["exist_row"],
        "cls_out_ext_label": cls_out_ext_label,
        "cls_out_col_ext": pred["exist_col"],
        "cls_out_col_ext_label": cls_out_col_ext_label,
        "labels_row_float": data_label["labels_row_float"],
        "labels_col_float": data_label["labels_col_float"],
    }
    if "seg_out" in pred.keys():
        res_dict["seg_out"] = pred["seg_out"]
        res_dict["seg_label"] = data_label["seg_images"]

    return res_dict


def inference_curvelanes(net, data_label):
    pred = net(data_label["images"])
    cls_out_ext_label = (data_label["labels_row"] != -1).long()
    cls_out_col_ext_label = (data_label["labels_col"] != -1).long()

    res_dict = {
        "cls_out": pred["loc_row"],
        "cls_label": data_label["labels_row"],
        "cls_out_col": pred["loc_col"],
        "cls_label_col": data_label["labels_col"],
        "cls_out_ext": pred["exist_row"],
        "cls_out_ext_label": cls_out_ext_label,
        "cls_out_col_ext": pred["exist_col"],
        "cls_out_col_ext_label": cls_out_col_ext_label,
        "seg_label": data_label["seg_images"],
        "seg_out_row": pred["lane_token_row"],
        "seg_out_col": pred["lane_token_col"],
    }
    if "seg_out" in pred.keys():
        res_dict["seg_out"] = pred["seg_out"]
        res_dict["seg_label"] = data_label["segs"]
    return res_dict


def calc_loss(loss_dict, results, logger, global_step, epoch):
    loss = 0

    for i in range(len(loss_dict["name"])):
        if loss_dict["weight"][i] == 0:
            continue

        data_src = loss_dict["data_src"][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict["op"][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar("loss/" + loss_dict["name"][i], loss_cur, global_step)

        loss += loss_cur * loss_dict["weight"][i]

    return loss
