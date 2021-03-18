import os
import sys
import time
from argparse import ArgumentParser

import torch as th

import numpy as np

import yaml
from shutil import copyfile

from util.cuda import get_max_available_gpu
from util.run import *

if __name__ == "__main__":
    """
    config
    """
    argp = ArgumentParser()
    argp.add_argument("--config", type=str, help="Config file name")
    argp.add_argument("--save", type=bool, default=False, help="Save model")
    argp.add_argument("--cuda", type=int, help="CUDA device")
    args = argp.parse_args()

    if args.cuda is not None:
        device_id = args.cuda
    else:
        device_id, _ = get_max_available_gpu()
    device = th.device("cuda:" + str(device_id) if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")

    timestr = time.strftime("%m%d-%H%M%S")

    config_file = os.path.join("./", "config", args.config + ".yaml")
    config_yaml = yaml.load(open(config_file), Loader=yaml.FullLoader)

    embedding_config = config_yaml["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    layer_idx = embedding_config["layer"]

    probe_name = config_yaml["probe"]["name"]
    if "Poincare" in probe_name:
        default_dtype = th.float64
        th.set_default_dtype(default_dtype)
    elif "Euclidean" in probe_name:
        default_dtype = th.float32
        th.set_default_dtype(default_dtype)

    log_dir = os.path.join(
        config_yaml["log"]["dir"], task_name, embedding_name, probe_name, timestr
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    copyfile(config_file, os.path.join(log_dir, "config.yaml"))

    if task_name == "distance":
        runDistance(
            config_yaml,
            device,
            default_dtype,
            log_dir=log_dir,
            layer_idx=layer_idx,
            save=args.save,
        )
    elif task_name == "depth":
        runDepth(
            config_yaml,
            device,
            default_dtype,
            log_dir=log_dir,
            layer_idx=layer_idx,
            save=args.save,
        )
    elif task_name == "both":
        runBoth(
            config_yaml,
            device,
            default_dtype,
            log_dir=log_dir,
            layer_idx=layer_idx,
            save=args.save,
        )
