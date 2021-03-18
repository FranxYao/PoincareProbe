import sys

sys.path.append("..")

import os
import time
import pickle

import torch as th
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dill
from tqdm import tqdm

from probe.euclid import *
from probe.hyper import *
from .data import custom_pad
from .loss import *
from .evalu import *
from .train import *


def constructProbe(probe_config, device, default_dtype):
    probe_name = probe_config["name"]
    Probe = getattr(sys.modules[__name__], probe_name)

    if "Euclidean" in probe_name:
        return Probe(
            device=device,
            dim_in=probe_config["dim_in"],
            dim_out=probe_config["dim_out"],
            default_dtype=default_dtype,
        )
    elif "Poincare" in probe_name:
        return Probe(
            device=device,
            curvature=probe_config["c"],
            dim_in=probe_config["dim_in"],
            dim_out=probe_config["dim_out"],
            dim_hidden=probe_config["dim_hidden"],
            default_dtype=default_dtype,
        )


def runDistance(
    config, device, default_dtype, log_dir: str, layer_idx: int, save: bool
):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_dataset_path = os.path.join(
        embedding_dir, task_name, embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_dataset = th.load(train_dataset_path, pickle_module=dill)
    dev_dataset = th.load(dev_dataset_path, pickle_module=dill)
    test_dataset = th.load(test_dataset_path, pickle_module=dill)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_uuas, avg_distance_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        distance_loss = L1DistanceLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_loss, dev_loss = train(
                train_data_loader,
                probe,
                distance_loss,
                optimizer,
                task="distance",
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                # metrics on dev
                dev_distance_prediction_batches =\
                    prediction(dev_data_loader, probe, task="distance")
                _, gold_edge_lens, pred_edge_lens, gold_edge_recall, gold_edge_cnt =\
                    reportUUAS(dev_distance_prediction_batches, dev_data_loader)
                pickle.dump(gold_edge_lens, open(log_dir + '/gold_edge_run%d_epoch%d_lens.pkl' % (run_idx, epoch), 'wb'))
                pickle.dump(pred_edge_lens, open(log_dir + '/pred_edge_run%d_epoch%d_lens.pkl' % (run_idx, epoch), 'wb'))
                pickle.dump(gold_edge_cnt, open(log_dir + '/gold_edge_run%d_epoch%d_cnt.pkl' % (run_idx, epoch), 'wb'))
                pickle.dump(gold_edge_recall, open(log_dir + '/pred_edge_run%d_epoch%d_recall.pkl' % (run_idx, epoch), 'wb'))

                # metrics on test
                start_time = time.time()
                test_distance_prediction_batches = prediction(
                    test_data_loader, probe, task="distance"
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_uuas, gold_edge_lens, pred_edge_lens, gold_edge_recall, gold_edge_cnt = reportUUAS(
                    test_distance_prediction_batches, test_data_loader
                )
                
                test_distance_dspr, test_distance_dspr_list = reportDistanceSpearmanr(
                    test_distance_prediction_batches, test_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Distance Loss: {train_loss:.4f}")
                    f.write(f"\tDev Distance Loss: {dev_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest UUAS: {test_uuas:.4f}")
                    f.write(f"\tTest Distance DSpr.: {test_distance_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_uuas.append(test_uuas)
        avg_distance_dspr.append(test_distance_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg UUAS: {np.mean(avg_uuas):.4f}\n")
        f.write(f"Avg Distance DSpr.: {np.mean(avg_distance_dspr):.4f}\n")


def runDepth(config, device, default_dtype, log_dir: str, layer_idx: int, save: bool):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    max_layer_idx = embedding_config["layer_max"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_dataset_path = os.path.join(
        embedding_dir, task_name, embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_dataset = th.load(train_dataset_path, pickle_module=dill)
    dev_dataset = th.load(dev_dataset_path, pickle_module=dill)
    test_dataset = th.load(test_dataset_path, pickle_module=dill)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_acc, avg_depth_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        depth_loss = L1DepthLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_depth_loss, dev_depth_loss = train(
                train_data_loader,
                probe,
                depth_loss,
                optimizer,
                task="depth",
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                start_time = time.time()

                test_depth_prediction_batches = prediction(
                    test_data_loader, probe, task="depth"
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_acc = reportRootAcc(
                    test_depth_prediction_batches, test_data_loader
                )
                test_depth_dspr, test_depth_dspr_list = reportDepthSpearmanr(
                    test_depth_prediction_batches, test_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Depth Loss: {train_depth_loss:.4f}")
                    f.write(f"\tDev Depth Loss: {dev_depth_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest Acc: {test_acc:.4f}")
                    f.write(f"\tTest Depth DSpr.: {test_depth_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_acc.append(test_acc)
        avg_depth_dspr.append(test_depth_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg Acc: {np.mean(avg_acc):.4f}\n")
        f.write(f"Avg Depth DSpr.: {np.mean(avg_depth_dspr):.4f}\n")


def runBoth(config, device, default_dtype, log_dir: str, layer_idx: int, save: bool):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    max_layer_idx = embedding_config["layer_max"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_distance_dataset = th.load(train_distance_dataset_path, pickle_module=dill)
    dev_distance_dataset = th.load(dev_distance_dataset_path, pickle_module=dill)
    test_distance_dataset = th.load(test_distance_dataset_path, pickle_module=dill)

    train_distance_data_loader = DataLoader(
        train_distance_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_pad,
    )
    dev_distance_data_loader = DataLoader(
        dev_distance_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_pad,
    )
    test_distance_data_loader = DataLoader(
        test_distance_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_pad,
    )

    train_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "test-layer-" + str(layer_idx) + ".pt",
    )

    train_depth_dataset = th.load(train_depth_dataset_path, pickle_module=dill)
    dev_depth_dataset = th.load(dev_depth_dataset_path, pickle_module=dill)
    test_depth_dataset = th.load(test_depth_dataset_path, pickle_module=dill)

    train_depth_data_loader = DataLoader(
        train_depth_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_depth_data_loader = DataLoader(
        dev_depth_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_depth_data_loader = DataLoader(
        test_depth_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_uuas, avg_distance_dspr = [], []
    avg_acc, avg_depth_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        distance_loss = L1DistanceLoss()
        depth_loss = L1DepthLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_distance_loss, dev_distance_loss = train(
                train_distance_data_loader,
                probe,
                distance_loss,
                optimizer,
                task="distance",
                dev_data_loader=dev_distance_data_loader,
            )
            train_depth_loss, dev_depth_loss = train(
                train_depth_data_loader,
                probe,
                depth_loss,
                optimizer,
                task="depth",
                dev_data_loader=dev_depth_data_loader,
            )
            scheduler.step(dev_distance_loss + dev_depth_loss)

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                start_time = time.time()

                test_distance_prediction_batches = prediction(
                    test_distance_data_loader, probe, task="distance"
                )

                test_depth_prediction_batches = prediction(
                    test_depth_data_loader, probe, task="depth"
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_uuas = reportUUAS(
                    test_distance_prediction_batches, test_distance_data_loader
                )
                test_distance_dspr, test_distance_dspr_list = reportDistanceSpearmanr(
                    test_distance_prediction_batches, test_distance_data_loader
                )

                test_acc = reportRootAcc(
                    test_depth_prediction_batches, test_depth_data_loader
                )
                test_depth_dspr, test_depth_dspr_list = reportDepthSpearmanr(
                    test_depth_prediction_batches, test_depth_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Distance Loss: {train_distance_loss:.4f}")
                    f.write(f"\tDev Distance Loss: {dev_distance_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTrain Depth Loss: {train_depth_loss:.4f}")
                    f.write(f"\tDev Depth Loss: {dev_depth_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest UUAS: {test_uuas:.4f}")
                    f.write(f"\tTest Distance DSpr.: {test_distance_dspr:.4f}")
                    f.write("\n")
                    f.write(f"\tTest Acc: {test_acc:.4f}")
                    f.write(f"\tTest Depth DSpr.: {test_depth_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_uuas.append(test_uuas)
        avg_distance_dspr.append(test_distance_dspr)
        avg_acc.append(test_acc)
        avg_depth_dspr.append(test_depth_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg UUAS: {np.mean(avg_uuas):.4f}\n")
        f.write(f"Avg Distance DSpr.: {np.mean(avg_distance_dspr):.4f}\n")
        f.write(f"Avg Acc: {np.mean(avg_acc):.4f}\n")
        f.write(f"Avg Depth DSpr.: {np.mean(avg_depth_dspr):.4f}\n")


def runDistanceWithLinear(
    config,
    device,
    default_dtype,
    log_dir: str,
    pretrained_bert_dir: str,
    layer_idx: int,
    save: bool,
):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_dataset_path = os.path.join(
        embedding_dir, task_name, embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_dataset = th.load(train_dataset_path, pickle_module=dill)
    dev_dataset = th.load(dev_dataset_path, pickle_module=dill)
    test_dataset = th.load(test_dataset_path, pickle_module=dill)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_uuas, avg_distance_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        distance_loss = L1DistanceLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_loss, dev_loss = trainLinear(
                train_data_loader,
                probe,
                distance_loss,
                optimizer,
                task="distance",
                pretrained_bert_dir=pretrained_bert_dir,
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                start_time = time.time()

                test_distance_prediction_batches = predictionLinear(
                    test_data_loader,
                    probe,
                    task="distance",
                    pretrained_bert_dir=pretrained_bert_dir,
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_uuas = reportUUAS(
                    test_distance_prediction_batches, test_data_loader
                )
                test_distance_dspr, test_distance_dspr_list = reportDistanceSpearmanr(
                    test_distance_prediction_batches, test_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Distance Loss: {train_loss:.4f}")
                    f.write(f"\tDev Distance Loss: {dev_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest UUAS: {test_uuas:.4f}")
                    f.write(f"\tTest Distance DSpr.: {test_distance_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_uuas.append(test_uuas)
        avg_distance_dspr.append(test_distance_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg UUAS: {np.mean(avg_uuas):.4f}\n")
        f.write(f"Avg Distance DSpr.: {np.mean(avg_distance_dspr):.4f}\n")


def runDepthWithLinear(
    config,
    device,
    default_dtype,
    log_dir: str,
    pretrained_bert_dir: str,
    layer_idx: int,
    save: bool,
):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    max_layer_idx = embedding_config["layer_max"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_dataset_path = os.path.join(
        embedding_dir, task_name, embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_dataset_path = os.path.join(
        embedding_dir,
        task_name,
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_dataset = th.load(train_dataset_path, pickle_module=dill)
    dev_dataset = th.load(dev_dataset_path, pickle_module=dill)
    test_dataset = th.load(test_dataset_path, pickle_module=dill)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_data_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_acc, avg_depth_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        depth_loss = L1DepthLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_depth_loss, dev_depth_loss = trainLinear(
                train_data_loader,
                probe,
                depth_loss,
                optimizer,
                task="depth",
                pretrained_bert_dir=pretrained_bert_dir,
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                start_time = time.time()

                test_depth_prediction_batches = predictionLinear(
                    test_data_loader,
                    probe,
                    task="depth",
                    pretrained_bert_dir=pretrained_bert_dir,
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_acc = reportRootAcc(
                    test_depth_prediction_batches, test_data_loader
                )
                test_depth_dspr, test_depth_dspr_list = reportDepthSpearmanr(
                    test_depth_prediction_batches, test_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Depth Loss: {train_depth_loss:.4f}")
                    f.write(f"\tDev Depth Loss: {dev_depth_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest Acc: {test_acc:.4f}")
                    f.write(f"\tTest Depth DSpr.: {test_depth_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_acc.append(test_acc)
        avg_depth_dspr.append(test_depth_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg Acc: {np.mean(avg_acc):.4f}\n")
        f.write(f"Avg Depth DSpr.: {np.mean(avg_depth_dspr):.4f}\n")


def runBothWithLinear(
    config,
    device,
    default_dtype,
    log_dir: str,
    pretrained_bert_dir: str,
    layer_idx: int,
    save: bool,
):
    embedding_config = config["embedding"]
    embedding_name = embedding_config["name"]
    task_name = embedding_config["task"]
    max_layer_idx = embedding_config["layer_max"]
    embedding_dir = embedding_config["dir"]

    run_config = config["run"]
    run_num = run_config["num"]
    epoch_num = run_config["epoch"]
    batch_size = run_config["batch_size"]
    lr = float(run_config["lr"])
    stop_lr = float(run_config["stop_lr"])

    train_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_distance_dataset_path = os.path.join(
        embedding_dir,
        "distance",
        embedding_name,
        "test-layer-" + str(layer_idx) + ".pt",
    )

    train_distance_dataset = th.load(train_distance_dataset_path, pickle_module=dill)
    dev_distance_dataset = th.load(dev_distance_dataset_path, pickle_module=dill)
    test_distance_dataset = th.load(test_distance_dataset_path, pickle_module=dill)

    train_distance_data_loader = DataLoader(
        train_distance_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_pad,
    )
    dev_distance_data_loader = DataLoader(
        dev_distance_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_pad,
    )
    test_distance_data_loader = DataLoader(
        test_distance_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_pad,
    )

    train_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "train-layer-" + str(layer_idx) + ".pt",
    )
    dev_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "dev-layer-" + str(layer_idx) + ".pt",
    )
    test_depth_dataset_path = os.path.join(
        embedding_dir, "depth", embedding_name, "test-layer-" + str(layer_idx) + ".pt",
    )

    train_depth_dataset = th.load(train_depth_dataset_path, pickle_module=dill)
    dev_depth_dataset = th.load(dev_depth_dataset_path, pickle_module=dill)
    test_depth_dataset = th.load(test_depth_dataset_path, pickle_module=dill)

    train_depth_data_loader = DataLoader(
        train_depth_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_pad
    )
    dev_depth_data_loader = DataLoader(
        dev_depth_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )
    test_depth_data_loader = DataLoader(
        test_depth_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_pad
    )

    avg_uuas, avg_distance_dspr = [], []
    avg_acc, avg_depth_dspr = [], []
    for run_idx in tqdm(range(run_num), desc="[Run]"):
        probe = constructProbe(config["probe"], device, default_dtype)
        probe.to(device)

        distance_loss = L1DistanceLoss()
        depth_loss = L1DepthLoss()
        optimizer = Adam(probe.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=run_config["scheduler"]["factor"],
            patience=run_config["scheduler"]["patience"],
        )

        for epoch in tqdm(range(epoch_num), desc="[Epoch]"):
            start_time = time.time()

            train_distance_loss, dev_distance_loss = trainLinear(
                train_distance_data_loader,
                probe,
                distance_loss,
                optimizer,
                task="distance",
                pretrained_bert_dir=pretrained_bert_dir,
                dev_data_loader=dev_distance_data_loader,
            )
            train_depth_loss, dev_depth_loss = trainLinear(
                train_depth_data_loader,
                probe,
                depth_loss,
                optimizer,
                task="depth",
                pretrained_bert_dir=pretrained_bert_dir,
                dev_data_loader=dev_depth_data_loader,
            )
            scheduler.step(dev_distance_loss + dev_depth_loss)

            if optimizer.param_groups[0]["lr"] < stop_lr or epoch == epoch_num - 1:
                train_secs = int(time.time() - start_time)
                train_mins = train_secs / 60
                train_secs = train_secs % 60

                start_time = time.time()

                test_distance_prediction_batches = predictionLinear(
                    test_distance_data_loader,
                    probe,
                    task="distance",
                    pretrained_bert_dir=pretrained_bert_dir,
                )

                test_depth_prediction_batches = predictionLinear(
                    test_depth_data_loader,
                    probe,
                    task="depth",
                    pretrained_bert_dir=pretrained_bert_dir,
                )

                pred_secs = int(time.time() - start_time)
                pred_mins = pred_secs / 60
                pred_secs = pred_secs % 60

                test_uuas = reportUUAS(
                    test_distance_prediction_batches, test_distance_data_loader
                )
                test_distance_dspr, test_distance_dspr_list = reportDistanceSpearmanr(
                    test_distance_prediction_batches, test_distance_data_loader
                )

                test_acc = reportRootAcc(
                    test_depth_prediction_batches, test_depth_data_loader
                )
                test_depth_dspr, test_depth_dspr_list = reportDepthSpearmanr(
                    test_depth_prediction_batches, test_depth_data_loader
                )

                # log
                log_file = os.path.join(log_dir, "layer_" + str(layer_idx) + ".log")
                with open(log_file, "a") as f:
                    f.write(f"Run: {run_idx + 1} | Epoch: {epoch + 1}\n")
                    f.write(f"\tLearning Rate: {optimizer.param_groups[0]['lr']}\n")
                    f.write(
                        f"Train time in {train_mins:.0f} minutes, {train_secs:.0f} seconds\n"
                    )
                    f.write(
                        f"Pred time in {pred_mins:.0f} minutes, {pred_secs:.0f} seconds\n"
                    )
                    f.write(f"\tTrain Distance Loss: {train_distance_loss:.4f}")
                    f.write(f"\tDev Distance Loss: {dev_distance_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTrain Depth Loss: {train_depth_loss:.4f}")
                    f.write(f"\tDev Depth Loss: {dev_depth_loss:.4f}")
                    f.write("\n")
                    f.write(f"\tTest UUAS: {test_uuas:.4f}")
                    f.write(f"\tTest Distance DSpr.: {test_distance_dspr:.4f}")
                    f.write("\n")
                    f.write(f"\tTest Acc: {test_acc:.4f}")
                    f.write(f"\tTest Depth DSpr.: {test_depth_dspr:.4f}")
                    f.write("\n")
                    f.write("-" * 50 + "\n")
                break

        avg_uuas.append(test_uuas)
        avg_distance_dspr.append(test_distance_dspr)
        avg_acc.append(test_acc)
        avg_depth_dspr.append(test_depth_dspr)

        if save:
            model_file = os.path.join(
                log_dir, "layer_" + str(layer_idx) + "-run_" + str(run_idx) + ".pt"
            )
            th.save(probe.state_dict(), model_file)

    with open(log_file, "a") as f:
        f.write(f"Avg UUAS: {np.mean(avg_uuas):.4f}\n")
        f.write(f"Avg Distance DSpr.: {np.mean(avg_distance_dspr):.4f}\n")
        f.write(f"Avg Acc: {np.mean(avg_acc):.4f}\n")
        f.write(f"Avg Depth DSpr.: {np.mean(avg_depth_dspr):.4f}\n")
