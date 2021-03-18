import math
import random

import torch as th
import geoopt as gt
from geoopt.manifolds.stereographic import math as pmath

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.path import Path
import numpy as np

from .task import ParseDepthTask
from .evalu import prims_matrix_to_edges


def vizParseTree(dataset, inst_i):
    inst = dataset[inst_i]
    length = len(inst[0].sentence)
    sentence = inst[0].sentence
    xpos = inst[0].xpos_sentence
    label_distance = inst[1]
    label_depth = ParseDepthTask().labels(inst[0]).numpy()

    loc_x = []
    i = 0
    while i < length:
        if (
            i > 0
            and i + 1 < length
            and label_depth[i] > label_depth[i + 1]
            and label_depth[i - 1] == label_depth[i]
        ):
            loc_x.extend([i + 1, i])
            i += 2
        else:
            loc_x.extend([i])
            i += 1
    rand_loc_y = np.array([random.uniform(-0.2, 0.2) for i in range(length)])
    loc_depth = label_depth + rand_loc_y
    gold_edges = prims_matrix_to_edges(label_distance, sentence, xpos)

    return sentence, loc_x, loc_depth, gold_edges


def vizInstance(dataset, inst_i, probe):
    inst = dataset[inst_i]
    length = len(inst[0].sentence)
    embeddings = inst[0].embeddings
    if not isinstance(embeddings, th.Tensor):
        embeddings = th.Tensor(embeddings)
    sentence = inst[0].sentence
    xpos = inst[0].xpos_sentence
    label_distance = inst[1]
    label_depth = ParseDepthTask().labels(inst[0]).numpy()

    dtype = probe.proj.dtype
    device = probe.proj.device
    with th.no_grad():
        viz = probe.project(embeddings.to(dtype).to(device))
        pred_distance = probe(
            embeddings.unsqueeze(0).to(dtype).to(device), task="distance"
        )
        pred_depth = probe(embeddings.unsqueeze(0).to(dtype).to(device), task="depth")

    viz = viz.cpu().numpy()
    pred_distance = pred_distance.squeeze(0).cpu().numpy()
    pred_depth = pred_depth.squeeze(0).cpu().numpy()

    gold_edges = prims_matrix_to_edges(label_distance, sentence, xpos)
    pred_edges = prims_matrix_to_edges(pred_distance, sentence, xpos)

    uspan_correct = len(
        set([tuple(sorted(x)) for x in gold_edges]).intersection(
            set([tuple(sorted(x)) for x in pred_edges])
        )
    )
    uspan_total = len(gold_edges)
    uuas = uspan_correct / uspan_total

    label_root_idx = label_depth.argmin()
    root_idx = pred_depth.argmin()
    root_acc = root_idx == label_root_idx
    root_depth = pred_depth[label_root_idx]

    return (
        viz,
        sentence,
        uuas,
        root_acc,
        label_root_idx,
        root_depth,
        gold_edges,
        pred_edges,
    )


def plotParseTree(sentence, loc_x, loc_depth, gold_edges, stop_words=",.:;?()``\"''"):
    fig, ax = plt.subplots()

    for i, word in enumerate(sentence):
        if not word in stop_words:
            ax.scatter(
                loc_x[i],
                -loc_depth[i],
                facecolors="lightgray",
                edgecolors="gray",
                alpha=0.9,
                lw=2,
            )
            ax.annotate(word, (loc_x[i] + 0.2, -loc_depth[i]), fontsize=14)

    for i, j in gold_edges:
        x1, x2 = loc_x[i], loc_x[j]
        y1, y2 = -loc_depth[i], -loc_depth[j]

        verts = [
            (x1, y1),  # P0
            (x1, (y1 + y2) / 2),  # P1
            (x2, (y1 + y2) / 2),  # P2
            (x2, y2),  # P3
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

        path = Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor="gray",
            alpha=0.6,
            joinstyle="bevel",
            ls="-",
            lw=1,
        )
        ax.add_patch(patch)
        xs, ys = zip(*verts)

    ax.axis("off")
    return fig, ax


def plotEuclidean(
    viz, root_idx, sentence, pred_edges, gold_edges=None, stop_words=",.:;?()``\"''"
):
    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for i, word in enumerate(sentence):
        if not word in stop_words:
            if i == root_idx:
                ax.scatter(
                    viz[i, 0],
                    viz[i, 1],
                    facecolors="#1192e8",
                    edgecolors="#4c4c4c",
                    s=150,
                    alpha=1,
                    lw=0.5,
                )
            else:
                ax.scatter(
                    viz[i, 0],
                    viz[i, 1],
                    facecolors="#6f6f6f",
                    edgecolors="#4c4c4c",
                    alpha=0.6,
                    lw=2,
                )
            ax.annotate(word, (viz[i, 0], viz[i, 1]), fontsize=14)

    for i, j in gold_edges:
        x1, x2 = viz[i, 0], viz[j, 0]
        y1, y2 = viz[i, 1], viz[j, 1]
        plt.plot([x1, x2], [y1, y2], c="#ffb000", alpha=1, ls="-", lw=2)

    for i, j in pred_edges:
        x1, x2 = viz[i, 0], viz[j, 0]
        y1, y2 = viz[i, 1], viz[j, 1]
        plt.plot([x1, x2], [y1, y2], c="#4589ff", alpha=1, ls="--", lw=2)

    return fig, ax


def getGeodesic(p_x, p_y, q_x, q_y):
    p_xy = p_x ** 2 + p_y ** 2 + 1
    q_xy = q_x ** 2 + q_y ** 2 + 1
    denominator = 2 * (p_x * q_y - p_y * q_x)

    x_0 = q_y * p_xy - p_y * q_xy
    x_0 /= denominator + 1e-5
    y_0 = -q_x * p_xy + p_x * q_xy
    y_0 /= denominator + 1e-5

    r = np.sqrt(x_0 ** 2 + y_0 ** 2 - 1)

    return x_0, y_0, r


def getTheta(x, y, x_0, y_0):
    delta_y = y - y_0
    delta_x = x - x_0

    angle_in_degrees = math.atan2(delta_y, delta_x) * 180 / math.pi

    return angle_in_degrees % 360


def plotPoincare(
    viz,
    radius,
    root_idx,
    sentence,
    pred_edges,
    gold_edges=None,
    stop_words=",.:;?()``\"''",
    geodesic: bool = True,
):
    fig, ax = plt.subplots()

    ax.axis("square")
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    bound_circle = Circle(
        (0, 0), radius - 5e-3, color="#c6c6c6", alpha=1, ls="-", lw=1, fill=False
    )
    ax.add_artist(bound_circle)

    for i, word in enumerate(sentence):
        if not word in stop_words:
            if i == root_idx:
                ax.scatter(
                    viz[i, 0],
                    viz[i, 1],
                    facecolors="#1192e8",
                    edgecolors="#4c4c4c",
                    s=150,
                    alpha=1,
                    lw=0.5,
                )
            else:
                ax.scatter(
                    viz[i, 0],
                    viz[i, 1],
                    facecolors="#6f6f6f",
                    edgecolors="#4c4c4c",
                    alpha=1,
                    lw=2,
                )
            ax.annotate(word, (viz[i, 0], viz[i, 1]), fontsize=14)

    for i, j in gold_edges:
        if geodesic:
            # plot geodesic
            p_x, p_y = viz[i, 0], viz[i, 1]
            q_x, q_y = viz[j, 0], viz[j, 1]
            x_0, y_0, r = getGeodesic(p_x, p_y, q_x, q_y)
            theta1 = getTheta(p_x, p_y, x_0, y_0)
            theta2 = getTheta(q_x, q_y, x_0, y_0)
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            if abs(theta1 - theta2) > 180:
                theta1, theta2 = theta2, theta1

            arc = Arc(
                (x_0, y_0),
                width=2 * r,
                height=2 * r,
                theta1=theta1,
                theta2=theta2,
                color="#ffb000",
                alpha=1,
                ls="-",
                lw=2,
                fill=False,
            )
            ax.add_artist(arc)

            p_x, p_y = viz[i, 0], viz[i, 1]
            q_x, q_y = viz[j, 0], viz[j, 1]
            x_0, y_0, r = getGeodesic(p_x, p_y, q_x, q_y)

            circle = Circle(
                (x_0, y_0),
                radius=r,
                color="#ffb000",
                alpha=0.1,
                ls=":",
                lw=2,
                fill=False,
            )

            ax.add_artist(circle)
            circle.set_clip_path(bound_circle)
        else:
            # simple straight line
            x1, x2 = viz[i, 0], viz[j, 0]
            y1, y2 = viz[i, 1], viz[j, 1]
            plt.plot([x1, x2], [y1, y2], c="#ffb000", alpha=1, ls="--", lw=2)

    for i, j in pred_edges:
        if geodesic:
            # plot geodesic
            p_x, p_y = viz[i, 0], viz[i, 1]
            q_x, q_y = viz[j, 0], viz[j, 1]
            x_0, y_0, r = getGeodesic(p_x, p_y, q_x, q_y)
            theta1 = getTheta(p_x, p_y, x_0, y_0)
            theta2 = getTheta(q_x, q_y, x_0, y_0)
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            if abs(theta1 - theta2) > 180:
                theta1, theta2 = theta2, theta1

            arc = Arc(
                (x_0, y_0),
                width=2 * r,
                height=2 * r,
                theta1=theta1,
                theta2=theta2,
                color="#4589ff",
                alpha=1,
                ls="--",
                lw=2,
                fill=False,
            )
            ax.add_artist(arc)

            circle = Circle(
                (x_0, y_0),
                radius=r,
                color="#4589ff",
                alpha=0.1,
                ls=":",
                lw=1,
                fill=False,
            )

            ax.add_artist(circle)
            circle.set_clip_path(bound_circle)
        else:
            # simple straight line
            x1, x2 = viz[i, 0], viz[j, 0]
            y1, y2 = viz[i, 1], viz[j, 1]
            plt.plot([x1, x2], [y1, y2], c="#1192e8", alpha=0.9, ls="-", lw=1)

    return fig, ax
