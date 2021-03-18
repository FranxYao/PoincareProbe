import os
import math
import random

import torch as th
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import geoopt as gt

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.path import Path
import matplotlib.patches as patches

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.path import Path
import matplotlib.patches as patches


def getTransformed(probe, bert, input_ids, attention_mask, token_type_ids):
    input_ids, token_type_ids, attention_mask = (
        input_ids.to(probe.device),
        token_type_ids.to(probe.device),
        attention_mask.to(probe.device),
    )
    with th.no_grad():
        outputs = bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        sequence_output = (
            hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
        )
        transformed = th.matmul(sequence_output, probe.proj)
        transformed = probe.ball.expmap0(transformed)
        transformed = probe.ball.mobius_matvec(probe.trans, transformed)
        transformed = transformed[attention_mask == 1]

    return transformed


def getTransformedEuclid(probe, bert, input_ids, attention_mask, token_type_ids):
    input_ids, token_type_ids, attention_mask = (
        input_ids.to(probe.device),
        token_type_ids.to(probe.device),
        attention_mask.to(probe.device),
    )
    with th.no_grad():
        outputs = bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        sequence_output = (
            hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
        )
        transformed = th.matmul(sequence_output, probe.proj)
        transformed = transformed[attention_mask == 1]

    return transformed


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
    transformed_pca,
    pos_pca,
    neg_pca,
    radius,
    probe,
    transformed_pt,
    tokenizer,
    text_input_ids,
    force: bool = False,
    viz_text=[],
):

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.axis("square")
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius))
    bound_circle = Circle(
        (0, 0), radius - 5e-3, color="#c6c6c6", alpha=1, ls="-", lw=1, fill=False
    )
    ax.add_artist(bound_circle)

    pos_color = "#ffb000"
    ax.scatter(
        pos_pca[0, 0],
        pos_pca[0, 1],
        facecolors=pos_color,
        edgecolors=pos_color,
        alpha=1,
        lw=2,
        s=150,
    )
    ax.annotate(
        "[POS]", (pos_pca[0, 0], pos_pca[0, 1] + 0.1), fontsize=16, fontweight="bold",
    )

    neg_color = "#648fff"
    ax.scatter(
        neg_pca[0, 0],
        neg_pca[0, 1],
        facecolors=neg_color,
        edgecolors=neg_color,
        alpha=1,
        lw=2,
        s=150,
    )
    ax.annotate(
        "[NEG]", (neg_pca[0, 0], neg_pca[0, 1] - 0.1), fontsize=16, fontweight="bold",
    )

    for i in range(1, transformed_pca.shape[0] - 1):
        ax.scatter(
            transformed_pca[i, 0],
            transformed_pca[i, 1],
            facecolors="#6f6f6f",
            edgecolors="#4c4c4c",
            alpha=0.6,
            lw=2,
        )
        x1, y1 = transformed_pca[i, 0], transformed_pca[i, 1]
        pos_dis = probe.ball.dist(transformed_pt[i], probe.pos)
        neg_dis = probe.ball.dist(transformed_pt[i], probe.neg)
        if abs(pos_dis - neg_dis) > 0.1:
            if pos_dis < neg_dis:
                x2, y2 = pos_pca[0, 0], pos_pca[0, 1]
                line_color = pos_color
            else:
                x2, y2 = neg_pca[0, 0], neg_pca[0, 1]
                line_color = neg_color

            p_x, p_y = x1, y1
            q_x, q_y = x2, y2
            x_0, y_0, r = getGeodesic(p_x, p_y, q_x, q_y)
            theta1 = getTheta(p_x, p_y, x_0, y_0)
            theta2 = getTheta(q_x, q_y, x_0, y_0)
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            if abs(theta1 - theta2) > 180:
                theta1, theta2 = theta2, theta1

            # plot geodesic
            circle = Circle(
                (x_0, y_0),
                radius=r,
                color=line_color,
                alpha=0.3,
                ls=":",
                lw=1,
                fill=False,
            )

            ax.add_artist(circle)
            circle.set_clip_path(bound_circle)

            arc = Arc(
                (x_0, y_0),
                width=2 * r,
                height=2 * r,
                theta1=theta1,
                theta2=theta2,
                color=line_color,
                alpha=0.9,
                ls="-",
                lw=2,
                fill=False,
            )
            ax.add_artist(arc)

            if len(viz_text) == 0:
                anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
                ax.annotate(
                    anno_text,
                    (transformed_pca[i, 0], transformed_pca[i, 1]),
                    fontsize=14,
                )
        else:
            if pos_dis < neg_dis:
                x2, y2 = pos_pca[0, 0], pos_pca[0, 1]
                line_color = pos_color
            else:
                x2, y2 = neg_pca[0, 0], neg_pca[0, 1]
                line_color = neg_color

            p_x, p_y = x1, y1
            q_x, q_y = x2, y2
            x_0, y_0, r = getGeodesic(p_x, p_y, q_x, q_y)
            theta1 = getTheta(p_x, p_y, x_0, y_0)
            theta2 = getTheta(q_x, q_y, x_0, y_0)
            if theta1 > theta2:
                theta1, theta2 = theta2, theta1
            if abs(theta1 - theta2) > 180:
                theta1, theta2 = theta2, theta1

            # plot geodesic
            circle = Circle(
                (x_0, y_0),
                radius=r,
                color=line_color,
                alpha=0.3,
                ls=":",
                lw=1,
                fill=False,
            )

            ax.add_artist(circle)
            circle.set_clip_path(bound_circle)

            arc = Arc(
                (x_0, y_0),
                width=2 * r,
                height=2 * r,
                theta1=theta1,
                theta2=theta2,
                color=line_color,
                alpha=0.9,
                ls="--",
                lw=1,
                fill=False,
            )
            ax.add_artist(arc)

            if len(viz_text) == 0 and force:
                anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
                ax.annotate(
                    anno_text,
                    (transformed_pca[i, 0], transformed_pca[i, 1]),
                    fontsize=14,
                )

        anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
        if len(viz_text) > 0 and anno_text in viz_text:
            ax.annotate(
                anno_text, (transformed_pca[i, 0], transformed_pca[i, 1]), fontsize=14
            )

    return fig, ax


def plotEuclidean(
    transformed_pca,
    pos_pca,
    neg_pca,
    probe,
    transformed_pt,
    tokenizer,
    text_input_ids,
    force: bool = False,
    viz_text=[],
):

    fig, ax = plt.subplots()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    pos_color = "#ffb000"
    ax.scatter(
        pos_pca[0, 0],
        pos_pca[0, 1],
        facecolors=pos_color,
        edgecolors=pos_color,
        alpha=1,
        lw=2,
        s=150,
    )
    ax.annotate(
        "[POS]", (pos_pca[0, 0], pos_pca[0, 1] + 1), fontsize=16, fontweight="bold",
    )

    neg_color = "#648fff"
    ax.scatter(
        neg_pca[0, 0],
        neg_pca[0, 1],
        facecolors=neg_color,
        edgecolors=neg_color,
        alpha=1,
        lw=2,
        s=150,
    )
    ax.annotate(
        "[NEG]", (neg_pca[0, 0], neg_pca[0, 1] - 1), fontsize=16, fontweight="bold",
    )

    for i in range(1, transformed_pca.shape[0] - 1):
        ax.scatter(
            transformed_pca[i, 0],
            transformed_pca[i, 1],
            facecolors="#6f6f6f",
            edgecolors="#4c4c4c",
            alpha=0.6,
            lw=2,
        )
        x1, y1 = transformed_pca[i, 0], transformed_pca[i, 1]
        with th.no_grad():
            pos_dis = ((transformed_pt[i] - probe.pos) ** 2).sum()
            neg_dis = ((transformed_pt[i] - probe.neg) ** 2).sum()
        if abs(pos_dis - neg_dis) > 0.1:
            if pos_dis < neg_dis:
                x2, y2 = pos_pca[0, 0], pos_pca[0, 1]
                line_color = pos_color
            else:
                x2, y2 = neg_pca[0, 0], neg_pca[0, 1]
                line_color = neg_color

            plt.plot([x1, x2], [y1, y2], c=line_color, alpha=0.9, ls="-", lw=2)

            if len(viz_text) == 0:
                anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
                ax.annotate(
                    anno_text,
                    (transformed_pca[i, 0] - 0.03, transformed_pca[i, 1] + 0.02),
                    fontsize=14,
                )
        else:
            if pos_dis < neg_dis:
                x2, y2 = pos_pca[0, 0], pos_pca[0, 1]
                line_color = pos_color
            else:
                x2, y2 = neg_pca[0, 0], neg_pca[0, 1]
                line_color = neg_color

            plt.plot([x1, x2], [y1, y2], c=line_color, alpha=0.9, ls="--", lw=1)
            if len(viz_text) == 0 and force:
                anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
                ax.annotate(
                    anno_text,
                    (transformed_pca[i, 0], transformed_pca[i, 1]),
                    fontsize=14,
                )

        anno_text = tokenizer.ids_to_tokens[text_input_ids[0, i].item()]
        if len(viz_text) > 0 and anno_text in viz_text:
            ax.annotate(
                anno_text,
                (transformed_pca[i, 0] - 0.4, transformed_pca[i, 1] + 0.1),
                fontsize=14,
            )

    return fig, ax


def vizSentence(input_sent, probe, bert, tokenizer, force: bool = False, viz_text=[]):
    inputs = tokenizer(input_sent, return_tensors="pt")
    text_input_ids, text_token_type_ids, text_attention_mask = (
        inputs["input_ids"],
        inputs["token_type_ids"],
        inputs["attention_mask"],
    )
    text_input_ids, text_token_type_ids, text_attention_mask = (
        text_input_ids.to(probe.device),
        text_token_type_ids.to(probe.device),
        text_attention_mask.to(probe.device),
    )
    with th.no_grad():
        outputs = bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        sequence_output = (
            hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
        )
        logits = probe(sequence_output)

    print(
        " ".join(
            tokenizer.convert_ids_to_tokens(
                text_input_ids[0][text_attention_mask[0] == 1][1:-1]
            )
        )
    )
    print(f"Pred: {logits.argmax().item()}")

    transformed_pt = getTransformed(
        probe, bert, text_input_ids, text_attention_mask, text_token_type_ids
    )
    transformed = transformed_pt.squeeze().cpu().numpy()

    pos = probe.pos.detach().cpu().numpy().reshape(1, -1)
    neg = probe.neg.detach().cpu().numpy().reshape(1, -1)
    embedding = np.concatenate((pos, neg, transformed), axis=0)

    pca = PCA(n_components=2)
    pca.fit(embedding)

    pos_pca = pca.transform(pos)
    neg_pca = pca.transform(neg)
    transformed_pca = pca.transform(transformed)

    radius = round(abs(transformed_pca).max(), 1) + 1e-1
    fig, ax = plotPoincare(
        transformed_pca,
        pos_pca,
        neg_pca,
        radius,
        probe,
        transformed_pt,
        tokenizer,
        text_input_ids,
        force=force,
        viz_text=viz_text,
    )

    anchored_text = "Pred: " + ("[P]" if logits.argmax() == 1 else "[N]")
    at = AnchoredText(
        anchored_text, frameon=True, loc="lower right", prop=dict(fontsize="14"),
    )
    ax.add_artist(at)

    return fig, ax


def vizSentenceEuclid(
    input_sent, probe, bert, tokenizer, force: bool = False, viz_text=[]
):
    inputs = tokenizer(input_sent, return_tensors="pt")
    text_input_ids, text_token_type_ids, text_attention_mask = (
        inputs["input_ids"],
        inputs["token_type_ids"],
        inputs["attention_mask"],
    )
    text_input_ids, text_token_type_ids, text_attention_mask = (
        text_input_ids.to(probe.device),
        text_token_type_ids.to(probe.device),
        text_attention_mask.to(probe.device),
    )
    with th.no_grad():
        outputs = bert(
            text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = outputs[2]
        sequence_output = (
            hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
        )
        logits = probe(sequence_output)

    print(
        " ".join(
            tokenizer.convert_ids_to_tokens(
                text_input_ids[0][text_attention_mask[0] == 1][1:-1]
            )
        )
    )
    print(f"Pred: {logits.argmax().item()}")

    transformed_pt = getTransformedEuclid(
        probe, bert, text_input_ids, text_attention_mask, text_token_type_ids
    )
    transformed = transformed_pt.squeeze().cpu().numpy()

    pos = probe.pos.detach().cpu().numpy().reshape(1, -1)
    neg = probe.neg.detach().cpu().numpy().reshape(1, -1)
    embedding = np.concatenate((pos, neg, transformed), axis=0)

    pca = PCA(n_components=2)
    pca.fit(embedding)

    pos_pca = pca.transform(pos)
    neg_pca = pca.transform(neg)
    transformed_pca = pca.transform(transformed)

    fig, ax = plotEuclidean(
        transformed_pca,
        pos_pca,
        neg_pca,
        probe,
        transformed_pt,
        tokenizer,
        text_input_ids,
        force=force,
        viz_text=viz_text,
    )

    anchored_text = "Pred: " + ("[P]" if logits.argmax() == 1 else "[N]")
    at = AnchoredText(
        anchored_text, frameon=True, loc="lower right", prop=dict(fontsize="14"),
    )
    ax.add_artist(at)

    return fig, ax
