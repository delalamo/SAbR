#!/usr/bin/env python
"""
Train SoftAlign-like model on SageMaker.

Assumes:
  - data/aln_SCOPE
  - data/dicti_inputs_SCOPE_colab.pkl
  - local modules: Input_MPNN, END_TO_END_MODELS, Loss_functions, utils

Outputs:
  - checkpoints/epoch_{:03d}.pkl
  - checkpoints/metrics.json
  - checkpoints/val_curve.png
"""

import argparse
import json
import os
import pickle
import re
import time
from typing import Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax.example_libraries.optimizers import adam
from softalign import END_TO_END_MODELS as ete
from softalign import Loss_functions as loss_
from softalign import utils
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm.auto import tqdm

ORIG_END_TO_END_CALL = ete.END_TO_END.__call__

# -------------------
# Change to RBF kernel
# -------------------


def rbf_sim(h_V1, h_V2, sigma=1.0):
    # h_V1: (n, i, a), h_V2: (n, j, a)
    diff = h_V1[:, :, None, :] - h_V2[:, None, :, :]  # shape (n, i, j, a)
    sqdist = jnp.sum(diff**2, axis=-1)  # shape (n, i, j)
    return jnp.exp(-sqdist / (2 * sigma**2))  # RBF kernel


def new_fxn(self, x1, x2, lens, t):
    X1, mask1, res1, ch1 = x1
    X2, mask2, res2, ch2 = x2
    h_V1 = self.MPNN(X1, mask1, res1, ch1)
    h_V2 = self.MPNN(X2, mask2, res2, ch2)
    # encodings
    gap = hk.get_parameter(
        "gap", shape=[1], init=hk.initializers.RandomNormal(0.1, -1)
    )
    if self.affine:
        popen = hk.get_parameter(
            "open", shape=[1], init=hk.initializers.RandomNormal(0.1, -3)
        )
    #######
    # sim_matrix = jnp.einsum("nia,nja->nij",h_V1,h_V2)
    sim_matrix = rbf_sim(h_V1, h_V2)
    if self.affine:
        scores, soft_aln = self.my_sw_func(
            sim_matrix, lens, gap[0], popen[0], t
        )
    else:
        scores, soft_aln = self.my_sw_func(sim_matrix, lens, gap[0], t)
    return soft_aln, sim_matrix, scores


def set_alignment_kernel(use_rbf: bool):
    """Toggle between original dot-product kernel and RBF."""
    if use_rbf:
        ete.END_TO_END.__call__ = new_fxn
    else:
        ete.END_TO_END.__call__ = ORIG_END_TO_END_CALL


# -------------------
# Data preparation
# -------------------


def load_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=",")
    df = df.drop_duplicates()
    # Normalize column names if needed
    if "name1" not in df.columns:
        # Original notebook-style mapping; adjust if your file differs
        # Here we assume: col1=name1, col2=name2, col3=TMS, col4=aln, col5=fold
        df = df.rename(
            columns={
                0: "fold",
                1: "name1",
                2: "name2",
                3: "TMS",
                4: "aln",
            }
        )
    return df


def load_inputs_dict(path: str) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_input_data(
    pairs_df: pd.DataFrame,
    inputs_dict: Dict,
    max_len: int,
    tms_threshold: float,
) -> Tuple[list, ...]:
    X1 = []
    X2 = []
    mask1 = []
    mask2 = []
    chain1 = []
    chain2 = []
    res1 = []
    res2 = []
    id1 = []
    id2 = []
    tmaln = []

    for _, row in pairs_df.iterrows():
        pr1, pr2 = row["name1"], row["name2"]
        if pr1 not in inputs_dict or pr2 not in inputs_dict:
            continue
        if row["TMS"] <= tms_threshold:
            continue

        _X1, _mask1, _chain1, _res1 = inputs_dict[pr1]
        _X2, _mask2, _chain2, _res2 = inputs_dict[pr2]

        if len(_X1[0]) > max_len or len(_X2[0]) > max_len:
            continue

        id1.append(pr1)
        id2.append(pr2)
        X1.append(_X1[0])
        X2.append(_X2[0])
        mask1.append(_mask1[0])
        mask2.append(_mask2[0])
        chain1.append(_chain1[0])
        chain2.append(_chain2[0])
        res1.append(_res1[0])
        res2.append(_res2[0])

        aln_indices = np.array(
            [int(s) for s in re.findall(r"-?\d+", str(row["aln"]))]
        )
        tmaln.append(aln_indices)

    return X1, X2, mask1, mask2, chain1, chain2, res1, res2, id1, id2, tmaln


def format_lr_tag(lr: float) -> str:
    """Produce a filesystem-friendly tag such as lr1e-03."""
    return f"lr{lr:.0e}".replace("+", "")


def save_train_loss_plot(
    losses: List[float], step: int, out_dir: str, run_tag: str
):
    """Plot per-step training losses and save to disk."""
    if not losses:
        return
    plt.figure()
    x = np.arange(len(losses))
    plt.plot(x, losses, color="#6699ff", alpha=0.4, label="loss (per step)")
    if len(losses) >= 5:
        frac = 0.1
        smoothed = lowess(losses, x, frac=frac, it=0, return_sorted=False)
        label = f"LOESS (frac={frac:.2f})"
        plt.plot(x, smoothed, color="#004a99", alpha=1.0, label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(out_dir, f"train_loss_steps_{run_tag}.png")
    plt.savefig(fname, dpi=200)
    plt.close()


def train_val_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    """Fold-aware split if 'fold' exists, else random split."""
    rng = np.random.RandomState(seed)
    if "fold" in df.columns:
        folds = df["fold"].unique()
        rng.shuffle(folds)
        n_val = max(1, int(len(folds) * val_frac))
        val_folds = set(folds[:n_val])
        train_df = df[~df["fold"].isin(val_folds)]
        val_df = df[df["fold"].isin(val_folds)]
    else:
        perm = rng.permutation(len(df))
        n_val = int(len(df) * val_frac)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# -------------------
# Model
# -------------------


def model_end_to_end(
    x1,
    x2,
    lens,
    t,
    node_features,
    edge_features,
    hidden_dim,
    num_encoder_layers,
    k_neighbors,
    categorical,
    nb_clusters,
    affine,
    soft_max,
):
    if categorical:
        model = ete.END_TO_END_SEQ_KMEANS(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            nb_clusters=nb_clusters,
            affine=affine,
            soft_max=soft_max,
            dropout=0.0,
        )
    else:
        model = ete.END_TO_END(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            affine=affine,
            soft_max=soft_max,
            dropout=0.0,
        )
    return model(x1, x2, lens, t)


def build_model(config):
    def forward(x1, x2, lens, t):
        return model_end_to_end(
            x1,
            x2,
            lens,
            t,
            node_features=config["encoding_dim"],
            edge_features=config["encoding_dim"],
            hidden_dim=config["encoding_dim"],
            num_encoder_layers=config["num_layers"],
            k_neighbors=config["num_neighbors"],
            categorical=config["categorical"],
            nb_clusters=config["nb_clusters"],
            affine=True,
            soft_max=config["soft_max"],
        )

    return hk.transform(forward)


# -------------------
# Training utilities
# -------------------


def make_batches(
    X_1,
    X_2,
    mask_1,
    mask_2,
    res_1,
    res_2,
    chain_1,
    chain_2,
    tmaln,
    batch_size,
    max_size,
):
    n = len(X_1)
    indices = np.random.permutation(n)
    for i in range(0, n, batch_size):
        idx = indices[i : i + batch_size]
        if len(idx) < batch_size:
            continue
        X1, mask1, res1, chain1, X2, mask2, res2, chain2, TMALN, lens = (
            utils.pad_tmalign(
                [X_1[j] for j in idx],
                [mask_1[j] for j in idx],
                [res_1[j] for j in idx],
                [chain_1[j] for j in idx],
                [X_2[j] for j in idx],
                [mask_2[j] for j in idx],
                [res_2[j] for j in idx],
                [chain_2[j] for j in idx],
                [tmaln[j] for j in idx],
                max_size,
            )
        )
        yield (X1, mask1, res1, chain1, X2, mask2, res2, chain2, TMALN, lens)


def make_train_step(model, categorical):
    loss_fn = (
        loss_.CrossEntropyLoss_CAT if categorical else loss_.CrossEntropyLoss
    )

    @jax.jit
    def _train_step(step_i, opt_state, batch):
        params = get_params(opt_state)
        (loss, aux), grads = jax.value_and_grad(
            loss_fn, argnums=0, has_aux=True
        )(params, batch, model)
        new_opt_state = opt_update(step_i, grads, opt_state)
        return new_opt_state, loss, aux

    return _train_step


def make_eval_step(model, categorical):
    loss_fn = (
        loss_.CrossEntropyLoss_CAT if categorical else loss_.CrossEntropyLoss
    )

    @jax.jit
    def _eval_step(params, batch):
        (loss, aux) = loss_fn(params, batch, model)
        return loss, aux

    return _eval_step


# -------------------
# Main
# -------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional identifier to override the default run tag.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_size", type=int, default=300)
    parser.add_argument(
        "--use_rbf",
        action="store_true",
        help="Use RBF kernel instead of dot product",
    )
    parser.add_argument("--tms_threshold", type=float, default=0.5)
    parser.add_argument("--encoding_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_neighbors", type=int, default=30)
    parser.add_argument("--categorical", action="store_true")
    parser.add_argument("--nb_clusters", type=int, default=16)
    parser.add_argument("--soft_max", action="store_true")
    parser.add_argument("--T_start", type=float, default=1.0)
    parser.add_argument("--T_end", type=float, default=1.0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15000,
        help="Maximum number of training steps before stopping (0 disables).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    lr_tag = format_lr_tag(args.lr)
    kernel_tag = "rbf" if args.use_rbf else "dot"
    if not args.run_tag:
        args.run_tag = f"{lr_tag}_{kernel_tag}"
    run_tag = args.run_tag
    config = vars(args)
    set_alignment_kernel(args.use_rbf)

    # --- Load data ---
    pairs = load_pairs(os.path.join(args.data_dir, "aln_SCOPE"))
    inputs_dict = load_inputs_dict(
        os.path.join(args.data_dir, "dicti_inputs_SCOPE_colab.pkl")
    )

    train_df, val_df = train_val_split(pairs, val_frac=0.1, seed=42)

    train_data = prepare_input_data(
        train_df, inputs_dict, args.max_size, args.tms_threshold
    )
    val_data = prepare_input_data(
        val_df, inputs_dict, args.max_size, args.tms_threshold
    )

    (tr_X1, tr_X2, tr_m1, tr_m2, tr_c1, tr_c2, tr_r1, tr_r2, _, _, tr_tmaln) = (
        train_data
    )
    (va_X1, va_X2, va_m1, va_m2, va_c1, va_c2, va_r1, va_r2, _, _, va_tmaln) = (
        val_data
    )

    # --- Build model ---
    model = build_model(config)
    key = jax.random.PRNGKey(0)

    # One dummy batch for init
    init_batch = next(
        make_batches(
            tr_X1,
            tr_X2,
            tr_m1,
            tr_m2,
            tr_r1,
            tr_r2,
            tr_c1,
            tr_c2,
            tr_tmaln,
            batch_size=args.batch_size,
            max_size=args.max_size,
        )
    )
    X1, m1, r1, c1, X2, m2, r2, c2, TMALN, lens = init_batch
    params = model.init(
        key, (X1, m1, r1, c1), (X2, m2, r2, c2), lens, args.T_start
    )

    # --- Optimizer ---
    global opt_init, opt_update, get_params
    opt_init, opt_update, get_params = adam(args.lr)
    opt_state = opt_init(params)
    train_step = make_train_step(model, args.categorical)
    eval_step = make_eval_step(model, args.categorical)

    history = {"train_loss": [], "val_loss": [], "steps_completed": 0}
    train_loss_steps: List[float] = []
    step_i = 0
    steps_cap = (
        args.max_steps if args.max_steps and args.max_steps > 0 else None
    )
    stop_training = False
    batches_per_epoch = len(tr_X1) // args.batch_size
    if batches_per_epoch <= 0:
        batches_per_epoch = 1
    schedule_steps = (
        steps_cap if steps_cap is not None else batches_per_epoch * args.epochs
    )
    schedule_steps = max(1, schedule_steps)

    def temperature_at_step(step_idx: int) -> float:
        frac = min(step_idx / schedule_steps, 1.0)
        return args.T_start + frac * (args.T_end - args.T_start)

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        tic = time.time()
        T_epoch = temperature_at_step(step_i)

        # Train
        train_losses = []
        train_total = len(tr_X1) // args.batch_size
        train_batches = make_batches(
            tr_X1,
            tr_X2,
            tr_m1,
            tr_m2,
            tr_r1,
            tr_r2,
            tr_c1,
            tr_c2,
            tr_tmaln,
            batch_size=args.batch_size,
            max_size=args.max_size,
        )
        train_bar = tqdm(
            train_batches,
            total=train_total,
            desc=f"Epoch {epoch} train ({run_tag})",
            leave=False,
        )
        for batch in train_bar:
            # Append t & lens inside batch as in original loss_data format
            X1, m1, r1, c1, X2, m2, r2, c2, TMALN, lens = batch
            T_step = temperature_at_step(step_i)
            loss_data = (X1, m1, r1, c1, X2, m2, r2, c2, TMALN, lens, T_step)
            opt_state, loss, _ = train_step(step_i, opt_state, loss_data)
            step_i += 1
            loss_float = float(loss)
            train_losses.append(loss_float)
            train_bar.set_postfix(loss=f"{loss_float:.4f}")
            train_loss_steps.append(loss_float)
            if step_i % 500 == 0:
                save_train_loss_plot(
                    train_loss_steps, step_i, args.out_dir, run_tag
                )
            if steps_cap is not None and step_i >= steps_cap:
                stop_training = True
                break
        train_bar.close()

        # Validation
        params_epoch = get_params(opt_state)
        val_losses = []
        val_total = len(va_X1) // args.batch_size
        val_batches = make_batches(
            va_X1,
            va_X2,
            va_m1,
            va_m2,
            va_r1,
            va_r2,
            va_c1,
            va_c2,
            va_tmaln,
            batch_size=args.batch_size,
            max_size=args.max_size,
        )
        val_bar = tqdm(
            val_batches,
            total=val_total,
            desc=f"Epoch {epoch} val ({run_tag})",
            leave=False,
        )
        for batch in val_bar:
            X1, m1, r1, c1, X2, m2, r2, c2, TMALN, lens = batch
            loss_data = (X1, m1, r1, c1, X2, m2, r2, c2, TMALN, lens, 1e-4)
            vloss, _ = eval_step(params_epoch, loss_data)
            vloss_float = float(vloss)
            val_losses.append(vloss_float)
            val_bar.set_postfix(loss=f"{vloss_float:.4f}")
        val_bar.close()

        mean_tr = float(np.mean(train_losses)) if train_losses else float("nan")
        mean_va = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(mean_tr)
        history["val_loss"].append(mean_va)

        # Save checkpoint
        ckpt_path = os.path.join(
            args.out_dir, f"{run_tag}_epoch_{epoch:03d}.pkl"
        )
        with open(ckpt_path, "wb") as f:
            pickle.dump(params_epoch, f)

        # Update metrics
        history["steps_completed"] = step_i
        metrics_path = os.path.join(args.out_dir, f"metrics_{run_tag}.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        print(
            (
                f"[epoch {epoch:03d}] T={T_epoch:.3f} "
                f"train_loss={mean_tr:.4f} val_loss={mean_va:.4f}"
            )
        )
        print(
            (
                f"lr={args.lr:g} kernel={kernel_tag} "
                f"time={time.time()-tic:.1f}s saved={ckpt_path}"
            )
        )
        if stop_training:
            print(f"Reached max_steps={steps_cap}. Ending training loop.")
            break

    # Final per-step train loss plot
    save_train_loss_plot(train_loss_steps, step_i, args.out_dir, run_tag)

    # Plot validation curve
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"val_curve_{run_tag}.png"), dpi=200)


if __name__ == "__main__":
    main()
