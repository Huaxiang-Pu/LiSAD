"""Pretraining entry point for LiSAD physical and detector backbones."""

import sys

sys.path.append(".")

import os
import argparse
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from datasets.dataloader import get_fobss_loader, get_ebai_loader
from models.battery import LiBPINN
from config import settings
from scripts.utils import set_seed, to_device
from scripts.loss_balancer import LogCovLossBalancer
from scripts.utils import print_logging


# ------------------ Builders / helpers ------------------
def get_device(device_str: str = "cuda") -> torch.device:
    """Resolve the requested runtime device with CUDA fallback handling."""
    cuda_ok = torch.cuda.is_available() and (device_str == "cuda")
    return torch.device("cuda" if cuda_ok else "cpu")


def build_model(args, device: torch.device) -> LiBPINN:
    """Instantiate the LiBPINN model from centralized configuration."""
    # update model config with args if needed
    settings.model.n_cell = args.n_cell

    # instantiate model from centralized settings
    model = LiBPINN(**settings.model.model_dump())
    return model.to(device)


def build_balancer(device: torch.device):
    """Instantiate the loss balancer configured in ``config.settings``."""
    bal = LogCovLossBalancer(**settings.balancer.model_dump())
    return bal.to(device)


def build_optimizer(
    model: torch.nn.Module, balancer: torch.nn.Module, args
) -> optim.Optimizer:
    """Create the optimizer with separate groups for network, physics and balancer."""
    # split physics vs nn params
    phys_params, nn_params = [], []
    for name, param in model.named_parameters():
        if any(k in name for k in ["R0", "R1", "R2", "C1", "C2"]):
            phys_params.append(param)
        else:
            nn_params.append(param)

    return optim.AdamW(
        [
            {"params": nn_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": phys_params, "lr": args.lr * 0.5, "weight_decay": 0.0},
            {
                "params": balancer.parameters(),
                "lr": args.lr * 0.1,
                "weight_decay": 0.0,
            },
        ]
    )


# --------------------------------------------------------

# ============================================================
# Train
# ============================================================


def pretrain(args):
    """Run end-to-end pretraining, validation and checkpointing."""
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # Device setup
    device = get_device(args.device)

    # Data
    if args.dataset == "fobss":
        train_loader, val_loader = get_fobss_loader(
            root_dir=args.root_dir,
            foldername=args.foldername,
            slave=args.slave,
            interval=args.interval,
            method=args.method,
            split_val=args.split_val,
            window_length=args.window_length,
            step=args.step,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    elif args.dataset == "ebai":
        train_loader, val_loader = get_ebai_loader(
            root_dir=args.root_dir,
            filenames=args.filenames,
            split_val=args.split_val,
            window_length=args.window_length,
            step=args.step,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    # Model, balancer and optimizer (constructed from centralized settings)
    model = build_model(args, device)
    balancer = build_balancer(device)
    optimizer = build_optimizer(model, balancer, args)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )

    best_val = math.inf
    best_ckpt = os.path.join(
        args.save_dir, args.prefix + "-" + args.dataset + "-best.pt"
    )

    # -----------------------------
    # Epoch loop
    # -----------------------------

    history = {
        "train_losses": None,
        "val_losses": [],
        # store loss names (list of strings) for later analysis/plotting
        "loss_names": None,
    }
    start_time = time.time()
    # previous-step loss values used for delta/trend display
    prev_loss_values = None

    for epoch in range(args.epochs):
        model.train()
        balancer.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            if args.dataset == "fobss":
                I, V, T = batch
                I, V, T = (
                    to_device(I, device),
                    to_device(V, device),
                    to_device(T, device),
                )
                Tenv = None
            elif args.dataset == "ebai":
                I, V, T, Tenv = batch
                Tenv = Tenv[:, 1:, :]  # align time steps
                I, V, T, Tenv = (
                    to_device(I, device),
                    to_device(V, device),
                    to_device(T, device),
                    to_device(Tenv, device),
                )
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            out = model(I, V, T, Tenv, use_mean=False)
            losses = out["losses"]

            # record loss names once
            if history.get("loss_names") is None:
                if isinstance(losses, dict):
                    history["loss_names"] = list(losses.keys())
                    history["train_losses"] = {k: [] for k in history["loss_names"]}
                else:
                    history["loss_names"] = [f"loss_{i}" for i in range(len(losses))]
                    history["train_losses"] = {k: [] for k in history["loss_names"]}

            # Collect losses into named timeseries (support dict or list/tuple losses)
            if history.get("train_losses") is not None:
                if isinstance(losses, dict):
                    for name in history["loss_names"]:
                        val = losses.get(name)
                        try:
                            vf = (
                                float(val.item())
                                if hasattr(val, "item")
                                else float(val)
                            )
                        except Exception:
                            vf = None
                        history["train_losses"][name].append(vf)
                else:
                    for name, val in zip(history["loss_names"], losses):
                        try:
                            vf = (
                                float(val.item())
                                if hasattr(val, "item")
                                else float(val)
                            )
                        except Exception:
                            vf = None
                        history["train_losses"][name].append(vf)

            if args.use_balancer:
                # print("Using loss balancer.")
                # print(args.use_balancer)
                # uncertainty-weighted with balancer
                total_loss = balancer(losses)
            else:
                # baseline: simple sum
                if isinstance(losses, dict):
                    total_loss = sum(losses.values())

            # Backward
            total_loss.backward()

            # Gradient clipping (optional)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(balancer.parameters(), args.grad_clip)

            # Optimizer step
            optimizer.step()

            if args.use_scheduler:
                # Scheduler step (must come after optimizer.step())
                scheduler.step()

            # Logging
            running_loss += float(total_loss.detach())
            if (i + 1) % max(1, args.log_interval) == 0:
                avg_loss = running_loss / args.log_interval
                lv = balancer.weights_info()
                # print(lv)

                running_loss = 0.0

                print_logging(
                    epoch,
                    args.epochs,
                    i + 1,
                    len(train_loader),
                    avg_loss,
                    losses,
                    lv,
                    scheduler,
                    prev_losses=prev_loss_values,
                )

                # update prev_loss_values (list of floats) for next step
                if isinstance(losses, dict):
                    prev_loss_values = []
                    for v in losses.values():
                        try:
                            prev_loss_values.append(
                                float(v.item()) if hasattr(v, "item") else float(v)
                            )
                        except Exception:
                            prev_loss_values.append(None)
                else:
                    prev_loss_values = []
                    for v in losses:
                        try:
                            prev_loss_values.append(
                                float(v.item()) if hasattr(v, "item") else float(v)
                            )
                        except Exception:
                            prev_loss_values.append(None)

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        balancer.eval()

        if val_loader is not None and len(val_loader) > 0:
            val_total = 0.0
            n_batches = 0

            for batch in val_loader:
                if args.dataset == "fobss":
                    I, V, T = batch
                    I, V, T = (
                        to_device(I, device),
                        to_device(V, device),
                        to_device(T, device),
                    )
                    Tenv = None
                elif args.dataset == "ebai":
                    I, V, T, Tenv = batch
                    I, V, T, Tenv = (
                        to_device(I, device),
                        to_device(V, device),
                        to_device(T, device),
                        to_device(Tenv, device),
                    )
                    Tenv = Tenv[:, 1:, :]  # align time steps
                else:
                    raise ValueError(f"Unknown dataset: {args.dataset}")
                out = model(I, V, T, Tenv, use_mean=True)
                losses = out["losses"]
                if args.use_balancer:
                    val_loss = balancer(losses)
                else:
                    if isinstance(losses, dict):
                        val_loss = sum(losses.values())
                    else:
                        val_loss = sum(losses)
                val_total += val_loss.item()
                n_batches += 1

            val_avg = val_total / max(1, n_batches)
            print(f"[Validation] Epoch {epoch+1}: Loss = {val_avg:.6f}")
            history["val_losses"].append(val_avg)

            # Save best checkpoint
            if val_avg < best_val:
                best_val = val_avg
                best_time = time.time() - start_time
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "balancer_state": balancer.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "args": vars(args),
                        "best_val": best_val,
                        "best_time": best_time,
                    },
                    best_ckpt,
                )
                print(
                    f"[Checkpoint] New best saved at {best_ckpt} (val={best_val:.6f})"
                )
        else:
            print("[Validation] Skipped (no validation loader).")

        # -----------------------------
        # Save periodic checkpoints
        # -----------------------------
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.save_dir, f"{args.prefix}-{args.dataset}-epoch{epoch+1:03d}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "balancer_state": balancer.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                ckpt_path,
            )
            print(f"[Checkpoint] Epoch {epoch+1} checkpoint saved at: {ckpt_path}")

    # -----------------------------
    # Save last checkpoint
    # -----------------------------
    last_ckpt = os.path.join(
        args.save_dir, args.prefix + "-" + args.dataset + "_last.pt"
    )
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "balancer_state": balancer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "args": vars(args),
            "best_val": best_val,
        },
        last_ckpt,
    )
    print(f"[Done] Last checkpoint saved at: {last_ckpt}")
    if os.path.exists(best_ckpt):
        print(f"[Done] Best checkpoint: {best_ckpt} (val={best_val:.6f})")

    # Save history
    history_path = os.path.join(
        args.log_dir, args.prefix + "-" + args.dataset + "_history.pt"
    )
    torch.save(history, history_path)
    print(f"[Done] History saved at: {history_path}")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    # ============================================================
    # Args
    # ============================================================
    parser = argparse.ArgumentParser(description="Pretrain Battery Model")
    parser.add_argument(
        "--dataset", type=str, default="fobss", choices=["fobss", "ebai"]
    )
    parser.add_argument("--root_dir", type=str, default=str(settings.data.root_dir))
    parser.add_argument(
        "--foldername",
        type=str,
        default=str(settings.data.foldername),
    )
    parser.add_argument(
        "--filenames",
        type=str,
        default="BMS_Data_2026-01-12T11-52-18-112Z.csv,BMS_Data_2026-01-06T07-51-14-127Z.csv",
        help="Comma-separated list of filenames for ebai dataset",
    )
    parser.add_argument(
        "--slave",
        type=str,
        default=str(settings.data.slave),
        help="Slave number for voltage and temperature data",
    )
    parser.add_argument("--n_cell", type=int, default=settings.data.n_cell)
    parser.add_argument("--interval", type=float, default=settings.data.interval)
    parser.add_argument("--method", type=str, default=settings.data.method)
    parser.add_argument(
        "--split_val",
        type=float,
        default=settings.train.split_val,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--window_length", type=int, default=settings.data.window_length
    )
    parser.add_argument("--step", type=int, default=settings.train.step)
    parser.add_argument("--batch_size", type=int, default=settings.train.batch_size)
    parser.add_argument("--shuffle", type=bool, default=settings.train.shuffle)
    parser.add_argument("--num_workers", type=int, default=settings.train.num_workers)
    parser.add_argument("--pin_memory", type=bool, default=settings.train.pin_memory)
    parser.add_argument("--epochs", type=int, default=settings.train.epochs)
    parser.add_argument("--lr", type=float, default=settings.train.lr)
    parser.add_argument("--use_balancer", action="store_true", default=False)
    parser.add_argument("--use_scheduler", action="store_true", default=False)
    parser.add_argument(
        "--device", type=str, default=settings.train.device, choices=["cpu", "cuda"]
    )
    parser.add_argument("--save_dir", type=str, default=str(settings.save_dir))
    parser.add_argument("--log_dir", type=str, default=str(settings.log_dir))
    parser.add_argument("--seed", type=int, default=settings.train.seed)
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=settings.train.grad_clip,
        help="Max gradient norm; <=0 disables clipping",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=settings.train.weight_decay
    )
    parser.add_argument("--log_interval", type=int, default=settings.train.log_interval)

    parser.add_argument(
        "--save_interval", type=int, default=settings.train.save_interval
    )
    parser.add_argument("--prefix", type=str, default="pretrain-model")

    # Note: balancer and model hyperparameters are read from `config.settings`.
    args = parser.parse_args()

    pretrain(args)
