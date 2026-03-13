"""Shared utility functions for training scripts and experiment logging."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.table import Table


def set_seed(seed: int = 42):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(t, device):
    """Move a tensor-like object to the target device."""
    return t.to(device, non_blocking=True)


# pretty-print training status
def print_logging(
    epoch: int,
    epochs: int,
    step: int,
    total_steps: int,
    avg_loss: float,
    losses_obj,
    inv_info: dict,
    scheduler,
    prev_losses=None,
    param_group_names=("nn", "phys", "balancer"),
):
    """Render one training-status snapshot with loss trends and learned weights."""
    console = Console()
    # Build header
    console.print(
        f"[bold cyan]Epoch[/] {epoch+1}/{epochs} | [bold cyan]Step[/] {step}/{total_steps} | [bold]Avg Loss[/] {avg_loss:.4f}"
    )

    # Prepare loss table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Loss", style="dim", width=15)
    table.add_column("Value", justify="right")
    table.add_column("diag(Sigma)", justify="right")
    table.add_column("Delta", justify="right")

    # losses_obj may be dict (preferred) or list/tuple
    if isinstance(losses_obj, dict):
        keys = list(losses_obj.keys())
        vals = list(losses_obj.values())
    else:
        vals = list(losses_obj)
        keys = [f"loss_{i}" for i in range(len(vals))]

    inv_vars = inv_info.get("diag(Sigma)", []) if isinstance(inv_info, dict) else []

    # normalize prev_losses to list of floats if provided
    prev_list = None
    if prev_losses is not None:
        if isinstance(prev_losses, dict):
            prev_list = [prev_losses.get(k) for k in keys]
        else:
            prev_list = list(prev_losses)

    for idx, k in enumerate(keys):
        v = vals[idx]
        try:
            v_f = float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            v_f = None

        inv_f = round(inv_vars[idx], 4) if idx < len(inv_vars) else "-"

        # compute delta and trend arrow
        delta_str = "-"
        if v_f is not None and prev_list is not None and idx < len(prev_list):
            prev_v = prev_list[idx]
            try:
                prev_vf = (
                    float(prev_v)
                    if not hasattr(prev_v, "item")
                    else float(prev_v.item())
                )
            except Exception:
                prev_vf = None

            if prev_vf is not None:
                diff = v_f - prev_vf
                # arrow: down if decreased (good), up if increased
                if abs(diff) < 1e-12:
                    arrow = "→"
                    delta_str = f"{arrow} {diff:.6f}"
                elif diff < 0:
                    arrow = "↓"
                    delta_str = f"[blue]{arrow} {abs(diff):.6f}[/blue]"
                else:
                    arrow = "↑"
                    delta_str = f"[red]{arrow} {abs(diff):.6f}[/red]"

        table.add_row(
            k,
            (
                f"{v_f:.6f}"
                if isinstance(v_f, float)
                else (str(v_f) if v_f is not None else "-")
            ),
            str(inv_f),
            delta_str,
        )

    console.print(table)

    # Learning rates
    lrs = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else []
    lr_info = ", ".join(
        [f"{name}:{lr:.6g}" for name, lr in zip(param_group_names, lrs)]
    )
    console.print(f"[bold]LRs[/]: {lr_info}")
