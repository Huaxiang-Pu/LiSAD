"""Attack synthesis helpers for generating biased cell-voltage traces."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd


def make_bias_type(
    L: int,
    start_idx: int,
    end_idx: int,
    bias_type: str = "exp",
    tau_steps: float = None,
) -> np.ndarray:
    """Build a per-timestep attack intensity profile."""

    p = np.zeros(L, dtype=float)
    start_idx = max(0, min(start_idx, L))
    end_idx = max(0, min(end_idx, L))
    if end_idx <= start_idx:
        return p
    n = end_idx - start_idx
    r = np.arange(n, dtype=float)

    if bias_type == "step":
        p[start_idx:end_idx] = 1.0

    elif bias_type == "linear":
        p[start_idx:end_idx] = (r + 1) / n

    elif bias_type == "exp":
        tau = (
            tau_steps
            if (tau_steps is not None and tau_steps > 0)
            else max(1.0, n / 10.0)  # default tau is 1/10 of the length
        )
        v = np.exp(r / tau) - 1  # exponential growth starting from 0

        # normalize to 1
        p[start_idx:end_idx] = v / (v.max() + 1e-12)
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")

    return p


def attack(
    data: pd.DataFrame,
    cells: List[int],
    bias_per_cell: List[float],
    start_idx: int,
    end_idx: int,
    dt: float,
    bias_type: str = "exp",
    tau_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject a synthetic voltage bias into selected cell channels."""
    df = data.copy()  # shape (L, N_cells)
    L = data.shape[0]
    tau_steps = None if tau_seconds is None else max(1.0, tau_seconds / dt)
    end_idx = L if end_idx < 0 else end_idx
    p = make_bias_type(L, start_idx, end_idx, bias_type, tau_steps)[
        :, None
    ]  # shape (L,1)
    for c, v in zip(cells, bias_per_cell):
        df.iloc[:, c] += p[:, 0] * v

    # labels = p[:, 0] > 0.0  # boolean array
    # df_Label = pd.DataFrame(labels.astype(int), columns=['Label'])
    df_Label = pd.DataFrame(p[:, 0], columns=["Label"])
    return df, df_Label
