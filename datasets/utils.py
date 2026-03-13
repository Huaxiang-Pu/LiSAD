"""Time-series preprocessing and visualization helpers for battery datasets."""

import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import medfilt

from datasets.fobss import FOBSS


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report and interpolate missing values in a dataframe."""
    result_df = df.copy()
    missing_counts = result_df.isna().sum()
    total_missing = missing_counts.sum()
    if total_missing > 0:
        print("Missing values detected:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"Column '{col}': {count} missing values")
    else:
        print("No missing values detected.")
    if total_missing > 0:
        result_df = result_df.interpolate(method="index", limit_direction="both")
    return result_df


def apply_median_filter(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Apply a median filter to all numeric columns except ``timestamp``."""
    assert window % 2 == 1, "Median-filter window must be an odd integer."

    df_filtered = df.copy()

    # Filter each numeric feature independently and keep timestamps unchanged.
    for col in df.columns:
        if col == "timestamp":
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().all():
                continue

            df_filtered[col] = medfilt(df[col], kernel_size=window)

    return df_filtered


def generate_uniform_timestamps(timestamps: list, interval: float = None) -> list:
    """Create a uniform numeric timestamp grid from an irregular reference series."""
    ts_array = np.sort(np.array(timestamps, dtype=float))

    if interval is None:
        diffs = np.diff(ts_array)
        interval = np.median(diffs) if len(diffs) > 0 else 1.0
        print(f"Generated uniform timestamps with interval: {interval} s")

    start = ts_array[0]
    end = ts_array[-1]
    uniform_timestamps = np.arange(start, end + interval, interval, dtype=float)

    return uniform_timestamps.tolist()


def align_timestamps_nearest(df: pd.DataFrame, timestamps: list) -> pd.DataFrame:
    """Align samples to target timestamps with left-nearest lookup."""
    target_timestamps = np.sort(np.array(timestamps, dtype=float))
    df_timestamps = df["timestamp"].values.astype(float)

    indices = np.searchsorted(df_timestamps, target_timestamps, side="right")
    indices = np.clip(indices - 1, 0, len(df_timestamps) - 1)

    aligned_data = df.iloc[indices].copy()
    aligned_data["timestamp"] = target_timestamps

    return aligned_data.set_index("timestamp")


def align_timestamps_linear(df: pd.DataFrame, timestamps: list) -> pd.DataFrame:
    """Align samples to target timestamps with vectorized linear interpolation."""
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_timestamps = np.sort(np.array(timestamps, dtype=float))
    df_timestamps = df["timestamp"].values.astype(float)

    feature_cols = df.columns.drop("timestamp")
    df_values = df[feature_cols].values

    idx = np.searchsorted(df_timestamps, target_timestamps, side="left")
    left_idxs = np.clip(idx - 1, 0, len(df_timestamps) - 1)
    right_idxs = np.clip(idx, 0, len(df_timestamps) - 1)

    denom = df_timestamps[right_idxs] - df_timestamps[left_idxs]
    mask = denom != 0

    frac = np.zeros(len(target_timestamps))
    frac[mask] = (target_timestamps[mask] - df_timestamps[left_idxs[mask]]) / denom[
        mask
    ]

    interpolated = df_values[left_idxs] + frac[:, np.newaxis] * (
        df_values[right_idxs] - df_values[left_idxs]
    )

    return pd.DataFrame(interpolated, index=target_timestamps, columns=feature_cols)


def align_timestamps(
    df: pd.DataFrame, timestamps: list, method: str = "linear"
) -> pd.DataFrame:
    """Dispatch timestamp alignment to the requested interpolation method."""
    if method == "nearest":
        return align_timestamps_nearest(df, timestamps)
    if method == "linear":
        return align_timestamps_linear(df, timestamps)
    raise ValueError(f"Unknown method: {method}")


def load_fobss_data(foldername, slave="0", interval=10.0, method="linear"):
    """Convenience helper that loads and aligns one FOBSS profile."""
    fobss = FOBSS(
        root_dir="/home/safer/workspace/DATABASES/FOBSS", foldername=foldername
    )

    df_I = handle_missing_values(fobss.datasets["Battery_Current"])
    df_V = handle_missing_values(fobss.datasets[f"Slave_{slave}_Voltages"])
    df_T = handle_missing_values(fobss.datasets[f"Slave_{slave}_Temperatures"])

    uniform_timestamps = generate_uniform_timestamps(df_I["timestamp"], interval)
    print("len(uniform_timestamps): ", len(uniform_timestamps))

    df_I = align_timestamps(df_I, uniform_timestamps, method)
    df_V = align_timestamps(df_V, uniform_timestamps, method)
    df_T = align_timestamps(df_T, uniform_timestamps, method)

    return df_I, df_V, df_T


def visualize_dataset(df_I, df_V, df_T, profile_name="Battery Profile", save_path=None):
    """Plot current, voltage and mean temperature for one aligned profile."""
    fig, ax_I = plt.subplots(figsize=(10, 4))

    x = df_I.index.values
    y_I = df_I["current in A"].values
    T_mean = df_T.mean(axis=1).values

    points = np.array([x, y_I]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(T_mean.min(), T_mean.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(T_mean)
    lc.set_linewidth(2)
    ax_I.add_collection(lc)
    ax_I.autoscale()

    ax_I.set_xlabel(f"Timestamp\n{profile_name}")
    ax_I.set_ylabel("Current (A)")

    cbar = plt.colorbar(lc, ax=ax_I)
    cbar.set_label("Temperature (degC)")

    ax_V = ax_I.twinx()
    voltage_lines = []
    voltage_labels = []

    colors = plt.cm.get_cmap("tab20b", df_V.shape[1])
    for i, col in enumerate(df_V.columns):
        (line,) = ax_V.plot(
            x, df_V[col], linestyle="--", alpha=0.6, label=f"V{col}", color=colors(i)
        )
        voltage_lines.append(line)
        voltage_labels.append(f"Cell {i + 1} Voltage")

    ax_V.set_ylabel("Voltage (V)")

    current_proxy = mlines.Line2D(
        [], [], color="k", linewidth=2, label="Current (temperature-coded)"
    )

    fig.legend(
        [current_proxy] + voltage_lines,
        ["Current (temp-coded)"] + voltage_labels,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.03),
        ncol=6,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)


def visualize_dataset_with_label(
    df_I, df_V, df_T, df_Label, profile_name="Battery Profile", save_path=None
):
    """Visualize a profile together with a dense attack-intensity label curve."""
    fig, ax_I = plt.subplots(figsize=(10, 4))

    x = df_I.index.values
    y_I = df_I["current in A"].values
    T_mean = df_T.mean(axis=1).values

    points = np.array([x, y_I]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(T_mean.min(), T_mean.max())
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(T_mean)
    lc.set_linewidth(2)
    ax_I.add_collection(lc)
    ax_I.autoscale()

    y_min, y_max = ax_I.get_ylim()
    for i in range(len(df_Label) - 1):
        start = df_I.index[i]
        end = df_I.index[i + 1]
        intensity = df_Label["Label"].iloc[i]
        if intensity > 0:
            alpha = intensity * 0.5
            ax_I.fill_between(
                [start, end], y_min, y_max, color="lightcoral", alpha=alpha
            )

    ax_I.set_xlabel(f"Timestamp\n{profile_name}")
    ax_I.set_ylabel("Current (A)")

    cbar = plt.colorbar(lc, ax=ax_I)
    cbar.set_label("Temperature (degC)")

    ax_V = ax_I.twinx()
    voltage_lines = []
    voltage_labels = []

    for i, col in enumerate(df_V.columns):
        (line,) = ax_V.plot(x, df_V[col], linestyle="--", alpha=0.6, label=f"V{col}")
        voltage_lines.append(line)
        voltage_labels.append(f"Cell {i + 1} Voltage")

    ax_V.set_ylabel("Voltage (V)")

    current_proxy = mlines.Line2D(
        [], [], color="k", linewidth=2, label="Current (temperature-coded)"
    )

    fig.legend(
        [current_proxy] + voltage_lines,
        ["Current (temp-coded)"] + voltage_labels,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.03),
        ncol=6,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300)


def mkdatasets(
    df_I,
    df_V,
    df_T,
    df_Env=None,
    df_Label=None,
    ID: str = "D10C25",
    datasetname: str = "FOBSS",
    type: str = "train",
    save_dir: str = "data/",
    slave: str = "0",
    interval: float = 10.0,
):
    """Export aligned signals as a flat csv dataset used by downstream tasks."""
    os.makedirs(f"{save_dir}/{datasetname}/{type}", exist_ok=True)

    if df_Env is not None:
        df = pd.concat(
            [df_I.round(2), df_V.round(3), df_T.round(1), df_Env.round(1)], axis=1
        )
        df.columns = (
            ["I"]
            + [f"V{i + 1}" for i in range(df_V.shape[1])]
            + [f"T{i + 1}" for i in range(df_T.shape[1])]
            + ["Env"]
        )
    else:
        df = pd.concat([df_I.round(2), df_V.round(3), df_T.round(1)], axis=1)
        df.columns = (
            ["I"]
            + [f"V{i + 1}" for i in range(df_V.shape[1])]
            + [f"T{i + 1}" for i in range(df_T.shape[1])]
        )

    if df_Label is not None:
        if isinstance(df_Label, pd.DataFrame) is False:
            df_Label = pd.DataFrame(df_Label, columns=["Label"])
        df_Label.index = df.index
        df = pd.concat([df, df_Label.round(2)], axis=1)
        df.columns = list(df.columns[:-1]) + ["Label"]

    filename = (
        f"{save_dir}/{datasetname}/{type}/"
        f"{ID}-slave@{slave}-interval@{interval:.1f}s.csv"
    )

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
