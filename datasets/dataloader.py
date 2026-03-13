"""Dataset wrappers and loader builders used by training and evaluation."""

import sys

sys.path.append(".")

from typing import Tuple, Optional, List

import os
import math
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from datasets.fobss import FOBSS
from datasets.utils import *


class BatteryDataset(Dataset):
    """Slice synchronized battery signals into sliding-window training samples."""

    def __init__(self, df_I, df_V, df_T, df_Tenv=None, window_length=50, step=1):
        """Store aligned current, voltage and temperature series for sampling."""
        self.window_length = window_length
        self.step = step

        self.I = df_I.values
        self.V = df_V.values
        self.T = df_T.values
        if df_Tenv is not None:
            self.Tenv = df_Tenv.values

    def __len__(self):
        """Return the number of windows available under the current stride."""
        return (len(self.I) - self.window_length) // self.step + 1

    def __getitem__(self, idx):
        """Return one window of ``(I, V, T[, Tenv])`` tensors."""
        start_idx = idx * self.step
        end_idx = start_idx + self.window_length

        I_w = torch.from_numpy(self.I[start_idx:end_idx]).float()
        V_w = torch.from_numpy(self.V[start_idx:end_idx]).float()
        T_w = torch.from_numpy(self.T[start_idx:end_idx]).float()

        if hasattr(self, "Tenv"):
            Tenv_w = torch.from_numpy(self.Tenv[start_idx:end_idx]).float()
            return I_w, V_w, T_w, Tenv_w

        return I_w, V_w, T_w


def preprocess_fobss(fobss: FOBSS, slave="0", interval=10.0, method="linear"):
    """Clean, smooth and align one FOBSS profile for a selected slave board."""

    df_I = handle_missing_values(fobss.datasets["Battery_Current"])
    df_V = handle_missing_values(fobss.datasets[f"Slave_{slave}_Voltages"])
    df_T = handle_missing_values(fobss.datasets[f"Slave_{slave}_Temperatures"])
    # print(df_I.columns, df_V.columns, df_T.columns)

    df_I = apply_median_filter(df_I, window=7)
    df_V = apply_median_filter(df_V, window=7)
    df_T = apply_median_filter(df_T, window=9)

    uniform_timestamps = generate_uniform_timestamps(df_I["timestamp"], interval)

    df_I = align_timestamps(df_I, uniform_timestamps, method)
    df_V = align_timestamps(df_V, uniform_timestamps, method)
    df_T = align_timestamps(df_T, uniform_timestamps, method)

    # print(df_I.shape, df_V.shape, df_T.shape)

    return df_I, df_V, df_T


def _parse_list_arg(s: str) -> List[str]:
    """Normalize a comma-separated argument or iterable into a list of strings."""
    if isinstance(s, (list, tuple)):
        return [str(x).strip() for x in s]
    if isinstance(s, str):
        return [x.strip() for x in s.split(",") if x.strip() != ""]
    return [str(s)]


def _num_windows(n_rows: int, window_length: int, step: int) -> int:
    """Compute how many sliding windows fit in a sequence."""
    if n_rows < window_length:
        return 0
    return 1 + (n_rows - window_length) // step


def get_fobss_loader(
    root_dir: str = "/home/safer/workspace/DATABASES/FOBSS",
    foldername: str = "profile_-25A_10A_04_12_18,-10A_jump_17_12_18,-25A_jump_17_12_18,Profile 25A,Profile 25A Run 070618_2,Ri Jumps 25A,Ri Jumps 25A Run 2,Ri Jumps 25A Run 3,Ri Jumps 25A Run 4",
    slave: str = "0,1",
    interval: float = 5.0,
    method: str = "linear",
    split_val: float = 0.3,
    window_length: int = 100,
    step: int = 100,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build FOBSS train/validation dataloaders from one or more profiles."""
    if foldername == "all":
        folder_list = [
            d
            for d in os.listdir(root_dir + "/data/")
            if os.path.isdir(os.path.join(root_dir + "/data/", d))
        ]
    else:
        folder_list = _parse_list_arg(foldername)

    slave_list = _parse_list_arg(slave)

    train_sets: List[Dataset] = []
    val_sets: List[Dataset] = []

    for folder in folder_list:
        for slave in slave_list:
            print(f"Processing folder: {folder}, slave: {slave}")

            # 1) load and preprocess data
            fobss = FOBSS(root_dir, folder)
            df_I, df_V, df_T = preprocess_fobss(fobss, slave, interval, method)

            # 2) split data
            n = len(df_I)
            if n == 0:
                continue
            val_len = int(math.floor(split_val * n)) if split_val > 0 else 0
            train_len = n - val_len

            # Note: only create dataset if there is data
            n_train_win = _num_windows(train_len, window_length, step)
            if n_train_win > 0:
                train_sets.append(
                    BatteryDataset(
                        df_I=df_I.iloc[:train_len, :],
                        df_V=df_V.iloc[:train_len, :],
                        df_T=df_T.iloc[:train_len, :],
                        window_length=window_length,
                        step=step,
                    )
                )
            else:
                print(
                    f"[INFO] Folder '{folder}' has no train windows "
                    f"(rows={train_len} < window_length={window_length}), skip train part."
                )

            if val_len > 0:
                n_val_win = _num_windows(val_len, window_length, step)
                if n_val_win > 0:
                    val_sets.append(
                        BatteryDataset(
                            df_I=df_I.iloc[train_len:, :],
                            df_V=df_V.iloc[train_len:, :],
                            df_T=df_T.iloc[train_len:, :],
                            window_length=window_length,
                            step=step,
                        )
                    )
                else:
                    print(
                        f"[INFO] Folder '{folder}' has no val windows "
                        f"(rows={val_len} < window_length={window_length}), skip val part."
                    )

    # 3) combine datasets
    if len(train_sets) == 0:
        raise RuntimeError("No training data found after processing the given folders.")

    train_ds = train_sets[0] if len(train_sets) == 1 else ConcatDataset(train_sets)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    # 4) create val loader
    val_loader = None
    if split_val > 0 and len(val_sets) > 0:
        val_ds = val_sets[0] if len(val_sets) == 1 else ConcatDataset(val_sets)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    return train_loader, val_loader


def preprocess_ebai_data(
    root_folder: str = "../", filename: str = "", interval: float = 10.0
):
    """Load one EBAI csv file and align its channels to a uniform time grid."""

    # Load data
    file_path = os.path.join(root_folder, filename)
    data = pd.read_csv(file_path)

    # Convert timestamp to seconds since epoch
    data["timestamp"] = pd.to_datetime(data["timestamp"]).astype("int64") / 1e9

    df_I = pd.DataFrame(data["current_mA"].apply(lambda x: x / 1000))  # Convert mA to A
    df_I["timestamp"] = data["timestamp"].to_list()
    df_I.rename(columns={"current_mA": "current in A"}, inplace=True)

    df_V = pd.DataFrame(
        data[[f"cell{i}_mV".format(i) for i in range(1, 10)]].apply(lambda x: x / 1000)
    )  # Convert mV to V
    df_V["timestamp"] = data["timestamp"].to_list()

    df_T = pd.DataFrame(data[[f"cell{i}_C".format(i) for i in range(1, 10)]])
    df_T["timestamp"] = data["timestamp"].to_list()

    df_Tenv = pd.DataFrame(data["env_C"])
    df_Tenv["timestamp"] = data["timestamp"].to_list()

    # Apply median filter to smooth data
    df_I = apply_median_filter(df_I, window=7)
    df_V = apply_median_filter(df_V, window=7)
    df_T = apply_median_filter(df_T, window=9)

    # generate uniform timestamps
    uniform_timestamps = generate_uniform_timestamps(
        data["timestamp"], interval=interval
    )

    # align dataframes to uniform timestamps
    df_I = align_timestamps(df_I, uniform_timestamps, method="linear")
    df_V = align_timestamps(df_V, uniform_timestamps, method="linear")
    df_T = align_timestamps(df_T, uniform_timestamps, method="linear")
    df_Tenv = align_timestamps(df_Tenv, uniform_timestamps, method="linear")

    return df_I, df_V, df_T, df_Tenv


def get_ebai_loader(
    root_dir: str = "/home/safer/workspace/LiSAD/ourdata",
    filenames: str = "BMS_Data_2026-01-12T11-52-18-112Z,BMS_Data_2026-01-06T07-51-14-127Z",
    interval: float = 10.0,
    split_val: float = 0.3,
    window_length: int = 100,
    step: int = 1,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 16,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build EBAI train/validation dataloaders from one or more csv files."""
    file_list = _parse_list_arg(filenames)
    train_sets: List[Dataset] = []
    val_sets: List[Dataset] = []
    for file in file_list:
        # 1) load and preprocess data
        df_I, df_V, df_T, df_Tenv = preprocess_ebai_data(
            root_dir, file, interval=interval
        )

        # 2) split data
        n = len(df_I)
        if n == 0:
            continue
        val_len = int(math.floor(split_val * n)) if split_val > 0 else 0
        train_len = n - val_len
        # Note: only create dataset if there is data
        n_train_win = _num_windows(train_len, window_length, step)
        if n_train_win > 0:
            train_sets.append(
                BatteryDataset(
                    df_I=df_I.iloc[:train_len, :],
                    df_V=df_V.iloc[:train_len, :],
                    df_T=df_T.iloc[:train_len, :],
                    df_Tenv=df_Tenv.iloc[:train_len, :],
                    window_length=window_length,
                    step=step,
                )
            )
        else:
            print(
                f"[INFO] File '{file}' has no train windows "
                f"(rows={train_len} < window_length={window_length}), skip train part."
            )
        if val_len > 0:
            n_val_win = _num_windows(val_len, window_length, step)
            if n_val_win > 0:
                val_sets.append(
                    BatteryDataset(
                        df_I=df_I.iloc[train_len:, :],
                        df_V=df_V.iloc[train_len:, :],
                        df_T=df_T.iloc[train_len:, :],
                        df_Tenv=df_Tenv.iloc[train_len:, :],
                        window_length=window_length,
                        step=step,
                    )
                )
            else:
                print(
                    f"[INFO] File '{file}' has no val windows "
                    f"(rows={val_len} < window_length={window_length}), skip val part."
                )
        # 3) combine datasets
        if train_sets:
            train_dataset = torch.utils.data.ConcatDataset(train_sets)
        else:
            train_dataset = None

        if val_sets:
            val_dataset = torch.utils.data.ConcatDataset(val_sets)
        else:
            val_dataset = None

        # 4) create dataloaders
        train_loader = (
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            if train_dataset is not None
            else None
        )

        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            if val_dataset is not None
            else None
        )

        return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_fobss_loader()
    for I, V, T in train_loader:
        print(I.shape, V.shape, T.shape)
        break
