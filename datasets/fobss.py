"""FOBSS dataset access utilities.

This module wraps the FOBSS on-disk layout and exposes helpers for locating,
loading and inspecting the battery monitoring csv files used by the project.
"""

import os

import pandas as pd


def visualize_folder_structure(folder_path, prefix=""):
    """Recursively print the directory tree rooted at ``folder_path``."""
    items = os.listdir(folder_path)
    for index, item in enumerate(items):
        item_path = os.path.join(folder_path, item)
        is_last = index == len(items) - 1
        if os.path.isdir(item_path):
            print(prefix + ("`-- " if is_last else "|-- ") + item + "/")
            new_prefix = prefix + ("    " if is_last else "|   ")
            visualize_folder_structure(item_path, new_prefix)
        else:
            print(prefix + ("`-- " if is_last else "|-- ") + item)


def find_file_full_path(folder_dir, file_name):
    """Search ``folder_dir`` recursively and return the first matching file path."""
    for root, dirs, files in os.walk(folder_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


class FOBSS(object):
    """Loader for a single FOBSS battery profile folder."""

    def __init__(self, root_dir="./data/FOBSS", foldername=None):
        """Initialize the loader and eagerly read data when ``foldername`` is set."""
        self.root_dir = root_dir
        self.folder = None
        self.datasets = {}
        self.foldername = foldername
        if foldername is not None:
            self.folder = os.path.join(self.root_dir, f"data/{foldername}")
            self.load_all_data()

    def folder_structure_visual(self):
        """Print the folder tree of the active profile directory."""
        visualize_folder_structure(self.folder)

    def load_module_data(self, filename, header=2, delimiter=";"):
        """Load one pack-level FOBSS csv file such as battery current or voltage."""
        filepath = find_file_full_path(self.folder, f"{filename}.csv")
        data = pd.read_csv(filepath, header=header, delimiter=delimiter)
        data.columns = data.columns.str.replace("#time in s", "timestamp")
        return data

    def load_slave_data(self, slave_id, filename):
        """Load one slave-level csv file for cell temperatures or voltages."""
        filename = f"Slave_{slave_id}_Cell_{filename}.csv"
        filepath = find_file_full_path(self.folder, filename)
        slave_data = pd.read_csv(filepath, header=3, delimiter=";")
        slave_data.columns = slave_data.columns.str.replace("#", "timestamp")
        return slave_data

    def load_all_data(self):
        """Populate ``self.datasets`` with all standard FOBSS tables."""
        for equipment in ["Inverter", "Battery"]:
            for physical_quantity in ["Current", "Voltage"]:
                self.datasets[f"{equipment}_{physical_quantity}"] = (
                    self.load_module_data(f"{equipment}_{physical_quantity}")
                )

        for slave_id in range(4):
            for filename in ["Temperatures", "Voltages"]:
                self.datasets[f"Slave_{slave_id}_{filename}"] = self.load_slave_data(
                    slave_id, filename
                )

    def datasets_info(self):
        """Print a compact summary for each loaded dataframe."""
        for key, value in self.datasets.items():
            print(
                f'{key}>> \t| shape={value.shape} \t| mean={value.iloc[:, 1:].values.mean()} \t| std={value.iloc[:, 1:].values.std()} \t| start_time={value.iloc[0]["timestamp"]} \t| end_time={value.iloc[-1]["timestamp"]} \t|'
            )


if __name__ == "__main__":
    fobss = FOBSS(
        root_dir="/home/safer/workspace/databases/FOBSS",
        foldername="profile_-25A_10A_04_12_18",
    )
    fobss.folder_structure_visual()
    fobss.datasets_info()
