"""Detector wrapper that feeds PINN residual losses into a PyOD detector."""

from pyod.models.base import BaseDetector
import torch
import numpy as np
import pandas as pd
from datasets.dataloader import BatteryDataset


class LiSAD(BaseDetector):
    """Combine a pretrained PINN with a tabular anomaly detector."""

    def __init__(
        self,
        pinn,
        detector,
        window_length=200,
        n_cell=11,
        use_mean=True,
        contamination=0.1,
        device="auto",
    ):
        """Store detector components and resolve the execution device."""
        super().__init__(contamination=contamination)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.pinn = pinn.to(self.device)
        self.detector = detector
        self.window_length = window_length
        self.n_cell = n_cell
        self.use_mean = use_mean

    def fit(self, X, y=None):
        """Fit the downstream detector on loss features extracted from ``X``."""
        df_loss = self._get_loss(X, overlap=True)
        self.detector.fit(df_loss.values)

        self.decision_scores_ = self.detector.decision_function(df_loss.values)

        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Compute anomaly scores from PINN-derived loss features."""
        df_loss = self._get_loss(X, overlap=True)
        scores = self.detector.decision_function(df_loss.values)

        # pad scores to match the length of X
        # pad_length = self.window_length - 1
        # pad_value = scores[0]
        # scores = np.pad(
        #     scores, (pad_length, 0), "constant", constant_values=(pad_value, 0)
        # )

        return scores

    # def _get_loss(self, X, overlap=True):
    #     self.pinn.eval()
    #     Loss = []
    #     start_index = 0
    #     stop_index = X.shape[0] - self.window_length + 1
    #     step = 1 if overlap else self.window_length
    #     for i in range(start_index, stop_index, step):
    #         x_window = X[i : i + self.window_length]

    #         I = torch.from_numpy(x_window[:, 0:1]).unsqueeze(0).float().to(self.device)
    #         V = (
    #             torch.from_numpy(x_window[:, 1 : 1 + self.n_cell])
    #             .unsqueeze(0)
    #             .float()
    #             .to(self.device)
    #         )
    #         T = (
    #             torch.from_numpy(x_window[:, 1 + self.n_cell : 1 + 2 * self.n_cell])
    #             .unsqueeze(0)
    #             .float()
    #             .to(self.device)
    #         )
    #         if x_window.shape[1] > 1 + 2 * self.n_cell:
    #             Tenv = (
    #                 torch.from_numpy(x_window[:, 1 + 2 * self.n_cell :])[1:, :]
    #                 .unsqueeze(0)
    #                 .float()
    #                 .to(self.device)
    #             )
    #         else:
    #             Tenv = None
    #         losses = self.pinn(I, V, T, Tenv, use_mean=self.use_mean)["losses"]
    #         Loss.append({k: v.item() for k, v in losses.items()})
    #     df_loss = pd.DataFrame(Loss).drop(columns=["KL_loss"], axis=1)
    #     return df_loss

    def _get_loss(self, X, overlap=True):
        """Run the PINN and reshape per-timestep residuals into a dataframe."""
        I, V, T = (
            X[:, 0:1],
            X[:, 1 : 1 + self.n_cell],
            X[:, 1 + self.n_cell : 1 + 2 * self.n_cell],
        )
        I = torch.from_numpy(I).float().unsqueeze(0).to(self.device)
        V = torch.from_numpy(V).float().unsqueeze(0).to(self.device)
        T = torch.from_numpy(T).float().unsqueeze(0).to(self.device)
        if X.shape[1] > 1 + 2 * self.n_cell:
            Tenv = (
                torch.from_numpy(X[:, 1 + 2 * self.n_cell :])[1:, :]
                .float()
                .unsqueeze(0)
                .to(self.device)
            )
        else:
            Tenv = None
        losses = self.pinn(I, V, T, Tenv, use_mean=self.use_mean, reduce="none")[
            "losses"
        ]
        Loss = {}
        for name, loss in losses.items():
            if len(loss.shape) == 3:
                Loss[name] = loss.mean(dim=(0, 2)).cpu().detach().numpy()
                # print(f"{name}: {Loss[name].shape}")
                # pad to match the length of X
                pad_length = X.shape[0] - Loss[name].shape[0]
                pad_value = Loss[name][0]
                Loss[name] = np.pad(
                    Loss[name],
                    (pad_length, 0),
                    "constant",
                    constant_values=(pad_value, 0),
                )
                # print(f"{name}: {Loss[name].shape}")

        df_loss = pd.DataFrame(Loss)
        return df_loss
