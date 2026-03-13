"""Loss balancing module based on log-loss covariance modelling."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogCovLossBalancer(nn.Module):
    """Learn a covariance-aware weighting over multiple non-negative losses."""

    def __init__(
        self,
        n_losses: int,
        diag_eps: float = 1e-4,
        init_scale: float = 0.0,
        log_base: float = 10.0,
        log_eps: float = 1e-12,
        tau: float = 1.0,
    ):
        """Initialize the precision-matrix parameterization and transform settings."""
        super().__init__()

        self.n_losses = int(n_losses)
        self.diag_eps = float(diag_eps)
        self.init_scale = float(init_scale)
        self.log_base = float(log_base)
        self.log_eps = float(log_eps)
        self.tau = float(tau)

        shape = (self.n_losses, self.n_losses)
        init_tensor = torch.zeros(shape)
        if self.init_scale > 0.0:
            init_tensor = init_tensor + self.init_scale * torch.randn_like(init_tensor)
        self.raw_L: nn.Parameter = nn.Parameter(init_tensor)

        self.keys_order: list[str] | None = None

    def _build_cholesky(self) -> torch.Tensor:
        """Build a valid lower-triangular Cholesky factor for the precision matrix."""
        if self.raw_L is None:
            raise RuntimeError("raw_L is not initialized yet.")

        L_lower = torch.tril(self.raw_L)
        raw_diag = torch.diagonal(L_lower)
        diag_pos = F.softplus(raw_diag) + self.diag_eps
        strictly_lower = torch.tril(L_lower, diagonal=-1)
        return strictly_lower + torch.diag(diag_pos)

    def _precision_matrix(self) -> torch.Tensor:
        """Return the precision matrix defined by the learned Cholesky factor."""
        L_chol = self._build_cholesky()
        return L_chol @ L_chol.transpose(-1, -2)

    def _log_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured logarithmic transform to a non-negative scalar."""
        x = torch.clamp(x, min=0.0)
        x_shift = x + self.log_eps

        if self.log_base == math.e:
            return torch.log(x_shift)
        return torch.log(x_shift) / math.log(self.log_base)

    def forward(self, losses, cov_l2_reg: float = 0.5) -> torch.Tensor:
        """Combine a loss dictionary or sequence into one balanced scalar objective."""
        if isinstance(losses, dict):
            keys = list(losses.keys())
            vals = [losses[k] for k in keys]
        else:
            vals = list(losses)
            keys = [f"loss_{i}" for i in range(len(vals))]

        if self.keys_order is None:
            self.keys_order = keys

        n = len(self.keys_order)
        if n == 0:
            raise ValueError("No losses provided to LogCovLossBalancer.")
        if n != self.n_losses:
            raise RuntimeError(
                f"n_losses mismatch: configured={self.n_losses} but observed={n}"
            )
        if self.raw_L.shape != (n, n):
            raise RuntimeError(
                f"raw_L shape mismatch: expected ({n},{n}), got {tuple(self.raw_L.shape)}"
            )

        key2idx = {k: i for i, k in enumerate(self.keys_order)}
        L_log_list = [None] * n

        for k, v in zip(keys, vals):
            if k not in key2idx:
                raise KeyError(
                    f"Loss key '{k}' not found in keys_order {self.keys_order}."
                )
            idx = key2idx[k]

            Li = v
            if not torch.is_tensor(Li):
                Li = torch.tensor(Li, dtype=torch.float32, device=self.raw_L.device)
            if Li.ndim > 0:
                Li = Li.mean()

            L_log_list[idx] = self._log_transform(Li.to(self.raw_L.device))

        for i, Li_log in enumerate(L_log_list):
            if Li_log is None:
                raise RuntimeError(
                    f"Loss for key '{self.keys_order[i]}' is missing in this forward."
                )

        L_log_vec = torch.stack(L_log_list, dim=0)

        L_chol = self._build_cholesky()
        P = L_chol @ L_chol.transpose(-1, -2)

        PL = torch.mv(P, L_log_vec)
        PL = F.softmax(PL / self.tau, dim=0)
        quad = 0.5 * torch.dot(L_log_vec, PL)

        diag_L = torch.diagonal(L_chol)
        if torch.any(diag_L <= 0):
            diag_L = torch.clamp(diag_L, min=1e-8)

        log_det_term = -torch.log(diag_L).sum()
        total = quad + log_det_term

        if cov_l2_reg > 0.0:
            total = total + cov_l2_reg * (P.pow(2).sum())

        return total

    def weights_info(self):
        """Return diagnostic statistics of the learned log-loss covariance model."""
        with torch.no_grad():
            L_chol = self._build_cholesky()
            P = L_chol @ L_chol.transpose(-1, -2)
            Sigma = torch.linalg.inv(P)

            std = torch.sqrt(torch.diagonal(Sigma))
            denom = std.unsqueeze(0) * std.unsqueeze(1)
            corr = Sigma / (denom + 1e-12)
            Sigma_diag = torch.diagonal(Sigma)

            return {
                "precision": P.cpu().tolist(),
                "covariance": Sigma.cpu().tolist(),
                "diag(Sigma)": Sigma_diag.cpu().tolist(),
                "std": std.cpu().tolist(),
                "corr": corr.cpu().tolist(),
                "keys": list(self.keys_order) if self.keys_order is not None else [],
                "space": "log",
            }
