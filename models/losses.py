"""Physics-informed residual losses used to train the LiBPINN model."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


class PILoss:
    """Bundle the residual terms used by the physics-informed training objective."""

    def __init__(
        self,
        eta: float,
        dt: float,
        Qcap: float,
        m: float,
        cp: float,
        hA: float,
        I_th: float = 0.1,
        mono_margin: float = 1e-4,
        mono_beta: float = 10.0,
        adj: torch.Tensor = None,
        normalized_lap: bool = True,
        dUocdT_sign: float = -1.0,
        margin: float = 0.0,
    ):
        """Store physical constants and prepare optional graph regularizers."""
        self.eta = eta
        self.dt = dt
        self.Qcap = Qcap
        self.m = m
        self.cp = cp
        self.hA = hA
        self.adj = adj
        self.I_th = I_th
        self.mono_margin = mono_margin
        self.mono_beta = mono_beta
        self.dUocdT_sign = dUocdT_sign
        self.margin = margin
        self.Lap = (
            self._laplacian(self.adj, normalized=normalized_lap)
            if adj is not None
            else None
        )

    def get_losses(
        self,
        R0,
        R1,
        R2,
        C1,
        C2,
        I,
        V,
        V1,
        V2,
        T,
        SOC,
        dUocdSOC,
        dUocdT,
        Uoc,
        T_env=None,
        reduce="mean",
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Compute all enabled residual losses for a forward pass."""
        dVdt = torch.diff(V, n=1, dim=1)
        dIdt = torch.diff(I, n=1, dim=1)
        dV1dt = torch.diff(V1, n=1, dim=1)
        dV2dt = torch.diff(V2, n=1, dim=1)
        dTdt = torch.diff(T, n=1, dim=1)
        dSOCdt = torch.diff(SOC, n=1, dim=1)
        dUocdt = torch.diff(Uoc, n=1, dim=1)

        dUocdt_hat = dUocdSOC[:, :-1, :] * dSOCdt + dUocdT[:, :-1, :] * dTdt
        T_env = self._envtemp_proxy(T) if T_env is None else T_env

        return {
            "KVL_loss": self.loss_KVL(Uoc, R0, I, V1, V2, V, reduce),
            "dVdt_loss": self.loss_dVdt(
                dUocdt_hat, R0, dIdt, dV1dt, dV2dt, dVdt, reduce
            ),
            "RC1_loss": self.loss_RC(I, V1, R1, C1, reduce),
            "RC2_loss": self.loss_RC(I, V2, R2, C2, reduce),
            "dSOCdt_loss": self.loss_dSOCdt(dSOCdt, I, reduce),
            "dTdt_loss": self.loss_dTdt(I, R0, V1, V2, dUocdT, T, T_env, dTdt, reduce),
            "mono_loss": self.loss_mono(
                dUocdSOC, self.mono_margin, self.mono_beta, reduce
            ),
            "smooth_loss": self.loss_smooth([dSOCdt, dV1dt, dV2dt, dUocdt]),
        }

    def loss_KVL(self, Uoc, R0, I, V1, V2, V, reduce="mean"):
        """Kirchhoff-like terminal-voltage consistency loss."""
        residual = (Uoc - R0 * I - V1 - V2 - V).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_dUocdt(self, dUocdt_hat, dUocdt, reduce="mean"):
        """First-order Taylor consistency loss for open-circuit voltage dynamics."""
        residual = (dUocdt_hat - dUocdt).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_dVdt(self, dUocdt_hat, R0, dIdt, dV1dt, dV2dt, dVdt, reduce="mean"):
        """Discrete voltage-dynamics consistency loss."""
        residual = (dUocdt_hat - R0 * dIdt - dV1dt - dV2dt - dVdt).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_RC(self, I, V, R, C, reduce="mean"):
        """Equivalent-circuit update loss for one RC branch."""
        alpha = torch.exp(-self.dt / (R * C))
        V_hat = alpha * V[:, :-1, :] + (1.0 - alpha) * R * I[:, :-1, :]
        residual = (V_hat - V[:, 1:, :]).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_dSOCdt(self, dSOCdt, I, reduce="mean"):
        """Coulomb-counting consistency loss for state of charge."""
        dSOCdt_error = dSOCdt + (self.eta * self.dt / self.Qcap) * 0.5 * (
            I[:, 1:, :] + I[:, :-1, :]
        )
        residual = dSOCdt_error.pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_mono(self, dUocdSOC, margin: float, beta: float, reduce="mean"):
        """Penalize negative slope of Uoc with respect to SOC."""
        residual = F.softplus(-(dUocdSOC - margin), beta=beta).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_dTdt(self, I, R0, V1, V2, dUocdT, T, T_env, dTdt, reduce="mean"):
        """Thermal-model consistency loss on temperature increments."""
        thermal_rhs = (
            I[:, 1:, :].pow(2) * R0
            + I[:, 1:, :] * (V1[:, 1:, :] + V2[:, 1:, :])
            + I[:, 1:, :] * T[:, 1:, :] * dUocdT[:, 1:, :]
            - self.hA * (T[:, 1:, :] - T_env)
        )
        dTdt_hat = (self.dt / (self.m * self.cp)) * thermal_rhs
        residual = (dTdt_hat - dTdt).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_SOC_relax(self, dSOCdt, I0_mask, reduce="mean"):
        """Penalize SOC drift during near-zero-current resting periods."""
        residual = (I0_mask * dSOCdt).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_RC_relax(self, V, R, C, I0_mask, reduce="mean"):
        """Penalize non-decaying RC branch voltage during resting periods."""
        alpha = torch.exp(-self.dt / (R * C))
        V_hat = alpha * V[:, :-1, :]
        residual = I0_mask * (V_hat - V[:, 1:, :]).pow(2)
        return residual.mean() if reduce == "mean" else residual

    def loss_Uoc_Lap(self, Uoc, L, reduce: str = "mean"):
        """Apply graph-Laplacian smoothing to open-circuit voltage."""
        return self.loss_graph_Lap(Uoc, L, reduce=reduce)

    def loss_SOC_Lap(self, SOC, L, reduce: str = "mean"):
        """Apply graph-Laplacian smoothing to state of charge."""
        return self.loss_graph_Lap(SOC, L, reduce=reduce)

    def loss_graph_Lap(self, X, Lap, reduce: str = "mean"):
        """Compute a generic graph-Laplacian smoothness penalty over cells."""
        Lap = Lap.to(X.device, X.dtype)
        B, T, N = X.shape
        X_flat = X.reshape(-1, N)
        lap_res_flat = X_flat @ Lap.t()
        lap_res = lap_res_flat.view(B, T, N)
        lap_loss = lap_res.pow(2)
        return lap_loss.mean() if reduce == "mean" else lap_loss

    def loss_sign(
        self,
        X: torch.Tensor,
        sign: float = -1.0,
        margin: float = 0.0,
        reduce: str = "mean",
    ):
        """Penalize violations of an expected sign pattern."""
        q = sign * X
        violations = F.relu(margin - q)
        sign_loss = violations.pow(2)
        return sign_loss.mean() if reduce == "mean" else sign_loss

    def loss_smooth(self, *args):
        """Aggregate temporal smoothness penalties for several state sequences."""
        return self._temporal_grad(*args)

    @staticmethod
    def _temporal_grad(*xs):
        """Compute mean squared first-order temporal gradients."""
        if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
            xs = tuple(xs[0])

        xs = [x for x in xs if x is not None]
        example_tensor = next((t for t in xs if isinstance(t, torch.Tensor)), None)
        if example_tensor is None:
            return torch.tensor(0.0, dtype=torch.float32)

        total = None
        for x in xs:
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"_temporal_grad expects torch.Tensor inputs, got {type(x)}"
                )
            if x.size(1) < 2:
                continue
            diff = x[:, 1:, ...] - x[:, :-1, ...]
            loss = diff.pow(2).mean()
            total = loss if total is None else total + loss

        if total is None:
            return example_tensor.new_zeros(())
        return total

    @staticmethod
    def _envtemp_proxy(T, alpha=0.99):
        """Estimate ambient temperature from pack measurements when absent."""
        T_pack_med = T[:, :-1, :].median(dim=2, keepdim=True).values
        return alpha * T_pack_med + (1 - alpha) * T_pack_med.mean(
            dim=1, keepdim=True
        )

    @staticmethod
    def _laplacian(
        adj: torch.Tensor,
        normalized: bool = False,
        add_self_loops: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute the standard or normalized graph Laplacian from adjacency."""
        assert adj.dim() == 2 and adj.size(0) == adj.size(
            1
        ), "adj must be a square matrix of shape (N, N)"

        if add_self_loops:
            adj = adj.clone()
            idx = torch.arange(adj.size(0), device=adj.device)
            adj[idx, idx] = adj[idx, idx] + 1.0

        deg = adj.sum(dim=-1)

        if not normalized:
            return torch.diag(deg) - adj

        deg_inv_sqrt = (deg + eps).pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ adj @ D_inv_sqrt
        I = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        return I - A_norm
