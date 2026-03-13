"""Physics-informed battery model used as the core representation learner."""

from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat

from models.losses import PILoss
from models.modules import NodeAttnLSTM
from models.utils import BayesPosParamLogExp, BayesPosParamSoftplus


class LiBPINN(nn.Module):
    """Estimate latent battery states and residual losses from I/V/T sequences."""

    def __init__(
        self,
        n_cell: int = 11,
        dt: float = 1.0,
        m: float = 0.8,
        cp: float = 900.0,
        eta: float = 1.0,
        Uoc_min: float = 2.8,
        Uoc_max: float = 4.2,
        Qcap: float = 40 * 3600.0,
        hA: float = 10.0,
        R0_p: tuple = (5e-3, 0.05),
        R1_p: tuple = (1e-3, 0.05),
        R2_p: tuple = (5e-3, 0.05),
        C1_p: tuple = (1e3, 0.05),
        C2_p: tuple = (5e3, 0.05),
        D_in: int = 3,
        D_hidden: int = 12,
        n_lstm_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: float = 0.0,
        n_heads: int = 4,
        n_trans_layers: int = 1,
        dim_feedforward: int = 128,
        trans_dropout: float = 0.0,
        require_adj: bool = True,
        use_adj_bias: bool = True,
        alpha_init: float = 1.0,
        beta_init: float = 0.1,
        V1_clip: float = 0.03,
        V2_clip: float = 0.20,
        tau_SOC: float = 1.0,
        eps_SOC: float = 0.01,
        tau_Uoc: float = 1.0,
        eps_Uoc: float = 0.01,
        device: str = "cuda",
    ):
        """Initialize the model, Bayesian physical parameters and backbone blocks."""
        super().__init__()
        self.n_cell = n_cell
        self.dt = dt
        self.m = m
        self.cp = cp
        self.eta = eta
        self.Uoc_min = Uoc_min
        self.Uoc_max = Uoc_max
        self.Qcap = Qcap
        self.hA = hA
        self.R0_p = R0_p
        self.R1_p = R1_p
        self.R2_p = R2_p
        self.C1_p = C1_p
        self.C2_p = C2_p
        self.V1_clip = V1_clip
        self.V2_clip = V2_clip
        self.tau_SOC = tau_SOC
        self.eps_SOC = eps_SOC
        self.tau_Uoc = tau_Uoc
        self.eps_Uoc = eps_Uoc
        self.device = device

        self._init_phys_params()
        self.adj = self._adj_matrix().float().to(self.device)
        self.norm_vit = nn.LayerNorm(D_in)

        self.st_module = NodeAttnLSTM(
            d_in=D_in,
            d_hidden=D_hidden,
            n_nodes=n_cell,
            n_lstm_layers=n_lstm_layers,
            bidirectional=bidirectional,
            lstm_dropout=lstm_dropout,
            n_heads=n_heads,
            n_trans_layers=n_trans_layers,
            dim_feedforward=dim_feedforward,
            trans_dropout=trans_dropout,
            require_adj=require_adj,
            use_adj_bias=use_adj_bias,
            alpha_init=alpha_init,
            beta_init=beta_init,
        ).to(self.device)

        self.pv_estimator = nn.Sequential(
            nn.Linear(D_hidden, D_hidden),
            nn.GELU(),
            nn.Linear(D_hidden, 2),
        )
        self.soc_estimator = nn.Sequential(
            nn.Linear(D_hidden, D_hidden),
            nn.GELU(),
            nn.Linear(D_hidden, 1),
        )
        self.uoc_estimator = nn.Sequential(
            nn.Linear(2, D_hidden),
            nn.GELU(),
            nn.Linear(D_hidden, 1),
        )

        self.pi_loss = PILoss(
            eta=self.eta,
            dt=self.dt,
            Qcap=self.Qcap,
            m=self.m,
            cp=self.cp,
            hA=self.hA,
            adj=self.adj,
        )

    def forward(
        self,
        I: torch.Tensor,
        V: torch.Tensor,
        T: torch.Tensor,
        T_env: torch.Tensor = None,
        use_mean: bool = False,
        reduce: str = "mean",
        return_outputs: bool = True,
        return_alphas: bool = False,
    ) -> Dict[str, object]:
        """Estimate latent states and compute all physics-informed residual losses."""
        B, L, N = V.shape

        I = repeat(I, "b l 1 -> b l n", n=N)
        X = torch.cat([I.unsqueeze(-1), V.unsqueeze(-1), T.unsqueeze(-1)], dim=-1)
        X = self.norm_vit(X)

        if return_alphas:
            H, alphas = self.st_module.forward_with_attn(X, self.adj)
            alphas = alphas.detach()
        else:
            H = self.st_module(X, self.adj)
            alphas = None

        H_flat = rearrange(H, "b l n d -> (b l n) d")

        pv_flat = self.pv_estimator(H_flat)
        pv_reshaped = rearrange(pv_flat, "(b l n) c -> b l n c", b=B, l=L, n=N, c=2)
        V1_raw, V2_raw = torch.chunk(pv_reshaped, 2, dim=-1)
        V1_raw = V1_raw.squeeze(dim=-1)
        V2_raw = V2_raw.squeeze(dim=-1)
        V1 = self.V1_clip * torch.tanh(V1_raw)
        V2 = self.V2_clip * torch.tanh(V2_raw)

        soc_flat = self.soc_estimator(H_flat)
        SOC_raw = rearrange(soc_flat, "(b l n) 1 -> b l n", b=B, l=L, n=N)
        SOC_norm = torch.sigmoid(SOC_raw / self.tau_SOC)
        SOC = SOC_norm * (1.0 - 2.0 * self.eps_SOC) + self.eps_SOC

        # Build grad-bearing tensors so autograd can form dUoc/dSOC and dUoc/dT.
        SOC_for_grad = SOC if SOC.requires_grad else SOC.detach().requires_grad_(True)
        T_for_grad = T if T.requires_grad else T.detach().requires_grad_(True)
        SOC_T = torch.cat(
            (SOC_for_grad.unsqueeze(-1), T_for_grad.unsqueeze(-1)), dim=-1
        )

        SOC_T_flat = rearrange(SOC_T, "b l n d -> (b l n) d")
        Uoc_raw = self.uoc_estimator(SOC_T_flat)
        Uoc_raw = rearrange(Uoc_raw, "(b l n) 1 -> b l n", b=B, l=L, n=N)
        Uoc_norm = torch.sigmoid(Uoc_raw / self.tau_Uoc)
        Uoc_norm = Uoc_norm * (1.0 - 2.0 * self.eps_Uoc) + self.eps_Uoc
        Uoc = self.Uoc_min + (self.Uoc_max - self.Uoc_min) * Uoc_norm

        dUocdSOC = torch.autograd.grad(
            outputs=Uoc.sum(), inputs=SOC_for_grad, create_graph=True, retain_graph=True
        )[0]
        dUocdT = torch.autograd.grad(
            outputs=Uoc.sum(),
            inputs=T_for_grad,
            create_graph=True,
            retain_graph=True,
        )[0]

        R0, R1, R2, C1, C2 = self._phys_params(use_mean=use_mean)
        losses = self.pi_loss.get_losses(
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
            T_env,
            reduce,
        )
        losses["KL_loss"] = self._KL_loss()

        outputs: Dict[str, torch.Tensor] | None = None
        if return_outputs:
            outputs = {
                "SOC": SOC.detach(),
                "Uoc": Uoc.detach(),
                "V1": V1.detach(),
                "V2": V2.detach(),
                "R0": R0.detach(),
                "R1": R1.detach(),
                "R2": R2.detach(),
                "C1": C1.detach(),
                "C2": C2.detach(),
            }

        return {"losses": losses, "outputs": outputs, "alphas": alphas}

    def _init_phys_params(self):
        """Create Bayesian positive parameters for the equivalent-circuit model."""
        self.R0 = BayesPosParamSoftplus(
            torch.full((self.n_cell,), self.R0_p[0]),
            min_val=1e-6,
            prior_sigma=self.R0_p[1],
            device=self.device,
        )
        self.R1 = BayesPosParamSoftplus(
            torch.full((self.n_cell,), self.R1_p[0]),
            min_val=1e-7,
            prior_sigma=self.R1_p[1],
            device=self.device,
        )
        self.R2 = BayesPosParamSoftplus(
            torch.full((self.n_cell,), self.R2_p[0]),
            min_val=1e-6,
            prior_sigma=self.R2_p[1],
            device=self.device,
        )
        self.C1 = BayesPosParamLogExp(
            torch.full((self.n_cell,), self.C1_p[0]),
            min_val=1e-3,
            prior_sigma=self.C1_p[1],
            device=self.device,
        )
        self.C2 = BayesPosParamLogExp(
            torch.full((self.n_cell,), self.C2_p[0]),
            min_val=1e-3,
            prior_sigma=self.C2_p[1],
            device=self.device,
        )

    def _phys_params(self, use_mean: bool):
        """Expand sampled or mean physical parameters to ``(1, 1, N)``."""
        if use_mean and not self.training:
            R0 = self.R0.mean()[None, None, :]
            R1 = self.R1.mean()[None, None, :]
            R2 = self.R2.mean()[None, None, :]
            C1 = self.C1.mean()[None, None, :]
            C2 = self.C2.mean()[None, None, :]
        else:
            R0 = self.R0()[None, None, :]
            R1 = self.R1()[None, None, :]
            R2 = self.R2()[None, None, :]
            C1 = self.C1()[None, None, :]
            C2 = self.C2()[None, None, :]

        return R0, R1, R2, C1, C2

    def _adj_matrix(self):
        """Build the fixed chain-structured adjacency matrix between cells."""
        adj = torch.zeros(self.n_cell, self.n_cell)
        for i in range(self.n_cell - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        return adj

    def _KL_loss(self) -> torch.Tensor:
        """Aggregate KL losses from all Bayesian parameters."""
        return (
            self.R0.kl_loss()
            + self.R1.kl_loss()
            + self.C1.kl_loss()
            + self.R2.kl_loss()
            + self.C2.kl_loss()
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L, N = 32, 200, 11
    I = torch.randn(B, L, 1, device=device)
    V = torch.randn(B, L, N, device=device).abs() * 3.7
    T = torch.randn(B, L, N, device=device).abs() * 25

    model = LiBPINN().to(device)
    out = model(I, V, T, return_outputs=True)
