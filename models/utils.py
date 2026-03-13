"""Reusable Bayesian parameterizations for positive physical quantities."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorOrFloat = Union[torch.Tensor, float]


class BayesPosParamBase(nn.Module):
    """Base class for Bayesian positive parameters in latent Gaussian space."""

    def __init__(
        self,
        raw0: torch.Tensor,
        min_val: float,
        prior_mu: Optional[TensorOrFloat],
        prior_sigma: TensorOrFloat,
        device: str = "cuda",
    ):
        """Initialize variational posterior and prior hyperparameters."""
        super().__init__()
        raw0 = raw0.to(device)

        self.mu = nn.Parameter(raw0.clone())
        self.rho = nn.Parameter(torch.full_like(raw0, -3.0))
        self.min_val = float(min_val)

        if prior_mu is None:
            prior_mu = raw0.detach().clone()
        if not torch.is_tensor(prior_mu):
            prior_mu = torch.full_like(raw0, float(prior_mu))
        self.register_buffer("prior_mu", prior_mu.clone().detach().to(device))

        if not torch.is_tensor(prior_sigma):
            prior_sigma = torch.full_like(raw0, float(prior_sigma))
        self.register_buffer("prior_sigma", prior_sigma.clone().detach().to(device))

    def sample_raw(self):
        """Draw a differentiable sample in latent space."""
        eps = torch.randn_like(self.mu)
        sigma = F.softplus(self.rho)
        raw = self.mu + sigma * eps
        return raw, sigma

    def std(self) -> torch.Tensor:
        """Return the posterior standard deviation in latent space."""
        return F.softplus(self.rho)

    def kl_loss(self) -> torch.Tensor:
        """Apply variance-only KL regularization against the configured prior."""
        sigma = F.softplus(self.rho)
        prior_var = self.prior_sigma**2
        post_var = sigma**2
        kl = 0.5 * (
            (post_var / prior_var) - 1.0 + torch.log(prior_var / post_var + 1e-12)
        )
        return kl.sum()

    def transform(self, raw: torch.Tensor) -> torch.Tensor:
        """Transform a latent variable into the positive parameter space."""
        raise NotImplementedError

    def mean(self) -> torch.Tensor:
        """Return the deterministic estimate in transformed space."""
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Sample a positive parameter value in transformed space."""
        raw, _ = self.sample_raw()
        return self.transform(raw)


class BayesPosParamSoftplus(BayesPosParamBase):
    """Positive parameterization based on the softplus transform."""

    def __init__(
        self,
        init: torch.Tensor,
        min_val: float = 1e-9,
        prior_mu: Optional[TensorOrFloat] = None,
        prior_sigma: TensorOrFloat = 1.0,
        device: str = "cuda",
    ):
        """Initialize a softplus-parameterized positive latent variable."""
        init = init.to(device)
        raw0 = torch.log(torch.expm1(init - min_val) + 1e-12)
        super().__init__(
            raw0=raw0,
            min_val=min_val,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            device=device,
        )

    def transform(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a latent sample to a strictly positive resistance-like value."""
        return F.softplus(raw) + self.min_val

    def mean(self) -> torch.Tensor:
        """Return the deterministic posterior mean in transformed space."""
        return F.softplus(self.mu) + self.min_val


class BayesPosParamLogExp(BayesPosParamBase):
    """Positive parameterization based on the exponential transform."""

    def __init__(
        self,
        init: torch.Tensor,
        min_val: float = 1e-9,
        prior_mu: Optional[TensorOrFloat] = None,
        prior_sigma: TensorOrFloat = 0.5,
        device: str = "cuda",
    ):
        """Initialize an exp-parameterized positive latent variable."""
        init = init.to(device)
        raw0 = torch.log(init - min_val + 1e-12)
        super().__init__(
            raw0=raw0,
            min_val=min_val,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            device=device,
        )

    def transform(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a latent sample to a strictly positive capacity-like value."""
        return torch.exp(raw) + self.min_val

    def mean(self) -> torch.Tensor:
        """Return the transformed posterior mean under a log-normal approximation."""
        sigma = F.softplus(self.rho)
        return torch.exp(self.mu + 0.5 * sigma**2) + self.min_val
