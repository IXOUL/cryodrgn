"""
Gaussian primitive decoder for cryoDRGN.

Representation
--------------
For each latent conformation z, represent the real-space density as a sum of
isotropic Gaussian primitives

    rho_z(x) = sum_j a_j(z) * G(x; mu_j(z), sigma)

and evaluate its Fourier transform analytically at cryoDRGN's rotated Fourier
coordinates k.

cryoDRGN's lattice coordinates used by train_vae are frequencies in
cycles / pixel, so centers mu are represented in real-space pixel units.

The class follows cryoDRGN's decoder interface:
    forward(lattice) -> Hartley values
    eval_volume(...) -> real-space 3-D density volume
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cryodrgn import fft
from cryodrgn.decoders.base import Decoder


Norm = Sequence[Any]


class _GaussianParamGenerator(nn.Module):
    """Small MLP mapping latent z -> per-Gaussian parameter offsets."""

    def __init__(
        self,
        zdim: int,
        n_gaussians: int,
        hidden_dim: int = 256,
        nlayers: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        if zdim <= 0:
            raise ValueError("zdim must be > 0 for _GaussianParamGenerator")
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be > 0")
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        layers: list[nn.Module] = [
            nn.Linear(zdim, hidden_dim),
            activation(),
        ]

        for _ in range(nlayers - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    activation(),
                ]
            )

        self.output = nn.Linear(hidden_dim, n_gaussians * 4)
        layers.append(self.output)
        self.net = nn.Sequential(*layers)

        # Start all conformations from the same consensus-like Gaussian model.
        # z-dependent deformations are then learned from data.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class GaussianDecoder(Decoder):
    """
    Heterogeneous Gaussian primitive decoder compatible with cryoDRGN.

    Input
    -----
    lattice:
        Training path:
            shape (B, N, 3 + zdim)

        eval_volume path:
            shape (N, 3 + zdim)

        The first 3 values are rotated Fourier coordinates in cycles/pixel.
        The remaining values are the latent conformation z, repeated at every
        Fourier coordinate by HetOnlyVAE.cat_z().

    Output
    ------
    Hartley-space values with shape (B, N), or (N,) for unbatched input.

    Parameterization
    ----------------
    Each Gaussian j has

        center mu_j(z) in real-space pixels
        positive amplitude a_j(z)
        fixed isotropic sigma in real-space pixels

    The learned model is a shared base Gaussian cloud plus z-dependent offsets.
    This encourages Gaussian correspondence across conformations.
    """

    def __init__(
        self,
        zdim: int,
        D: int,
        n_gaussians: int = 512,
        hidden_dim: int = 256,
        nlayers: int = 3,
        sigma: float = 1.5,
        activation: Type[nn.Module] = nn.ReLU,
        chunk_size: int = 1024,
        init_radius_fraction: float = 0.60,
        center_delta_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if D < 3:
            raise ValueError(f"D must be >= 3, got {D}")
        if n_gaussians <= 0:
            raise ValueError("n_gaussians must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0.0 < init_radius_fraction < 1.0):
            raise ValueError("init_radius_fraction must lie in (0, 1)")

        self.zdim = int(zdim)
        self.D = int(D)
        self.n_gaussians = int(n_gaussians)
        self.sigma = float(sigma)
        self.chunk_size = int(chunk_size)
        self.center_delta_scale = float(center_delta_scale)

        # cryoDRGN evaluates an odd Fourier lattice and removes the last
        # positive-frequency sample before inverse transforming, so the final
        # real-space box is D - 1 pixels wide.
        self.box_size = D - 1
        self.half_box = self.box_size / 2.0

        # ------------------------------------------------------------------
        # Shared base Gaussian cloud
        # ------------------------------------------------------------------
        # Initialize centers approximately uniformly inside a 3-D ball.
        # They are stored in unconstrained "logit" coordinates and mapped
        # through tanh so that physical centers stay inside the box.
        directions = torch.randn(n_gaussians, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        radii = torch.rand(n_gaussians, 1).pow(1.0 / 3.0)
        radii = radii * init_radius_fraction * self.half_box
        init_centers = directions * radii

        normalized_centers = (init_centers / self.half_box).clamp(-0.98, 0.98)
        base_center_logits = torch.atanh(normalized_centers)
        self.base_center_logits = nn.Parameter(base_center_logits)

        # Independent positive amplitudes, with a global trainable scale.
        # Starting from logits=0 gives softplus(0)=~0.693.  The 1/G global
        # scale keeps the initial total density O(1) instead of O(G).
        self.base_amp_logits = nn.Parameter(torch.zeros(n_gaussians))
        self.log_global_amp_scale = nn.Parameter(
            torch.tensor(-math.log(float(n_gaussians)), dtype=torch.float32)
        )

        # z -> (delta_x, delta_y, delta_z, delta_amplitude) for each Gaussian.
        if self.zdim > 0:
            self.generator: Optional[nn.Module] = _GaussianParamGenerator(
                zdim=self.zdim,
                n_gaussians=self.n_gaussians,
                hidden_dim=hidden_dim,
                nlayers=nlayers,
                activation=activation,
            )
        else:
            self.generator = None

    # ----------------------------------------------------------------------
    # Gaussian parameter generation
    # ----------------------------------------------------------------------

    def gaussian_parameters(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generate Gaussian centers and amplitudes for latent codes z.

        Parameters
        ----------
        z:
            (B, zdim)

        Returns
        -------
        centers:
            (B, G, 3), real-space coordinates in pixels.
        amplitudes:
            (B, G), positive density amplitudes.
        """
        if z.ndim != 2:
            raise ValueError(f"Expected z with shape (B, zdim), got {tuple(z.shape)}")
        if z.shape[-1] != self.zdim:
            raise ValueError(
                f"Expected zdim={self.zdim}, got last dimension {z.shape[-1]}"
            )

        B = z.shape[0]

        if self.generator is None:
            delta = torch.zeros(
                B,
                self.n_gaussians,
                4,
                dtype=self.base_center_logits.dtype,
                device=self.base_center_logits.device,
            )
        else:
            delta = self.generator(z).view(B, self.n_gaussians, 4)

        delta_center = delta[..., :3]
        delta_amp = delta[..., 3]

        center_logits = (
            self.base_center_logits.unsqueeze(0)
            + self.center_delta_scale * delta_center
        )
        centers = self.half_box * torch.tanh(center_logits)

        amp_logits = self.base_amp_logits.unsqueeze(0) + delta_amp
        amplitudes = (
            torch.exp(self.log_global_amp_scale)
            * F.softplus(amp_logits)
        )

        return centers, amplitudes

    # ----------------------------------------------------------------------
    # Fourier / Hartley rendering
    # ----------------------------------------------------------------------

    def _fourier_chunk(
        self,
        coords: Tensor,
        centers: Tensor,
        amplitudes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate complex Fourier values for one coordinate chunk.

        coords:      (B, M, 3), cycles / pixel
        centers:     (B, G, 3), pixels
        amplitudes:  (B, G)

        Returns:
            real, imag: each (B, M)
        """
        # phase = 2*pi*k.mu
        phase = 2.0 * math.pi * torch.einsum(
            "bmc,bgc->bmg", coords, centers
        )

        # FT of isotropic Gaussian:
        # exp(-2*pi^2*sigma^2*||k||^2)
        k2 = torch.sum(coords * coords, dim=-1)
        envelope = torch.exp(
            -2.0 * (math.pi**2) * (self.sigma**2) * k2
        )

        amp = amplitudes.unsqueeze(1)

        real = envelope * torch.sum(
            amp * torch.cos(phase),
            dim=-1,
        )
        imag = -envelope * torch.sum(
            amp * torch.sin(phase),
            dim=-1,
        )

        return real, imag

    def fourier_values(
        self,
        coords: Tensor,
        z: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate F_z(k) at arbitrary Fourier coordinates.

        coords:
            (B, N, 3)
        z:
            (B, zdim)

        Returns:
            real, imag, each (B, N)
        """
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"Expected coords with shape (B, N, 3), got {tuple(coords.shape)}"
            )

        centers, amplitudes = self.gaussian_parameters(z)

        real_chunks = []
        imag_chunks = []

        for start in range(0, coords.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, coords.shape[1])
            real, imag = self._fourier_chunk(
                coords[:, start:end],
                centers,
                amplitudes,
            )
            real_chunks.append(real)
            imag_chunks.append(imag)

        return (
            torch.cat(real_chunks, dim=1),
            torch.cat(imag_chunks, dim=1),
        )

    def forward(self, lattice: Tensor) -> Tensor:
        """
        Evaluate the decoder in cryoDRGN Hartley space.

        cryoDRGN uses H(k) = Re(F(k)) - Im(F(k)).
        """
        squeeze_batch = False

        if lattice.ndim == 2:
            lattice = lattice.unsqueeze(0)
            squeeze_batch = True

        if lattice.ndim != 3:
            raise ValueError(
                "GaussianDecoder expects lattice of shape "
                "(B, N, 3+zdim) or (N, 3+zdim); "
                f"got {tuple(lattice.shape)}"
            )

        expected_dim = 3 + self.zdim
        if lattice.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected last dimension {expected_dim} "
                f"(3 coords + zdim={self.zdim}), got {lattice.shape[-1]}"
            )

        coords = lattice[..., :3]

        # HetOnlyVAE.cat_z() repeats the same z at every Fourier coordinate,
        # so reading z from one coordinate is sufficient.
        if self.zdim > 0:
            z = lattice[:, 0, 3:]
        else:
            z = torch.empty(
                lattice.shape[0],
                0,
                dtype=lattice.dtype,
                device=lattice.device,
            )

        real, imag = self.fourier_values(coords, z)
        hartley = real - imag

        if squeeze_batch:
            hartley = hartley.squeeze(0)

        return hartley

    # ----------------------------------------------------------------------
    # Volume evaluation for cryodrgn eval_vol / analyze
    # ----------------------------------------------------------------------

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate a D x D x D Fourier lattice and inverse Hartley transform it.

        This mirrors cryoDRGN's existing Fourier decoders so eval_vol/analyze
        can use GaussianDecoder without a separate rendering pipeline.
        """
        assert not self.training, "Call model.eval() before eval_volume()"
        assert extent <= 0.5

        device = coords.device
        dtype = coords.dtype

        if self.zdim > 0:
            if zval is None:
                # Convenient default for debugging; eval_vol normally supplies z.
                z = torch.zeros(self.zdim, dtype=dtype, device=device)
            else:
                if len(zval) != self.zdim:
                    raise ValueError(
                        f"Expected zval of length {self.zdim}, got {len(zval)}"
                    )
                z = torch.as_tensor(zval, dtype=dtype, device=device)
        else:
            z = torch.empty(0, dtype=dtype, device=device)

        vol_f = torch.zeros(
            (D, D, D),
            dtype=dtype,
            device=device,
        )

        z_slices = np.linspace(
            -extent,
            extent,
            D,
            endpoint=True,
            dtype=np.float32,
        )

        for i, dz in enumerate(z_slices):
            x = coords + torch.tensor(
                [0.0, 0.0, float(dz)],
                dtype=dtype,
                device=device,
            )

            # Match cryoDRGN Fourier decoders: only evaluate inside the
            # spherical Fourier support.
            keep = x.pow(2).sum(dim=1) <= extent**2

            x_keep = x[keep]

            if self.zdim > 0:
                x_keep = torch.cat(
                    [
                        x_keep,
                        z.expand(x_keep.shape[0], self.zdim),
                    ],
                    dim=-1,
                )

            with torch.no_grad():
                y = self.forward(x_keep)

            slice_ = torch.zeros(D * D, dtype=dtype, device=device)
            slice_[keep] = y
            vol_f[i] = slice_.view(D, D)

        # Preserve cryoDRGN's existing normalization convention.
        vol_f = vol_f * norm[1] + norm[0]

        # Remove the duplicated +Nyquist endpoint, exactly as the existing
        # Fourier decoders do before inverse Hartley transform.
        vol = fft.ihtn_center(vol_f[:-1, :-1, :-1])
        return vol
