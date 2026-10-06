"""
CryoKRAQEN-inspired Fourier-domain triplane decoder for cryoDRGN.

This implements only the triplane representation while preserving the current
cryoDRGN heterogeneous VAE pipeline:

    cryoDRGN encoder -> continuous latent z -> TriplaneDecoder

It does NOT reproduce CryoKRAQEN's decoder-only inference, quantized codebook,
Epanechnikov-kernel assignment, annealing schedule, or triplet/VQ losses.

Design choices motivated by CryoKRAQEN:
    * Fourier-domain triplane representation
    * three orthogonal planes: XY, YZ, ZX
    * plane resolution ~ floor(D / 2)
    * 3-layer MLP with hidden width 1024 by default
    * bilinear interpolation of triplane features
    * complex Fourier prediction, converted to cryoDRGN Hartley values

Expected input from HetOnlyVAE.cat_z:
    (..., 3 + zdim)
where the first 3 entries are Fourier coordinates and the remaining entries
are the repeated per-particle latent code.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cryodrgn import fft
from cryodrgn.decoders.base import Decoder


Norm = Sequence[Any]


class _TriplaneMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 1024,
        nlayers: int = 3,
        out_dim: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if nlayers < 1:
            raise ValueError("nlayers must be >= 1")

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(activation())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TriplaneDecoder(Decoder):
    """
    Fourier-domain triplane implicit decoder.

    Query q=(x,y,z):
        f_xy = P_xy(x,y)
        f_yz = P_yz(y,z)
        f_zx = P_zx(z,x)

    Then:
        [f_xy, f_yz, f_zx, latent] -> MLP -> (Re F(q), Im F(q))

    Hermitian symmetry is enforced so the corresponding real-space density
    is real, and the output returned to cryoDRGN is Hartley:
        H(q) = Re F(q) - Im F(q)
    """

    def __init__(
        self,
        zdim: int,
        D: int,
        plane_res: Optional[int] = None,
        plane_dim: int = 64,
        hidden_dim: int = 1024,
        nlayers: int = 3,
        activation: Type[nn.Module] = nn.ReLU,
        coord_extent: float = 0.5,
        chunk_size: int = 32768,
    ) -> None:
        super().__init__()

        if D < 3:
            raise ValueError(f"D must be >= 3, got {D}")
        if plane_dim <= 0:
            raise ValueError("plane_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if coord_extent <= 0:
            raise ValueError("coord_extent must be > 0")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.zdim = int(zdim)
        self.D = int(D)
        self.plane_res = int(D // 2 if plane_res is None else plane_res)
        self.plane_dim = int(plane_dim)
        self.coord_extent = float(coord_extent)
        self.chunk_size = int(chunk_size)

        if self.plane_res < 2:
            raise ValueError(f"plane_res must be >= 2, got {self.plane_res}")

        init_std = 0.01
        self.plane_xy = nn.Parameter(
            torch.randn(1, self.plane_dim, self.plane_res, self.plane_res)
            * init_std
        )
        self.plane_yz = nn.Parameter(
            torch.randn(1, self.plane_dim, self.plane_res, self.plane_res)
            * init_std
        )
        self.plane_zx = nn.Parameter(
            torch.randn(1, self.plane_dim, self.plane_res, self.plane_res)
            * init_std
        )

        mlp_in_dim = 3 * self.plane_dim + self.zdim
        self.decoder = _TriplaneMLP(
            in_dim=mlp_in_dim,
            hidden_dim=hidden_dim,
            nlayers=nlayers,
            out_dim=2,
            activation=activation,
        )

    def _normalize_coords(self, xy: Tensor) -> Tensor:
        return (xy / self.coord_extent).clamp(-1.0, 1.0)

    @staticmethod
    def _sample_plane(plane: Tensor, grid: Tensor) -> Tensor:
        B, N, _ = grid.shape
        expanded_plane = plane.expand(B, -1, -1, -1)
        grid_4d = grid.reshape(B, 1, N, 2)

        feat = F.grid_sample(
            expanded_plane,
            grid_4d,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return feat.squeeze(2).transpose(1, 2)

    def sample_triplane(self, coords: Tensor) -> Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(f"Expected coords [B,N,3], got {tuple(coords.shape)}")

        xy = self._normalize_coords(coords[..., [0, 1]])
        yz = self._normalize_coords(coords[..., [1, 2]])
        zx = self._normalize_coords(coords[..., [2, 0]])

        f_xy = self._sample_plane(self.plane_xy, xy)
        f_yz = self._sample_plane(self.plane_yz, yz)
        f_zx = self._sample_plane(self.plane_zx, zx)

        return torch.cat((f_xy, f_yz, f_zx), dim=-1)

    def _canonicalize_hermitian(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        """
        Choose a deterministic representative from each q / -q pair.
        """
        eps = torch.finfo(coords.dtype).eps * 8

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        z_pos = z > eps
        z_zero = z.abs() <= eps
        y_pos = y > eps
        y_zero = y.abs() <= eps
        x_pos = x > eps

        conjugate_mask = z_pos | (z_zero & y_pos) | (z_zero & y_zero & x_pos)
        canonical = torch.where(conjugate_mask.unsqueeze(-1), -coords, coords)
        return canonical, conjugate_mask

    def _decode_chunk(
        self,
        coords: Tensor,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        canonical, conjugate_mask = self._canonicalize_hermitian(coords)
        tri_feat = self.sample_triplane(canonical)

        if self.zdim > 0:
            z_feat = latent.unsqueeze(1).expand(
                latent.shape[0], canonical.shape[1], self.zdim
            )
            feat = torch.cat((tri_feat, z_feat), dim=-1)
        else:
            feat = tri_feat

        complex_out = self.decoder(feat)
        real = complex_out[..., 0]
        imag = complex_out[..., 1]

        # F(-q) = conj(F(q))
        imag = torch.where(conjugate_mask, -imag, imag)
        return real, imag

    def fourier_values(
        self,
        coords: Tensor,
        latent: Tensor,
    ) -> tuple[Tensor, Tensor]:
        real_chunks = []
        imag_chunks = []

        for start in range(0, coords.shape[1], self.chunk_size):
            end = min(start + self.chunk_size, coords.shape[1])
            real, imag = self._decode_chunk(coords[:, start:end], latent)
            real_chunks.append(real)
            imag_chunks.append(imag)

        return torch.cat(real_chunks, dim=1), torch.cat(imag_chunks, dim=1)

    def forward(self, lattice: Tensor) -> Tensor:
        squeeze_batch = False

        if lattice.ndim == 2:
            lattice = lattice.unsqueeze(0)
            squeeze_batch = True

        if lattice.ndim != 3:
            raise ValueError(
                "TriplaneDecoder expects [B,N,3+zdim] or [N,3+zdim], "
                f"got {tuple(lattice.shape)}"
            )

        expected_dim = 3 + self.zdim
        if lattice.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected last dim {expected_dim}, got {lattice.shape[-1]}"
            )

        coords = lattice[..., :3]

        if self.zdim > 0:
            latent = lattice[:, 0, 3:]
        else:
            latent = torch.empty(
                lattice.shape[0], 0, device=lattice.device, dtype=lattice.dtype
            )

        real, imag = self.fourier_values(coords, latent)
        hartley = real - imag

        if squeeze_batch:
            hartley = hartley.squeeze(0)

        return hartley

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        assert not self.training, "Call model.eval() before eval_volume()"
        assert extent <= 0.5

        device = coords.device
        dtype = coords.dtype

        if self.zdim > 0:
            if zval is None:
                latent = torch.zeros(self.zdim, dtype=dtype, device=device)
            else:
                if len(zval) != self.zdim:
                    raise ValueError(
                        f"Expected zval length {self.zdim}, got {len(zval)}"
                    )
                latent = torch.as_tensor(zval, dtype=dtype, device=device)
        else:
            latent = torch.empty(0, dtype=dtype, device=device)

        vol_f = torch.zeros((D, D, D), dtype=dtype, device=device)

        z_slices = np.linspace(
            -extent, extent, D, endpoint=True, dtype=np.float32
        )

        for i, dz in enumerate(z_slices):
            xyz = coords + torch.tensor(
                [0.0, 0.0, float(dz)],
                device=device,
                dtype=dtype,
            )

            keep = xyz.pow(2).sum(dim=1) <= extent**2
            query = xyz[keep]

            if self.zdim > 0:
                query = torch.cat(
                    (query, latent.expand(query.shape[0], self.zdim)),
                    dim=-1,
                )

            with torch.no_grad():
                values = self.forward(query)

            slice_ = torch.zeros(D * D, dtype=dtype, device=device)
            slice_[keep] = values
            vol_f[i] = slice_.view(D, D)

        vol_f = vol_f * norm[1] + norm[0]
        return fft.ihtn_center(vol_f[:-1, :-1, :-1])

    def extra_repr(self) -> str:
        return (
            f"zdim={self.zdim}, D={self.D}, "
            f"plane_res={self.plane_res}, plane_dim={self.plane_dim}, "
            f"coord_extent={self.coord_extent}"
        )
