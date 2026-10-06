# cryodrgn/decoders/base.py

from typing import Optional, Sequence, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


Norm = Sequence[Any]


class Decoder(nn.Module):
    """
    Base class for all cryoDRGN decoders.

    A decoder should implement:
      1. forward(...)      -- used during training/inference
      2. eval_volume(...) -- used to generate a 3D volume for visualization/evaluation
    """

    def eval_volume(
        self,
        coords: Tensor,
        D: int,
        extent: float,
        norm: Norm,
        zval: Optional[np.ndarray] = None,
    ) -> Tensor:
        """
        Evaluate the decoder on a D x D x D volume.

        Args:
            coords:
                Coordinates on one x-y lattice plane.
                Usually shape: (D^2, 3).

            D:
                Volume side length.

            extent:
                Coordinate range is approximately [-extent, extent].

            norm:
                Data normalization, typically (mean, std).

            zval:
                Optional latent conformation vector.

        Returns:
            Tensor representing the reconstructed 3D density volume.
        """
        raise NotImplementedError

    def get_voxel_decoder(self) -> Optional["Decoder"]:
        """
        Optional hook for decoders that can expose an equivalent voxel decoder.
        """
        return None