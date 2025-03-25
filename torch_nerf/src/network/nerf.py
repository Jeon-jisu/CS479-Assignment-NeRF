"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()
        self.pos_dim = pos_dim
        self.view_dir_dim = view_dir_dim
        self.feat_dim = feat_dim
        self.layers = nn.ModuleList(
            [
                nn.Linear(pos_dim, feat_dim),  # input layer 0
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 1st hidden layer 2
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 2nd hidden layer 4
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 3rd hidden layer 6
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 4th hidden layer 8
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 5th hidden layer, skip connection 10
                nn.ReLU(),
                nn.Linear(feat_dim + pos_dim, feat_dim),  # 6th hidden layer 12
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 7th hidden layer 14
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),  # 8th hidden layer 16
                # skip ReLU here
                nn.Linear(feat_dim, feat_dim + 1),  # 9th hidden layer, output sigma 17
                nn.ReLU(),
                nn.Linear(
                    feat_dim + view_dir_dim, feat_dim // 2
                ),  # output layer, output RGB color 19
                nn.ReLU(),
                nn.Linear(feat_dim // 2, 3),  # output layer, output RGB color 21
                nn.Sigmoid(),  # with a sigmoid activation
            ]
        )

        # # TODO
        # raise NotImplementedError("Task 1")

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[
        Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]
    ]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """
        x = self.layers[0](pos)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        x = self.layers[4](x)
        x = self.layers[5](x)
        x = self.layers[6](x)
        x = self.layers[7](x)
        x = self.layers[8](x)
        x = self.layers[9](x)

        x = torch.cat([x, pos], dim=1)
        x = self.layers[10](x)
        x = self.layers[11](x)
        x = self.layers[12](x)
        x = self.layers[13](x)
        x = self.layers[14](x)
        x = self.layers[15](x)
        x = self.layers[16](x)
        sigma_and_x = self.layers[17](x)
        sigma = sigma_and_x[:, 0:1]
        x = sigma_and_x[:, 1:]
        x = self.layers[18](x)
        x = torch.cat([x, view_dir], dim=1)
        x = self.layers[19](x)
        x = self.layers[20](x)
        radiance = self.layers[21](x)

        return sigma, radiance
