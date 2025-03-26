"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[
        Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]
    ]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # T_1 = exp(-0) = 1, T_2 = exp(- sig_1 - delta_)
        other_cumsum = torch.cumsum(sigma[:, :-1] * delta[:, :-1], dim=1)
        final_cumsum = torch.cat(
            [torch.zeros_like(other_cumsum[:, :1]), other_cumsum], dim=1
        )
        pass_rate = torch.exp(-final_cumsum)
        opacity = 1 - torch.exp(-sigma * delta)
        weights = pass_rate * opacity
        # print("weights.shape",weights.shape, "(1 - torch.exp(-sigma * delta))",(1 - torch.exp(-sigma * delta)).shape, "radiance.shape",radiance.shape,)

        color_element = weights.unsqueeze(-1) * radiance
        rgbs = torch.sum(color_element, dim=1)  # 누적합
        return rgbs, weights
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.
        # raise NotImplementedError("Task 3")
