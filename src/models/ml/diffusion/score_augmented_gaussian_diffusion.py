from logging import getLogger

import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.ml.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.ml.score.abstract_score import ScoreFramework

logger = getLogger(__name__)


class ScoreAugmentedGaussianDiffusion(GaussianDiffusion):
    """Score-Augmented Predictor-Driven Diffusion (SA-PDD).

    The network predicts the drift increment and the forward-process noise from
    two pointwise heads that share the same trunk. The score used in the
    reverse-lambda integration is obtained from the noise head, so no automatic
    differentiation through the action functional is needed.
    """

    def __init__(
        self,
        noise_estimate_fn: ScoreFramework,
        *,
        drift_loss_weight: float,
        score_loss_weight: float,
        **kwargs,
    ) -> None:
        assert drift_loss_weight >= 0.0
        assert score_loss_weight >= 0.0
        assert drift_loss_weight > 0.0 or score_loss_weight > 0.0

        super().__init__(noise_estimate_fn=noise_estimate_fn, **kwargs)

        self.drift_loss_weight = drift_loss_weight
        self.score_loss_weight = score_loss_weight
        logger.info(f"{self.drift_loss_weight=}, {self.score_loss_weight=}")

    def _losses(self, x_0: Tensor, t: Tensor) -> Tensor:
        noise = self._noise(x_0.shape)
        x_t, _, _ = self._calc_q_samples(x_0=x_0, t=t, noise=noise)
        vars = self._extract_params(self.var_t, t)

        dx_t = self.noise_estimate_fn.calc_velocity(x_t, return_dimensionless=True)
        v_t = self.noise_estimate_fn.calc_rhs(x_t, return_dimensionless=True)
        outputs = self.noise_estimate_fn.calc_closure_and_noise(x_t, t)
        closure, est_noise = outputs["closure"], outputs["noise"]
        assert est_noise.shape == noise.shape

        drift_om = v_t + closure

        stds = torch.sqrt(vars)[:, None, None, None] * self.std_ratio
        diffs = (dx_t - drift_om[:, :, :-1, :]) / stds

        drift_loss = torch.mean(diffs**2)
        score_loss = F.mse_loss(est_noise, noise)

        return self.drift_loss_weight * drift_loss + self.score_loss_weight * score_loss

    def _estimate_score(self, x_t: Tensor, t: Tensor) -> Tensor:
        with torch.no_grad():
            est_noise = self.noise_estimate_fn.calc_noise(
                x_dimensionless=x_t, diffusion_time=t
            )

            if self.spatial_dimension == "1d":
                _, _, _, n_l = est_noise.shape
                assert n_l == self.image_size
                noise_hat = torch.fft.fft(est_noise, dim=-1, norm="ortho")
                p_hats = self._extract_spectral_params_1d(self.P_hat_1d_t, t)  # (B, L)
                score_hat = p_hats[:, None, None, :] * noise_hat
                score = torch.fft.ifft(score_hat, dim=-1, norm="ortho").real

            elif self.spatial_dimension == "2d":
                n_b, n_c, n_f, n_l = est_noise.shape
                assert self.image_size_x is not None and self.image_size_y is not None
                assert n_l == self.image_size_y * self.image_size_x
                noise_hat = torch.fft.fft2(
                    self._2d_view(est_noise), dim=(-2, -1), norm="ortho"
                )
                p_hats = self._extract_spectral_params_2d(self.P_hat_t, t)  # (B,Ny,Nx)
                score_hat = p_hats[:, None, None, :, :] * noise_hat
                score_2d = torch.fft.ifft2(score_hat, dim=(-2, -1), norm="ortho").real
                score = score_2d.view(n_b, n_c, n_f, n_l)

            else:
                raise NotImplementedError()

        return score.detach()

    @torch.inference_mode()
    def _extrapolate_frames(self, arr: torch.Tensor) -> None:
        # SA-PDD reads the score from the noise head, so the initial density
        # r_lambda never enters the reverse-lambda update. The linear
        # extrapolation of frame 0 that PDD applies is therefore skipped.
        assert arr.ndim == 4  # (B, C, F, L)
        return
