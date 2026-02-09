import math
import sys
from logging import getLogger
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.math_helper.ou import ou_spectrum_1d_eigenvalues, ou_spectrum_2d_eigenvalues
from src.models.ml.score.abstract_score import ScoreFramework

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger(__name__)


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        noise_estimate_fn: ScoreFramework,
        *,
        channels: int,
        num_frames: int,
        image_size: int,
        #
        num_timesteps: int,
        laplacian_factor: float,
        mass_factor: float,
        #
        std_ratio: float,
        spatial_dimension: Literal["1d", "2d"],
        noise_amplitude_squared: Optional[float],
        noise_type: Literal["white", "zero"],
        #
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        x_length: float = 2.0 * math.pi,
        image_size_x: Optional[int] = None,
        image_size_y: Optional[int] = None,
    ):

        assert std_ratio > 0.0 and isinstance(std_ratio, float)
        assert noise_type in ["white", "zero"]
        assert spatial_dimension in ["1d", "2d"]

        if spatial_dimension == "1d":
            assert image_size_x is None and image_size_y is None
        else:
            assert image_size_x is not None and image_size_y is not None
            assert image_size == 0, "image_size is ignored for 2d spatial_dimension"

        super().__init__()

        self.noise_estimate_fn = noise_estimate_fn

        self.channels = channels
        self.num_frames = num_frames
        self.image_size = image_size
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y
        self.spatial_dimension: Literal["1d", "2d"] = spatial_dimension

        self.num_timesteps = num_timesteps
        self.laplacian_factor = laplacian_factor
        self.mass_factor = mass_factor

        self.std_ratio = std_ratio

        self.device = device
        self.dtype = dtype
        self.x_length = x_length
        self.noise_type: Literal["white", "zero"] = noise_type
        self.noise_amplitude_squared = noise_amplitude_squared

        self._set_params()

    def _set_params(self):

        def reg(name: str, val: Tensor):
            self.register_buffer(name, val.to(dtype=self.dtype, device=self.device))

        dt = 1.0 / self.num_timesteps
        ts = np.linspace(
            0.0, 1.0, self.num_timesteps + 1, dtype=np.float64, endpoint=True
        )

        self.dt = dt
        reg("ts", torch.from_numpy(ts[1:]))  # skip t == 0

        # Coefficient for Winner process, b x dW
        # Variance-preserving SDE is assumed for the zero laplacian case.
        self.b = math.sqrt(2.0 * self.mass_factor)
        self.b2 = self.b**2

        if self.noise_amplitude_squared is not None:
            logger.info(f"The input of {self.noise_amplitude_squared=} is not None.")
            logger.info(f"Before replacing, {self.b=}, {self.b2=}")
            self.b2 = float(self.noise_amplitude_squared)
            self.b = math.sqrt(self.b2)
            logger.info(f"After replacing, {self.b=}, {self.b2=}")

        if self.spatial_dimension == "2d":
            self._set_params_fft2(ts=ts)
            return

        if self.spatial_dimension == "1d":
            self._set_params_fft1d(ts=ts)
            return

    def _set_params_fft1d(self, ts: np.ndarray) -> None:

        lam_eff = ou_spectrum_1d_eigenvalues(
            N=self.image_size,
            a=self.laplacian_factor,
            m=self.mass_factor,
            h=float(self.x_length / self.image_size),
            device=self.device,
            dtype=self.dtype,
        )

        A_hats: list[Tensor] = []
        B_hats: list[Tensor] = []
        P_hats: list[Tensor] = []
        vars: list[float] = []

        for t in ts[:-1]:
            t_val = float(t)

            # A_hat(k) = exp(λ_eff(k) * t)
            A_hat = torch.exp(lam_eff * t_val)  # (L,)

            if t_val == 0.0:
                t_val = ts[1]  # to avoid zero variance
                logger.info(f"Set {t_val=} when t == 0 to avoid zero variance")

            # Q_hat(k) = Var[x_t(k)]
            denom = 2.0 * lam_eff
            Q_hat = torch.zeros_like(lam_eff, dtype=self.dtype, device=self.device)

            mask = torch.abs(denom) > 1e-14
            Q_hat[mask] = self.b2 * (torch.exp(denom[mask] * t_val) - 1.0) / denom[mask]
            Q_hat[~mask] = self.b2 * t_val

            B_hat = torch.sqrt(torch.clamp(Q_hat, min=0.0))

            Q_safe = torch.clamp(Q_hat, min=1e-12)
            P_hat = -1.0 / torch.sqrt(Q_safe)

            var = Q_safe.mean().item()

            A_hats.append(A_hat)
            B_hats.append(B_hat)
            P_hats.append(P_hat)
            vars.append(var)

        self.register_buffer(
            "A_hat_1d_t",
            torch.stack(A_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )  # (T, L)
        self.register_buffer(
            "B_hat_1d_t",
            torch.stack(B_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )  # (T, L)
        self.register_buffer(
            "P_hat_1d_t",
            torch.stack(P_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )  # (T, L)
        self.register_buffer(
            "var_t",
            torch.tensor(vars, device=self.device, dtype=self.dtype),
        )  # (T,)
        self.register_buffer(
            "M_hat_1d",
            lam_eff.to(device=self.device, dtype=self.dtype),
        )  # (L,)

        logger.info("Set 1d params for ETD1 backward SDE")
        # z = - M_hat_1d * dt
        z_1d = -1.0 * self.M_hat_1d * self.dt
        back_sde_A_step_1d = torch.exp(z_1d)  # (L,)

        back_sde_phi1_1d = torch.zeros_like(self.M_hat_1d)
        eps = 1e-8
        mask_small = torch.abs(z_1d) < eps
        back_sde_phi1_1d[mask_small] = 1.0
        back_sde_phi1_1d[~mask_small] = (back_sde_A_step_1d[~mask_small] - 1.0) / z_1d[
            ~mask_small
        ]

        self.register_buffer(
            "back_sde_A_step_1d",
            back_sde_A_step_1d.to(device=self.device, dtype=self.dtype),
        )  # (L,)
        self.register_buffer(
            "back_sde_phi1_1d",
            back_sde_phi1_1d.to(device=self.device, dtype=self.dtype),
        )  # (L,)

    def _set_params_fft2(self, ts: np.ndarray) -> None:
        assert self.image_size_x is not None and self.image_size_y is not None

        lam_eff = ou_spectrum_2d_eigenvalues(
            Ny=self.image_size_y,
            Nx=self.image_size_x,
            a=self.laplacian_factor,
            m=self.mass_factor,
            dy=self.x_length / self.image_size_y,
            dx=self.x_length / self.image_size_x,
            device=self.device,
            dtype=self.dtype,
        )  # (Ny, Nx)

        A_hats: list[Tensor] = []
        B_hats: list[Tensor] = []
        P_hats: list[Tensor] = []
        vars: list[float] = []

        for t in ts[:-1]:
            t_val = float(t)

            # A_hat(k) = exp(λ_eff(k) * t)
            A_hat = torch.exp(lam_eff * t_val)

            if t_val == 0.0:
                t_val = ts[1]  # to avoid zero variance
                logger.info(f"Set {t_val=} when t == 0 to avoid zero variance")

            # Q_hat(k) = Var[x_t(k)]
            # λ_eff != 0: b^2*(exp(2λt)-1)/(2λ)
            # λ_eff = 0 : b^2 * t
            denom = 2.0 * lam_eff
            Q_hat = torch.zeros_like(lam_eff, dtype=self.dtype, device=self.device)

            mask = torch.abs(denom) > 1e-14
            Q_hat[mask] = self.b2 * (torch.exp(denom[mask] * t_val) - 1.0) / denom[mask]
            Q_hat[~mask] = self.b2 * t_val

            # B_hat = sqrt(Q_hat)
            B_hat = torch.sqrt(torch.clamp(Q_hat, min=0.0))

            eps = 1e-12
            P_hat = -1.0 / torch.sqrt(torch.clamp(Q_hat, min=eps))

            var = torch.clamp(Q_hat, min=eps).mean().item()

            A_hats.append(A_hat)
            B_hats.append(B_hat)
            P_hats.append(P_hat)
            vars.append(var)

        self.register_buffer(
            "A_hat_t",
            torch.stack(A_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )  # (T, Ny, Nx)
        self.register_buffer(
            "B_hat_t",
            torch.stack(B_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )  # (T, Ny, Nx)
        self.register_buffer(
            "P_hat_t",
            torch.stack(P_hats, dim=0).to(device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "var_t",
            torch.tensor(vars, device=self.device, dtype=self.dtype),
        )  # (T,)

        self.register_buffer("M_hat", lam_eff.to(device=self.device, dtype=self.dtype))

        logger.info("Set 2d params for ETD1 backward SDE")
        z_2d = -1.0 * self.M_hat * self.dt  # (Ny, Nx)
        back_sde_A_step_2d = torch.exp(z_2d)

        back_sde_phi1_2d = torch.zeros_like(self.M_hat)
        eps = 1e-8
        mask_small = torch.abs(z_2d) < eps
        back_sde_phi1_2d[mask_small] = 1.0
        back_sde_phi1_2d[~mask_small] = (back_sde_A_step_2d[~mask_small] - 1.0) / z_2d[
            ~mask_small
        ]

        self.register_buffer(
            "back_sde_A_step_2d",
            back_sde_A_step_2d.to(device=self.device, dtype=self.dtype),
        )  # size = (Ny, Nx)
        self.register_buffer(
            "back_sde_phi1_2d",
            back_sde_phi1_2d.to(device=self.device, dtype=self.dtype),
        )  # size = (Ny, Nx)

    def _extract_matrix_params(self, matrix_params: Tensor, t: Tensor) -> Tensor:

        def select(arry):
            return torch.index_select(arry, dim=0, index=t)
            # Select diffusion times along batch dim

        (n_batches,) = t.shape

        selected = select(matrix_params)
        assert selected.shape == (n_batches, self.image_size, self.image_size)

        return selected.requires_grad_(False)

    def _extract_params(self, params: Tensor, t: Tensor) -> Tensor:

        def select(arry):
            return torch.index_select(arry, dim=0, index=t)
            # Select diffusion times along batch dim

        (n_batches,) = t.shape

        selected = select(params)
        assert selected.shape == (n_batches,)

        return selected.requires_grad_(False)

    def _extract_spectral_params(self, spectral_params: Tensor, t: Tensor) -> Tensor:
        (n_batches,) = t.shape
        selected = torch.index_select(spectral_params, dim=0, index=t)
        assert selected.shape == (n_batches, self.image_size_y, self.image_size_x)
        return selected.requires_grad_(False)

    def _extract_spectral_params_1d(self, spectral_params: Tensor, t: Tensor) -> Tensor:
        (n_batches,) = t.shape
        selected = torch.index_select(spectral_params, dim=0, index=t)
        assert selected.shape == (n_batches, self.image_size)
        return selected.requires_grad_(False)

    def _2d_view(self, x: Tensor) -> Tensor:
        B, C, F, L = x.shape
        return x.view(B, C, F, self.image_size_y, self.image_size_x)

    def _calc_q_samples(
        self, x_0: Tensor, t: Tensor, noise: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:

        if self.spatial_dimension == "1d":
            B, C, F, L = x_0.shape
            assert L == self.image_size
            assert noise.shape == (B, C, F, L)

            x0_hat = torch.fft.fft(x_0, dim=-1, norm="ortho")
            noise_hat = torch.fft.fft(noise, dim=-1, norm="ortho")

            A_hats = self._extract_spectral_params_1d(self.A_hat_1d_t, t)  # (B, L)
            B_hats = self._extract_spectral_params_1d(self.B_hat_1d_t, t)  # (B, L)

            A_bc = A_hats[:, None, None, :]  # (B,1,1,L)
            B_bc = B_hats[:, None, None, :]  # (B,1,1,L)

            a_hat = A_bc * x0_hat
            b_hat = B_bc * noise_hat

            x_t = torch.fft.ifft(a_hat + b_hat, dim=-1, norm="ortho").real

            return x_t, A_hats, B_hats

        # ---- 2D: fft2 backend ----
        # x_0, noise: (B, C, F, L)
        B, C, F, L = x_0.shape
        assert L == self.image_size_x * self.image_size_y
        assert noise.shape == (B, C, F, L)

        # (B, C, F, L) -> (B, C, F, Ny, Nx)
        x0_2d = self._2d_view(x_0)
        noise_2d = self._2d_view(noise)

        # FFT2
        x0_hat = torch.fft.fft2(x0_2d, dim=(-2, -1), norm="ortho")  # complex
        noise_hat = torch.fft.fft2(noise_2d, dim=(-2, -1), norm="ortho")

        A_hats = self._extract_spectral_params(self.A_hat_t, t)  # (B, Ny, Nx)
        B_hats = self._extract_spectral_params(self.B_hat_t, t)  # (B, Ny, Nx)

        # broadcast: (B,1,1,Ny,Nx) * (B,C,F,Ny,Nx)
        A_hats_bc = A_hats[:, None, None, :, :]
        B_hats_bc = B_hats[:, None, None, :, :]

        a_hat = A_hats_bc * x0_hat
        b_hat = B_hats_bc * noise_hat

        x_t_2d = torch.fft.ifft2(
            a_hat + b_hat, dim=(-2, -1), norm="ortho"
        ).real  # (B,C,F,Ny,Nx)

        x_t = x_t_2d.view(B, C, F, L)

        return x_t, A_hats, B_hats

    def _noise(self, shape) -> Tensor:
        if self.noise_type == "white":
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        elif self.noise_type == "zero":
            noise = torch.zeros(shape, dtype=self.dtype, device=self.device)
            return noise
        else:
            raise NotImplementedError()

    def _losses(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        #
        noise = self._noise(x_0.shape)
        x_t, _, _ = self._calc_q_samples(x_0=x_0, t=t, noise=noise)
        vars = self._extract_params(self.var_t, t)

        dx_t = self.noise_estimate_fn.calc_velocity(x_t, return_dimensionless=True)
        v_t = self.noise_estimate_fn.calc_rhs(x_t, return_dimensionless=True)
        closure = self.noise_estimate_fn.calc_closure(x_t, t, return_dimensionless=True)

        drift_om = v_t + closure

        stds = torch.sqrt(vars)[:, None, None, None] * self.std_ratio
        diffs = (dx_t - drift_om[:, :, :-1, :]) / stds

        return torch.mean(diffs**2)

    def _estimate_score(self, x_t: Tensor, t: Tensor) -> Tensor:

        vars = self._extract_params(self.var_t, t)

        score = self.noise_estimate_fn(
            x=x_t,
            time=t,
            create_graph=False,
            retain_graph=False,
            vars=vars,
            return_score=True,
        ).detach()

        return score

    def _estimate_p_mean_back_sde_etd(self, x_t: Tensor, t: Tensor) -> Tensor:
        score = self._estimate_score(x_t=x_t, t=t)

        with torch.inference_mode():
            dt = self.dt
            b2 = self.b2

            if self.spatial_dimension == "1d":
                B, C, F, L = x_t.shape
                assert L == self.image_size

                x_hat = torch.fft.fft(x_t, dim=-1, norm="ortho")  # (B,C,F,L)
                score_hat = torch.fft.fft(score, dim=-1, norm="ortho")  # (B,C,F,L)

                A_step = self.back_sde_A_step_1d  # (L,)
                phi1 = self.back_sde_phi1_1d  # (L,)

                A_bc = A_step[None, None, None, :]  # (1,1,1,L)
                phi_bc = phi1[None, None, None, :]  # (1,1,1,L)

                x_lin_hat = A_bc * x_hat
                score_contrib_hat = dt * b2 * phi_bc * score_hat

                mean_hat = x_lin_hat + score_contrib_hat
                mean = torch.fft.ifft(mean_hat, dim=-1, norm="ortho").real  # (B,C,F,L)
                return mean

            # 2D FFT backend
            assert self.spatial_dimension == "2d"
            B, C, F, L = x_t.shape
            assert L == self.image_size_y * self.image_size_x

            x_2d = self._2d_view(x_t)  # (B,C,F,Ny,Nx)
            score_2d = self._2d_view(score)  # (B,C,F,Ny,Nx)

            x_hat = torch.fft.fft2(x_2d, dim=(-2, -1), norm="ortho")  # (B,C,F,Ny,Nx)
            score_hat = torch.fft.fft2(score_2d, dim=(-2, -1), norm="ortho")

            A_step = self.back_sde_A_step_2d  # (Ny,Nx)
            phi1 = self.back_sde_phi1_2d  # (Ny,Nx)

            A_bc = A_step[None, None, None, :, :]  # (1,1,1,Ny,Nx)
            phi_bc = phi1[None, None, None, :, :]

            x_lin_hat = A_bc * x_hat
            score_contrib_hat = dt * b2 * phi_bc * score_hat

            mean_hat = x_lin_hat + score_contrib_hat
            mean_2d = torch.fft.ifft2(mean_hat, dim=(-2, -1), norm="ortho").real
            mean = mean_2d.view(B, C, F, L)

            return mean

    def _corrector_step(self, x_t: Tensor, t: Tensor, corrector_snr: float) -> Tensor:

        score = self._estimate_score(x_t=x_t, t=t)

        with torch.inference_mode():
            batches = x_t.shape[0]
            channels = x_t.shape[1]
            d = score[0, 0].numel()  # sum of time and space dims

            score_flat = score.view(batches, channels, -1)  # (batches, channels, d)
            score_norm = torch.linalg.norm(score_flat, dim=2)  # (batches, channels)
            assert score_norm.shape == (batches, channels)

            alpha = 2.0 * (corrector_snr**2) * d / torch.clamp(score_norm**2, min=1e-12)
            alpha = alpha[:, :, None, None]  # (B,C,1,1)

            means = x_t + alpha * score
            self._extrapolate_frames(means)

            noise = self._noise(x_t.shape)
            # no noise (last index is 2)
            mask = (1 - (t <= 2).float()).reshape(batches, *((1,) * (x_t.ndim - 1)))

            x = means + mask * torch.sqrt(2.0 * alpha) * noise
            x = self._remove_constant(x)

            return torch.clamp(x, min=-3, max=3)

    def _sample_from_p(
        self, x_t: Tensor, t: Tensor, num_corrector_steps: int, corrector_snr: float
    ) -> Tensor:

        n_batches, *_ = x_t.shape

        means = self._estimate_p_mean_back_sde_etd(x_t=x_t, t=t)

        with torch.inference_mode():
            noise = self._noise(x_t.shape)
            b_dW = self.b * math.sqrt(self.dt) * noise

            # extrapolate the first and last elements along the frame dim
            self._extrapolate_frames(means)

            # no noise (last index is 2)
            mask = (1 - (t <= 2).float()).reshape(n_batches, *((1,) * (x_t.ndim - 1)))

            x = means + mask * b_dW
            x = self._remove_constant(x)
            x = torch.clamp(x, min=-3, max=3)

        if num_corrector_steps > 0:
            for _ in range(num_corrector_steps):
                x = self._corrector_step(x_t=x, t=t, corrector_snr=corrector_snr)

        return x

    def _remove_constant(self, x: Tensor) -> Tensor:
        assert x.ndim == 4
        return x - torch.mean(x, dim=(2, 3), keepdim=True)

    @torch.inference_mode()
    def _extrapolate_frames(self, arr: torch.Tensor):
        # Extrapolate the first and last elements along the frames dimension (linear extrapolation)
        assert arr.ndim == 4  # (B, C, F, L)
        # First element extrapolation
        idx_first = 0
        idx_next = 1
        idx_next_next = 2
        y1_f = arr[..., idx_next, :]
        y2_f = arr[..., idx_next_next, :]
        slope_f = (y2_f - y1_f) / (idx_next_next - idx_next)
        arr[..., idx_first, :] = y1_f + slope_f * (idx_first - idx_next)

    def _p_loop(
        self,
        n_timesteps: int,
        n_batches: int,
        img: torch.Tensor,
        num_corrector_steps: int,
        corrector_snr: float,
    ) -> dict[int, torch.Tensor]:
        assert n_batches == img.shape[0]

        intermediates: dict[int, torch.Tensor] = {}
        intermediates[n_timesteps - 1] = img.cpu().detach().clone()

        for t in tqdm(reversed(range(2, n_timesteps)), total=n_timesteps):
            img = self._sample_from_p(
                x_t=img,
                t=torch.full((n_batches,), t, device=self.device, dtype=torch.long),
                num_corrector_steps=num_corrector_steps,
                corrector_snr=corrector_snr,
            )
            intermediates[t - 1] = img.cpu().detach().clone()
            # state at t-1 is computed at t

        return intermediates

    def _sample_from_p_loop(
        self,
        shape: tuple,
        num_corrector_steps: int,
        corrector_snr: float,
    ) -> dict[int, torch.Tensor]:

        n_batches = shape[0]

        last_index = torch.full(
            size=(n_batches,),
            fill_value=self.num_timesteps - 1,
            device=self.device,
            dtype=torch.long,
        )

        noise = self._noise(shape)

        if self.spatial_dimension == "1d" and self.ou_backend_1d == "dense":
            last_B = self._extract_matrix_params(self.B_t, last_index)
            assert last_B.shape == (n_batches, self.image_size, self.image_size)
            img = torch.einsum("bij,bcfj->bcfi", last_B, noise)

        elif self.spatial_dimension == "1d" and self.ou_backend_1d == "fft":
            B, C, F, L = noise.shape
            assert L == self.image_size
            noise_hat = torch.fft.fft(noise, dim=-1, norm="ortho")
            B_hats = self._extract_spectral_params_1d(
                self.B_hat_1d_t, last_index
            )  # (B,L)
            B_bc = B_hats[:, None, None, :]  # (B,1,1,L)
            img_hat = B_bc * noise_hat
            img = torch.fft.ifft(img_hat, dim=-1, norm="ortho").real

        else:
            # 2D: fft2 backend
            B, C, F, L = noise.shape
            assert self.image_size_y * self.image_size_x == L
            noise_2d = self._2d_view(noise)

            noise_hat = torch.fft.fft2(
                noise_2d, dim=(-2, -1), norm="ortho"
            )  # (B, C, F, Ny, Nx)

            # B_hat_t から最後の時刻のスペクトル係数を取り出す
            B_hats = self._extract_spectral_params(
                self.B_hat_t, last_index
            )  # (B, Ny, Nx)
            B_hats_bc = B_hats[:, None, None, :, :]  # (B, 1, 1, Ny, Nx)

            img_hat = B_hats_bc * noise_hat
            img_2d = torch.fft.ifft2(
                img_hat, dim=(-2, -1), norm="ortho"
            ).real  # (B, C, F, Ny, Nx)

            img = img_2d.view(B, C, F, L)
            img = self._remove_constant(img)

        return self._p_loop(
            n_timesteps=self.num_timesteps,
            n_batches=n_batches,
            img=img.clone().detach(),
            num_corrector_steps=num_corrector_steps,
            corrector_snr=corrector_snr,
        )

    # Public methods

    def sample(
        self, batch_size: int, num_corrector_steps: int = 0, corrector_snr: float = 0.0
    ) -> dict[int, torch.Tensor]:

        if self.spatial_dimension == "1d":
            L = self.image_size
        else:
            assert self.image_size_x is not None and self.image_size_y is not None
            L = self.image_size_x * self.image_size_y

        shape = (batch_size, self.channels, self.num_frames, L)

        _ = self.noise_estimate_fn.eval()

        out = self._sample_from_p_loop(
            shape=shape,
            num_corrector_steps=num_corrector_steps,
            corrector_snr=corrector_snr,
        )

        if self.spatial_dimension == "1d":
            return out

        reshaped: dict[int, torch.Tensor] = {}
        for k, v in out.items():
            B, C, F, L_flat = v.shape
            assert L_flat == self.image_size_y * self.image_size_x
            reshaped[k] = self._2d_view(v)

        return reshaped

    def forward(self, x_0: torch.Tensor, **kwargs) -> Tensor:

        if self.spatial_dimension == "1d":
            assert x_0.ndim == 4
            b, c, f, h = x_0.shape
            assert c == self.channels
            assert f == self.num_frames
            assert h == self.image_size
            x_flat = x_0

        elif self.spatial_dimension == "2d":
            b, c, f, Ny, Nx = x_0.shape
            assert c == self.channels
            assert f == self.num_frames
            assert Ny == self.image_size_y
            assert Nx == self.image_size_x
            x_flat = x_0.view(b, c, f, Ny * Nx)

        else:
            raise NotImplementedError()

        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

        return self._losses(x_0=x_flat, t=t)
