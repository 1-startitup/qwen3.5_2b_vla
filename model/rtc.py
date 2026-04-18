import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch
from torch import Tensor


class RTCAttentionSchedule(str, Enum):
    ZEROS = "zeros"
    ONES = "ones"
    LINEAR = "linear"
    EXP = "exp"


@dataclass
class RTCConfig:
    enabled: bool = False
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        if isinstance(self.prefix_attention_schedule, str):
            self.prefix_attention_schedule = RTCAttentionSchedule(self.prefix_attention_schedule.lower())
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")


class RTCProcessor:
    """Real-Time Chunking guidance wrapper aligned with the original LeRobot RTC logic."""

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self._debug_steps: list[dict] = []

    def track(self, **metadata) -> None:
        if not self.rtc_config.debug:
            return
        self._debug_steps.append(metadata)
        if len(self._debug_steps) > self.rtc_config.debug_maxlen:
            self._debug_steps = self._debug_steps[-self.rtc_config.debug_maxlen :]

    def get_all_debug_steps(self) -> list[dict]:
        return list(self._debug_steps)

    def is_debug_enabled(self) -> bool:
        return bool(self.rtc_config.debug)

    def reset_tracker(self) -> None:
        self._debug_steps.clear()

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Optional[Tensor],
        inference_delay: Optional[int],
        time: float | Tensor,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        execution_horizon: Optional[int] = None,
    ) -> Tensor:
        tau = 1 - time

        if prev_chunk_left_over is None or inference_delay is None:
            return original_denoise_step_partial(x_t)

        x_t = x_t.clone().detach()

        squeezed = False
        if x_t.ndim < 3:
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if prev_chunk_left_over.ndim < 3:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        execution_horizon = min(int(execution_horizon), int(prev_chunk_left_over.shape[1]))

        batch_size, chunk_size, action_dim = x_t.shape
        if prev_chunk_left_over.shape[1] < chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, chunk_size, action_dim, device=x_t.device, dtype=x_t.dtype)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        weights = self.get_prefix_weights(int(inference_delay), execution_horizon, chunk_size)
        weights = weights.to(device=x_t.device, dtype=x_t.dtype).unsqueeze(0).unsqueeze(-1)

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)
            x1_t = x_t - time * v_t
            err = (prev_chunk_left_over - x1_t) * weights
            correction = torch.autograd.grad(x1_t, x_t, err.detach(), retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight, device=x_t.device, dtype=x_t.dtype)
        tau_tensor = torch.as_tensor(tau, device=x_t.device, dtype=x_t.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / squared_one_minus_tau
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        self.track(
            time=time,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        return result

    def get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
            return weights

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
            return weights

        lin_weights = self._linweights(start, end, total)
        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
        weights = self._add_trailing_zeros(lin_weights, total, end)
        weights = self._add_leading_ones(weights, start, total)
        return weights

    def _linweights(self, start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    def _add_leading_ones(self, weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])
