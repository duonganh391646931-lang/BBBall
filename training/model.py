"""
Условная нейросеть BasketballActorCritic (OBS_DIM=244, NUM_DISCRETE_ACTIONS=23, CONT_ACTION_DIM=2)
с поддержкой:
  * произвольной глубины/ширины актора (192x2, 256x3 и др. — 100% совместимо с браузером);
  * расширенного централизованного критика [obs, team_mean, team_max] (только при обучении, 0 байт в браузере);
  * точной игровой IQ-температуры computeIqTemperature(iq) прямо внутри распределения политики;
  * экспорта обученных весов актора в JSON и во встроенный модуль defaultTrainedWeights.ts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

OBS_DIM = 244
NUM_DISCRETE_ACTIONS = 23
CONT_ACTION_DIM = 2
IQ_OBS_INDEX = 54  # st.iq / 115.0 в src/game/rl/encoder.ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_OUT = PROJECT_ROOT / "public" / "models" / "policy_latest.json"
DEFAULT_TS_WEIGHTS_OUT = PROJECT_ROOT / "src" / "game" / "rl" / "defaultTrainedWeights.ts"


def _init_linear(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def compute_iq_temperature(obs: torch.Tensor) -> torch.Tensor:
    """Точный тензорный эквивалент computeIqTemperature(iq) из src/game/rl/neuralPolicy.ts."""
    iq = obs[..., IQ_OBS_INDEX].float() * 115.0
    norm = torch.clamp(iq / 100.0, 0.0, 1.15)
    return torch.clamp(1.75 - 1.52 * norm, 0.22, 1.65)


class ResidualMLPBlock(nn.Module):
    """
    Послойно совпадает с CompiledResBlock в src/game/rl/neuralPolicy.ts (паритет < 1e-5):
      h = SiLU(LN1(FC1(x)))
      out = SiLU(x + LN2(FC2(h)))
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = _init_linear(nn.Linear(hidden_dim, hidden_dim), gain=math.sqrt(2.0))
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.fc2 = _init_linear(nn.Linear(hidden_dim, hidden_dim), gain=0.25)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.ln1(self.fc1(x)))
        return F.silu(x + self.ln2(self.fc2(h)))


class BasketballActorCritic(nn.Module):
    """
    Условная сеть Актор-Критик (CTDE MAPPO):
      - Актор: 244 входа -> ResMLP (например 192x2 или 256x3) -> 23 дискретных + 2 непрерывных действия.
      - Централизованный Критик:
          critic_context_mult=2 -> [obs, team_mean] (488 входов, обратная совместимость)
          critic_context_mult=3 -> [obs, team_mean, team_max] (732 входа, повышенная точность V(s))
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 192,
        num_blocks: int = 2,
        discrete_actions: int = NUM_DISCRETE_ACTIONS,
        cont_dim: int = CONT_ACTION_DIM,
        init_log_std: float = -0.70,
        critic_hidden_dim: int | None = None,
        critic_num_blocks: int | None = None,
        critic_context_mult: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.discrete_actions = discrete_actions
        self.cont_dim = cont_dim
        self.critic_hidden_dim = int(critic_hidden_dim or hidden_dim)
        self.critic_num_blocks = int(critic_num_blocks or num_blocks)
        self.critic_context_mult = int(critic_context_mult)

        self.input_linear = _init_linear(nn.Linear(obs_dim, hidden_dim), gain=math.sqrt(2.0))
        self.input_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])

        self.discrete_head = _init_linear(nn.Linear(hidden_dim, discrete_actions), gain=0.01)
        self.move_mean_head = _init_linear(nn.Linear(hidden_dim, cont_dim), gain=0.1)
        self.move_log_std = nn.Parameter(torch.full((cont_dim,), init_log_std, dtype=torch.float32))

        critic_in = obs_dim * self.critic_context_mult
        self.critic_input = _init_linear(nn.Linear(critic_in, self.critic_hidden_dim), gain=math.sqrt(2.0))
        self.critic_norm = nn.LayerNorm(self.critic_hidden_dim, eps=1e-5)
        self.critic_blocks = nn.ModuleList(
            [ResidualMLPBlock(self.critic_hidden_dim) for _ in range(self.critic_num_blocks)]
        )
        self.value_head = _init_linear(nn.Linear(self.critic_hidden_dim, 1), gain=1.0)

    def encode_actor(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.input_norm(self.input_linear(obs)))
        for blk in self.blocks:
            x = blk(x)
        return x

    def forward_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode_actor(obs)
        logits = self.discrete_head(feat)
        move_mean = torch.tanh(self.move_mean_head(feat))
        return logits, move_mean

    def forward_critic(self, obs: torch.Tensor, team_context: torch.Tensor | None = None) -> torch.Tensor:
        expected_ctx = self.obs_dim * (self.critic_context_mult - 1)
        if team_context is None or team_context.shape[-1] != expected_ctx:
            team_context = compute_team_context(
                obs, team_size=5, include_max=(self.critic_context_mult == 3)
            )
            if team_context.shape[-1] != expected_ctx:
                team_context = obs.repeat_interleave(self.critic_context_mult - 1, dim=-1)
        c_in = torch.cat([obs, team_context], dim=-1)
        v = F.silu(self.critic_norm(self.critic_input(c_in)))
        for blk in self.critic_blocks:
            v = blk(v)
        return self.value_head(v).squeeze(-1)

    def forward_actor_critic(
        self, obs: torch.Tensor, team_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, move_mean = self.forward_actor(obs)
        value = self.forward_critic(obs, team_context)
        return logits, move_mean, value

    def _masked_logits_fp32(
        self,
        logits: torch.Tensor,
        action_masks: torch.Tensor,
        obs: torch.Tensor | None = None,
        use_iq_temp: bool = False,
    ) -> torch.Tensor:
        mask_bool = action_masks.to(dtype=torch.bool)
        empty_rows = ~mask_bool.any(dim=-1, keepdim=True)
        if empty_rows.any():
            fallback = torch.zeros_like(mask_bool)
            fallback[..., 0] = True
            mask_bool = torch.where(empty_rows, fallback, mask_bool)
        scaled = logits.float()
        if use_iq_temp and obs is not None:
            tau = compute_iq_temperature(obs).unsqueeze(-1)
            scaled = scaled / tau
        return scaled.masked_fill(~mask_bool, -1e9)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor,
        team_context: torch.Tensor | None = None,
        deterministic: bool = False,
        use_iq_temp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, move_mean, value = self.forward_actor_critic(obs, team_context)
        masked_logits = self._masked_logits_fp32(logits, action_masks, obs=obs, use_iq_temp=use_iq_temp)
        cat_dist = Categorical(logits=masked_logits)

        mean_fp32 = move_mean.float()
        std_fp32 = torch.exp(torch.clamp(self.move_log_std.float(), -3.0, 0.5)).expand_as(mean_fp32)
        cont_dist = Normal(mean_fp32, std_fp32)

        if deterministic:
            disc_action = torch.argmax(masked_logits, dim=-1)
            move_action = mean_fp32
        else:
            disc_action = cat_dist.sample()
            move_action = cont_dist.sample()

        log_prob = cat_dist.log_prob(disc_action) + cont_dist.log_prob(move_action).sum(dim=-1)
        move_clipped = torch.clamp(move_action, -1.0, 1.0)
        return disc_action, move_action, move_clipped, log_prob, value.float()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor,
        disc_actions: torch.Tensor,
        raw_move_actions: torch.Tensor,
        team_context: torch.Tensor | None = None,
        use_iq_temp: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, move_mean, value = self.forward_actor_critic(obs, team_context)
        masked_logits = self._masked_logits_fp32(logits, action_masks, obs=obs, use_iq_temp=use_iq_temp)
        cat_dist = Categorical(logits=masked_logits)

        mean_fp32 = move_mean.float()
        std_fp32 = torch.exp(torch.clamp(self.move_log_std.float(), -3.0, 0.5)).expand_as(mean_fp32)
        cont_dist = Normal(mean_fp32, std_fp32)

        log_prob = cat_dist.log_prob(disc_actions) + cont_dist.log_prob(raw_move_actions.float()).sum(dim=-1)
        entropy = cat_dist.entropy() + 0.25 * cont_dist.entropy().sum(dim=-1)
        return log_prob, entropy, value.float(), masked_logits


def compute_team_context(obs: torch.Tensor, team_size: int = 5, include_max: bool = False) -> torch.Tensor:
    if obs.shape[-2] % team_size != 0:
        return torch.cat([obs, obs], dim=-1) if include_max else obs
    lead_shape = obs.shape[:-2]
    num_teams = obs.shape[-2] // team_size
    grouped = obs.reshape(*lead_shape, num_teams, team_size, obs.shape[-1])
    team_mean = grouped.mean(dim=-2, keepdim=True).expand_as(grouped).reshape_as(obs)
    if not include_max:
        return team_mean
    team_max = grouped.amax(dim=-2, keepdim=True).expand_as(grouped).reshape_as(obs)
    return torch.cat([team_mean, team_max], dim=-1)


def _linear_to_dict(layer: nn.Linear, precision: int = 6) -> dict[str, Any]:
    w = layer.weight.detach().cpu().float().reshape(-1).tolist()
    b = layer.bias.detach().cpu().float().reshape(-1).tolist()
    return {
        "inFeatures": int(layer.in_features),
        "outFeatures": int(layer.out_features),
        "weight": [round(float(v), precision) for v in w],
        "bias": [round(float(v), precision) for v in b],
    }


def _layernorm_to_dict(ln: nn.LayerNorm, precision: int = 6) -> dict[str, Any]:
    w = ln.weight.detach().cpu().float().reshape(-1).tolist()
    b = ln.bias.detach().cpu().float().reshape(-1).tolist()
    return {
        "features": int(ln.normalized_shape[0]),
        "weight": [round(float(v), precision) for v in w],
        "bias": [round(float(v), precision) for v in b],
        "eps": float(ln.eps),
    }


def build_actor_payload(
    model: nn.Module,
    model_name: str = "balbesbol-neural-v1",
    trained_steps: int = 0,
    precision: int = 5,
) -> dict[str, Any]:
    raw: BasketballActorCritic = getattr(model, "_orig_mod", model)
    return {
        "formatVersion": 1,
        "modelName": model_name,
        "trainedSteps": int(trained_steps),
        "obsDim": int(raw.obs_dim),
        "discreteActions": int(raw.discrete_actions),
        "contActionDim": int(raw.cont_dim),
        "hiddenDim": int(raw.hidden_dim),
        "inputLinear": _linear_to_dict(raw.input_linear, precision),
        "inputNorm": _layernorm_to_dict(raw.input_norm, precision),
        "blocks": [
            {
                "fc1": _linear_to_dict(blk.fc1, precision),
                "ln1": _layernorm_to_dict(blk.ln1, precision),
                "fc2": _linear_to_dict(blk.fc2, precision),
                "ln2": _layernorm_to_dict(blk.ln2, precision),
            }
            for blk in raw.blocks
        ],
        "discreteHead": _linear_to_dict(raw.discrete_head, precision),
        "moveMeanHead": _linear_to_dict(raw.move_mean_head, precision),
    }


def export_actor_to_json(
    model: nn.Module,
    out_path: str | Path = DEFAULT_JSON_OUT,
    model_name: str = "balbesbol-neural-v1",
    trained_steps: int = 0,
    embed_in_ts: bool = True,
    ts_out_path: str | Path = DEFAULT_TS_WEIGHTS_OUT,
) -> Path:
    """
    Сохраняет слепок весов актора в public/models/policy_latest.json И автоматически вшивает
    его в src/game/rl/defaultTrainedWeights.ts для тихой работы в браузере без лишних кнопок.
    """
    payload = build_actor_payload(model, model_name=model_name, trained_steps=trained_steps)
    compact_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(compact_json, encoding="utf-8")

    if embed_in_ts:
        ts_target = Path(ts_out_path)
        ts_target.parent.mkdir(parents=True, exist_ok=True)
        ts_code = (
            "import { PolicyWeightsJSON } from './schema';\n\n"
            "/**\n"
            f" * Автоматически сгенерированный слепок обученной нейросети ({model_name}, шагов: {trained_steps:,}).\n"
            " * Подхватывается движком «БалбесБола» тихо при старте игры без дополнительных меню.\n"
            " */\n"
            f"export const DEFAULT_EMBEDDED_POLICY_WEIGHTS: PolicyWeightsJSON = {compact_json};\n"
        )
        ts_target.write_text(ts_code, encoding="utf-8")

    return target
