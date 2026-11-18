#!/usr/bin/env python3
"""
Объективная оценка силы ИИ «БалбесБола» на ЭТАЛОННОМ движке src/game/sim.ts:
  * против встроенного эвристического ИИ (stepPlayerAI) в игровом контроллере 'neural';
  * очные матчи двух нейросетей ('neural' vs 'neural' с сменой сторон 50/50);
  * использует лёгкие ESM-воркеры (node --max-old-space-size=256) без расхода памяти на tsx.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (  # noqa: E402
    CONT_ACTION_DIM,
    OBS_DIM,
    BasketballActorCritic,
    compute_team_context,
    export_actor_to_json,
)
from vec_env import SubprocBBBallVecEnv, resolve_node_worker_cmd  # noqa: E402


def _load_linear(layer: torch.nn.Linear, d: dict) -> None:
    layer.weight.data.copy_(torch.tensor(d["weight"], dtype=torch.float32).reshape(d["outFeatures"], d["inFeatures"]))
    layer.bias.data.copy_(torch.tensor(d["bias"], dtype=torch.float32))


def _load_ln(ln: torch.nn.LayerNorm, d: dict) -> None:
    ln.weight.data.copy_(torch.tensor(d["weight"], dtype=torch.float32))
    ln.bias.data.copy_(torch.tensor(d["bias"], dtype=torch.float32))


def load_policy(
    path: str | Path,
    device: torch.device,
    upgrade_critic: tuple[int, int, int] | None = None,
) -> BasketballActorCritic:
    """
    Загружает .pt чекпоинт (актор+критик) или .json слепок актора.
    Если задан upgrade_critic=(critic_hidden, critic_blocks, critic_mult), переносит веса актора
    в модель с новым расширенным критиком (или сохраняет критика, если размеры уже совпадают).
    """
    path = Path(path)
    if path.suffix == ".pt":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        hidden = int(ck.get("hidden_dim", sd["input_linear.weight"].shape[0]))
        blocks = int(ck.get("num_blocks", len({k.split(".")[1] for k in sd if k.startswith("blocks.")})))
        c_hid = int(sd["critic_input.weight"].shape[0])
        c_mult = int(sd["critic_input.weight"].shape[1] // OBS_DIM)
        c_blk = len({k.split(".")[1] for k in sd if k.startswith("critic_blocks.")})
        if upgrade_critic is not None and (c_hid, c_blk, c_mult) != upgrade_critic:
            m = BasketballActorCritic(
                obs_dim=OBS_DIM,
                hidden_dim=hidden,
                num_blocks=blocks,
                critic_hidden_dim=upgrade_critic[0],
                critic_num_blocks=upgrade_critic[1],
                critic_context_mult=upgrade_critic[2],
            )
            actor_sd = {k: v for k, v in sd.items() if not k.startswith(("critic_", "value_head"))}
            m.load_state_dict(actor_sd, strict=False)
        else:
            m = BasketballActorCritic(
                obs_dim=OBS_DIM,
                hidden_dim=hidden,
                num_blocks=blocks,
                critic_hidden_dim=c_hid,
                critic_num_blocks=c_blk,
                critic_context_mult=c_mult,
            )
            m.load_state_dict(sd)
    else:
        d = json.loads(path.read_text(encoding="utf-8"))
        c_args = (
            {"critic_hidden_dim": upgrade_critic[0], "critic_num_blocks": upgrade_critic[1], "critic_context_mult": upgrade_critic[2]}
            if upgrade_critic is not None
            else {}
        )
        m = BasketballActorCritic(
            obs_dim=int(d["obsDim"]),
            hidden_dim=int(d["hiddenDim"]),
            num_blocks=len(d["blocks"]),
            **c_args,
        )
        _load_linear(m.input_linear, d["inputLinear"])
        _load_ln(m.input_norm, d["inputNorm"])
        for blk, bd in zip(m.blocks, d["blocks"]):
            _load_linear(blk.fc1, bd["fc1"])
            _load_ln(blk.ln1, bd["ln1"])
            _load_linear(blk.fc2, bd["fc2"])
            _load_ln(blk.ln2, bd["ln2"])
        _load_linear(m.discrete_head, d["discreteHead"])
        _load_linear(m.move_mean_head, d["moveMeanHead"])
    return m.to(device).eval()


def game_eval(
    weights_json: str | Path,
    games: int = 48,
    procs: int = 4,
    seed: int = 1,
    quarter: int = 45,
    opponent_json: str | Path | None = None,
    mod_prob: float = 0.0,
) -> dict:
    """
    Оценка РОВНО как в игре: контроллер 'neural' из sim.ts (IQ-температура, шум восприятия, 15 Гц)
    против встроенного ИИ (opponent_json=None) ИЛИ против другой нейросети (opponent_json=Path).
    Запускается через лёгкий `node --max-old-space-size=256 training/.dist/rl_eval_game.mjs`.
    """
    root = Path(__file__).resolve().parent.parent
    base_cmd = resolve_node_worker_cmd("rl_eval_game", heap_mb=256)
    per = max(2, int(math.ceil(games / procs)))
    per += per % 2
    opp_arg = str(Path(opponent_json).resolve()) if opponent_json is not None else "heuristic"

    def _spawn(i: int) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                *base_cmd,
                f"--weights={Path(weights_json).resolve()}",
                f"--opponent={opp_arg}",
                f"--games={per}",
                f"--seed={seed * 101 + i}",
                f"--quarter={quarter}",
                f"--modProb={mod_prob}",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    ps = [_spawn(i) for i in range(procs)]
    diffs: list[float] = []
    for i, pr in enumerate(ps):
        for _ in range(3):
            out, err = pr.communicate()
            lines = [ln for ln in (out or "").strip().splitlines() if ln.startswith("{")]
            if lines:
                diffs.extend(json.loads(lines[-1])["diffs"])
                break
            print(f"  [game-eval] процесс {i} без результата (код {pr.returncode}): {(err or '')[-240:]!r}; повтор", flush=True)
            time.sleep(1.5)
            pr = _spawn(i)
    if len(diffs) < 2:
        raise RuntimeError("game_eval: не удалось получить результаты")

    arr = np.asarray(diffs[:games] if len(diffs) >= games else diffs, dtype=np.float64)
    return {
        "games": int(len(arr)),
        "win_rate": float((arr > 0).mean()),
        "tie_rate": float((arr == 0).mean()),
        "mean_diff": float(arr.mean()),
        "ci95": float(1.96 * arr.std(ddof=1) / math.sqrt(len(arr))),
    }


@torch.no_grad()
def _act(model: BasketballActorCritic, obs: torch.Tensor, masks: torch.Tensor, deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
    da, _, mc, _, _ = model.act(obs, masks, deterministic=deterministic)
    return da.to(torch.uint8).cpu().numpy(), mc.float().cpu().numpy()


def evaluate(
    policy: BasketballActorCritic,
    opponent: BasketballActorCritic | None,
    games: int,
    device: torch.device,
    num_workers: int = 4,
    quarter_duration: int = 45,
    deterministic: bool = False,
    seed: int = 777,
    verbose: bool = True,
) -> dict:
    diffs: list[float] = []
    per_side = max(1, games // 2)
    for side in (0, 1):
        envs_per_w = max(1, math.ceil(per_side / num_workers))
        n_envs = envs_per_w * num_workers
        controlled: str | int = side if opponent is None else "both"
        with SubprocBBBallVecEnv(
            num_workers=num_workers,
            envs_per_worker=envs_per_w,
            controlled_teams=controlled,
            frame_skip=4,
            quarter_duration=quarter_duration,
            randomize_rosters=True,
            randomize_tactics=True,
            seed=seed + side * 31337,
        ) as env:
            o_np, m_np = env.reset(seed=seed + side * 31337)
            finished = np.zeros(n_envs, dtype=bool)
            side_diffs: list[float] = []
            if opponent is not None:
                team_of_agent = np.tile(np.repeat([0, 1], 5), n_envs)
                pol_rows = torch.from_numpy(np.nonzero(team_of_agent == side)[0]).to(device)
                opp_rows = torch.from_numpy(np.nonzero(team_of_agent != side)[0]).to(device)
            while not finished.all():
                obs = torch.from_numpy(o_np).to(device)
                masks = torch.from_numpy(m_np).to(device)
                if opponent is None:
                    disc, move = _act(policy, obs, masks, deterministic)
                else:
                    disc = np.zeros((env.total_agents,), dtype=np.uint8)
                    move = np.zeros((env.total_agents, CONT_ACTION_DIM), dtype=np.float32)
                    pd, pm = _act(policy, obs[pol_rows], masks[pol_rows], deterministic)
                    od, om = _act(opponent, obs[opp_rows], masks[opp_rows], deterministic)
                    disc[pol_rows.cpu().numpy()] = pd
                    move[pol_rows.cpu().numpy()] = pm
                    disc[opp_rows.cpu().numpy()] = od
                    move[opp_rows.cpu().numpy()] = om
                o_np, m_np, _, _, episodes = env.step(disc, move)
                for ep in episodes:
                    gi = int(ep["globalEnvIdx"])
                    if finished[gi]:
                        continue
                    finished[gi] = True
                    s = ep["score"]
                    side_diffs.append(float(s[side] - s[1 - side]))
            diffs.extend(side_diffs[:per_side])
    arr = np.asarray(diffs, dtype=np.float64)
    wins = float((arr > 0).mean())
    ties = float((arr == 0).mean())
    ci = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    res = {"games": len(arr), "win_rate": wins, "tie_rate": ties, "mean_diff": float(arr.mean()), "ci95": ci}
    if verbose:
        print(f"  игр={res['games']}  побед={wins*100:.1f}%  ничьих={ties*100:.1f}%  разница={res['mean_diff']:+.2f} ± {ci:.2f}")
    return res


def _ensure_json(pol_path: Path) -> Path:
    if pol_path.suffix != ".pt":
        return pol_path
    out = Path(tempfile.gettempdir()) / f"bbball_eval_{pol_path.stem}.json"
    export_actor_to_json(load_policy(pol_path, torch.device("cpu")), out_path=out, embed_in_ts=False)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Оценка ИИ БалбесБола на эталонном sim.ts")
    p.add_argument("--policy", type=Path, nargs="+", required=True, help=".pt или .json (можно несколько)")
    p.add_argument("--opponent", type=Path, default=None, help="Политика-соперник; по умолчанию встроенный эвристический ИИ")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--quarter", type=int, default=45)
    p.add_argument("--mod-prob", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--mode", choices=["game", "python"], default="game")
    args = p.parse_args()

    if args.mode == "game":
        opp_json = _ensure_json(args.opponent) if args.opponent else None
        opp_name = args.opponent.name if args.opponent else "встроенный ИИ (stepPlayerAI)"
        for pol_path in args.policy:
            t0 = time.perf_counter()
            js = _ensure_json(pol_path)
            r = game_eval(js, args.games, args.workers, args.seed, args.quarter, opponent_json=opp_json, mod_prob=args.mod_prob)
            print(
                f"[game-eval] {pol_path.name} vs {opp_name}: игр={r['games']} "
                f"побед={r['win_rate']*100:.1f}% ничьих={r['tie_rate']*100:.1f}% "
                f"разница={r['mean_diff']:+.2f}±{r['ci95']:.2f} ({time.perf_counter()-t0:.0f}с)"
            )
        return 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opp = load_policy(args.opponent, device) if args.opponent else None
    for pol_path in args.policy:
        t0 = time.perf_counter()
        evaluate(load_policy(pol_path, device), opp, args.games, device, args.workers, args.quarter, args.deterministic, args.seed)
        print(f"  ({time.perf_counter() - t0:.0f} с)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
