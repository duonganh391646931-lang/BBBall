#!/usr/bin/env python3
"""
Конвейер обучения ИИ «БалбесБола» v3 (Этапы 1–4 автономного 5-часового плана):
  1) Полное выравнивание Обучение == Реальная игра:
       - инерционный шум восприятия IQ (iqPerceptionNoise=True в sim.ts);
       - точная индивидуальная температура игрока computeIqTemperature(iq) внутри BC и PPO;
       - плавное затухание move_log_std к детерминированному движению игры.
  2) Имитационное обучение (BC) в VRAM (bf16) с покрытием модификаторов матча и несмещённым
     взвешиванием по активным состояниям w(s) (masks[:, 1:].any(-1)).
  3) CTDE-PPO на эталонном sim.ts с расширенным централизованным критиком [obs, team_mean, team_max] (384x3),
     адаптивным KL-контроллером и плавным якорем учителя.
  4) Лиговая самоигра (League Self-Play): смесь эвристики (50%) + текущей сети (35%) + пула чемпионов (15%)
     внутри тех же лёгких ESM-воркеров без расхода лишней памяти.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import game_eval, load_policy  # noqa: E402
from gpu_limiter import GPUResourceLimiter  # noqa: E402
from model import (  # noqa: E402
    CONT_ACTION_DIM,
    DEFAULT_JSON_OUT,
    DEFAULT_TS_WEIGHTS_OUT,
    NUM_DISCRETE_ACTIONS,
    OBS_DIM,
    BasketballActorCritic,
    compute_iq_temperature,
    compute_team_context,
    export_actor_to_json,
)
from vec_env import SubprocBBBallVecEnv, resolve_node_worker_cmd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD_PATH = ROOT / "training" / "leaderboard.json"


# ============================================================================= системная память Windows
class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def get_free_memory_gb() -> tuple[float, float]:
    """Возвращает (свободно физ. ОЗУ ГБ, свободно Commit Charge ГБ)."""
    if os.name != "nt":
        return 8.0, 8.0
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPhys / (1024**3), st.ullAvailPageFile / (1024**3)


def append_leaderboard(entry: dict) -> None:
    rows: list[dict] = []
    if LEADERBOARD_PATH.exists():
        try:
            rows = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    entry["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rows.append(entry)
    LEADERBOARD_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================= аргументы
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="БалбесБол v3: IQ-aligned BC + CTDE-PPO + League Self-Play")
    p.add_argument("--run-name", type=str, default="v3_run")
    p.add_argument("--init", type=Path, default=None, help=".pt/.json для старта (иначе с нуля)")
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--critic-hidden", type=int, default=384)
    p.add_argument("--critic-blocks", type=int, default=3)
    p.add_argument("--max-gpu-util", type=float, default=0.70)
    p.add_argument("--max-minutes", type=float, default=0.0, help="Мягкий лимит времени фазы в минутах (0 = без лимита)")
    p.add_argument("--seed", type=int, default=2027)

    # Фаза BC
    p.add_argument("--bc-samples", type=int, default=0, help="0 = пропустить обучение BC")
    p.add_argument("--data-dir", type=Path, default=ROOT / "training" / "data" / "bc_v3")
    p.add_argument("--datagen-samples", type=int, default=4_000_000, help="Размер датасета учителя на диске")
    p.add_argument("--datagen-procs", type=int, default=6)
    p.add_argument("--reuse-data", action="store_true")
    p.add_argument("--bc-epochs", type=float, default=10.0)
    p.add_argument("--bc-batch", type=int, default=8192)
    p.add_argument("--bc-lr", type=float, default=1.2e-3)
    p.add_argument("--tactical-sample-frac", type=float, default=0.60,
                   help="Доля батча BC из тактически активных состояний masks[:,1:].any(-1)")

    # Фаза PPO / League
    p.add_argument("--ppo-updates", type=int, default=200)
    p.add_argument("--mode", choices=["heuristic", "league"], default="heuristic")
    p.add_argument("--league-pool", type=Path, nargs="*", default=[], help="JSON-веса чемпионов для пула Лиги")
    p.add_argument("--gate-opponent", type=Path, default=None, help="JSON ориентира для очной проверки neural-vs-neural")
    p.add_argument("--workers-per-side", type=int, default=3)
    p.add_argument("--envs-per-worker", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=6144)
    p.add_argument("--lr", type=float, default=1.8e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.15)
    p.add_argument("--target-kl", type=float, default=0.018)
    p.add_argument("--ent-coef", type=float, default=0.0025)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--critic-warmup", type=int, default=8)
    p.add_argument("--move-std-start", type=float, default=-1.65)
    p.add_argument("--move-std-end", type=float, default=-2.25)
    p.add_argument("--bc-coef-start", type=float, default=0.18)
    p.add_argument("--bc-coef-end", type=float, default=0.02)
    p.add_argument("--bc-anchor-vram", type=int, default=500_000, help="Сколько примеров учителя держать в VRAM во время PPO")
    p.add_argument("--mod-prob", type=float, default=0.15, help="Доля матчей с модификаторами физики")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-games", type=int, default=64)
    p.add_argument("--eval-procs", type=int, default=4)
    p.add_argument("--quarter", type=int, default=45)

    # Сохранение
    p.add_argument("--save-dir", type=Path, default=ROOT / "training" / "checkpoints_v3")
    p.add_argument("--export-live", action="store_true", help="Сразу обновлять policy_latest.json и defaultTrainedWeights.ts при новом рекорде")
    p.add_argument("--min-export-diff", type=float, default=0.30, help="Минимальная разница vs эвристика для перезаписи боевых весов игры")
    return p.parse_args()


# ============================================================================= вспомогательные классы
class RunningMeanStd:
    def __init__(self, device: torch.device) -> None:
        self.mean = torch.zeros((), device=device, dtype=torch.float64)
        self.var = torch.ones((), device=device, dtype=torch.float64)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        x = x.double().reshape(-1)
        b_mean, b_var, b_cnt = x.mean(), x.var(unbiased=False), x.numel()
        delta = b_mean - self.mean
        tot = self.count + b_cnt
        self.mean = self.mean + delta * b_cnt / tot
        m2 = self.var * self.count + b_var * b_cnt + delta.pow(2) * self.count * b_cnt / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return float(torch.sqrt(self.var).item()) + 1e-8


def save_ckpt(model: BasketballActorCritic, path: Path, steps: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "num_blocks": model.num_blocks,
            "trained_steps": int(steps),
            **(extra or {}),
        },
        path,
    )


def bc_iq_loss(
    model: BasketballActorCritic,
    obs: torch.Tensor,
    masks: torch.Tensor,
    disc: torch.Tensor,
    move: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BC-потеря с учётом игровой температуры computeIqTemperature(iq) каждого игрока."""
    logits, move_pred = model.forward_actor(obs)
    tau = compute_iq_temperature(obs).unsqueeze(-1)
    scaled_logits = (logits.float() / tau).masked_fill(~masks, -1e9)
    l_d = F.cross_entropy(scaled_logits, disc.long())
    l_m = F.smooth_l1_loss(move_pred.float(), move.float(), beta=0.12)
    return l_d, l_m, scaled_logits


def _tactical_mask(masks_gpu: torch.Tensor) -> torch.Tensor:
    """Состояния с активным тактическим выбором (кроме вездесущего SCREEN_SET=17), зависит ТОЛЬКО от s."""
    return masks_gpu[:, 1:17].any(dim=-1) | masks_gpu[:, 18:].any(dim=-1)
def ensure_teacher_dataset(args: argparse.Namespace) -> list[Path]:
    data_root: Path = args.data_dir
    data_root.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, args.datagen_procs)
    per_proc = int(math.ceil(args.datagen_samples / n_shards))
    dirs = [data_root / f"shard{i}" for i in range(n_shards)]
    if args.reuse_data and all((d / "meta.json").exists() and (d / "disc.u8").exists() for d in dirs):
        total = sum(int((d / "disc.u8").stat().st_size) for d in dirs)
        if total >= int(args.datagen_samples * 0.9):
            print(f"[Data] Повторно используем готовый датасет учителя: {total:,} переходов в {data_root}", flush=True)
            return dirs

    cmd_base = resolve_node_worker_cmd("rl_datagen", heap_mb=256)
    print(f"[Data] Генерация {per_proc * n_shards:,} переходов учителя (iqNoise=1, modProb=0.22, {n_shards} воркеров)...", flush=True)
    t0 = time.perf_counter()
    procs = []
    logs = []
    for i, d in enumerate(dirs):
        lf = open(data_root / f"gen_{i}.log", "w", encoding="utf-8")
        logs.append(lf)
        procs.append(
            subprocess.Popen(
                [
                    *cmd_base,
                    f"--samples={per_proc}",
                    f"--out={d}",
                    f"--seed={args.seed * 1000 + i * 7919}",
                    f"--quarterDuration={args.quarter}",
                    "--iqNoise=1",
                    "--modProb=0.22",
                ],
                cwd=str(ROOT),
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        )
    for pr in procs:
        rc = pr.wait()
        if rc != 0:
            raise RuntimeError(f"rl_datagen завершился с кодом {rc}")
    for lf in logs:
        lf.close()
    print(f"[Data] Сгенерировано {per_proc * n_shards:,} примеров за {time.perf_counter()-t0:.1f}с", flush=True)
    return dirs


def load_teacher_to_vram(
    dirs: list[Path],
    device: torch.device,
    max_samples: int,
    max_gpu_util: float = 0.70,
) -> dict[str, torch.Tensor]:
    """
    Читает шарды по одному через крошечный временный буфер и сразу переносит в VRAM (bf16/uint8),
    адаптируя объём под свободный Commit Charge Windows и лимит VRAM (<= 70%), чтобы 0 МБ данных лежало в ОЗУ.
    """
    free_ram_gb, free_commit_gb = get_free_memory_gb()
    vram_cap_gb = 4.0
    if device.type == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        vram_cap_gb = total_vram_gb * max_gpu_util - 1.1  # запас 1.1 ГБ под модель, градиенты и активации
    # На Windows каждый 1 ГБ VRAM резервирует 1 ГБ Commit Charge (~520 байт на 1 переход в bf16)
    safe_gb = max(0.5, min(vram_cap_gb, free_commit_gb - 1.35))
    safe_samples = int(safe_gb * (1024**3) / 530)
    target_n = min(max_samples, max(400_000, safe_samples))

    avail_per_shard = [int((d / "disc.u8").stat().st_size) for d in dirs]
    total_disk = sum(avail_per_shard)
    take_total = min(target_n, total_disk)
    print(
        f"[VRAM] Загрузка датасета в VRAM: {take_total:,} / {total_disk:,} примеров "
        f"(свободно RAM={free_ram_gb:.2f} ГБ, Commit={free_commit_gb:.2f} ГБ)",
        flush=True,
    )

    obs_gpu = torch.empty((take_total, OBS_DIM), dtype=torch.bfloat16, device=device)
    masks_gpu = torch.empty((take_total, NUM_DISCRETE_ACTIONS), dtype=torch.bool, device=device)
    disc_gpu = torch.empty((take_total,), dtype=torch.uint8, device=device)
    move_gpu = torch.empty((take_total, CONT_ACTION_DIM), dtype=torch.bfloat16, device=device)

    cursor = 0
    per_shard_quota = int(math.ceil(take_total / len(dirs)))
    for d, n_sh in zip(dirs, avail_per_shard):
        if cursor >= take_total:
            break
        k = min(n_sh, per_shard_quota, take_total - cursor)
        # Читаем ровно k записей и немедленно освобождаем массив numpy
        o_np = np.fromfile(d / "obs.f32", dtype="<f4", count=k * OBS_DIM).reshape(k, OBS_DIM)
        obs_gpu[cursor : cursor + k].copy_(torch.from_numpy(o_np).to(device=device, dtype=torch.bfloat16))
        del o_np

        m_np = np.fromfile(d / "masks.u8", dtype=np.uint8, count=k * NUM_DISCRETE_ACTIONS).reshape(k, NUM_DISCRETE_ACTIONS)
        masks_gpu[cursor : cursor + k].copy_(torch.from_numpy(m_np).to(device=device, dtype=torch.bool))
        del m_np

        d_np = np.fromfile(d / "disc.u8", dtype=np.uint8, count=k)
        disc_gpu[cursor : cursor + k].copy_(torch.from_numpy(d_np).to(device=device))
        del d_np

        mv_np = np.fromfile(d / "move.f32", dtype="<f4", count=k * CONT_ACTION_DIM).reshape(k, CONT_ACTION_DIM)
        move_gpu[cursor : cursor + k].copy_(torch.from_numpy(mv_np).to(device=device, dtype=torch.bfloat16))
        del mv_np

        cursor += k
        gc.collect()

    obs_gpu = obs_gpu[:cursor]
    masks_gpu = masks_gpu[:cursor]
    disc_gpu = disc_gpu[:cursor]
    move_gpu = move_gpu[:cursor]
    masks_gpu.scatter_(1, disc_gpu.long().unsqueeze(1), True)

    tact_idx = torch.nonzero(_tactical_mask(masks_gpu), as_tuple=False).squeeze(-1)
    vram_mb = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    print(
        f"[VRAM] Загружено {cursor:,} переходов в VRAM ({vram_mb:.0f} MiB занято в GPU, "
        f"тактически активных состояний: {tact_idx.numel():,} = {tact_idx.numel()/max(1,cursor)*100:.1f}%)",
        flush=True,
    )
    return {"obs": obs_gpu, "masks": masks_gpu, "disc": disc_gpu, "move": move_gpu, "tact_idx": tact_idx}


def shrink_bc_vram(data: dict[str, torch.Tensor] | None, keep_n: int, device: torch.device) -> dict[str, torch.Tensor] | None:
    """Сжимает буфер учителя в VRAM перед стартом воркеров PPO, освобождая гигабайты VRAM и Commit Charge."""
    if data is None:
        return None
    n = int(data["disc"].numel())
    if n <= keep_n:
        return data
    perm = torch.randperm(n, device=device)[:keep_n]
    compact = {
        "obs": data["obs"][perm].contiguous(),
        "masks": data["masks"][perm].contiguous(),
        "disc": data["disc"][perm].contiguous(),
        "move": data["move"][perm].contiguous(),
    }
    compact["tact_idx"] = torch.nonzero(_tactical_mask(compact["masks"]), as_tuple=False).squeeze(-1)
    data.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vram_mb = torch.cuda.memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    print(f"[VRAM] Якорь учителя сжат до {keep_n:,} примеров (VRAM занято: {vram_mb:.0f} MiB)", flush=True)
    return compact


# ============================================================================= Фаза BC
def run_bc_phase(model, data, args, device, limiter, amp_dtype, use_bf16) -> int:
    n = int(data["disc"].numel())
    n_val = min(60_000, n // 40)
    perm = torch.randperm(n, device=device)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_tact_mask = _tactical_mask(data["masks"][tr_idx])
    tr_tact = tr_idx[tr_tact_mask]

    steps = max(50, int(args.bc_epochs * tr_idx.numel() / args.bc_batch))
    n_tact_mb = int(args.bc_batch * args.tactical_sample_frac) if tr_tact.numel() > 0 else 0
    n_unif_mb = args.bc_batch - n_tact_mb

    opt = torch.optim.AdamW(model.parameters(), lr=args.bc_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.bc_lr, total_steps=steps, pct_start=0.06)
    print(
        f"[BC] Модель {model.hidden_dim}x{model.num_blocks}: {tr_idx.numel():,} train / {n_val:,} val | "
        f"{steps} шагов по {args.bc_batch} (тактических {n_tact_mb} + общих {n_unif_mb}, IQ-temp=ON)",
        flush=True,
    )
    t0 = time.perf_counter()
    model.train()
    for it in range(1, steps + 1):
        limiter.begin_work()
        if n_tact_mb > 0:
            b1 = tr_tact[torch.randint(0, tr_tact.numel(), (n_tact_mb,), device=device)]
            b2 = tr_idx[torch.randint(0, tr_idx.numel(), (n_unif_mb,), device=device)]
            b = torch.cat([b1, b2], dim=0)
        else:
            b = tr_idx[torch.randint(0, tr_idx.numel(), (args.bc_batch,), device=device)]

        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
            l_d, l_m, _ = bc_iq_loss(model, data["obs"][b], data["masks"][b], data["disc"][b], data["move"][b])
            loss = l_d + 2.4 * l_m

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        limiter.end_work_and_throttle()

        if it % max(1, steps // 8) == 0 or it == steps:
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                accs, nonnoop, lm = [], [], []
                for s in range(0, n_val, 8192):
                    v = val_idx[s : s + 8192]
                    _, vm, ml = bc_iq_loss(model, data["obs"][v], data["masks"][v], data["disc"][v], data["move"][v])
                    pred = ml.argmax(-1)
                    targ = data["disc"][v].long()
                    accs.append((pred == targ).float().mean().item())
                    nz = targ != 0
                    if nz.any():
                        nonnoop.append((pred[nz] == targ[nz]).float().mean().item())
                    lm.append(vm.item())
            model.train()
            print(
                f"  [BC {it:4d}/{steps}] ce={l_d.item():.4f} | val acc={np.mean(accs)*100:.2f}% "
                f"acc(не NOOP)={np.mean(nonnoop)*100:.1f}% move_l1={np.mean(lm):.4f} "
                f"GPU_EMA={limiter.ema_gpu_util:.1f}% ({time.perf_counter()-t0:.0f}с)",
                flush=True,
            )
    return steps * args.bc_batch


# ============================================================================= оценка и гейткипер
def evaluate_candidate(
    model: BasketballActorCritic,
    args: argparse.Namespace,
    tag: str,
    steps_done: int,
) -> dict:
    t0 = time.perf_counter()
    model.eval()
    cand_json = args.save_dir / "_eval_candidate.json"
    export_actor_to_json(
        model,
        out_path=cand_json,
        model_name=f"balbesbol-{args.run_name}-{model.hidden_dim}x{model.num_blocks}",
        trained_steps=steps_done,
        embed_in_ts=False,
    )
    res_h = game_eval(cand_json, games=args.eval_games, procs=args.eval_procs, seed=4242, quarter=args.quarter)
    score = res_h["mean_diff"]
    h2h_info = ""
    res_g = None
    if args.gate_opponent is not None and args.gate_opponent.exists():
        res_g = game_eval(
            cand_json,
            games=max(48, (args.eval_games * 3) // 4),
            procs=args.eval_procs,
            seed=8484,
            quarter=args.quarter,
            opponent_json=args.gate_opponent,
        )
        # В режиме Лиги ценим очный перевес над сильным нейро-чемпионом (55%) + разгром эвристики (45%)
        w_h = 0.45 if args.mode == "league" else 0.70
        score = w_h * res_h["mean_diff"] + (1.0 - w_h) * res_g["mean_diff"]
        h2h_info = f" | vs {args.gate_opponent.stem}: побед={res_g['win_rate']*100:.1f}% разн={res_g['mean_diff']:+.2f}"

    print(
        f"  [EVAL {tag}] vs эвристика: побед={res_h['win_rate']*100:.1f}% "
        f"разница={res_h['mean_diff']:+.2f}±{res_h['ci95']:.2f}{h2h_info} "
        f"-> рейтинг={score:+.2f} ({time.perf_counter()-t0:.0f}с)",
        flush=True,
    )
    out = {
        "run": args.run_name,
        "tag": tag,
        "arch": f"{model.hidden_dim}x{model.num_blocks}",
        "steps": int(steps_done),
        "vs_heuristic": res_h,
        "vs_gate": res_g,
        "combined_score": float(score),
    }
    append_leaderboard(out)
    return out


# ============================================================================= Фаза PPO / League Self-Play
def run_ppo_phase(
    model: BasketballActorCritic,
    bc_anchor: dict[str, torch.Tensor] | None,
    args: argparse.Namespace,
    device: torch.device,
    limiter: GPUResourceLimiter,
    amp_dtype: torch.dtype,
    use_bf16: bool,
    best_record: dict,
    steps_done: int,
    t_phase_start: float,
) -> int:
    W, E, T = args.workers_per_side, args.envs_per_worker, args.rollout_steps
    envs = [
        SubprocBBBallVecEnv(
            num_workers=W,
            envs_per_worker=E,
            controlled_teams=side,
            frame_skip=4,
            quarter_duration=args.quarter,
            randomize_rosters=True,
            randomize_tactics=True,
            modifier_prob=args.mod_prob,
            iq_perception_noise=True,
            seed=args.seed + 10_000 * (side + 1),
            worker_heap_mb=256,
        )
        for side in (0, 1)
    ]
    try:
        A = sum(e.total_agents for e in envs)
        print(
            f"[PPO/{args.mode}] {2*W*E} матчей параллельно ({2*W} лёгких .mjs воркеров), "
            f"{A} агентов, rollout {T} -> {T*A:,} шагов/обновление | критик {model.critic_hidden_dim}x{model.critic_num_blocks}",
            flush=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
        ret_rms = RunningMeanStd(device)
        disc_ret = torch.zeros(A, device=device, dtype=torch.float64)
        kl_lr_mult = 1.0

        self_snap_json = args.save_dir / "_self_latest.json"
        league_paths: list[str] = [str(p.resolve()) for p in args.league_pool if p.exists()]

        def refresh_league_opponents() -> None:
            if args.mode != "league":
                return
            export_actor_to_json(model, out_path=self_snap_json, embed_in_ts=False)
            self_path = str(self_snap_json.resolve())
            for env_group in envs:
                n_e = env_group.num_envs
                assign: list[str | None] = []
                for idx in range(n_e):
                    r = random.random()
                    if r < 0.40:
                        assign.append(None)  # 40% — встроенный эвристический ИИ (защита от забывания)
                    elif r < 0.80 or not league_paths:
                        assign.append(self_path)  # 40% — текущая версия сети (живой Self-Play)
                    else:
                        assign.append(random.choice(league_paths))  # 20% — пул чемпионов Лиги
                env_group.set_opponents(assign)

        def reset_all():
            o, m = zip(*[e.reset(seed=args.seed + 31 * i) for i, e in enumerate(envs)])
            return torch.from_numpy(np.concatenate(o)).to(device), torch.from_numpy(np.concatenate(m)).to(device)

        def step_all(disc: torch.Tensor, move: torch.Tensor):
            d_np = disc.to(torch.uint8).cpu().numpy()
            m_np = move.float().cpu().numpy()
            outs, s = [], 0
            for e in envs:
                outs.append(e.step(d_np[s : s + e.total_agents], m_np[s : s + e.total_agents]))
                s += e.total_agents
            o = np.concatenate([x[0] for x in outs])
            m = np.concatenate([x[1] for x in outs])
            r = np.concatenate([x[2] for x in outs])
            d = np.concatenate([x[3] for x in outs])
            eps = []
            for side, x in enumerate(outs):
                for ep in x[4]:
                    sc = ep["score"]
                    eps.append(float(sc[side] - sc[1 - side]))
            return (
                torch.from_numpy(o).to(device),
                torch.from_numpy(m).to(device),
                torch.from_numpy(r).to(device),
                torch.from_numpy(d.astype(np.float32)).to(device),
                eps,
            )

        include_max = model.critic_context_mult == 3
        buf_obs = torch.zeros((T, A, OBS_DIM), dtype=torch.bfloat16 if use_bf16 else torch.float32, device=device)
        buf_masks = torch.zeros((T, A, NUM_DISCRETE_ACTIONS), dtype=torch.bool, device=device)
        buf_disc = torch.zeros((T, A), dtype=torch.long, device=device)
        buf_move = torch.zeros((T, A, CONT_ACTION_DIM), dtype=torch.float32, device=device)
        buf_logp = torch.zeros((T, A), dtype=torch.float32, device=device)
        buf_val = torch.zeros((T, A), dtype=torch.float32, device=device)
        buf_rew = torch.zeros((T, A), dtype=torch.float32, device=device)
        buf_done = torch.zeros((T, A), dtype=torch.float32, device=device)

        obs, masks = reset_all()
        if args.mode == "league":
            refresh_league_opponents()

        recent_eps: list[float] = []
        n_bc = int(bc_anchor["disc"].numel()) if bc_anchor is not None else 0

        for upd in range(1, args.ppo_updates + 1):
            if args.max_minutes > 0 and (time.perf_counter() - t_phase_start) / 60.0 >= args.max_minutes:
                print(f"[PPO] Достигнут бюджет времени фазы ({args.max_minutes:.1f} мин) на обновлении {upd-1}.", flush=True)
                break

            frac = (upd - 1) / max(1, args.ppo_updates - 1)
            cos_decay = 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * frac))
            lr_now = args.lr * cos_decay * kl_lr_mult
            for g in opt.param_groups:
                g["lr"] = lr_now

            move_std_now = args.move_std_start + (args.move_std_end - args.move_std_start) * frac
            with torch.no_grad():
                model.move_log_std.fill_(move_std_now)

            bc_coef = args.bc_coef_start + (args.bc_coef_end - args.bc_coef_start) * frac
            warm = upd <= args.critic_warmup

            if args.mode == "league" and upd > 1 and upd % 5 == 1:
                refresh_league_opponents()

            # ---------------- сбор траекторий (с IQ-температурой!)
            model.eval()
            t_roll = time.perf_counter()
            for t in range(T):
                limiter.begin_work()
                ctx = compute_team_context(obs, team_size=5, include_max=include_max)
                with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                    da, rma, cma, lpa, va = model.act(obs, masks, team_context=ctx, use_iq_temp=True)
                buf_obs[t] = obs.to(buf_obs.dtype)
                buf_masks[t] = masks.bool()
                buf_disc[t] = da
                buf_move[t] = rma
                buf_logp[t] = lpa
                buf_val[t] = va
                limiter.end_work_and_throttle()

                obs, masks, rew, done, eps = step_all(da, cma)
                buf_rew[t] = rew
                buf_done[t] = done
                recent_eps.extend(eps)
                disc_ret = disc_ret * args.gamma * (1.0 - done.double()) + rew.double()
                ret_rms.update(disc_ret)
            t_roll = time.perf_counter() - t_roll
            steps_done += T * A

            # ---------------- GAE
            t_upd = time.perf_counter()
            r_scale = ret_rms.std
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                next_ctx = compute_team_context(obs, team_size=5, include_max=include_max)
                next_v = model.forward_critic(obs, next_ctx).float()
            adv = torch.zeros((T, A), device=device)
            gae = torch.zeros(A, device=device)
            rew_n = buf_rew / r_scale
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else buf_val[t + 1]
                nonterm = 1.0 - buf_done[t]
                delta = rew_n[t] + args.gamma * nv * nonterm - buf_val[t]
                gae = delta + args.gamma * args.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = adv + buf_val

            # ---------------- PPO обновление
            N = T * A
            obs_f32 = buf_obs.float()
            f_ctx = compute_team_context(obs_f32, team_size=5, include_max=include_max).reshape(N, -1)
            f_obs = obs_f32.reshape(N, OBS_DIM)
            f_masks = buf_masks.reshape(N, -1)
            f_disc = buf_disc.reshape(N)
            f_move = buf_move.reshape(N, -1)
            f_logp = buf_logp.reshape(N)
            f_adv = adv.reshape(N)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
            f_ret = ret.reshape(N)
            f_val = buf_val.reshape(N)

            stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "bc": []}
            stop = False
            model.train()
            for ep_i in range(args.ppo_epochs):
                perm = torch.randperm(N, device=device)
                for s in range(0, N, args.minibatch_size):
                    limiter.begin_work()
                    mb = perm[s : s + args.minibatch_size]
                    a_mb = f_adv[mb]
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                        new_logp, ent, new_v, _ = model.evaluate_actions(
                            f_obs[mb], f_masks[mb], f_disc[mb], f_move[mb], team_context=f_ctx[mb], use_iq_temp=True
                        )
                        logratio = torch.clamp(new_logp - f_logp[mb], -20.0, 20.0)
                        ratio = logratio.exp()
                        with torch.no_grad():
                            kl = ((ratio - 1.0) - logratio).mean()
                            clipfrac = ((ratio - 1.0).abs() > args.clip).float().mean()
                        pg = torch.max(
                            -a_mb * ratio, -a_mb * torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                        ).mean()
                        v_clip = f_val[mb] + torch.clamp(new_v - f_val[mb], -0.25, 0.25)
                        vf = 0.5 * torch.max((new_v - f_ret[mb]).pow(2), (v_clip - f_ret[mb]).pow(2)).mean()
                        if warm:
                            loss = args.vf_coef * vf
                        else:
                            loss = pg + args.vf_coef * vf - args.ent_coef * ent.mean()
                            if bc_coef > 0 and n_bc > 0:
                                t_idx = bc_anchor["tact_idx"]
                                if t_idx.numel() > 0:
                                    b1 = t_idx[torch.randint(0, t_idx.numel(), (1536,), device=device)]
                                    b2 = torch.randint(0, n_bc, (1536,), device=device)
                                    bi = torch.cat([b1, b2], dim=0)
                                else:
                                    bi = torch.randint(0, n_bc, (3072,), device=device)
                                l_d, l_m, _ = bc_iq_loss(
                                    model, bc_anchor["obs"][bi], bc_anchor["masks"][bi], bc_anchor["disc"][bi], bc_anchor["move"][bi]
                                )
                                l_bc = l_d + 2.2 * l_m
                                loss = loss + bc_coef * l_bc
                                stats["bc"].append(float(l_bc.item()))

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if warm:
                        for n_, p_ in model.named_parameters():
                            if not n_.startswith(("critic_", "value_head")) and p_.grad is not None:
                                p_.grad = None
                    else:
                        # move_log_std управляется расписанием, не даём градиенту раздувать σ
                        if model.move_log_std.grad is not None:
                            model.move_log_std.grad = None
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    limiter.end_work_and_throttle()

                    stats["pg"].append(float(pg.item()))
                    stats["vf"].append(float(vf.item()))
                    stats["ent"].append(float(ent.mean().item()))
                    stats["kl"].append(float(kl.item()))
                    stats["clip"].append(float(clipfrac.item()))
                    if not warm and kl.item() > 1.6 * args.target_kl:
                        stop = True
                        break
                if stop:
                    break

            mean_kl = float(np.mean(stats["kl"])) if stats["kl"] else 0.0
            if not warm:
                if mean_kl > 1.4 * args.target_kl:
                    kl_lr_mult = max(0.35, kl_lr_mult / 1.15)
                elif mean_kl < 0.45 * args.target_kl:
                    kl_lr_mult = min(1.50, kl_lr_mult * 1.05)

            with torch.no_grad():
                ev = 1.0 - float(torch.var(f_ret - f_val) / (torch.var(f_ret) + 1e-8))
            del obs_f32, f_ctx
            t_upd = time.perf_counter() - t_upd

            if upd % 5 == 0 or upd == 1 or warm:
                last = recent_eps[-64:]
                ep_str = f"{np.mean(last):+.1f} ({len(recent_eps)} игр)" if last else "n/a"
                sps = T * A / max(1e-6, t_roll + t_upd)
                print(
                    f"  [PPO {upd:3d}/{args.ppo_updates}{' W' if warm else ''}] очки(64)={ep_str} "
                    f"rew={buf_rew.mean().item():+.4f} EV={ev:.2f} KL={mean_kl:.4f} clip={np.mean(stats['clip']):.3f} "
                    f"ent={np.mean(stats['ent']):.2f} bc={np.mean(stats['bc']) if stats['bc'] else 0:.3f} "
                    f"lr={lr_now:.1e} σ={math.exp(move_std_now):.2f} GPU={limiter.ema_gpu_util:.0f}% | "
                    f"{sps:,.0f} шаг/с ({t_roll:.1f}+{t_upd:.1f}с)",
                    flush=True,
                )
            save_ckpt(model, args.save_dir / "latest.pt", steps_done)

            if not warm and (upd % args.eval_every == 0 or upd == args.ppo_updates):
                ev_res = evaluate_candidate(model, args, f"upd_{upd}", steps_done)
                if ev_res["combined_score"] > best_record["combined_score"]:
                    best_record.update(ev_res)
                    save_ckpt(model, args.save_dir / "best.pt", steps_done, {"eval": ev_res})
                    best_json = args.save_dir / "best.json"
                    export_actor_to_json(
                        model,
                        out_path=best_json,
                        model_name=f"balbesbol-{args.run_name}-{model.hidden_dim}x{model.num_blocks}",
                        trained_steps=steps_done,
                        embed_in_ts=False,
                    )
                    if args.mode == "league" and str(best_json.resolve()) not in league_paths:
                        snap_copy = args.save_dir / f"champ_upd{upd}.json"
                        snap_copy.write_text(best_json.read_text(encoding="utf-8"), encoding="utf-8")
                        league_paths.append(str(snap_copy.resolve()))
                    if args.export_live and ev_res["vs_heuristic"]["mean_diff"] >= args.min_export_diff:
                        export_actor_to_json(
                            model,
                            out_path=DEFAULT_JSON_OUT,
                            model_name=f"balbesbol-v3-{model.hidden_dim}x{model.num_blocks}",
                            trained_steps=steps_done,
                            embed_in_ts=True,
                            ts_out_path=DEFAULT_TS_WEIGHTS_OUT,
                        )
                        save_ckpt(model, ROOT / "training" / "checkpoints" / "balbesbol_trained_latest.pt", steps_done, {"eval": ev_res})
                    print(
                        f"  >>> НОВЫЙ РЕКОРД [{args.run_name}]: vs эвристика {ev_res['vs_heuristic']['mean_diff']:+.2f} "
                        f"({ev_res['vs_heuristic']['win_rate']*100:.1f}% побед), рейтинг={ev_res['combined_score']:+.2f}",
                        flush=True,
                    )
    finally:
        for e in envs:
            e.close()
    return steps_done


# ============================================================================= main
def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    limiter = GPUResourceLimiter(device=device, max_fraction=args.max_gpu_util)
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    args.save_dir.mkdir(parents=True, exist_ok=True)
    critic_spec = (args.critic_hidden, args.critic_blocks, 3)

    if args.init is not None and args.init.exists():
        model = load_policy(args.init, device, upgrade_critic=critic_spec)
        steps_done = 0
        if args.init.suffix == ".pt":
            ck = torch.load(args.init, map_location="cpu", weights_only=False)
            steps_done = int(ck.get("trained_steps", 0))
        print(f"[{args.run_name}] Загружены стартовые веса {args.init} ({model.hidden_dim}x{model.num_blocks})", flush=True)
    else:
        model = BasketballActorCritic(
            obs_dim=OBS_DIM,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            critic_hidden_dim=args.critic_hidden,
            critic_num_blocks=args.critic_blocks,
            critic_context_mult=3,
        ).to(device)
        steps_done = 0
        print(f"[{args.run_name}] Создана новая модель {model.hidden_dim}x{model.num_blocks} (критик {args.critic_hidden}x{args.critic_blocks})", flush=True)

    t_global = time.perf_counter()
    best_record = {"combined_score": -1e9}

    bc_data = None
    need_teacher = args.bc_samples > 0 or (args.ppo_updates > 0 and args.bc_coef_start > 0 and args.bc_anchor_vram > 0)
    if need_teacher:
        shard_dirs = ensure_teacher_dataset(args)
        load_cap = args.bc_samples if args.bc_samples > 0 else args.bc_anchor_vram
        bc_data = load_teacher_to_vram(shard_dirs, device, max_samples=load_cap, max_gpu_util=args.max_gpu_util)

    if args.bc_samples > 0 and bc_data is not None:
        steps_done += run_bc_phase(model, bc_data, args, device, limiter, amp_dtype, use_bf16)
        save_ckpt(model, args.save_dir / "after_bc.pt", steps_done)
        ev = evaluate_candidate(model, args, "after_bc", steps_done)
        best_record.update(ev)
        save_ckpt(model, args.save_dir / "best.pt", steps_done, {"eval": ev})
        export_actor_to_json(model, out_path=args.save_dir / "best.json", embed_in_ts=False)
    elif args.ppo_updates > 0:
        ev = evaluate_candidate(model, args, "init", steps_done)
        best_record.update(ev)
        save_ckpt(model, args.save_dir / "best.pt", steps_done, {"eval": ev})
        export_actor_to_json(model, out_path=args.save_dir / "best.json", embed_in_ts=False)

    if args.ppo_updates > 0:
        bc_data = shrink_bc_vram(bc_data, args.bc_anchor_vram, device)
        steps_done = run_ppo_phase(
            model, bc_data, args, device, limiter, amp_dtype, use_bf16, best_record, steps_done, time.perf_counter()
        )

    print(
        f"\n[{args.run_name} ЗАВЕРШЁН за {(time.perf_counter()-t_global)/60:.1f} мин] "
        f"Лучший рейтинг: {best_record['combined_score']:+.2f} | чекпоинт: {args.save_dir / 'best.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
