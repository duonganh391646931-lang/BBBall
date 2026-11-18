#!/usr/bin/env python3
"""
Улучшенный конвейер обучения ИИ «БалбесБола» (v2) — всё обучение на ЭТАЛОННОМ движке src/game/sim.ts.

Почему v2 (по результатам замеров evaluate.py: прежние веса проигрывали встроенному ИИ ~25-45 очков):
  * Фаза A: крупное имитационное обучение (BC) на миллионах примеров учителя из sim.ts (параллельный datagen),
    без перевзвешивания классов (честная MLE-копия учителя), датасет целиком в VRAM (bf16).
  * Фаза B: PPO прямо в sim.ts против встроенного эвристического ИИ (за обе стороны площадки):
      - разогрев критика (политика заморожена), чтобы шумные преимущества не разрушили BC-политику;
      - BC-регуляризация с затуханием (кросс-энтропия к действиям учителя);
      - нормализация наград по скользящему std дисконтированного возврата;
      - контроль KL (ранняя остановка эпох), линейное затухание LR, логирование clipfrac / explained variance;
      - регулярная оценка (evaluate.py) и сохранение/экспорт ТОЛЬКО лучшей модели по разнице очков.

Пример:
  python training/train_sim.py --init training/baselines/run1_21M.pt --bc-samples 2000000 --ppo-updates 300
"""

from __future__ import annotations

import argparse
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
    compute_team_context,
    export_actor_to_json,
)
from vec_env import SubprocBBBallVecEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="БалбесБол v2: BC + PPO на эталонном sim.ts")
    p.add_argument("--init", type=Path, default=None, help=".pt/.json для старта (иначе с нуля)")
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--max-gpu-util", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=2027)
    # Фаза A
    p.add_argument("--bc-samples", type=int, default=2_000_000, help="0 = пропустить BC")
    p.add_argument("--bc-epochs", type=float, default=12.0)
    p.add_argument("--bc-batch", type=int, default=4096)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--datagen-procs", type=int, default=8)
    p.add_argument("--reuse-data", action="store_true", help="Не генерировать заново, если датасет уже есть")
    # Фаза B
    p.add_argument("--ppo-updates", type=int, default=300, help="0 = пропустить PPO")
    p.add_argument("--workers-per-side", type=int, default=4)
    p.add_argument("--envs-per-worker", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.15)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--critic-warmup", type=int, default=8, help="Обновлений только критика в начале PPO")
    p.add_argument("--move-log-std", type=float, default=None,
                   help="Задать log σ шума движения в начале PPO (в игре движение детерминировано; -1.6 ≈ σ 0.2)")
    p.add_argument("--bc-coef-start", type=float, default=0.5)
    p.add_argument("--bc-coef-end", type=float, default=0.02)
    p.add_argument("--eval-every", type=int, default=15)
    p.add_argument("--eval-games", type=int, default=32)
    p.add_argument("--quarter", type=int, default=45)
    # Выход
    p.add_argument("--save-dir", type=Path, default=ROOT / "training" / "checkpoints_v2")
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    p.add_argument("--ts-out", type=Path, default=DEFAULT_TS_WEIGHTS_OUT)
    p.add_argument("--no-export", action="store_true", help="Не перезаписывать веса игры")
    return p.parse_args()


# ============================================================================= утилиты
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


def save_ckpt(model: BasketballActorCritic, path: Path, args: argparse.Namespace, steps: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": model.hidden_dim,
            "num_blocks": model.num_blocks,
            "trained_steps": steps,
            **(extra or {}),
        },
        path,
    )


def run_eval(model: BasketballActorCritic, args: argparse.Namespace, device: torch.device, tag: str) -> dict:
    """Оценка как в игре (контроллер 'neural' sim.ts против встроенного ИИ) на фиксированных сидах."""
    t0 = time.perf_counter()
    model.eval()
    tmp = args.save_dir / "_eval_tmp.json"
    export_actor_to_json(model, out_path=tmp, embed_in_ts=False)
    res = game_eval(tmp, games=args.eval_games, procs=8, seed=4242, quarter=args.quarter)
    print(
        f"  [EVAL {tag}] vs встроенный ИИ: побед={res['win_rate']*100:.1f}% "
        f"разница={res['mean_diff']:+.2f}±{res['ci95']:.2f} ({time.perf_counter()-t0:.0f}с)",
        flush=True,
    )
    return res


# ============================================================================= Фаза A: данные учителя
def generate_bc_data(args: argparse.Namespace) -> list[Path]:
    data_root = ROOT / "training" / "data" / "bc_v2"
    per_proc = int(math.ceil(args.bc_samples / args.datagen_procs))
    dirs = [data_root / f"shard{i}" for i in range(args.datagen_procs)]
    if args.reuse_data and all((d / "meta.json").exists() for d in dirs):
        sizes = [int((d / "disc.u8").stat().st_size) for d in dirs]
        if min(sizes) >= per_proc:
            print(f"[A] Повторно используем датасет: {sum(sizes):,} примеров")
            return dirs
    tsx = ROOT / "node_modules" / ".bin" / ("tsx.cmd" if os.name == "nt" else "tsx")
    t0 = time.perf_counter()
    procs = []
    for i, d in enumerate(dirs):
        log = open(data_root.parent / f"bc_v2_gen{i}.log", "w", encoding="utf-8") if data_root.parent.exists() else subprocess.DEVNULL
        procs.append(
            subprocess.Popen(
                [str(tsx), str(ROOT / "scripts" / "rl_datagen.ts"), f"--samples={per_proc}", f"--out={d}",
                 f"--seed={args.seed * 1000 + i * 7919}", f"--quarterDuration={args.quarter}"],
                cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            )
        )
    for pr in procs:
        if pr.wait() != 0:
            raise RuntimeError("rl_datagen.ts завершился с ошибкой")
    print(f"[A] Сгенерировано {per_proc * len(dirs):,} примеров учителя за {time.perf_counter()-t0:.0f}с ({len(dirs)} процессов)")
    return dirs


def load_bc_to_gpu(dirs: list[Path], device: torch.device) -> dict[str, torch.Tensor]:
    """Шардами грузим в VRAM (obs в bf16), чтобы в ОЗУ одновременно жил максимум один шард."""
    obs_l, mask_l, disc_l, move_l = [], [], [], []
    for d in dirs:
        n = int((d / "disc.u8").stat().st_size)
        obs_l.append(torch.from_numpy(np.fromfile(d / "obs.f32", dtype="<f4").reshape(n, OBS_DIM)).to(device).to(torch.bfloat16))
        mask_l.append(torch.from_numpy(np.fromfile(d / "masks.u8", dtype=np.uint8).reshape(n, NUM_DISCRETE_ACTIONS)).to(device).bool())
        disc_l.append(torch.from_numpy(np.fromfile(d / "disc.u8", dtype=np.uint8)).to(device).long())
        move_l.append(torch.from_numpy(np.fromfile(d / "move.f32", dtype="<f4").reshape(n, CONT_ACTION_DIM)).to(device))
    data = {"obs": torch.cat(obs_l), "masks": torch.cat(mask_l), "disc": torch.cat(disc_l), "move": torch.cat(move_l)}
    data["masks"].scatter_(1, data["disc"].unsqueeze(1), True)  # действие учителя всегда допустимо
    return data


def bc_loss(model: BasketballActorCritic, obs: torch.Tensor, masks: torch.Tensor, disc: torch.Tensor, move: torch.Tensor):
    logits, move_pred = model.forward_actor(obs)
    ml = logits.float().masked_fill(~masks, -1e9)
    l_d = F.cross_entropy(ml, disc)
    l_m = F.smooth_l1_loss(move_pred.float(), move, beta=0.15)
    return l_d, l_m, ml


def phase_bc(model, data, args, device, limiter, amp_dtype, use_bf16) -> None:
    n = data["disc"].numel()
    n_val = min(50_000, n // 50)
    perm = torch.randperm(n, device=device)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    steps = int(args.bc_epochs * tr_idx.numel() / args.bc_batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.bc_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.bc_lr, total_steps=steps, pct_start=0.05)
    print(f"[A] BC: {tr_idx.numel():,} train / {n_val:,} val, {steps} шагов по {args.bc_batch}")
    t0 = time.perf_counter()
    model.train()
    for it in range(1, steps + 1):
        limiter.begin_work()
        b = tr_idx[torch.randint(0, tr_idx.numel(), (args.bc_batch,), device=device)]
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
            l_d, l_m, _ = bc_loss(model, data["obs"][b], data["masks"][b], data["disc"][b], data["move"][b])
            loss = l_d + 2.2 * l_m
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
                    v = val_idx[s:s + 8192]
                    vd, vm, ml = bc_loss(model, data["obs"][v], data["masks"][v], data["disc"][v], data["move"][v])
                    pred = ml.argmax(-1)
                    accs.append((pred == data["disc"][v]).float().mean().item())
                    nz = data["disc"][v] != 0
                    if nz.any():
                        nonnoop.append((pred[nz] == data["disc"][v][nz]).float().mean().item())
                    lm.append(vm.item())
            model.train()
            print(
                f"  [BC {it}/{steps}] train_ce={l_d.item():.4f} | val acc={np.mean(accs)*100:.2f}% "
                f"acc(не NOOP)={np.mean(nonnoop)*100:.1f}% move_l1={np.mean(lm):.4f} ({time.perf_counter()-t0:.0f}с)",
                flush=True,
            )


# ============================================================================= Фаза B: PPO в sim.ts
def phase_ppo(model, bc_data, args, device, limiter, amp_dtype, use_bf16, best: dict, steps_done: int) -> int:
    W, E, T = args.workers_per_side, args.envs_per_worker, args.rollout_steps
    envs = [
        SubprocBBBallVecEnv(num_workers=W, envs_per_worker=E, controlled_teams=side, frame_skip=4,
                            quarter_duration=args.quarter, randomize_rosters=True, randomize_tactics=True,
                            seed=args.seed + 1000 * (side + 1))
        for side in (0, 1)
    ]
    try:
        A = sum(e.total_agents for e in envs)
        print(f"[B] PPO в sim.ts vs встроенный ИИ: {2*W*E} матчей параллельно, {A} агентов, rollout {T} -> {T*A:,} сэмплов/обновление")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5, weight_decay=0.0)
        if args.move_log_std is not None:
            with torch.no_grad():
                model.move_log_std.fill_(args.move_log_std)
            print(f"[B] log σ движения = {args.move_log_std} (σ={math.exp(args.move_log_std):.2f})")
        ret_rms = RunningMeanStd(device)
        disc_ret = torch.zeros(A, device=device, dtype=torch.float64)

        def reset_all():
            o, m = zip(*[e.reset(seed=args.seed + 17 * i) for i, e in enumerate(envs)])
            return torch.from_numpy(np.concatenate(o)).to(device), torch.from_numpy(np.concatenate(m)).to(device)

        def step_all(disc: torch.Tensor, move: torch.Tensor):
            d_np = disc.to(torch.uint8).cpu().numpy()
            m_np = move.float().cpu().numpy()
            outs, s = [], 0
            for e in envs:
                outs.append(e.step(d_np[s:s + e.total_agents], m_np[s:s + e.total_agents]))
                s += e.total_agents
            o = np.concatenate([x[0] for x in outs]); m = np.concatenate([x[1] for x in outs])
            r = np.concatenate([x[2] for x in outs]); d = np.concatenate([x[3] for x in outs])
            eps = []
            for side, x in enumerate(outs):
                for ep in x[4]:
                    sc = ep["score"]
                    eps.append(float(sc[side] - sc[1 - side]))
            return (torch.from_numpy(o).to(device), torch.from_numpy(m).to(device),
                    torch.from_numpy(r).to(device), torch.from_numpy(d.astype(np.float32)).to(device), eps)

        buf = {
            "obs": torch.zeros((T, A, OBS_DIM), device=device),
            "masks": torch.zeros((T, A, NUM_DISCRETE_ACTIONS), dtype=torch.bool, device=device),
            "disc": torch.zeros((T, A), dtype=torch.long, device=device),
            "move": torch.zeros((T, A, CONT_ACTION_DIM), device=device),
            "logp": torch.zeros((T, A), device=device),
            "val": torch.zeros((T, A), device=device),
            "rew": torch.zeros((T, A), device=device),
            "done": torch.zeros((T, A), device=device),
        }
        obs, masks = reset_all()
        recent_eps: list[float] = []
        t_start = time.perf_counter()
        n_bc = bc_data["disc"].numel() if bc_data is not None else 0

        for upd in range(1, args.ppo_updates + 1):
            frac = (upd - 1) / max(1, args.ppo_updates - 1)
            lr_now = args.lr * (1.0 - 0.9 * frac)
            for g in opt.param_groups:
                g["lr"] = lr_now
            bc_coef = args.bc_coef_start + (args.bc_coef_end - args.bc_coef_start) * frac
            warm = upd <= args.critic_warmup

            # ---------------- сбор траекторий
            model.eval()
            t_roll = time.perf_counter()
            for t in range(T):
                with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                    da, rma, cma, lpa, va = model.act(obs, masks, team_context=compute_team_context(obs))
                buf["obs"][t] = obs; buf["masks"][t] = masks.bool(); buf["disc"][t] = da
                buf["move"][t] = rma; buf["logp"][t] = lpa; buf["val"][t] = va
                obs, masks, rew, done, eps = step_all(da, cma)
                buf["rew"][t] = rew; buf["done"][t] = done
                recent_eps.extend(eps)
                # нормализация наград по std дисконтированного возврата
                disc_ret = disc_ret * args.gamma * (1.0 - done.double()) + rew.double()
                ret_rms.update(disc_ret)
            t_roll = time.perf_counter() - t_roll
            steps_done += T * A

            # ---------------- GAE (в нормированных единицах награды)
            t_upd = time.perf_counter()
            r_scale = ret_rms.std
            with torch.no_grad(), torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                next_v = model.forward_critic(obs, compute_team_context(obs)).float()
            adv = torch.zeros((T, A), device=device)
            gae = torch.zeros(A, device=device)
            rew_n = buf["rew"] / r_scale
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else buf["val"][t + 1]
                nonterm = 1.0 - buf["done"][t]
                delta = rew_n[t] + args.gamma * nv * nonterm - buf["val"][t]
                gae = delta + args.gamma * args.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = adv + buf["val"]

            # ---------------- оптимизация
            N = T * A
            f_obs = buf["obs"].reshape(N, OBS_DIM)
            f_ctx = compute_team_context(buf["obs"]).reshape(N, OBS_DIM)
            f_masks = buf["masks"].reshape(N, -1); f_disc = buf["disc"].reshape(N)
            f_move = buf["move"].reshape(N, -1); f_logp = buf["logp"].reshape(N)
            f_adv = adv.reshape(N); f_ret = ret.reshape(N); f_val = buf["val"].reshape(N)
            stats = {"pg": [], "vf": [], "ent": [], "kl": [], "clip": [], "bc": []}
            stop = False
            model.train()
            for ep_i in range(args.ppo_epochs):
                perm = torch.randperm(N, device=device)
                for s in range(0, N, args.minibatch_size):
                    limiter.begin_work()
                    mb = perm[s:s + args.minibatch_size]
                    a_mb = f_adv[mb]
                    a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                        new_logp, ent, new_v, _ = model.evaluate_actions(f_obs[mb], f_masks[mb], f_disc[mb], f_move[mb], team_context=f_ctx[mb])
                        logratio = torch.clamp(new_logp - f_logp[mb], -20, 20)
                        ratio = logratio.exp()
                        with torch.no_grad():
                            kl = ((ratio - 1) - logratio).mean()
                            clipfrac = ((ratio - 1).abs() > args.clip).float().mean()
                        pg = torch.max(-a_mb * ratio, -a_mb * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)).mean()
                        v_clip = f_val[mb] + torch.clamp(new_v - f_val[mb], -0.2, 0.2)
                        vf = 0.5 * torch.max((new_v - f_ret[mb]).pow(2), (v_clip - f_ret[mb]).pow(2)).mean()
                        if warm:
                            loss = args.vf_coef * vf
                        else:
                            loss = pg + args.vf_coef * vf - args.ent_coef * ent.mean()
                        if bc_coef > 0 and n_bc > 0 and not warm:
                            bi = torch.randint(0, n_bc, (min(4096, args.minibatch_size // 2),), device=device)
                            l_d, l_m, _ = bc_loss(model, bc_data["obs"][bi], bc_data["masks"][bi], bc_data["disc"][bi], bc_data["move"][bi])
                            l_bc = l_d + 2.2 * l_m
                            loss = loss + bc_coef * l_bc
                            stats["bc"].append(float(l_bc.item()))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if warm:  # во время разогрева двигаем только критика
                        for n_, p_ in model.named_parameters():
                            if not n_.startswith(("critic", "value_head")) and p_.grad is not None:
                                p_.grad = None
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    limiter.end_work_and_throttle()
                    stats["pg"].append(float(pg.item())); stats["vf"].append(float(vf.item()))
                    stats["ent"].append(float(ent.mean().item())); stats["kl"].append(float(kl.item()))
                    stats["clip"].append(float(clipfrac.item()))
                    if not warm and kl.item() > 1.5 * args.target_kl:
                        stop = True
                        break
                if stop:
                    break
            with torch.no_grad():
                ev = 1.0 - float(torch.var(f_ret - f_val) / (torch.var(f_ret) + 1e-8))
            t_upd = time.perf_counter() - t_upd

            if upd % 5 == 0 or upd == 1 or warm:
                last = recent_eps[-64:]
                ep_str = f"{np.mean(last):+.1f} ({len(recent_eps)} игр)" if last else "n/a"
                sps = T * A / max(1e-6, t_roll + t_upd)
                print(
                    f"  [PPO {upd}/{args.ppo_updates}{' warmup' if warm else ''}] разн.очков(посл.64)={ep_str} "
                    f"rew={buf['rew'].mean().item():+.4f} EV={ev:.2f} KL={np.mean(stats['kl']):.4f} "
                    f"clip={np.mean(stats['clip']):.3f} ent={np.mean(stats['ent']):.2f} "
                    f"bc={np.mean(stats['bc']) if stats['bc'] else 0:.3f} epochs={ep_i+1} lr={lr_now:.1e} | "
                    f"{sps:,.0f} агент-шаг/с (сбор {t_roll:.1f}с, обуч {t_upd:.1f}с)",
                    flush=True,
                )
            save_ckpt(model, args.save_dir / "latest.pt", args, steps_done)

            if upd % args.eval_every == 0 or upd == args.ppo_updates:
                res = run_eval(model, args, device, f"upd {upd}")
                if res["mean_diff"] > best["mean_diff"]:
                    best.update(res)
                    save_ckpt(model, args.save_dir / "best.pt", args, steps_done, {"eval": res})
                    print(f"  >>> Новая лучшая модель: {res['mean_diff']:+.2f} (сохранена best.pt)", flush=True)
        print(f"[B] PPO завершён за {time.perf_counter()-t_start:.0f}с")
    finally:
        for e in envs:
            e.close()
    return steps_done


# ============================================================================= main
def main() -> int:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    limiter = GPUResourceLimiter(device=device, max_fraction=args.max_gpu_util)
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.init:
        model = load_policy(args.init, device)
        print(f"Старт с весов: {args.init}")
    else:
        model = BasketballActorCritic(obs_dim=OBS_DIM, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks).to(device)
    steps_done = 0
    t0 = time.perf_counter()

    best = {"mean_diff": -1e9}
    bc_data = None
    if args.bc_samples > 0:
        dirs = generate_bc_data(args)
        bc_data = load_bc_to_gpu(dirs, device)
        phase_bc(model, bc_data, args, device, limiter, amp_dtype, use_bf16)
        steps_done += int(args.bc_epochs * bc_data["disc"].numel())
        save_ckpt(model, args.save_dir / "after_bc.pt", args, steps_done)
        res = run_eval(model, args, device, "после BC")
        best.update(res)
        save_ckpt(model, args.save_dir / "best.pt", args, steps_done, {"eval": res})
    else:
        res = run_eval(model, args, device, "старт")
        best.update(res)
        save_ckpt(model, args.save_dir / "best.pt", args, steps_done, {"eval": res})
        if args.bc_coef_start > 0:
            dirs = generate_bc_data(args) if not args.reuse_data else generate_bc_data(args)
            bc_data = load_bc_to_gpu(dirs, device)

    if args.ppo_updates > 0:
        steps_done = phase_ppo(model, bc_data, args, device, limiter, amp_dtype, use_bf16, best, steps_done)

    best_model = load_policy(args.save_dir / "best.pt", device)
    print(f"\n[ИТОГ] Лучшая модель: разница {best['mean_diff']:+.2f}, побед {best.get('win_rate', 0)*100:.1f}% "
          f"({time.perf_counter()-t0:.0f}с всего)")
    if not args.no_export:
        export_actor_to_json(best_model, out_path=args.json_out, model_name=f"balbesbol-v2-{best_model.hidden_dim}x{best_model.num_blocks}",
                             trained_steps=steps_done, embed_in_ts=True, ts_out_path=args.ts_out)
        print(f"  -> экспортировано в {args.json_out} и {args.ts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
