#!/usr/bin/env python3
"""
Главный конвейер обучения нейросетевого ИИ «БалбесБола» под NVIDIA RTX 5060 (и с поддержкой CPU-фоллбэка).
Реализует утверждённый Вариант А («Имитация учителя -> 3-ступенчатая Самоигра PPO с Лигой -> Доводка на sim.ts»)
с жёстким ограничением использования видеокарты НЕ БОЛЕЕ 70% (--max-gpu-util 0.70 по умолчанию).
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gpu_limiter import GPUResourceLimiter
from model import (
    CONT_ACTION_DIM,
    DEFAULT_JSON_OUT,
    DEFAULT_TS_WEIGHTS_OUT,
    NUM_DISCRETE_ACTIONS,
    OBS_DIM,
    BasketballActorCritic,
    compute_team_context,
    export_actor_to_json,
)
from tensor_env import TensorBasketballArena
from vec_env import SubprocBBBallVecEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обучение ИИ БалбесБола (лимит GPU <= 70%)")
    p.add_argument("--max-gpu-util", type=float, default=0.70, help="Макс. доля ресурсов GPU/VRAM (0.70 = 70%%)")
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--num-arenas", type=int, default=0, help="0 = авто (1536 на RTX 5060 <=70%%, 64 на CPU)")
    p.add_argument("--bc-iters", type=int, default=180, help="Шагов имитационного разогрева (Вариант А)")
    p.add_argument("--ppo-updates", type=int, default=60, help="Суммарно обновлений PPO по 3 ступеням Curriculum")
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=4096)
    p.add_argument("--sim-finetune-updates", type=int, default=12, help="Доводка паритета на эталонном sim.ts")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--league-prob", type=float, default=0.30)
    p.add_argument("--save-dir", type=Path, default=Path("training/checkpoints"))
    p.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    p.add_argument("--ts-out", type=Path, default=DEFAULT_TS_WEIGHTS_OUT)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def run_ppo_epoch(
    model: BasketballActorCritic,
    optimizer: torch.optim.Optimizer,
    limiter: GPUResourceLimiter,
    b_obs: torch.Tensor,
    b_ctx: torch.Tensor,
    b_masks: torch.Tensor,
    b_disc: torch.Tensor,
    b_move: torch.Tensor,
    b_logp: torch.Tensor,
    b_adv: torch.Tensor,
    b_ret: torch.Tensor,
    b_val: torch.Tensor,
    ppo_epochs: int,
    minibatch_size: int,
    amp_dtype: torch.dtype,
    use_bf16: bool,
    ent_coef: float = 0.01,
) -> tuple[float, float, float]:
    model.train()
    n_samples = int(b_obs.shape[0])
    mb_size = min(n_samples, minibatch_size)
    pg_list: list[float] = []
    vf_list: list[float] = []
    ent_list: list[float] = []

    for _ in range(ppo_epochs):
        perm = torch.randperm(n_samples, device=b_obs.device)
        for start in range(0, n_samples, mb_size):
            limiter.begin_work()
            mb = perm[start : start + mb_size]
            adv = b_adv[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(b_obs.device.type, dtype=amp_dtype, enabled=use_bf16):
                new_logp, entropy, new_val, _ = model.evaluate_actions(
                    b_obs[mb], b_masks[mb], b_disc[mb], b_move[mb], team_context=b_ctx[mb]
                )
                logratio = torch.clamp(new_logp - b_logp[mb], -10.0, 10.0)
                ratio = logratio.exp()
                approx_kl = float(((ratio - 1.0) - logratio).mean().item())

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg1, pg2).mean()

                v_clip = b_val[mb] + torch.clamp(new_val - b_val[mb], -0.2, 0.2)
                vf_loss = 0.5 * torch.max((new_val - b_ret[mb]).pow(2), (v_clip - b_ret[mb]).pow(2)).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + 0.5 * vf_loss - ent_coef * ent_loss

            if approx_kl <= 0.04:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

            pg_list.append(float(pg_loss.item()))
            vf_list.append(float(vf_loss.item()))
            ent_list.append(float(ent_loss.item()))
            limiter.end_work_and_throttle()

            if approx_kl > 0.04:
                return float(np.mean(pg_list)), float(np.mean(vf_list)), float(np.mean(ent_list))

    return float(np.mean(pg_list)), float(np.mean(vf_list)), float(np.mean(ent_list))


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
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    num_arenas = args.num_arenas if args.num_arenas > 0 else (1536 if use_cuda else 64)
    cap_pct = int(round(args.max_gpu_util * 100))

    print(
        f"=== Старт обучения ИИ «БалбесБола» | Устройство: {device} | "
        f"Лимит ресурсов: <= {cap_pct}% | Параллельных арен: {num_arenas} ==="
    )

    model = BasketballActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    total_trained_steps = 0
    t_global = time.perf_counter()

    # =========================================================================
    # ФАЗА 1 (Вариант А): Имитация учителя на реальном sim.ts + тензорной арене
    # =========================================================================
    print("\n[Фаза 1/3] Стартовая имитация учителя (Behavior Cloning на эталонном движке + Tensor Arena)...")
    bc_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Собираем точные траектории эталонного учителя stepPlayerAI из TypeScript + тензорной арены
    arena = TensorBasketballArena(num_envs=num_arenas, device=device, curriculum_stage=1, seed=args.seed)
    obs_t, masks_t = arena.reset_all()

    import os
    from subprocess import run as subproc_run

    project_root = Path(__file__).resolve().parent.parent
    tsx_bin = project_root / "node_modules" / ".bin" / ("tsx.cmd" if os.name == "nt" else "tsx")
    ts_bc_dir = project_root / "training" / "data" / "bc_warm"
    subproc_run(
        [
            str(tsx_bin),
            str(project_root / "scripts" / "rl_datagen.ts"),
            f"--samples={24000 if not use_cuda else 120000}",
            f"--out={ts_bc_dir}",
        ],
        cwd=str(project_root),
        check=True,
    )
    n_ts = int((ts_bc_dir / "disc.u8").stat().st_size)
    ts_obs = torch.from_numpy(np.fromfile(ts_bc_dir / "obs.f32", dtype="<f4").reshape(n_ts, OBS_DIM)).to(device)
    ts_masks = torch.from_numpy(np.fromfile(ts_bc_dir / "masks.u8", dtype=np.uint8).reshape(n_ts, NUM_DISCRETE_ACTIONS)).to(device)
    ts_disc = torch.from_numpy(np.fromfile(ts_bc_dir / "disc.u8", dtype=np.uint8).astype(np.int64)).to(device)
    ts_move = torch.from_numpy(np.fromfile(ts_bc_dir / "move.f32", dtype="<f4").reshape(n_ts, CONT_ACTION_DIM)).to(device)

    counts = torch.bincount(ts_disc, minlength=NUM_DISCRETE_ACTIONS).float()
    cls_weights = torch.clamp(torch.pow(counts.sum() / torch.clamp(counts, min=1.0), 0.28), 0.7, 3.2)
    cls_weights = (cls_weights / cls_weights[0]).to(device)

    model.train()
    for it in range(1, args.bc_iters + 1):
        limiter.begin_work()
        # Половина батча из эталонного sim.ts, половина из живой TensorBasketballArena
        t_disc_a, t_move_a = arena.teacher_act(obs_t, masks_t)
        idx_ts = torch.randint(0, n_ts, (min(2048, n_ts),), device=device)

        mix_obs = torch.cat([obs_t, ts_obs[idx_ts]], dim=0)
        mix_masks = torch.cat([masks_t, ts_masks[idx_ts]], dim=0).bool()
        mix_disc = torch.cat([t_disc_a, ts_disc[idx_ts]], dim=0)
        mix_move = torch.cat([t_move_a, ts_move[idx_ts]], dim=0)
        mix_masks.scatter_(1, mix_disc.unsqueeze(1), True)

        bc_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
            logits, move_pred = model.forward_actor(mix_obs)
            masked_logits = logits.float().masked_fill(~mix_masks, -1e9)
            loss_d = F.cross_entropy(masked_logits, mix_disc, weight=cls_weights)
            loss_m = F.smooth_l1_loss(move_pred.float(), mix_move.float(), beta=0.15)
            loss = loss_d + 2.2 * loss_m

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bc_opt.step()

        obs_t, masks_t, _, _, _ = arena.step(t_disc_a, t_move_a)
        total_trained_steps += int(mix_obs.shape[0])
        limiter.end_work_and_throttle()

        if it % max(1, args.bc_iters // 4) == 0 or it == args.bc_iters:
            acc = float((masked_logits.argmax(dim=-1) == mix_disc).float().mean().item()) * 100.0
            print(
                f"  [BC {it:03d}/{args.bc_iters}] loss_disc={loss_d.item():.4f} "
                f"loss_move={loss_m.item():.4f} acc={acc:.1f}% | "
                f"нагрузка<=70% (EMA={limiter.ema_gpu_util:.1f}%)"
            )

    # =========================================================================
    # ФАЗА 2: 3-ступенчатая самоигра PPO в TensorBasketballArena + Пул Лиги
    # =========================================================================
    print("\n[Фаза 2/3] Самоигра PPO по 3 ступеням Curriculum (Ступень 1 -> Ступень 2 -> Ступень 3)...")
    ppo_opt = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5)
    league_pool: deque[dict[str, torch.Tensor]] = deque(maxlen=10)
    league_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    opp_model = copy.deepcopy(model).to(device)
    opp_model.eval()

    T = args.rollout_steps
    M = arena.total_agents
    obs_dtype = torch.bfloat16 if use_bf16 else torch.float32
    buf_obs = torch.zeros((T, M, OBS_DIM), dtype=obs_dtype, device=device)
    buf_ctx = torch.zeros((T, M, OBS_DIM), dtype=obs_dtype, device=device)
    buf_masks = torch.zeros((T, M, NUM_DISCRETE_ACTIONS), dtype=torch.uint8, device=device)
    buf_disc = torch.zeros((T, M), dtype=torch.long, device=device)
    buf_move = torch.zeros((T, M, CONT_ACTION_DIM), dtype=torch.float32, device=device)
    buf_logp = torch.zeros((T, M), dtype=torch.float32, device=device)
    buf_rew = torch.zeros((T, M), dtype=torch.float32, device=device)
    buf_done = torch.zeros((T, M), dtype=torch.float32, device=device)
    buf_val = torch.zeros((T, M), dtype=torch.float32, device=device)
    buf_learn = torch.ones((T, M), dtype=torch.bool, device=device)

    cur_stage = 1
    cur_obs, cur_masks = arena.reset_all()
    cur_done = torch.zeros((M,), dtype=torch.float32, device=device)

    for upd in range(1, args.ppo_updates + 1):
        target_stage = 1 if upd <= args.ppo_updates // 3 else (2 if upd <= (2 * args.ppo_updates) // 3 else 3)
        if target_stage != cur_stage:
            cur_stage = target_stage
            arena.set_stage(cur_stage)
            cur_obs, cur_masks = arena.get_obs_and_masks()
            cur_done.zero_()
            print(f"  >>> Переход на Ступень {cur_stage}/3 учебной программы!")

        # Лиговый отбор соперников в 30% параллельных арен
        learner_flags = torch.ones((M,), dtype=torch.bool, device=device)
        opp_idx: torch.Tensor | None = None
        if len(league_pool) > 0 and args.league_prob > 0:
            opp_model.load_state_dict(random.choice(list(league_pool)))
            env_hist = torch.rand((num_arenas,), device=device) < args.league_prob
            if env_hist.any():
                grid = torch.ones((num_arenas, 10), dtype=torch.bool, device=device)
                grid[env_hist, 5:10] = False
                learner_flags = grid.reshape(-1)
                opp_idx = torch.nonzero(~learner_flags, as_tuple=False).squeeze(-1)

        model.eval()
        ep_pts_sum = 0.0
        ep_pts_cnt = 0

        for t in range(T):
            limiter.begin_work()
            ctx = compute_team_context(cur_obs)
            buf_obs[t] = cur_obs.to(obs_dtype)
            buf_ctx[t] = ctx.to(obs_dtype)
            buf_masks[t] = cur_masks
            buf_done[t] = cur_done
            buf_learn[t] = learner_flags

            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_bf16):
                da, rma, cma, lpa, va = model.act(cur_obs, cur_masks, team_context=ctx, deterministic=False)
                if opp_idx is not None and opp_idx.numel() > 0:
                    oda, orma, ocma, _, _ = opp_model.act(cur_obs[opp_idx], cur_masks[opp_idx], deterministic=False)
                    da[opp_idx] = oda
                    rma[opp_idx] = orma
                    cma[opp_idx] = ocma

            buf_disc[t] = da
            buf_move[t] = rma
            buf_logp[t] = lpa
            buf_val[t] = va

            cur_obs, cur_masks, rew_t, done_u8, info = arena.step(da, cma)
            buf_rew[t] = rew_t
            cur_done = done_u8.float()
            if info["done_count"] > 0:
                ep_pts_sum += info["avg_total_points"]
                ep_pts_cnt += 1
            limiter.end_work_and_throttle()

        total_trained_steps += T * M

        # GAE (gamma=0.995, lambda=0.95)
        limiter.begin_work()
        with torch.no_grad():
            next_val = model.forward_critic(cur_obs, compute_team_context(cur_obs)).float()
            adv = torch.zeros_like(buf_rew)
            gae = torch.zeros((M,), dtype=torch.float32, device=device)
            for t in reversed(range(T)):
                non_term = 1.0 - (cur_done if t == T - 1 else buf_done[t + 1])
                nv = next_val if t == T - 1 else buf_val[t + 1]
                delta = buf_rew[t] + 0.995 * nv * non_term - buf_val[t]
                gae = delta + 0.995 * 0.95 * non_term * gae
                adv[t] = gae
            ret = adv + buf_val
        limiter.end_work_and_throttle()

        v_idx = torch.nonzero(buf_learn.reshape(-1), as_tuple=False).squeeze(-1)
        pg_l, vf_l, ent_l = run_ppo_epoch(
            model,
            ppo_opt,
            limiter,
            buf_obs.reshape(-1, OBS_DIM)[v_idx].float(),
            buf_ctx.reshape(-1, OBS_DIM)[v_idx].float(),
            buf_masks.reshape(-1, NUM_DISCRETE_ACTIONS)[v_idx],
            buf_disc.reshape(-1)[v_idx],
            buf_move.reshape(-1, CONT_ACTION_DIM)[v_idx],
            buf_logp.reshape(-1)[v_idx],
            adv.reshape(-1)[v_idx],
            ret.reshape(-1)[v_idx],
            buf_val.reshape(-1)[v_idx],
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            amp_dtype=amp_dtype,
            use_bf16=use_bf16,
            ent_coef=0.010 if cur_stage < 3 else 0.006,
        )

        if upd % 12 == 0:
            league_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        if upd % max(1, args.ppo_updates // 6) == 0 or upd == args.ppo_updates:
            avg_r = float(buf_rew.mean().item())
            pts_str = f"{ep_pts_sum / ep_pts_cnt:.1f}" if ep_pts_cnt > 0 else "n/a"
            alloc_mb, cap_mb = limiter.vram_status_mb()
            vram_info = f"VRAM {alloc_mb:.0f}/{cap_mb:.0f} MiB" if use_cuda else "CPU<=70%"
            print(
                f"  [TensorPPO {upd:03d}/{args.ppo_updates} | Ст.{cur_stage}] "
                f"steps={total_trained_steps:,} rew={avg_r:+.4f} pts={pts_str} ent={ent_l:.2f} | "
                f"{vram_info} util={limiter.ema_gpu_util:.1f}%"
            )

    # =========================================================================
    # ФАЗА 3: Финальная доводка паритета на эталонном движке src/game/sim.ts
    # =========================================================================
    if args.sim_finetune_updates > 0:
        print(
            f"\n[Фаза 3/3] Доводка физического паритета ({args.sim_finetune_updates} обновлений) "
            "напрямую в эталонном движке src/game/sim.ts..."
        )
        ft_opt = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.35, eps=1e-5)
        n_workers = 2 if not use_cuda else 4
        envs_per_w = 4 if not use_cuda else 8
        with SubprocBBBallVecEnv(
            num_workers=n_workers,
            envs_per_worker=envs_per_w,
            controlled_teams="both",
            frame_skip=4,
            quarter_duration=45,
            randomize_rosters=True,
            randomize_tactics=True,
            seed=args.seed + 99,
        ) as sim_env:
            FT_T = 48
            FT_M = sim_env.total_agents
            s_obs = torch.zeros((FT_T, FT_M, OBS_DIM), dtype=torch.float32, device=device)
            s_ctx = torch.zeros((FT_T, FT_M, OBS_DIM), dtype=torch.float32, device=device)
            s_masks = torch.zeros((FT_T, FT_M, NUM_DISCRETE_ACTIONS), dtype=torch.uint8, device=device)
            s_disc = torch.zeros((FT_T, FT_M), dtype=torch.long, device=device)
            s_move = torch.zeros((FT_T, FT_M, CONT_ACTION_DIM), dtype=torch.float32, device=device)
            s_logp = torch.zeros((FT_T, FT_M), dtype=torch.float32, device=device)
            s_rew = torch.zeros((FT_T, FT_M), dtype=torch.float32, device=device)
            s_done = torch.zeros((FT_T, FT_M), dtype=torch.float32, device=device)
            s_val = torch.zeros((FT_T, FT_M), dtype=torch.float32, device=device)

            o_np, m_np = sim_env.reset(seed=args.seed + 99)
            co = torch.from_numpy(o_np).to(device)
            cm = torch.from_numpy(m_np).to(device)
            cd = torch.zeros((FT_M,), dtype=torch.float32, device=device)

            for ft_u in range(1, args.sim_finetune_updates + 1):
                model.eval()
                for t in range(FT_T):
                    limiter.begin_work()
                    ctx = compute_team_context(co)
                    s_obs[t], s_ctx[t], s_masks[t], s_done[t] = co, ctx, cm, cd
                    da, rma, cma, lpa, va = model.act(co, cm, team_context=ctx, deterministic=False)
                    s_disc[t], s_move[t], s_logp[t], s_val[t] = da, rma, lpa, va
                    limiter.end_work_and_throttle()

                    no_np, nm_np, r_np, d_np, _ = sim_env.step(da.cpu().numpy(), cma.cpu().numpy())
                    s_rew[t] = torch.from_numpy(r_np).to(device)
                    co = torch.from_numpy(no_np).to(device)
                    cm = torch.from_numpy(nm_np).to(device)
                    cd = torch.from_numpy(d_np.astype(np.float32)).to(device)

                total_trained_steps += FT_T * FT_M
                with torch.no_grad():
                    nv_last = model.forward_critic(co, compute_team_context(co)).float()
                    adv_s = torch.zeros_like(s_rew)
                    gae_s = torch.zeros((FT_M,), dtype=torch.float32, device=device)
                    for t in reversed(range(FT_T)):
                        nt = 1.0 - (cd if t == FT_T - 1 else s_done[t + 1])
                        nv = nv_last if t == FT_T - 1 else s_val[t + 1]
                        dl = s_rew[t] + 0.995 * nv * nt - s_val[t]
                        gae_s = dl + 0.995 * 0.95 * nt * gae_s
                        adv_s[t] = gae_s
                    ret_s = adv_s + s_val

                run_ppo_epoch(
                    model,
                    ft_opt,
                    limiter,
                    s_obs.reshape(-1, OBS_DIM),
                    s_ctx.reshape(-1, OBS_DIM),
                    s_masks.reshape(-1, NUM_DISCRETE_ACTIONS),
                    s_disc.reshape(-1),
                    s_move.reshape(-1, CONT_ACTION_DIM),
                    s_logp.reshape(-1),
                    adv_s.reshape(-1),
                    ret_s.reshape(-1),
                    s_val.reshape(-1),
                    ppo_epochs=2,
                    minibatch_size=min(2048, FT_T * FT_M),
                    amp_dtype=amp_dtype,
                    use_bf16=use_bf16,
                    ent_coef=0.005,
                )
                if ft_u % max(1, args.sim_finetune_updates // 3) == 0 or ft_u == args.sim_finetune_updates:
                    print(
                        f"  [SimFineTune {ft_u:02d}/{args.sim_finetune_updates}] "
                        f"steps={total_trained_steps:,} rew_step={s_rew.mean().item():+.4f}"
                    )

    # Сохраняем чекпоинт PyTorch + экспортируем тихие веса в веб-движок (JSON + defaultTrainedWeights.ts)
    ckpt_path = args.save_dir / "balbesbol_trained_latest.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "trained_steps": total_trained_steps,
        },
        ckpt_path,
    )
    export_actor_to_json(
        model,
        out_path=args.json_out,
        model_name=f"balbesbol-v1-{args.hidden_dim}x{args.num_blocks}",
        trained_steps=total_trained_steps,
        embed_in_ts=True,
        ts_out_path=args.ts_out,
    )

    elapsed = time.perf_counter() - t_global
    print(
        f"\n[ГОТОВО за {elapsed:.1f}с] Суммарно обучено: {total_trained_steps:,} агент-шагов.\n"
        f"  -> PyTorch чекпоинт: {ckpt_path}\n"
        f"  -> Встроенные веса движка: {args.ts_out}\n"
        f"  -> JSON слепок: {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
