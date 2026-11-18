#!/usr/bin/env python3
"""
Этапы 2–3 плана обучения: Многоагентный PPO (CTDE MAPPO) + League Self-Play + KL-якорь учителя.
Спроектировано под бюджет видеопамяти и Tensor Cores видеокарты NVIDIA GeForce RTX 5060 (8 ГБ GDDR7).

Примеры запуска:
  # Этап 2: Дообучение против эвристики с мягким KL-якорем к BC-политике
  python training/train_ppo.py --mode vs_heuristic --init-checkpoint training/checkpoints/bc_init.pt \
      --bc-anchor training/checkpoints/bc_init.pt --total-steps 15000000

  # Этап 3: Лиговый Self-Play с пулом исторических соперников и рандомизацией модификаторов
  python training/train_ppo.py --mode selfplay --init-checkpoint training/checkpoints/ppo_stage2.pt \
      --league-prob 0.35 --chaos-curriculum --total-steps 50000000
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
from torch.utils.tensorboard import SummaryWriter

from gpu_limiter import GPUResourceLimiter
from model import (
    CONT_ACTION_DIM,
    NUM_DISCRETE_ACTIONS,
    OBS_DIM,
    BasketballActorCritic,
    compute_team_context,
    export_actor_to_json,
)
from vec_env import SubprocBBBallVecEnv

CHAOS_MODIFIER_POOL = [
    "two_balls",
    "ice_court",
    "low_gravity",
    "paint_trampolines",
    "explosive_ball",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTDE MAPPO + League Self-Play Trainer (RTX 5060, <=70% GPU)")
    p.add_argument("--mode", choices=["vs_heuristic", "selfplay"], default="selfplay")
    p.add_argument("--init-checkpoint", type=Path, default=Path("training/checkpoints/bc_init.pt"))
    p.add_argument("--bc-anchor", type=Path, default=None, help="Чекпоинт BC для KL-регуляризации")
    p.add_argument("--save-dir", type=Path, default=Path("training/checkpoints"))
    p.add_argument("--json-out", type=Path, default=Path("public/models/policy_latest.json"))
    p.add_argument("--log-dir", type=Path, default=Path("training/runs"))
    p.add_argument("--run-name", type=str, default="rtx5060_mappo")
    p.add_argument("--max-gpu-util", type=float, default=0.70, help="Верхний лимит ресурсов GPU (по умолчанию 0.70 = 70%)")

    # Параллелизм симуляции и размерности сети
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--envs-per-worker", type=int, default=16)
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--quarter-duration", type=int, default=45)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-blocks", type=int, default=2)

    # Гиперпараметры MAPPO под RTX 5060 (8 ГБ VRAM, бюджет <= 70%)
    p.add_argument("--total-steps", type=int, default=30_000_000)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.012)
    p.add_argument("--kl-coef", type=float, default=0.04, help="Начальный вес KL-якоря к BC")
    p.add_argument("--target-kl", type=float, default=0.025)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # League Self-Play и Curriculum
    p.add_argument("--league-prob", type=float, default=0.35, help="Доля матчей против пула прошлых версий")
    p.add_argument("--snapshot-interval", type=int, default=40, help="Каждые N обновлений пополнять пул Лиги")
    p.add_argument("--export-interval", type=int, default=20, help="Каждые N обновлений обновлять JSON")
    p.add_argument("--chaos-curriculum", action="store_true", help="Подмешивать аркадные модификаторы")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_weights_if_exists(model: BasketballActorCritic, ckpt_path: Path | None, device: torch.device) -> int:
    if ckpt_path is None or not ckpt_path.exists():
        return 0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[PPO] Частичная загрузка весов (missing={len(missing)}, unexpected={len(unexpected)})")
    steps = int(ckpt.get("trained_steps", 0))
    print(f"[PPO] Загружен чекпоинт {ckpt_path} (шагов ранее: {steps:,})")
    return steps


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    limiter = GPUResourceLimiter(device, max_fraction=args.max_gpu_util)
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if use_cuda:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"[PPO] GPU активирован (лимит <= {args.max_gpu_util*100:.0f}%): {torch.cuda.get_device_name(device)} | AMP={amp_dtype}")
    else:
        print(f"[PPO] Запуск на CPU (лимит потоков <= {args.max_gpu_util*100:.0f}%): {limiter.status_summary()}")

    controlled_teams: str | int = "both" if args.mode == "selfplay" else 0
    vec_env = SubprocBBBallVecEnv(
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        controlled_teams=controlled_teams,
        frame_skip=args.frame_skip,
        quarter_duration=args.quarter_duration,
        randomize_rosters=True,
        randomize_tactics=True,
        seed=args.seed,
    )

    total_agents = vec_env.total_agents
    num_envs = vec_env.num_envs
    agents_per_env = vec_env.agents_per_env
    T = args.rollout_steps
    batch_transitions = T * total_agents

    print(
        f"[PPO] Среда запущена: {num_envs} паралл. матчей ({args.num_workers}x{args.envs_per_worker}), "
        f"{total_agents} агентов/шаг, батч роллаута = {batch_transitions:,} переходов"
    )

    model = BasketballActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)
    prior_steps = load_weights_if_exists(model, args.init_checkpoint, device)

    # Опциональный замороженный якорь BC-политики для предотвращения распада командного баскетбола
    anchor_model: BasketballActorCritic | None = None
    if args.bc_anchor is not None and args.bc_anchor.exists():
        anchor_model = BasketballActorCritic(
            obs_dim=OBS_DIM,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
        ).to(device)
        load_weights_if_exists(anchor_model, args.bc_anchor, device)
        anchor_model.eval()
        for p in anchor_model.parameters():
            p.requires_grad_(False)

    # Пул исторических чекпоинтов для League Self-Play (храним state_dict в ОЗУ, один слот на GPU)
    league_pool: deque[dict[str, torch.Tensor]] = deque(maxlen=12)
    opponent_model: BasketballActorCritic | None = None
    if args.mode == "selfplay":
        league_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        opponent_model = copy.deepcopy(model).to(device)
        opponent_model.eval()

    train_net: torch.nn.Module = model
    if use_cuda and not args.no_compile and hasattr(torch, "compile"):
        try:
            train_net = torch.compile(model, mode="default")
        except Exception as exc:
            print(f"[PPO] torch.compile пропущен: {exc}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5, fused=use_cuda)

    # Буферы роллаута прямо на целевом устройстве (~160 МБ в VRAM при T=128, M=1280)
    buf_obs = torch.zeros((T, total_agents, OBS_DIM), dtype=torch.float32, device=device)
    buf_ctx = torch.zeros((T, total_agents, OBS_DIM), dtype=torch.float32, device=device)
    buf_masks = torch.zeros((T, total_agents, NUM_DISCRETE_ACTIONS), dtype=torch.uint8, device=device)
    buf_disc = torch.zeros((T, total_agents), dtype=torch.long, device=device)
    buf_move = torch.zeros((T, total_agents, CONT_ACTION_DIM), dtype=torch.float32, device=device)
    buf_logprob = torch.zeros((T, total_agents), dtype=torch.float32, device=device)
    buf_rewards = torch.zeros((T, total_agents), dtype=torch.float32, device=device)
    buf_dones = torch.zeros((T, total_agents), dtype=torch.float32, device=device)
    buf_values = torch.zeros((T, total_agents), dtype=torch.float32, device=device)
    buf_learner_mask = torch.ones((T, total_agents), dtype=torch.bool, device=device)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir / f"{args.run_name}_{int(time.time())}"))

    obs_np, masks_np = vec_env.reset(seed=args.seed)
    cur_obs = torch.from_numpy(obs_np).to(device)
    cur_masks = torch.from_numpy(masks_np).to(device)
    cur_done = torch.zeros((total_agents,), dtype=torch.float32, device=device)

    num_updates = max(1, args.total_steps // batch_transitions)
    agent_steps_done = 0
    recent_margin: deque[float] = deque(maxlen=100)
    recent_points: deque[float] = deque(maxlen=100)
    recent_win: deque[float] = deque(maxlen=100)
    recent_violations: deque[float] = deque(maxlen=100)

    try:
        for update in range(1, num_updates + 1):
            t_start = time.perf_counter()
            frac = 1.0 - (update - 1.0) / num_updates
            cur_lr = args.lr * max(0.1, frac)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            # Настраиваем долю сред против исторического соперника из пула Лиги
            learner_agent_flags = torch.ones((total_agents,), dtype=torch.bool, device=device)
            use_hist_opp = (
                args.mode == "selfplay"
                and opponent_model is not None
                and len(league_pool) > 0
                and args.league_prob > 0.0
            )
            opp_agent_indices: torch.Tensor | None = None
            if use_hist_opp:
                sampled_sd = random.choice(list(league_pool))
                opponent_model.load_state_dict(sampled_sd)
                env_flags = np.random.rand(num_envs) < args.league_prob
                if env_flags.any():
                    # В выбранных матчах Команда 1 (агенты 5..9 внутри env) играет исторической политикой
                    flag_grid = np.ones((num_envs, agents_per_env), dtype=bool)
                    flag_grid[env_flags, 5:10] = False
                    learner_agent_flags = torch.from_numpy(flag_grid.reshape(-1)).to(device)
                    opp_agent_indices = torch.nonzero(~learner_agent_flags, as_tuple=False).squeeze(-1)

            # Динамический curriculum аркадных модификаторов на поздних этапах обучения
            if args.chaos_curriculum and update % 25 == 1:
                if random.random() < 0.45:
                    chosen_mods = random.sample(CHAOS_MODIFIER_POOL, k=random.choice([1, 2]))
                else:
                    chosen_mods = []
                vec_env.set_curriculum(modifiers=chosen_mods, randomize_tactics=True)

            model.eval()
            for step_idx in range(T):
                team_ctx = compute_team_context(cur_obs)
                buf_obs[step_idx] = cur_obs
                buf_ctx[step_idx] = team_ctx
                buf_masks[step_idx] = cur_masks
                buf_dones[step_idx] = cur_done
                buf_learner_mask[step_idx] = learner_agent_flags

                with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_cuda and use_bf16):
                    disc_a, raw_move_a, clipped_move_a, logp_a, val_a = model.act(
                        cur_obs, cur_masks, team_context=team_ctx, deterministic=False
                    )
                    if opp_agent_indices is not None and opp_agent_indices.numel() > 0 and opponent_model is not None:
                        opp_disc, opp_raw_move, opp_clip_move, _, _ = opponent_model.act(
                            cur_obs[opp_agent_indices],
                            cur_masks[opp_agent_indices],
                            deterministic=False,
                        )
                        disc_a[opp_agent_indices] = opp_disc
                        raw_move_a[opp_agent_indices] = opp_raw_move
                        clipped_move_a[opp_agent_indices] = opp_clip_move

                buf_disc[step_idx] = disc_a
                buf_move[step_idx] = raw_move_a
                buf_logprob[step_idx] = logp_a
                buf_values[step_idx] = val_a

                next_obs_np, next_masks_np, rewards_np, dones_np, episodes = vec_env.step(
                    disc_a.cpu().numpy(), clipped_move_a.cpu().numpy()
                )
                buf_rewards[step_idx] = torch.from_numpy(rewards_np).to(device)
                cur_obs = torch.from_numpy(next_obs_np).to(device)
                cur_masks = torch.from_numpy(next_masks_np).to(device)
                cur_done = torch.from_numpy(dones_np.astype(np.float32)).to(device)

                for ep in episodes:
                    s0, s1 = ep["score"]
                    recent_margin.append(float(s0 - s1))
                    recent_points.append(float(s0 + s1))
                    recent_win.append(1.0 if ep["winner"] == 0 else 0.0)
                    v0, v1 = ep.get("violations", [0, 0])
                    recent_violations.append(float(v0 + v1))

            agent_steps_done += batch_transitions

            # Расчёт обобщённой оценки преимущества (GAE) на GPU
            with torch.no_grad():
                next_ctx = compute_team_context(cur_obs)
                next_value = model.forward_critic(cur_obs, next_ctx).float()
                advantages = torch.zeros_like(buf_rewards)
                last_gae = torch.zeros((total_agents,), dtype=torch.float32, device=device)

                for t in reversed(range(T)):
                    if t == T - 1:
                        next_non_terminal = 1.0 - cur_done
                        next_val = next_value
                    else:
                        next_non_terminal = 1.0 - buf_dones[t + 1]
                        next_val = buf_values[t + 1]
                    delta = buf_rewards[t] + args.gamma * next_val * next_non_terminal - buf_values[t]
                    last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae

                returns = advantages + buf_values

            # Оставляем только переходы обучаемых агентов (отсекаем замороженных соперников из пула)
            valid_idx = torch.nonzero(buf_learner_mask.reshape(-1), as_tuple=False).squeeze(-1)
            b_obs = buf_obs.reshape(-1, OBS_DIM)[valid_idx]
            b_ctx = buf_ctx.reshape(-1, OBS_DIM)[valid_idx]
            b_masks = buf_masks.reshape(-1, NUM_DISCRETE_ACTIONS)[valid_idx]
            b_disc = buf_disc.reshape(-1)[valid_idx]
            b_move = buf_move.reshape(-1, CONT_ACTION_DIM)[valid_idx]
            b_logprob = buf_logprob.reshape(-1)[valid_idx]
            b_adv = advantages.reshape(-1)[valid_idx]
            b_ret = returns.reshape(-1)[valid_idx]
            b_val = buf_values.reshape(-1)[valid_idx]

            # Обновление параметров сети (PPO Clipped Objective + Value Clipping + KL Anchor)
            model.train()
            n_samples = int(b_obs.shape[0])
            kl_anchor_weight = args.kl_coef * max(0.0, 1.0 - (update / max(1.0, num_updates * 0.6)))

            pg_losses: list[float] = []
            vf_losses: list[float] = []
            ent_losses: list[float] = []
            approx_kls: list[float] = []

            early_stop = False
            for _ in range(args.ppo_epochs):
                perm = torch.randperm(n_samples, device=device)
                for start in range(0, n_samples, args.minibatch_size):
                    limiter.begin_work()
                    mb = perm[start : start + args.minibatch_size]
                    mb_adv = b_adv[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_cuda and use_bf16):
                        new_logprob, entropy, new_val, masked_logits = train_net.evaluate_actions(
                            b_obs[mb], b_masks[mb], b_disc[mb], b_move[mb], team_context=b_ctx[mb]
                        )
                        logratio = new_logprob - b_logprob[mb]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                        approx_kls.append(approx_kl)

                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        v_clipped = b_val[mb] + torch.clamp(
                            new_val - b_val[mb], -args.clip_coef, args.clip_coef
                        )
                        vf_loss = 0.5 * torch.max(
                            (new_val - b_ret[mb]).pow(2),
                            (v_clipped - b_ret[mb]).pow(2),
                        ).mean()

                        ent_loss = entropy.mean()
                        loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent_loss

                        if anchor_model is not None and kl_anchor_weight > 1e-6:
                            with torch.no_grad():
                                ref_logits, _ = anchor_model.forward_actor(b_obs[mb])
                                ref_masked = anchor_model._masked_logits_fp32(ref_logits, b_masks[mb])
                                ref_logp = F.log_softmax(ref_masked, dim=-1)
                            cur_logp = F.log_softmax(masked_logits, dim=-1)
                            kl_bc = F.kl_div(cur_logp, ref_logp, log_target=True, reduction="batchmean")
                            loss = loss + kl_anchor_weight * kl_bc

                    if approx_kl > args.target_kl * 1.5:
                        limiter.end_work_and_throttle()
                        early_stop = True
                        break

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    limiter.end_work_and_throttle()

                    pg_losses.append(float(pg_loss.item()))
                    vf_losses.append(float(vf_loss.item()))
                    ent_losses.append(float(ent_loss.item()))

                if early_stop:
                    break

            if args.mode == "selfplay" and update % args.snapshot_interval == 0:
                league_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

            dt_upd = time.perf_counter() - t_start
            sps = int(batch_transitions / max(1e-6, dt_upd))
            mean_margin = float(np.mean(recent_margin)) if recent_margin else 0.0
            mean_pts = float(np.mean(recent_points)) if recent_points else 0.0
            win_rate = float(np.mean(recent_win)) * 100.0 if recent_win else 0.0
            mean_viol = float(np.mean(recent_violations)) if recent_violations else 0.0
            mean_kl = float(np.mean(approx_kls)) if approx_kls else 0.0

            global_total_steps = prior_steps + agent_steps_done
            writer.add_scalar("charts/SPS", sps, global_total_steps)
            writer.add_scalar("charts/win_rate_team0", win_rate, global_total_steps)
            writer.add_scalar("charts/score_margin", mean_margin, global_total_steps)
            writer.add_scalar("charts/total_match_points", mean_pts, global_total_steps)
            writer.add_scalar("charts/violations_per_match", mean_viol, global_total_steps)
            writer.add_scalar("losses/policy_loss", float(np.mean(pg_losses or [0.0])), global_total_steps)
            writer.add_scalar("losses/value_loss", float(np.mean(vf_losses or [0.0])), global_total_steps)
            writer.add_scalar("losses/entropy", float(np.mean(ent_losses or [0.0])), global_total_steps)
            writer.add_scalar("losses/approx_kl", mean_kl, global_total_steps)

            if update % 5 == 0 or update == 1 or update == num_updates:
                print(
                    f"[Upd {update:04d}/{num_updates:04d}] "
                    f"steps={agent_steps_done:,} ({sps:,} шаг/с) | "
                    f"Win0={win_rate:4.1f}% ΔScore={mean_margin:+5.1f} Pts={mean_pts:4.1f} Viol={mean_viol:3.1f} | "
                    f"KL={mean_kl:.4f} Ent={float(np.mean(ent_losses or [0.0])):.3f}"
                )

            if update % args.export_interval == 0 or update == num_updates:
                ckpt_file = args.save_dir / f"{args.run_name}_latest.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "hidden_dim": args.hidden_dim,
                        "num_blocks": args.num_blocks,
                        "trained_steps": global_total_steps,
                        "stage": args.mode,
                    },
                    ckpt_file,
                )
                export_actor_to_json(
                    model,
                    args.json_out,
                    model_name=f"{args.run_name}-{args.hidden_dim}x{args.num_blocks}",
                    trained_steps=global_total_steps,
                )

    finally:
        writer.close()
        vec_env.close()

    print(f"[PPO] Обучение завершено! Актуальные веса в браузере: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
