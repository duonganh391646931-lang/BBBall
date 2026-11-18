#!/usr/bin/env python3
"""
Этап 1 плана обучения: Имитационное обучение (Behavior Cloning) на траекториях эвристического учителя.
Оптимизировано под NVIDIA GeForce RTX 5060 (bfloat16 Tensor Cores, TF32, torch.compile, memmap).

Пример запуска:
  npm run rl:datagen -- --samples=500000
  python training/train_bc.py --data-dir training/data/bc_dataset --epochs 20 --batch-size 4096
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import (
    CONT_ACTION_DIM,
    NUM_DISCRETE_ACTIONS,
    OBS_DIM,
    BasketballActorCritic,
    export_actor_to_json,
)


class MemmapBCDataset(Dataset):
    """Чтение бинарного датасета rl_datagen.ts без копирования всего массива в ОЗУ."""

    def __init__(self, data_dir: Path, indices: np.ndarray) -> None:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        total = int(meta["samples"])
        files = meta["files"]

        self.obs = np.memmap(data_dir / files["obs"], dtype="<f4", mode="r", shape=(total, OBS_DIM))
        self.masks = np.memmap(
            data_dir / files["masks"], dtype=np.uint8, mode="r", shape=(total, NUM_DISCRETE_ACTIONS)
        )
        self.disc = np.memmap(data_dir / files["disc"], dtype=np.uint8, mode="r", shape=(total,))
        self.move = np.memmap(
            data_dir / files["move"], dtype="<f4", mode="r", shape=(total, CONT_ACTION_DIM)
        )
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        row = int(self.indices[idx])
        return (
            np.array(self.obs[row], dtype=np.float32),
            np.array(self.masks[row], dtype=np.uint8),
            int(self.disc[row]),
            np.array(self.move[row], dtype=np.float32),
        )


def compute_smoothed_class_weights(disc_memmap: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(disc_memmap, dtype=np.int64), minlength=num_classes).astype(
        np.float64
    )
    total = max(1.0, float(counts.sum()))
    # Сглаженная балансировка редких действий (пасы, броски, перехваты, подбор) относительно NOOP_MOVE
    weights = np.power(total / np.maximum(counts, 1.0), 0.30)
    weights = np.clip(weights / max(1e-6, weights[0]), 0.6, 3.5)
    weights[counts == 0] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior Cloning Warm-Start (RTX 5060)")
    p.add_argument("--data-dir", type=Path, default=Path("training/data/bc_dataset"))
    p.add_argument("--checkpoint-out", type=Path, default=Path("training/checkpoints/bc_init.pt"))
    p.add_argument("--json-out", type=Path, default=Path("public/models/policy_latest.json"))
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-blocks", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--move-loss-coef", type=float, default=2.0)
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta_path = args.data_dir / "meta.json"
    if not meta_path.exists():
        print(
            f"[ОШИБКА] Датасет не найден в {args.data_dir}.\n"
            "Сначала соберите траектории эвристического учителя:\n"
            "  npm run rl:datagen -- --samples=250000"
        )
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    total_samples = int(meta["samples"])
    print(f"[BC] Загружен датасет: {total_samples:,} примеров из {args.data_dir}")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if use_cuda:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[BC] Устройство: {torch.cuda.get_device_name(device)} | AMP: {amp_dtype}")
    else:
        print("[BC] Устройство: CPU (CUDA не обнаружена)")

    perm = np.random.permutation(total_samples)
    val_size = max(512, int(total_samples * args.val_split))
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = MemmapBCDataset(args.data_dir, train_idx)
    val_ds = MemmapBCDataset(args.data_dir, val_idx)

    class_weights = compute_smoothed_class_weights(train_ds.disc, NUM_DISCRETE_ACTIONS).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=use_cuda,
    )

    model = BasketballActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    ).to(device)

    forward_model: torch.nn.Module = model
    if use_cuda and not args.no_compile and hasattr(torch, "compile"):
        try:
            forward_model = torch.compile(model, mode="default")
        except Exception as exc:
            print(f"[BC] torch.compile пропущен: {exc}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_cuda,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05
    )

    best_val_loss = float("inf")
    global_steps = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_disc_loss = 0.0
        train_move_loss = 0.0
        train_correct = 0
        train_seen = 0

        for obs_b, masks_b, disc_b, move_b in train_loader:
            obs_t = obs_b.to(device, non_blocking=True)
            masks_t = masks_b.to(device, dtype=torch.bool, non_blocking=True)
            disc_t = disc_b.to(device, dtype=torch.long, non_blocking=True)
            move_t = move_b.to(device, non_blocking=True)

            # Страхуемся от невозможного маскирования целевого действия учителя
            masks_t.scatter_(1, disc_t.unsqueeze(1), True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_cuda and use_bf16):
                logits, move_pred = forward_model.forward_actor(obs_t)
                masked_logits = logits.float().masked_fill(~masks_t, -1e9)
                loss_disc = F.cross_entropy(masked_logits, disc_t, weight=class_weights)
                loss_move = F.smooth_l1_loss(move_pred.float(), move_t.float(), beta=0.15)
                loss = loss_disc + args.move_loss_coef * loss_move

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bsz = int(obs_t.shape[0])
            global_steps += bsz
            train_seen += bsz
            train_disc_loss += float(loss_disc.item()) * bsz
            train_move_loss += float(loss_move.item()) * bsz
            train_correct += int((masked_logits.argmax(dim=-1) == disc_t).sum().item())

        scheduler.step()

        # Валидация
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_non_noop_correct = 0
        val_non_noop_total = 0
        val_seen = 0

        with torch.no_grad():
            for obs_b, masks_b, disc_b, move_b in val_loader:
                obs_t = obs_b.to(device, non_blocking=True)
                masks_t = masks_b.to(device, dtype=torch.bool, non_blocking=True)
                disc_t = disc_b.to(device, dtype=torch.long, non_blocking=True)
                move_t = move_b.to(device, non_blocking=True)
                masks_t.scatter_(1, disc_t.unsqueeze(1), True)

                logits, move_pred = model.forward_actor(obs_t)
                masked_logits = logits.float().masked_fill(~masks_t, -1e9)
                l_d = F.cross_entropy(masked_logits, disc_t, weight=class_weights)
                l_m = F.smooth_l1_loss(move_pred.float(), move_t.float(), beta=0.15)
                bsz = int(obs_t.shape[0])
                val_seen += bsz
                val_loss_sum += float((l_d + args.move_loss_coef * l_m).item()) * bsz

                preds = masked_logits.argmax(dim=-1)
                val_correct += int((preds == disc_t).sum().item())
                active_mask = disc_t != 0
                if active_mask.any():
                    val_non_noop_total += int(active_mask.sum().item())
                    val_non_noop_correct += int((preds[active_mask] == disc_t[active_mask]).sum().item())

        val_loss = val_loss_sum / max(1, val_seen)
        dt_ep = time.perf_counter() - t0
        act_acc = 100.0 * val_non_noop_correct / max(1, val_non_noop_total)
        print(
            f"[Эпоха {epoch:02d}/{args.epochs:02d}] "
            f"disc={train_disc_loss / max(1, train_seen):.4f} "
            f"move={train_move_loss / max(1, train_seen):.4f} | "
            f"val_loss={val_loss:.4f} acc={100.0 * val_correct / max(1, val_seen):.1f}% "
            f"tactical_acc={act_acc:.1f}% ({dt_ep:.2f}с)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "hidden_dim": args.hidden_dim,
                    "num_blocks": args.num_blocks,
                    "trained_steps": global_steps,
                    "stage": "bc_warmstart",
                },
                args.checkpoint_out,
            )
            export_actor_to_json(
                model,
                args.json_out,
                model_name=f"bc-warmstart-{args.hidden_dim}x{args.num_blocks}",
                trained_steps=global_steps,
            )

    print(
        f"[BC] Готово! Лучший чекпоинт: {args.checkpoint_out} | "
        f"Веса для браузера: {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
