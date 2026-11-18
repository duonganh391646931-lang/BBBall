#!/usr/bin/env python3
"""
Диагностика окружения и микробенчмарк Tensor Cores для NVIDIA GeForce RTX 5060 (Blackwell, sm_120).
Проверяет поддержку CUDA 12.8+, bfloat16, TF32 и torch.compile перед запуском BC / MAPPO.
"""

from __future__ import annotations

import platform
import sys
import time


def main() -> int:
    print("=== БалбесБол RL: Диагностика GPU-стека (целевой профиль: RTX 5060 8 ГБ) ===")
    print(f"Python:   {sys.version.split()[0]} ({platform.system()} {platform.machine()})")

    try:
        import torch
    except ImportError:
        print(
            "[ОШИБКА] PyTorch не установлен. Установите стек CUDA 12.8:\n"
            "  pip install -r training/requirements-rtx5060.txt"
        )
        return 1

    print(f"PyTorch:  {torch.__version__} (CUDA build: {torch.version.cuda})")

    if not torch.cuda.is_available():
        print(
            "[ВНИМАНИЕ] CUDA-устройство не обнаружено (доступен только CPU).\n"
            "Для обучения на RTX 5060 убедитесь, что установлен драйвер NVIDIA >= 572.xx\n"
            "и сборка PyTorch >= 2.7.0+cu128."
        )
        return 0

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    cc_major, cc_minor = props.major, props.minor
    vram_gb = props.total_memory / (1024**3)
    bf16_ok = torch.cuda.is_bf16_supported()

    print(f"GPU:      {props.name}")
    print(f"Arch:     sm_{cc_major}{cc_minor} (Compute Capability {cc_major}.{cc_minor})")
    print(f"VRAM:     {vram_gb:.2f} GiB | Мультипроцессоров (SM): {props.multi_processor_count}")
    print(f"bfloat16: {'ДА' if bf16_ok else 'НЕТ'}")

    if cc_major >= 12:
        arch_list = [a.lower() for a in torch.cuda.get_arch_list()]
        has_sm120 = any("sm_120" in a or "compute_120" in a for a in arch_list)
        if not has_sm120:
            print(
                "[ОШИБКА] Обнаружена архитектура Blackwell (sm_120), но текущая сборка PyTorch\n"
                f"скомпилирована только для {arch_list}. Переустановите PyTorch с cu128:\n"
                "  pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
            )
            return 1
        print("[OK] Архитектура Blackwell sm_120 полностью поддерживается сборкой PyTorch.")

    if vram_gb < 7.0:
        print(
            f"[ВНИМАНИЕ] Доступно {vram_gb:.1f} GiB VRAM (< 8 GiB). Рекомендуется уменьшить "
            "--minibatch-size до 4096."
        )

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from gpu_limiter import GPUResourceLimiter
    from model import NUM_DISCRETE_ACTIONS, OBS_DIM, BasketballActorCritic

    limiter = GPUResourceLimiter(device, max_fraction=0.70)
    model = BasketballActorCritic(hidden_dim=192, num_blocks=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"Модель:   BasketballActorCritic (192x2 ResMLP, OBS={OBS_DIM}, ACT={NUM_DISCRETE_ACTIONS}), "
        f"параметров: {param_count:,}"
    )

    batch_size = 8192
    obs = torch.randn(batch_size, OBS_DIM, device=device)
    team_ctx = torch.randn(batch_size, OBS_DIM, device=device)
    masks = torch.ones(batch_size, NUM_DISCRETE_ACTIONS, dtype=torch.bool, device=device)
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

    # Прогрев и проверка torch.compile
    compiled_fn = model
    if hasattr(torch, "compile"):
        try:
            compiled_fn = torch.compile(model, mode="default")
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _ = compiled_fn.forward_actor_critic(obs, team_ctx)
            torch.cuda.synchronize()
            print("[OK] Граф torch.compile успешно скомпилирован.")
        except Exception as exc:
            print(f"[ИНФО] torch.compile пропущен (fallback на eager): {exc}")
            compiled_fn = model

    # Замер скорости прохода вперёд+назад в смешанной точности с ограничителем <= 70% GPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    iters = 50
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iters):
        limiter.begin_work()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            logits, move_mean, value = compiled_fn.forward_actor_critic(obs, team_ctx)
            masked_logits = logits.masked_fill(~masks, -1e9)
            loss = masked_logits.float().pow(2).mean() + move_mean.float().pow(2).mean() + value.float().pow(2).mean()
        loss.backward()
        optimizer.step()
        limiter.end_work_and_throttle()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    samples_per_sec = (batch_size * iters) / max(1e-6, elapsed)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(
        f"[БЕНЧМАРК <=70% GPU] Forward+Backward+AdamW (batch={batch_size}, {amp_dtype}): "
        f"{samples_per_sec:,.0f} агент-шагов/с | Пик VRAM: {peak_mb:.1f} MiB | {limiter.status_summary()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
