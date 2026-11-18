#!/usr/bin/env python3
"""
Автоматический тест строгого паритета между PyTorch (BasketballActorCritic)
и браузерным TypeScript-движком (NeuralPolicy в src/game/rl/neuralPolicy.ts) с точностью < 1e-5.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from model import DEFAULT_JSON_OUT, OBS_DIM, BasketballActorCritic

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ckpt_path = PROJECT_ROOT / "training" / "checkpoints" / "balbesbol_trained_latest.pt"
    if not ckpt_path.exists() or not DEFAULT_JSON_OUT.exists():
        print("[ОШИБКА] Сначала запустите обучение для генерации чекпоинта и JSON.")
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = BasketballActorCritic(
        obs_dim=OBS_DIM,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_blocks=int(ckpt["num_blocks"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Важно: сверяем с квантованными до 5 знаков весами из экспортированного JSON (как в браузере)
    exported = json.loads(DEFAULT_JSON_OUT.read_text(encoding="utf-8"))

    def load_lin(mod: torch.nn.Linear, d: dict) -> None:
        mod.weight.data.copy_(torch.tensor(d["weight"], dtype=torch.float32).reshape(d["outFeatures"], d["inFeatures"]))
        mod.bias.data.copy_(torch.tensor(d["bias"], dtype=torch.float32))

    def load_ln(mod: torch.nn.LayerNorm, d: dict) -> None:
        mod.weight.data.copy_(torch.tensor(d["weight"], dtype=torch.float32))
        mod.bias.data.copy_(torch.tensor(d["bias"], dtype=torch.float32))

    load_lin(model.input_linear, exported["inputLinear"])
    load_ln(model.input_norm, exported["inputNorm"])
    for blk, bd in zip(model.blocks, exported["blocks"]):
        load_lin(blk.fc1, bd["fc1"])
        load_ln(blk.ln1, bd["ln1"])
        load_lin(blk.fc2, bd["fc2"])
        load_ln(blk.ln2, bd["ln2"])
    load_lin(model.discrete_head, exported["discreteHead"])
    load_lin(model.move_mean_head, exported["moveMeanHead"])

    rng = np.random.default_rng(5060)
    test_obs = rng.uniform(-1.0, 1.0, size=(8, OBS_DIM)).astype(np.float32)

    with torch.no_grad():
        py_logits, py_move = model.forward_actor(torch.from_numpy(test_obs))
    py_logits_np = py_logits.numpy()
    py_move_np = py_move.numpy()

    node_code = f"""
import fs from 'node:fs';
import {{ NeuralPolicy }} from './src/game/rl/neuralPolicy';
const weights = JSON.parse(fs.readFileSync({json.dumps(str(DEFAULT_JSON_OUT))}, 'utf8'));
const policy = new NeuralPolicy(weights);
const batch = {json.dumps(test_obs.tolist())};
const out = batch.map((row) => {{
  const res = policy.forwardRaw(new Float32Array(row));
  return {{ logits: Array.from(res.logits), move: Array.from(res.moveMean) }};
}});
process.stdout.write(JSON.stringify(out));
"""
    import os

    tsx_bin = PROJECT_ROOT / "node_modules" / ".bin" / ("tsx.cmd" if os.name == "nt" else "tsx")
    proc = subprocess.run(
        [str(tsx_bin), "-e", node_code],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    ts_results = json.loads(proc.stdout)

    max_logit_diff = 0.0
    max_move_diff = 0.0
    for i, row in enumerate(ts_results):
        max_logit_diff = max(max_logit_diff, float(np.max(np.abs(py_logits_np[i] - np.array(row["logits"], dtype=np.float32)))))
        max_move_diff = max(max_move_diff, float(np.max(np.abs(py_move_np[i] - np.array(row["move"], dtype=np.float32)))))

    print(
        f"[ПАРИТЕТ PyTorch <-> TypeScript] max|Δlogits| = {max_logit_diff:.2e}, "
        f"max|ΔmoveMean| = {max_move_diff:.2e}"
    )
    assert max_logit_diff < 1e-5, f"Расхождение логитов {max_logit_diff} >= 1e-5!"
    assert max_move_diff < 1e-5, f"Расхождение вектора движения {max_move_diff} >= 1e-5!"
    print("[OK] Строгий паритет PyTorch и браузерного TypeScript-движка (< 1e-5) подтверждён!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
