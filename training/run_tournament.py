#!/usr/bin/env python3
"""
Финальный турнир Этапа 5: объективное сравнение всех ключевых поколений ИИ «БалбесБола»
на эталонном движке src/game/sim.ts:
  1) 128 классических матчей против встроенного эвристического ИИ (stepPlayerAI);
  2) 64 матча с 100% активными модификаторами хаоса (Два мяча, Гололёд, Луна, Батуты, Взрывной мяч);
  3) Круговой финал между сильнейшими нейросетевыми поколениями ('neural' vs 'neural', 96 игр на пару);
  4) Автоматический выбор и экспорт абсолютного чемпиона в policy_latest.json и defaultTrainedWeights.ts.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import game_eval, load_policy  # noqa: E402
from model import DEFAULT_JSON_OUT, DEFAULT_TS_WEIGHTS_OUT, export_actor_to_json  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    t0 = time.perf_counter()
    candidates = [
        ("repo_original_2.6M", ROOT / "training" / "baselines" / "repo_original_2.6M.json", None),
        ("v2_start_plus0.29", ROOT / "training" / "baselines" / "v2_best_plus0.29.json", ROOT / "training" / "baselines" / "v2_best_plus0.29.pt"),
        ("s2_bc_256x3", ROOT / "training" / "checkpoints_s2_256x3" / "best.json", ROOT / "training" / "checkpoints_s2_256x3" / "best.pt"),
        ("s3_ppo_256x3", ROOT / "training" / "baselines" / "stage3a_best_plus24.30.json", ROOT / "training" / "baselines" / "stage3a_best_plus24.30.pt"),
        ("s4a_league_256x3", ROOT / "training" / "baselines" / "stage4a_best_plus30.31.json", ROOT / "training" / "baselines" / "stage4a_best_plus30.31.pt"),
        ("s4b_gm_256x3", ROOT / "training" / "checkpoints_s4b_256x3" / "best.json", ROOT / "training" / "checkpoints_s4b_256x3" / "best.pt"),
    ]
    active = [(name, js, pt) for name, js, pt in candidates if js.exists()]
    print(f"=== ФИНАЛЬНЫЙ ТУРНИР БАЛБЕСБОЛА ({len(active)} участников) ===", flush=True)

    summary: dict[str, dict] = {}
    for name, js, pt in active:
        t1 = time.perf_counter()
        std_res = game_eval(js, games=128, procs=4, seed=9001, quarter=45, mod_prob=0.0)
        chaos_res = game_eval(js, games=64, procs=4, seed=9501, quarter=45, mod_prob=1.0)
        summary[name] = {
            "json": str(js),
            "pt": str(pt) if pt and pt.exists() else None,
            "vs_heuristic_128": std_res,
            "vs_heuristic_chaos_64": chaos_res,
            "h2h": {},
        }
        print(
            f"  [{name:20s}] Классика(128): побед={std_res['win_rate']*100:5.1f}% разн={std_res['mean_diff']:+6.2f}±{std_res['ci95']:.2f} | "
            f"Хаос-моды(64): побед={chaos_res['win_rate']*100:5.1f}% разн={chaos_res['mean_diff']:+6.2f}±{chaos_res['ci95']:.2f} "
            f"({time.perf_counter()-t1:.0f}с)",
            flush=True,
        )

    # Круговой финал между топ-поколениями v3 + контрольные матчи против старта сессии и оригинала
    finalists = [n for n in ("s3_ppo_256x3", "s4a_league_256x3", "s4b_gm_256x3") if n in summary]
    print(f"\n=== ОЧНЫЙ КРУГОВОЙ ФИНАЛ ('neural' vs 'neural', 96 матчей на пару): {finalists} ===", flush=True)
    for i, a in enumerate(finalists):
        for b in finalists[i + 1 :]:
            r = game_eval(summary[a]["json"], games=96, procs=4, seed=7700 + i * 31, quarter=45, opponent_json=summary[b]["json"])
            summary[a]["h2h"][b] = r
            summary[b]["h2h"][a] = {
                "games": r["games"],
                "win_rate": 1.0 - r["win_rate"] - r["tie_rate"],
                "tie_rate": r["tie_rate"],
                "mean_diff": -r["mean_diff"],
                "ci95": r["ci95"],
            }
            print(
                f"  {a} vs {b}: побед {a}={r['win_rate']*100:.1f}%, разница={r['mean_diff']:+.2f}±{r['ci95']:.2f}",
                flush=True,
            )

    # Выбор абсолютного чемпиона среди финалистов
    best_name = finalists[0]
    best_score = -1e9
    for name in finalists:
        std_d = summary[name]["vs_heuristic_128"]["mean_diff"]
        ch_d = summary[name]["vs_heuristic_chaos_64"]["mean_diff"]
        h2h_vals = [v["mean_diff"] for v in summary[name]["h2h"].values()]
        h2h_mean = sum(h2h_vals) / max(1, len(h2h_vals))
        comp = 0.50 * std_d + 0.20 * ch_d + 0.30 * h2h_mean * 2.5
        summary[name]["composite_rating"] = comp
        print(f"  [Итоговый балл] {name:20s}: std={std_d:+.2f}, chaos={ch_d:+.2f}, h2h={h2h_mean:+.2f} -> балл={comp:+.2f}", flush=True)
        if comp > best_score:
            best_score = comp
            best_name = name

    # Контрольные серии чемпиона против старта сессии и оригинала репозитория (по 64 игры)
    champ_js = Path(summary[best_name]["json"])
    for ref_name in ("v2_start_plus0.29", "repo_original_2.6M"):
        if ref_name in summary:
            r = game_eval(champ_js, games=64, procs=4, seed=5555, quarter=45, opponent_json=summary[ref_name]["json"])
            summary[best_name]["h2h"][ref_name] = r
            print(
                f"  [Чемпион {best_name} vs {ref_name}] побед={r['win_rate']*100:.1f}% разница={r['mean_diff']:+.2f}±{r['ci95']:.2f}",
                flush=True,
            )

    # Экспорт чемпиона в игру и чекпоинт паритета
    champ_pt_str = summary[best_name]["pt"]
    champ_model = load_policy(champ_pt_str if champ_pt_str else champ_js, torch.device("cpu"))
    trained_steps = 0
    if champ_pt_str:
        ck = torch.load(champ_pt_str, map_location="cpu", weights_only=False)
        trained_steps = int(ck.get("trained_steps", 0))
        dst_pt = ROOT / "training" / "checkpoints" / "balbesbol_trained_latest.pt"
        dst_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(champ_pt_str, dst_pt)

    export_actor_to_json(
        champ_model,
        out_path=DEFAULT_JSON_OUT,
        model_name=f"balbesbol-v3-champion-{champ_model.hidden_dim}x{champ_model.num_blocks}",
        trained_steps=trained_steps,
        embed_in_ts=True,
        ts_out_path=DEFAULT_TS_WEIGHTS_OUT,
    )

    out_report = {
        "champion": best_name,
        "champion_composite_rating": best_score,
        "trained_steps": trained_steps,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "participants": summary,
    }
    out_path = ROOT / "training" / "tournament_results.json"
    out_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ЧЕМПИОН ТУРНИРА: {best_name}] Веса вшиты в {DEFAULT_JSON_OUT} и {DEFAULT_TS_WEIGHTS_OUT} ({time.perf_counter()-t0:.0f}с)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
