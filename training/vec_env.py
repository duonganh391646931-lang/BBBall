"""
Многопроцессная векторизованная среда SubprocBBBallVecEnv.
Запускает пул лёгких воркеров Node.js (training/.dist/rl_bridge.mjs) с автоматическим
перезапуском упавших воркеров (auto-respawn) и поддержкой Лиги нейросетевых соперников.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRIDGE_TS = PROJECT_ROOT / "scripts" / "rl_bridge.ts"
DIST_DIR = PROJECT_ROOT / "training" / ".dist"
BRIDGE_MJS = DIST_DIR / "rl_bridge.mjs"


def ensure_rl_workers_built() -> bool:
    """Собирает лёгкие ESM-бандлы через esbuild, если они отсутствуют или устарели."""
    build_script = PROJECT_ROOT / "scripts" / "build_rl_workers.mjs"
    node_bin = shutil.which("node.exe" if os.name == "nt" else "node") or "node"
    if not build_script.exists():
        return False
    need_build = not BRIDGE_MJS.exists()
    if not need_build:
        mjs_mtime = BRIDGE_MJS.stat().st_mtime
        for src in [
            BRIDGE_TS,
            PROJECT_ROOT / "scripts" / "rl_datagen.ts",
            PROJECT_ROOT / "scripts" / "rl_eval_game.ts",
            PROJECT_ROOT / "src" / "game" / "rl" / "env.ts",
            PROJECT_ROOT / "src" / "game" / "rl" / "actions.ts",
            PROJECT_ROOT / "src" / "game" / "rl" / "encoder.ts",
            PROJECT_ROOT / "src" / "game" / "rl" / "neuralPolicy.ts",
        ]:
            if src.exists() and src.stat().st_mtime > mjs_mtime:
                need_build = True
                break
    if need_build:
        res = subprocess.run([node_bin, str(build_script)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode != 0:
            return False
    return BRIDGE_MJS.exists()


def resolve_node_worker_cmd(script_name: str, heap_mb: int = 256) -> list[str]:
    """Возвращает команду чистого `node --max-old-space-size=...` для .mjs (или фоллбэк на tsx)."""
    node_bin = shutil.which("node.exe" if os.name == "nt" else "node") or "node"
    if ensure_rl_workers_built():
        mjs = DIST_DIR / f"{script_name}.mjs"
        if mjs.exists():
            return [node_bin, f"--max-old-space-size={heap_mb}", str(mjs)]
    is_win = os.name == "nt"
    ts_path = PROJECT_ROOT / "scripts" / f"{script_name}.ts"
    local_bin = PROJECT_ROOT / "node_modules" / ".bin" / ("tsx.cmd" if is_win else "tsx")
    if local_bin.exists():
        return [str(local_bin), str(ts_path)]
    npx_path = shutil.which("npx.cmd" if is_win else "npx") or "npx"
    return [npx_path, "tsx", str(ts_path)]


class SubprocBBBallVecEnv:
    """
    Параллельный пул симуляторов «БалбесБола» с бинарной передачей тензоров (Base64 Float32/Uint8)
    и автоматическим перезапуском воркера при внезапном сбое дочернего процесса.
    """

    def __init__(
        self,
        num_workers: int = 4,
        envs_per_worker: int = 8,
        controlled_teams: str | int = "both",
        frame_skip: int = 4,
        quarter_duration: int = 45,
        max_episode_steps: int = 4500,
        randomize_rosters: bool = True,
        randomize_tactics: bool = True,
        modifier_prob: float = 0.0,
        iq_perception_noise: bool = False,
        modifiers: list[str] | None = None,
        seed: int = 42,
        worker_heap_mb: int = 256,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.envs_per_worker = max(1, int(envs_per_worker))
        self.num_envs = self.num_workers * self.envs_per_worker
        self.controlled_teams = controlled_teams
        self.closed = False
        self._cmd = resolve_node_worker_cmd("rl_bridge", heap_mb=worker_heap_mb)
        self._base_init = {
            "cmd": "init",
            "numEnvs": self.envs_per_worker,
            "controlledTeams": controlled_teams,
            "frameSkip": frame_skip,
            "quarterDuration": quarter_duration,
            "maxEpisodeSteps": max_episode_steps,
            "randomizeRosters": randomize_rosters,
            "randomizeTactics": randomize_tactics,
            "modifierProb": float(modifier_prob),
            "iqPerceptionNoise": bool(iq_perception_noise),
            "modifiers": modifiers or [],
        }
        self._base_seed = int(seed)
        self._respawn_count = 0
        self._worker_opponents: list[list[str | None] | None] = [None] * self.num_workers
        self._procs: list[subprocess.Popen[str]] = []
        self._err_files: list[Any] = []

        for w_idx in range(self.num_workers):
            proc, ef = self._spawn_one(w_idx)
            self._procs.append(proc)
            self._err_files.append(ef)

        init_replies = [self._recv_raw(w_idx) for w_idx in range(self.num_workers)]
        first = init_replies[0]
        if not first.get("ok"):
            raise RuntimeError(f"Ошибка инициализации rl_bridge: {first}")

        self.agents_per_env: int = int(first["numAgentsPerEnv"])
        self.agents_per_worker: int = int(first["totalAgents"])
        self.total_agents: int = self.num_workers * self.agents_per_worker
        self.obs_dim: int = int(first["obsDim"])
        self.num_discrete: int = int(first["numDiscrete"])
        self.cont_dim: int = int(first["contDim"])

        self._obs_buf = np.zeros((self.total_agents, self.obs_dim), dtype=np.float32)
        self._masks_buf = np.zeros((self.total_agents, self.num_discrete), dtype=np.uint8)
        self._rewards_buf = np.zeros((self.total_agents,), dtype=np.float32)
        self._dones_buf = np.zeros((self.total_agents,), dtype=np.uint8)

    def _spawn_one(self, w_idx: int) -> tuple[subprocess.Popen[str], Any]:
        ef = open(PROJECT_ROOT / "training" / f".rl_bridge_w{w_idx}.log", "w", encoding="utf-8")
        proc = subprocess.Popen(
            self._cmd,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=ef,
            text=True,
            bufsize=1,
        )
        init_msg = dict(self._base_init)
        init_msg["seed"] = self._base_seed + w_idx * 100_003 + self._respawn_count * 7919
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(init_msg) + "\n")
        proc.stdin.flush()
        return proc, ef

    def _respawn_worker(self, w_idx: int) -> dict[str, Any]:
        """Перезапускает упавший воркер и возвращает стартовый снимок его сред."""
        self._respawn_count += 1
        old = self._procs[w_idx]
        try:
            old.kill()
        except Exception:
            pass
        try:
            self._err_files[w_idx].close()
        except Exception:
            pass
        proc, ef = self._spawn_one(w_idx)
        self._procs[w_idx] = proc
        self._err_files[w_idx] = ef
        init_rep = self._recv_raw(w_idx)
        if not init_rep.get("ok"):
            raise RuntimeError(f"Не удалось перезапустить воркер #{w_idx}: {init_rep}")
        if self._worker_opponents[w_idx] is not None:
            self._send(w_idx, {"cmd": "set_opponents", "envPaths": self._worker_opponents[w_idx]})
            self._recv_raw(w_idx)
        self._send(w_idx, {"cmd": "reset", "seed": self._base_seed + w_idx * 100_003 + self._respawn_count * 997})
        return self._recv_raw(w_idx)

    def _send(self, worker_idx: int, payload: dict[str, Any]) -> None:
        proc = self._procs[worker_idx]
        if proc.stdin is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except OSError:
            pass

    def _recv_raw(self, worker_idx: int) -> dict[str, Any]:
        proc = self._procs[worker_idx]
        if proc.stdout is None:
            raise RuntimeError(f"stdout воркера #{worker_idx} закрыт")
        line = proc.stdout.readline()
        if not line:
            raise EOFError(f"Воркер rl_bridge #{worker_idx} закрыл поток")
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(f"Ошибка в воркере rl_bridge #{worker_idx}: {msg['error']}")
        return msg

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        for w_idx in range(self.num_workers):
            msg: dict[str, Any] = {"cmd": "reset"}
            if seed is not None:
                msg["seed"] = int(seed) + w_idx * 100_003
            self._send(w_idx, msg)

        for w_idx in range(self.num_workers):
            try:
                rep = self._recv_raw(w_idx)
            except (EOFError, OSError):
                rep = self._respawn_worker(w_idx)
            s = w_idx * self.agents_per_worker
            e = s + self.agents_per_worker
            self._obs_buf[s:e] = np.frombuffer(base64.b64decode(rep["obsB64"]), dtype="<f4").reshape(
                self.agents_per_worker, self.obs_dim
            )
            self._masks_buf[s:e] = np.frombuffer(base64.b64decode(rep["masksB64"]), dtype=np.uint8).reshape(
                self.agents_per_worker, self.num_discrete
            )

        return self._obs_buf.copy(), self._masks_buf.copy()

    def step(
        self, disc_actions: np.ndarray, move_actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        disc_u8 = np.asarray(disc_actions, dtype=np.uint8).reshape(self.total_agents)
        move_f32 = np.asarray(move_actions, dtype="<f4").reshape(self.total_agents, self.cont_dim)

        for w_idx in range(self.num_workers):
            s = w_idx * self.agents_per_worker
            e = s + self.agents_per_worker
            self._send(
                w_idx,
                {
                    "cmd": "step",
                    "discB64": base64.b64encode(disc_u8[s:e].tobytes()).decode("ascii"),
                    "moveB64": base64.b64encode(move_f32[s:e].tobytes()).decode("ascii"),
                },
            )

        episodes: list[dict[str, Any]] = []
        for w_idx in range(self.num_workers):
            s = w_idx * self.agents_per_worker
            e = s + self.agents_per_worker
            try:
                rep = self._recv_raw(w_idx)
                self._obs_buf[s:e] = np.frombuffer(base64.b64decode(rep["obsB64"]), dtype="<f4").reshape(
                    self.agents_per_worker, self.obs_dim
                )
                self._masks_buf[s:e] = np.frombuffer(base64.b64decode(rep["masksB64"]), dtype=np.uint8).reshape(
                    self.agents_per_worker, self.num_discrete
                )
                self._rewards_buf[s:e] = np.frombuffer(base64.b64decode(rep["rewardsB64"]), dtype="<f4")
                self._dones_buf[s:e] = np.frombuffer(base64.b64decode(rep["donesB64"]), dtype=np.uint8)
                for ep in rep.get("episodes", ()):
                    ep["globalEnvIdx"] = w_idx * self.envs_per_worker + int(ep.get("envIdx", 0))
                    episodes.append(ep)
            except (EOFError, OSError):
                print(f"  [vec_env] Воркер #{w_idx} перезапущен автоматически (auto-respawn)", flush=True)
                rst = self._respawn_worker(w_idx)
                self._obs_buf[s:e] = np.frombuffer(base64.b64decode(rst["obsB64"]), dtype="<f4").reshape(
                    self.agents_per_worker, self.obs_dim
                )
                self._masks_buf[s:e] = np.frombuffer(base64.b64decode(rst["masksB64"]), dtype=np.uint8).reshape(
                    self.agents_per_worker, self.num_discrete
                )
                self._rewards_buf[s:e] = 0.0
                self._dones_buf[s:e] = 1  # прерванный эпизод помечаем завершённым для корректного GAE

        return (
            self._obs_buf.copy(),
            self._masks_buf.copy(),
            self._rewards_buf.copy(),
            self._dones_buf.copy(),
            episodes,
        )

    def set_opponents(self, env_paths: list[str | None]) -> None:
        """Назначает соперников для каждой из num_envs сред (None = эвристика, str = путь к JSON политики)."""
        assert len(env_paths) == self.num_envs
        for w_idx in range(self.num_workers):
            sub = env_paths[w_idx * self.envs_per_worker : (w_idx + 1) * self.envs_per_worker]
            self._worker_opponents[w_idx] = list(sub)
            self._send(w_idx, {"cmd": "set_opponents", "envPaths": sub})
        for w_idx in range(self.num_workers):
            try:
                self._recv_raw(w_idx)
            except (EOFError, OSError):
                self._respawn_worker(w_idx)

    def set_curriculum(
        self,
        modifiers: list[str] | None = None,
        quarter_duration: int | None = None,
        randomize_tactics: bool | None = None,
        modifier_prob: float | None = None,
        iq_perception_noise: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {"cmd": "set_curriculum"}
        if modifiers is not None:
            payload["modifiers"] = modifiers
        if quarter_duration is not None:
            payload["quarterDuration"] = int(quarter_duration)
        if randomize_tactics is not None:
            payload["randomizeTactics"] = bool(randomize_tactics)
        if modifier_prob is not None:
            payload["modifierProb"] = float(modifier_prob)
        if iq_perception_noise is not None:
            payload["iqPerceptionNoise"] = bool(iq_perception_noise)

        for w_idx in range(self.num_workers):
            self._send(w_idx, payload)
        for w_idx in range(self.num_workers):
            try:
                self._recv_raw(w_idx)
            except (EOFError, OSError):
                self._respawn_worker(w_idx)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for w_idx, proc in enumerate(self._procs):
            if proc.poll() is None:
                try:
                    self._send(w_idx, {"cmd": "close"})
                except Exception:
                    pass
        for proc in self._procs:
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        for ef in self._err_files:
            try:
                ef.close()
            except Exception:
                pass

    def __enter__(self) -> SubprocBBBallVecEnv:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
