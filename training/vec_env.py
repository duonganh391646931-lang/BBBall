"""
Многопроцессная векторизованная среда SubprocBBBallVecEnv.
Запускает пул параллельных воркеров Node.js (scripts/rl_bridge.ts) для загрузки всех ядер CPU
и бесперебойной подачи батчей переходов на видеокарту NVIDIA RTX 5060.
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
BRIDGE_SCRIPT = PROJECT_ROOT / "scripts" / "rl_bridge.ts"


def _resolve_tsx_command() -> list[str]:
    is_win = os.name == "nt"
    local_bin = PROJECT_ROOT / "node_modules" / ".bin" / ("tsx.cmd" if is_win else "tsx")
    if local_bin.exists():
        return [str(local_bin), str(BRIDGE_SCRIPT)]
    npx_path = shutil.which("npx.cmd" if is_win else "npx") or "npx"
    return [npx_path, "tsx", str(BRIDGE_SCRIPT)]


class SubprocBBBallVecEnv:
    """
    Параллельный пул симуляторов «БалбесБола» с бинарной передачей тензоров (Base64 Float32/Uint8).
    Пример конфигурации под 6–8 ядер CPU + RTX 5060:
      num_workers=8, envs_per_worker=16 -> 128 параллельных матчей (1 280 агентов за шаг).
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
        modifiers: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.envs_per_worker = max(1, int(envs_per_worker))
        self.num_envs = self.num_workers * self.envs_per_worker
        self.controlled_teams = controlled_teams
        self.closed = False

        cmd = _resolve_tsx_command()
        self._procs: list[subprocess.Popen[str]] = []

        for w_idx in range(self.num_workers):
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._procs.append(proc)
            init_msg = {
                "cmd": "init",
                "numEnvs": self.envs_per_worker,
                "controlledTeams": controlled_teams,
                "frameSkip": frame_skip,
                "quarterDuration": quarter_duration,
                "maxEpisodeSteps": max_episode_steps,
                "randomizeRosters": randomize_rosters,
                "randomizeTactics": randomize_tactics,
                "modifiers": modifiers or [],
                "seed": int(seed) + w_idx * 100_003,
            }
            self._send(w_idx, init_msg)

        init_replies = [self._recv(w_idx) for w_idx in range(self.num_workers)]
        first = init_replies[0]
        if not first.get("ok"):
            raise RuntimeError(f"Ошибка инициализации rl_bridge.ts: {first}")

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

    def _send(self, worker_idx: int, payload: dict[str, Any]) -> None:
        proc = self._procs[worker_idx]
        if proc.stdin is None:
            raise RuntimeError(f"stdin воркера #{worker_idx} закрыт")
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def _recv(self, worker_idx: int) -> dict[str, Any]:
        proc = self._procs[worker_idx]
        if proc.stdout is None:
            raise RuntimeError(f"stdout воркера #{worker_idx} закрыт")
        line = proc.stdout.readline()
        if not line:
            err_tail = ""
            if proc.stderr is not None:
                err_tail = proc.stderr.read()
            raise RuntimeError(f"Воркер rl_bridge #{worker_idx} неожиданно завершился.\n{err_tail}")
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
            rep = self._recv(w_idx)
            s = w_idx * self.agents_per_worker
            e = s + self.agents_per_worker
            obs_w = np.frombuffer(base64.b64decode(rep["obsB64"]), dtype="<f4").reshape(
                self.agents_per_worker, self.obs_dim
            )
            masks_w = np.frombuffer(base64.b64decode(rep["masksB64"]), dtype=np.uint8).reshape(
                self.agents_per_worker, self.num_discrete
            )
            self._obs_buf[s:e] = obs_w
            self._masks_buf[s:e] = masks_w

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
            rep = self._recv(w_idx)
            s = w_idx * self.agents_per_worker
            e = s + self.agents_per_worker

            self._obs_buf[s:e] = np.frombuffer(
                base64.b64decode(rep["obsB64"]), dtype="<f4"
            ).reshape(self.agents_per_worker, self.obs_dim)
            self._masks_buf[s:e] = np.frombuffer(
                base64.b64decode(rep["masksB64"]), dtype=np.uint8
            ).reshape(self.agents_per_worker, self.num_discrete)
            self._rewards_buf[s:e] = np.frombuffer(
                base64.b64decode(rep["rewardsB64"]), dtype="<f4"
            )
            self._dones_buf[s:e] = np.frombuffer(
                base64.b64decode(rep["donesB64"]), dtype=np.uint8
            )

            for ep in rep.get("episodes", ()):
                ep["globalEnvIdx"] = w_idx * self.envs_per_worker + int(ep.get("envIdx", 0))
                episodes.append(ep)

        return (
            self._obs_buf.copy(),
            self._masks_buf.copy(),
            self._rewards_buf.copy(),
            self._dones_buf.copy(),
            episodes,
        )

    def set_curriculum(
        self,
        modifiers: list[str] | None = None,
        quarter_duration: int | None = None,
        randomize_tactics: bool | None = None,
    ) -> None:
        payload: dict[str, Any] = {"cmd": "set_curriculum"}
        if modifiers is not None:
            payload["modifiers"] = modifiers
        if quarter_duration is not None:
            payload["quarterDuration"] = int(quarter_duration)
        if randomize_tactics is not None:
            payload["randomizeTactics"] = bool(randomize_tactics)

        for w_idx in range(self.num_workers):
            self._send(w_idx, payload)
        for w_idx in range(self.num_workers):
            self._recv(w_idx)

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

    def __enter__(self) -> SubprocBBBallVecEnv:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
