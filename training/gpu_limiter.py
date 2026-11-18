"""
Контроллер ограничения ресурсов GPU/CPU (GPUResourceLimiter).
Гарантирует использование НЕ БОЛЕЕ заданной доли ресурсов видеокарты (по умолчанию <= 70%):
  1) Жёсткий потолок видеопамяти через torch.cuda.set_per_process_memory_fraction(0.70) (<= 5.6 GiB на 8 ГБ RTX 5060).
  2) Замкнутый контур скважности (Duty-Cycle Pacer) по точным таймерам CUDA Events и живому датчику NVML
     (libnvidia-ml.so.1 / nvml.dll), удерживающий среднюю загрузку ядер GPU <= 70%.
"""

from __future__ import annotations

import ctypes
import os
import time
from typing import Any

import torch


class _NVMLSensor:
    """Лёгкая ctypes-обёртка над системной библиотекой NVML без внешних pip-зависимостей."""

    class _UtilStruct(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    def __init__(self, device_index: int = 0) -> None:
        self.available = False
        self._lib: Any = None
        self._handle = ctypes.c_void_p()

        candidates = (
            ["nvml.dll"]
            if os.name == "nt"
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        )
        for name in candidates:
            try:
                lib = ctypes.CDLL(name)
                if lib.nvmlInit_v2() == 0:
                    if lib.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(self._handle)) == 0:
                        self._lib = lib
                        self.available = True
                        break
            except Exception:
                continue

    def read_gpu_util_percent(self) -> float | None:
        if not self.available or self._lib is None:
            return None
        util = self._UtilStruct()
        res = self._lib.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util))
        if res == 0:
            return float(util.gpu)
        return None


class GPUResourceLimiter:
    """
    Ограничитель вычислительной нагрузки и видеопамяти GPU на уровне <= max_fraction (по умолчанию 0.70 = 70%).
    """

    def __init__(
        self,
        device: torch.device,
        max_fraction: float = 0.70,
        safety_margin: float = 0.02,
    ) -> None:
        self.device = device
        self.max_fraction = float(max(0.15, min(1.0, max_fraction)))
        # Целевая рабочая точка чуть ниже порога (например 68% при лимите 70%), чтобы исключить выбросы > 70%
        self.target_duty = max(0.12, self.max_fraction - safety_margin)
        self.is_cuda = device.type == "cuda" and torch.cuda.is_available()

        # Ограничиваем потоки CPU пропорционально лимиту, чтобы не перегревать систему
        cpu_total = os.cpu_count() or 2
        allowed_threads = max(1, int(round(cpu_total * self.max_fraction)))
        torch.set_num_threads(allowed_threads)

        self._nvml: _NVMLSensor | None = None
        self._start_evt: torch.cuda.Event | None = None
        self._end_evt: torch.cuda.Event | None = None
        self._host_t0: float = 0.0
        self._gain: float = 1.0
        self.ema_gpu_util: float = self.target_duty * 100.0
        self.total_gpu_ms: float = 0.0
        self.total_sleep_ms: float = 0.0

        if self.is_cuda:
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            try:
                torch.cuda.set_per_process_memory_fraction(self.max_fraction, dev_idx)
            except Exception:
                pass
            self._nvml = _NVMLSensor(dev_idx)
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)

    def begin_work(self) -> None:
        self._host_t0 = time.perf_counter()
        if self.is_cuda and self._start_evt is not None:
            self._start_evt.record()

    def end_work_and_throttle(self) -> float:
        """
        Замеряет время активной работы ядер GPU и вставляет точную паузу разгрузки,
        удерживая суммарный коэффициент заполнения (Duty Cycle) не выше max_fraction (70%).
        Возвращает длительность вставленной паузы в секундах.
        """
        if self.max_fraction >= 0.99:
            return 0.0

        if not self.is_cuda or self._start_evt is None or self._end_evt is None:
            # На CPU мягко выдерживаем скважность рабочего окна при необходимости
            busy_s = time.perf_counter() - self._host_t0
            sleep_s = busy_s * ((1.0 - self.max_fraction) / self.max_fraction) * 0.5
            if sleep_s > 0.001:
                time.sleep(min(sleep_s, 0.15))
                self.total_sleep_ms += sleep_s * 1000.0
                return sleep_s
            return 0.0

        self._end_evt.record()
        torch.cuda.synchronize(self.device)
        gpu_ms = float(self._start_evt.elapsed_time(self._end_evt))
        wall_ms = (time.perf_counter() - self._host_t0) * 1000.0
        non_gpu_ms = max(0.0, wall_ms - gpu_ms)
        self.total_gpu_ms += gpu_ms

        # Обратная связь по реальному датчику загрузки GPU NVML (если доступен)
        nvml_util = self._nvml.read_gpu_util_percent() if self._nvml is not None else None
        if nvml_util is not None:
            self.ema_gpu_util = 0.8 * self.ema_gpu_util + 0.2 * nvml_util
            cap_pct = self.max_fraction * 100.0
            if self.ema_gpu_util > cap_pct - 1.0:
                self._gain = min(2.5, self._gain * 1.08)
            elif self.ema_gpu_util < cap_pct - 8.0:
                self._gain = max(0.85, self._gain * 0.96)
        else:
            denom = max(1e-3, self.total_gpu_ms + self.total_sleep_ms)
            self.ema_gpu_util = 100.0 * (self.total_gpu_ms / denom)

        # Требуемое полное окно T_window = gpu_ms / target_duty -> пауза = T_window - wall_ms
        required_window_ms = (gpu_ms / self.target_duty) * self._gain
        sleep_ms = max(0.0, required_window_ms - gpu_ms - non_gpu_ms)

        if sleep_ms >= 0.5:
            sleep_s = min(sleep_ms / 1000.0, 0.25)
            time.sleep(sleep_s)
            self.total_sleep_ms += sleep_s * 1000.0
            return sleep_s
        return 0.0

    def status_summary(self) -> str:
        alloc_mb, cap_mb = self.vram_status_mb()
        return (
            f"limit={self.max_fraction * 100:.0f}% util(EMA)={self.ema_gpu_util:.1f}% "
            f"VRAM={alloc_mb:.0f}/{cap_mb:.0f} MiB sleep={self.total_sleep_ms / 1000.0:.1f}s"
        )

    def vram_status_mb(self) -> tuple[float, float]:
        if not self.is_cuda:
            return 0.0, 0.0
        alloc_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
        total_cap_mb = (
            torch.cuda.get_device_properties(self.device).total_memory
            * self.max_fraction
            / (1024**2)
        )
        return alloc_mb, total_cap_mb
