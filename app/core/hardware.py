"""Détection matérielle et recommandation de modèles (cf. cahier des charges §31).

Aucune supposition n'est faite sur la présence d'un GPU NVIDIA : Liliana doit
tourner sur CPU, plus lentement mais correctement.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class HardwareInfo:
    os_name: str
    os_version: str
    architecture: str
    python_version: str
    cpu_count: int
    total_ram_gb: float
    has_cuda: bool
    gpu_name: str | None
    vram_gb: float | None
    recommendations: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _total_ram_gb() -> float:
    """RAM totale en Go, sans dépendance externe."""
    # POSIX (Linux, macOS récents)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024**3), 1)
    except (ValueError, OSError, AttributeError):
        pass

    # Windows
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024**3), 1)
        except Exception:  # pragma: no cover - dépend de l'OS
            logger.debug("Impossible de lire la RAM sous Windows", exc_info=True)

    return 0.0


def _detect_gpu() -> tuple[bool, str | None, float | None]:
    """Retourne (cuda_disponible, nom_gpu, vram_go) via nvidia-smi si présent."""
    if not shutil.which("nvidia-smi"):
        return False, None, None
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return False, None, None

    first_line = output.splitlines()[0] if output else ""
    if "," not in first_line:
        return False, None, None
    name, _, memory = first_line.partition(",")
    try:
        vram_gb = round(float(memory.strip()) / 1024, 1)
    except ValueError:
        vram_gb = None
    return True, name.strip(), vram_gb


def _recommend(ram_gb: float, has_cuda: bool, vram_gb: float | None) -> dict[str, str]:
    """Recommande une taille de modèle LLM/STT selon la machine."""
    budget_gb = vram_gb if (has_cuda and vram_gb) else ram_gb

    if budget_gb >= 16:
        llm = "a 7B-8B instruct model (e.g. `ollama pull qwen2.5:7b-instruct`)"
        stt = "medium"
    elif budget_gb >= 8:
        llm = "a 3B-4B instruct model (e.g. `ollama pull qwen2.5:3b-instruct`)"
        stt = "small"
    elif budget_gb >= 4:
        llm = "a 1B-2B instruct model (e.g. `ollama pull qwen2.5:1.5b-instruct`)"
        stt = "base"
    else:
        llm = "a sub-1B instruct model; expect slow answers"
        stt = "tiny"

    return {
        "llm_model": llm,
        "stt_model": stt,
        "stt_device": "cuda" if has_cuda else "cpu",
        "stt_compute_type": "float16" if has_cuda else "int8",
        "note": (
            "No GPU detected: Liliana will run on CPU. It works, it is just "
            "slower — prefer the smaller models above."
            if not has_cuda
            else "GPU detected: STT and the LLM can run accelerated."
        ),
    }


def detect_hardware() -> HardwareInfo:
    """Inspecte la machine et propose une configuration adaptée."""
    has_cuda, gpu_name, vram_gb = _detect_gpu()
    ram_gb = _total_ram_gb()
    info = HardwareInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        total_ram_gb=ram_gb,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        recommendations=_recommend(ram_gb, has_cuda, vram_gb),
    )
    logger.info(
        "Hardware: %s %s, %s CPU, %.1f GB RAM, GPU=%s",
        info.os_name,
        info.architecture,
        info.cpu_count,
        info.total_ram_gb,
        info.gpu_name or "none",
    )
    return info


def resolve_stt_device(configured_device: str, configured_compute: str) -> tuple[str, str]:
    """Résout ``auto`` en valeurs concrètes pour faster-whisper."""
    if configured_device != "auto" and configured_compute != "auto":
        return configured_device, configured_compute

    has_cuda, _, _ = _detect_gpu()
    device = configured_device if configured_device != "auto" else ("cuda" if has_cuda else "cpu")
    if configured_compute != "auto":
        compute = configured_compute
    else:
        compute = "float16" if device == "cuda" else "int8"
    return device, compute
