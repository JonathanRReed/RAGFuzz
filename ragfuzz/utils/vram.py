"""VRAM monitoring and throttling for local providers."""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class VRAMInfo:
    """Information about GPU VRAM."""

    total_mb: int
    free_mb: int
    used_mb: int
    gpu_count: int


class VRAMMonitor:
    """Monitor GPU VRAM for NVIDIA and Apple Silicon."""

    def __init__(self, threshold_mb: int = 1024):
        """Initialize VRAM monitor.

        Args:
            threshold_mb: Minimum free VRAM threshold in MB.
        """
        self.threshold_mb = threshold_mb
        self._platform = self._detect_platform()
        self._last_check: VRAMInfo | None = None

    def _detect_platform(self) -> str:
        """Detect current platform.

        Returns:
            Platform identifier: "nvidia", "mac", or "none".
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "nvidia"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and "Apple" in result.stdout:
                import logging

                logger = logging.getLogger("ragfuzz")
                logger.warning(
                    "VRAM monitoring on Apple Silicon is not fully supported. "
                    "Only system RAM is measured. GPU memory monitoring requires Metal APIs."
                )
                return "mac"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return "none"

    def get_vram_info(self) -> VRAMInfo:
        """Get current VRAM information.

        Returns:
            VRAMInfo object with current memory stats.
        """
        if self._platform == "nvidia":
            return self._get_nvidia_vram()
        elif self._platform == "mac":
            return self._get_mac_vram()
        else:
            return VRAMInfo(total_mb=0, free_mb=0, used_mb=0, gpu_count=0)

    def _get_nvidia_vram(self) -> VRAMInfo:
        """Get VRAM info from nvidia-smi.

        Returns:
            VRAMInfo object.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            lines = result.stdout.strip().split("\n")
            total_mb = 0
            free_mb = 0
            used_mb = 0

            for line in lines:
                parts = line.split(",")
                if len(parts) >= 3:
                    total_mb += int(parts[0].strip())
                    free_mb += int(parts[1].strip())
                    used_mb += int(parts[2].strip())

            self._last_check = VRAMInfo(
                total_mb=total_mb,
                free_mb=free_mb,
                used_mb=used_mb,
                gpu_count=len(lines),
            )
            return self._last_check

        except Exception:
            return VRAMInfo(total_mb=0, free_mb=0, used_mb=0, gpu_count=0)

    def _get_mac_vram(self) -> VRAMInfo:
        """Get unified memory info from system_profiler (Apple Silicon).

        Note: On Apple Silicon, unified memory is measured. This is system RAM,
        not GPU-specific memory. GPU memory monitoring requires Metal APIs.

        Returns:
            VRAMInfo object with system memory stats.
        """
        try:
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            page_size = 4096
            free_pages = 0
            inactive_pages = 0

            for line in result.stdout.split("\n"):
                if "Pages free:" in line:
                    free_pages = int(line.split(":")[1].strip().strip("."))
                elif "Pages inactive:" in line:
                    inactive_pages = int(line.split(":")[1].strip().strip("."))

            free_mb = (free_pages + inactive_pages) * page_size // (1024 * 1024)

            result2 = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            total_mb = int(result2.stdout.strip()) // (1024 * 1024)
            used_mb = total_mb - free_mb

            self._last_check = VRAMInfo(
                total_mb=total_mb,
                free_mb=free_mb,
                used_mb=used_mb,
                gpu_count=1,
            )
            return self._last_check

        except Exception:
            return VRAMInfo(total_mb=0, free_mb=0, used_mb=0, gpu_count=0)

    def should_throttle(self) -> bool:
        """Check if we should throttle based on VRAM.

        Returns:
            True if VRAM is below threshold.
        """
        vram = self.get_vram_info()
        return vram.free_mb < self.threshold_mb

    async def wait_for_vram(self, check_interval: float = 1.0) -> None:
        """Wait until sufficient VRAM is available.

        Args:
            check_interval: Time between checks in seconds.
        """
        while self.should_throttle():
            _ = self.get_vram_info()  # noqa: F841
            await asyncio.sleep(check_interval)

    def get_stats(self) -> dict[str, Any]:
        """Get VRAM stats for reporting.

        Returns:
            Dictionary with VRAM statistics.
        """
        vram = self.get_vram_info()
        return {
            "platform": self._platform,
            "total_mb": vram.total_mb,
            "free_mb": vram.free_mb,
            "used_mb": vram.used_mb,
            "gpu_count": vram.gpu_count,
            "threshold_mb": self.threshold_mb,
            "should_throttle": self.should_throttle(),
        }
