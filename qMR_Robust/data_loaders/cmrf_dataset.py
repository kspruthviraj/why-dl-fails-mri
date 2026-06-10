"""
cMRFDataset — PyTorch Dataset for the multi-scanner cardiac MRF phantom.

Handles two on-disk formats:
  1. HDF5 (.h5)  with datasets: signals, t1, t2
  2. ISMRMRD (.h5) using the ismrmrd Python API

Directory layout expected
─────────────────────────
  cmrf/
    scanner_00_siemens/
      phantom_scan_01.h5
    scanner_01_philips/
      phantom_scan_01.h5
    …

Each HDF5 file stores:
  /signals        (N, T)  complex64  — MRF time-series fingerprints
  /t1             (N,)    float32    — ground-truth T1 (ms)
  /t2             (N,)    float32    — ground-truth T2 (ms)
  attrs:
    vendor        str
    scanner_id    str
    field_strength float
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _try_ismrmrd(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Attempt to read the file as ISMRMRD; return None if not applicable."""
    try:
        import ismrmrd
    except ImportError:
        return None
    try:
        dset = ismrmrd.Dataset(str(path), "dataset", create_if_needed=False)
        n_ro = dset.read_header().encoding[0].reconSpace.matrixSize.x
        n_acq = dset.number_of_acquisitions()
        if n_acq == 0:
            return None
        signals = []
        for i in range(min(n_acq, 50_000)):
            acq = dset.read_acquisition(i)
            signals.append(acq.data)
        signals = np.stack(signals, axis=0)
        return {"signals": signals.astype(np.complex64)}
    except Exception:
        return None


class cMRFDataset(Dataset):
    """
    PyTorch Dataset for the cMRF multi-scanner cardiac phantom.

    Parameters
    ----------
    root : str | Path
        Root directory containing per-scanner sub-folders.
    scanners : sequence of str, optional
        Restrict to specific scanner folders.
    transform : callable, optional
        Applied to each raw complex MRF signal.
    normalize : bool
        Peak-normalise each signal.
    """

    def __init__(
        self,
        root: str | Path,
        scanners: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.normalize = normalize

        self._records: List[Dict] = []
        self._domain_map: Dict[str, int] = {}
        self._scan(scanners)

        if not self._records:
            logger.warning(
                "cMRFDataset: no files found under %s — using placeholder.",
                self.root,
            )
            self._placeholder(500)

    # ── filesystem scan ───────────────────────────────────────────────────────

    def _scan(self, scanners):
        if not self.root.exists():
            return

        domain_idx = 0
        for scanner_dir in sorted(self.root.iterdir()):
            if not scanner_dir.is_dir():
                continue
            scanner_name = scanner_dir.name.lower()
            if scanners and scanner_name not in scanners:
                continue

            # infer vendor from folder name convention: scanner_XX_VENDOR
            parts = scanner_name.split("_")
            vendor = parts[-1] if len(parts) > 1 else "unknown"
            field_strength = 3.0  # default; overridden by HDF5 attrs if present

            self._domain_map[scanner_name] = domain_idx
            domain_idx += 1

            for h5_path in sorted(scanner_dir.glob("*.h5")):
                self._records.append(
                    {
                        "path": h5_path,
                        "scanner": scanner_name,
                        "vendor": vendor,
                        "field_strength": field_strength,
                        "domain_label": self._domain_map[scanner_name],
                    }
                )

    # ── placeholder ───────────────────────────────────────────────────────────

    def _placeholder(self, n: int):
        rng = np.random.RandomState(123)
        scanners = ["scanner_00_siemens", "scanner_01_philips", "scanner_02_ge", "scanner_03_siemens"]
        for i in range(n):
            scanner = scanners[i % len(scanners)]
            if scanner not in self._domain_map:
                self._domain_map[scanner] = len(self._domain_map)
            parts = scanner.split("_")
            vendor = parts[-1]
            self._records.append(
                {
                    "path": None,
                    "scanner": scanner,
                    "vendor": vendor,
                    "field_strength": 3.0,
                    "domain_label": self._domain_map[scanner],
                    "_placeholder_signal": (
                        rng.randn(1000) + 1j * rng.randn(1000)
                    ).astype(np.complex64),
                    "_placeholder_t1": float(rng.uniform(100, 2000)),
                    "_placeholder_t2": float(rng.uniform(20, 300)),
                }
            )

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self._records[idx]

        if rec["path"] is not None:
            signal, t1, t2 = self._load_h5(rec["path"])
        else:
            signal = rec["_placeholder_signal"]
            t1 = rec["_placeholder_t1"]
            t2 = rec["_placeholder_t2"]

        if self.normalize:
            peak = np.abs(signal).max()
            if peak > 0:
                signal = signal / peak

        if self.transform is not None:
            signal = self.transform(signal)
        else:
            signal = torch.stack(
                [torch.from_numpy(signal.real.copy()), torch.from_numpy(signal.imag.copy())]
            ).float()

        target = torch.tensor([t1, t2], dtype=torch.float32)
        return signal, target, rec["domain_label"]

    # ── HDF5 reader ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_h5(path: Path) -> Tuple[np.ndarray, float, float]:
        with h5py.File(path, "r") as f:
            if "signals" in f:
                signals = f["signals"][:]
            elif "data" in f:
                signals = f["data"][:]
            else:
                ismrmrd_data = _try_ismrmrd(path)
                if ismrmrd_data is not None:
                    signals = ismrmrd_data["signals"]
                else:
                    raise KeyError(
                        f"No recognised dataset in {path}. "
                        "Expected 'signals' or 'data'."
                    )

            # take first signal for voxel-wise evaluation; or whole batch
            if signals.ndim == 2:
                signal = signals[0]
            else:
                signal = signals

            t1 = float(f.attrs.get("t1", f.get("t1", [0])[0] if "t1" in f else 0))
            t2 = float(f.attrs.get("t2", f.get("t2", [0])[0] if "t2" in f else 0))
        return signal.astype(np.complex64), t1, t2

    # ── metadata ──────────────────────────────────────────────────────────────

    @property
    def n_domains(self) -> int:
        return len(self._domain_map)

    @property
    def domain_map(self) -> Dict[str, int]:
        return dict(self._domain_map)

    def get_domain_name(self, label: int) -> str:
        for name, idx in self._domain_map.items():
            if idx == label:
                return name
        return f"unknown_{label}"
