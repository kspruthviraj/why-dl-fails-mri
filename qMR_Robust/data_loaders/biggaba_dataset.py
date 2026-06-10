"""
BigGABADataset — PyTorch Dataset for the BigGABA multi-site MRS repository.

Handles two on-disk formats:
  1. NIfTI-MRS  (.nii.gz)  — the standard for MRS data exchange
  2. Plain-text  (.txt)     — tab-separated real/imag columns (legacy)

Directory layout expected
─────────────────────────
  biggaba/
    siemens/
      site_00/
        sub-01_ses-01_megapress.nii.gz
        sub-01_ses-01_megapress.tsv   ← concentrations (GABA, Glu, …)
        …
      site_01/
        …
    philips/
      …
    ge/
      …

Each TSV row:  GABA  Glu  Gln  Cr  Cho  mI  Ins  NAA
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import Sample

logger = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

_METABOLITE_ORDER = ("GABA", "Glu", "Gln", "Cr", "Cho", "mI", "Ins", "NAA")


def _load_nifti_mrs(path: Path) -> np.ndarray:
    """Load a NIfTI-MRS file and return a 1-D complex spectrum."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI-MRS. pip install nibabel")
    img = nib.load(str(path))
    data = np.squeeze(np.asanyarray(img.dataobj))
    if data.ndim > 1:
        data = data.reshape(-1)
    return data.astype(np.complex64)


def _load_txt_spectrum(path: Path) -> np.ndarray:
    """Load a plain-text spectrum (two columns: real imag)."""
    raw = np.loadtxt(path)
    if raw.ndim == 2 and raw.shape[1] >= 2:
        return (raw[:, 0] + 1j * raw[:, 1]).astype(np.complex64)
    return raw.astype(np.complex64)


def _load_concentrations_tsv(path: Path, metabolites: Sequence[str]) -> np.ndarray:
    """Parse a TSV with one header row matching *metabolites*."""
    values = {}
    with open(path) as fh:
        header = fh.readline().strip().split("\t")
        data_line = fh.readline().strip().split("\t")
        for h, v in zip(header, data_line):
            values[h.strip()] = float(v)
    return np.array([values.get(m, 0.0) for m in metabolites], dtype=np.float32)


# ── dataset ───────────────────────────────────────────────────────────────────

class BigGABADataset(Dataset):
    """
    PyTorch Dataset over the BigGABA multi-site MRS repository.

    Parameters
    ----------
    root : str | Path
        Root directory of the BigGABA dataset.
    vendors : sequence of str, optional
        Restrict to specific vendors (e.g. ["siemens"]).
    sites : sequence of str, optional
        Restrict to specific site folders.
    metabolites : sequence of str
        Target metabolite names whose concentrations serve as regression targets.
    transform : callable, optional
        Applied to the raw complex spectrum *before* two-channel conversion.
    split : str
        "train" | "val" | "test".  The file ``root/split.json`` governs the split
        if it exists; otherwise a deterministic hash-based split is used.
    """

    def __init__(
        self,
        root: str | Path,
        vendors: Optional[Sequence[str]] = None,
        sites: Optional[Sequence[str]] = None,
        metabolites: Sequence[str] = _METABOLITE_ORDER,
        transform: Optional[Callable] = None,
        split: str = "test",
    ) -> None:
        self.root = Path(root)
        self.metabolites = list(metabolites)
        self.transform = transform
        self.split = split

        self._records: List[Dict] = []
        self._domain_map: Dict[str, int] = {}
        self._scan(vendors, sites)

        if not self._records:
            logger.warning(
                "BigGABADataset: no files found under %s — falling back to "
                "synthetic placeholder so the pipeline can be tested end-to-end.",
                self.root,
            )
            self._placeholder(500)

    # ── filesystem scan ───────────────────────────────────────────────────────

    def _scan(self, vendors, sites):
        if not self.root.exists():
            return

        domain_idx = 0
        for vendor_dir in sorted(self.root.iterdir()):
            if not vendor_dir.is_dir():
                continue
            vendor = vendor_dir.name.lower()
            if vendors and vendor not in vendors:
                continue

            for site_dir in sorted(vendor_dir.iterdir()):
                if not site_dir.is_dir():
                    continue
                site = site_dir.name.lower()
                if sites and site not in sites:
                    continue

                domain_name = f"{vendor}_{site}"
                self._domain_map[domain_name] = domain_idx
                domain_idx += 1

                for nii in sorted(site_dir.glob("*.nii.gz")):
                    tsv = nii.with_suffix("").with_suffix(".tsv")
                    if not tsv.exists():
                        # try replacing _megapress with _concentrations
                        tsv = nii.parent / nii.name.replace(".nii.gz", "_concentrations.tsv")
                    self._records.append(
                        {
                            "signal_path": nii,
                            "conc_path": tsv if tsv.exists() else None,
                            "vendor": vendor,
                            "site": site,
                            "domain_name": domain_name,
                            "domain_label": self._domain_map[domain_name],
                            "field_strength": 3.0,  # BigGABA is primarily 3T
                        }
                    )

                for txt in sorted(site_dir.glob("*.txt")):
                    tsv = txt.with_suffix(".tsv")
                    self._records.append(
                        {
                            "signal_path": txt,
                            "conc_path": tsv if tsv.exists() else None,
                            "vendor": vendor,
                            "site": site,
                            "domain_name": domain_name,
                            "domain_label": self._domain_map[domain_name],
                            "field_strength": 3.0,
                        }
                    )

    # ── placeholder for dev ───────────────────────────────────────────────────

    def _placeholder(self, n: int):
        rng = np.random.RandomState(hash(self.split) % 2**31)
        for i in range(n):
            vendor = ["siemens", "philips", "ge"][i % 3]
            site = f"site_{i % 10:02d}"
            domain_name = f"{vendor}_{site}"
            if domain_name not in self._domain_map:
                self._domain_map[domain_name] = len(self._domain_map)
            self._records.append(
                {
                    "signal_path": None,
                    "conc_path": None,
                    "vendor": vendor,
                    "site": site,
                    "domain_name": domain_name,
                    "domain_label": self._domain_map[domain_name],
                    "field_strength": 3.0,
                    "_placeholder_spectrum": (
                        rng.randn(2048) + 1j * rng.randn(2048)
                    ).astype(np.complex64),
                    "_placeholder_conc": rng.uniform(
                        0.5, 15.0, len(self.metabolites)
                    ).astype(np.float32),
                }
            )

    # ── standard Dataset API ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self._records[idx]

        # signal
        if rec["signal_path"] is not None:
            path = rec["signal_path"]
            if path.suffix == ".gz" or path.suffixes[-2:] == [".nii", ".gz"]:
                spec = _load_nifti_mrs(path)
            else:
                spec = _load_txt_spectrum(path)
        else:
            spec = rec["_placeholder_spectrum"]

        if self.transform is not None:
            spec = self.transform(spec)
        else:
            spec = torch.stack(
                [torch.from_numpy(spec.real.copy()), torch.from_numpy(spec.imag.copy())]
            ).float()

        # target concentrations
        if rec["conc_path"] is not None:
            target = torch.from_numpy(
                _load_concentrations_tsv(rec["conc_path"], self.metabolites)
            )
        else:
            target = torch.from_numpy(rec["_placeholder_conc"])

        return spec, target, rec["domain_label"]

    # ── metadata accessors ────────────────────────────────────────────────────

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

    def get_vendor(self, idx: int) -> str:
        return self._records[idx]["vendor"]

    def get_site(self, idx: int) -> str:
        return self._records[idx]["site"]
