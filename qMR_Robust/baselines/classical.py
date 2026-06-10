"""
Classical Baselines for qMRI — mandatory credibility check.

Implements:
  1. Dictionary Matching (inner-product) for MRF — the gold standard since Ma et al. 2013.
  2. Basis-Set Linear Fitting for MRS — the approach used by LCModel/Osprey.

These serve as non-neural-network reference points. If DL fails under vendor
shift but dictionary matching survives, that is a critical finding.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy.optimize import nnls

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MRF: Dictionary Matching (Inner-Product)
# ──────────────────────────────────────────────────────────────────────────────

class DictionaryMatcher:
    """
    Classical MRF dictionary matching via normalized inner product.

    This is the original MRF quantification method (Ma et al., Nature 2013).
    For each observed signal, find the dictionary entry with the highest
    absolute inner product (i.e., cosine similarity after normalization).

    Parameters
    ----------
    dictionary : np.ndarray, shape (N_dict, T)
        Complex MRF dictionary. Each row is a simulated signal for a known
        (T1, T2, M0) combination.
    parameters : np.ndarray, shape (N_dict, 3)
        Ground truth parameters [T1, T2, M0] for each dictionary entry.
    """

    def __init__(self, dictionary: np.ndarray, parameters: np.ndarray):
        self.dictionary = dictionary.astype(np.complex64)
        self.parameters = parameters.astype(np.float32)

        norms = np.linalg.norm(self.dictionary, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.dictionary_normalized = self.dictionary / norms

    def match(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a single signal against the dictionary.

        Parameters
        ----------
        signal : np.ndarray, shape (T,) complex

        Returns
        -------
        best_params : np.ndarray, shape (3,) — [T1, T2, M0] of best match
        similarity : float — cosine similarity of best match
        """
        sig_norm = signal / (np.linalg.norm(signal) + 1e-8)
        similarities = np.abs(self.dictionary_normalized @ sig_norm.conj())
        best_idx = np.argmax(similarities)
        return self.parameters[best_idx], similarities[best_idx]

    def match_batch(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a batch of signals against the dictionary.

        Parameters
        ----------
        signals : np.ndarray, shape (N, T) complex

        Returns
        -------
        best_params : np.ndarray, shape (N, 3)
        similarities : np.ndarray, shape (N,)
        """
        norms = np.linalg.norm(signals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        signals_norm = signals / norms

        # Similarity matrix: (N_signals, N_dict)
        similarities = np.abs(signals_norm @ self.dictionary_normalized.conj().T)
        best_indices = np.argmax(similarities, axis=1)
        best_sims = similarities[np.arange(len(signals)), best_indices]
        best_params = self.parameters[best_indices]

        return best_params, best_sims

    def predict_torch(self, signal_tensor: Tensor) -> Tensor:
        """
        PyTorch-compatible interface for use in the evaluation pipeline.

        Parameters
        ----------
        signal_tensor : Tensor, shape (B, 2, L) — real/imag channels

        Returns
        -------
        Tensor, shape (B, 2) — [T1, T2] predictions
        """
        signal_np = signal_tensor[:, 0].cpu().numpy() + 1j * signal_tensor[:, 1].cpu().numpy()
        params, _ = self.match_batch(signal_np)
        return torch.from_numpy(params[:, :2].astype(np.float32))


def build_dictionary_matcher_from_hdf5(dict_path: str) -> DictionaryMatcher:
    """Load a pre-computed MRF dictionary from HDF5."""
    import h5py
    with h5py.File(dict_path, "r") as f:
        dictionary = f["signals"][:]
        parameters = f["parameters"][:]
    return DictionaryMatcher(dictionary, parameters)


# ──────────────────────────────────────────────────────────────────────────────
# MRS: Basis-Set Linear Fitting (LCModel / Osprey style)
# ──────────────────────────────────────────────────────────────────────────────

class BasisSetFitter:
    """
    Classical MRS basis-set linear fitting via Non-Negative Least Squares.

    This replicates the core algorithm of LCModel (Provencher, 1993) and
    Osprey (Oeltzschner, 2020): fit the observed spectrum as a non-negative
    linear combination of simulated metabolite basis functions plus a
    spline baseline.

    Parameters
    ----------
    basis_set : np.ndarray, shape (N_metabolites, N_points)
        Complex basis functions for each metabolite.
    metabolite_names : list of str
        Names matching the basis set rows.
    n_baseline_components : int
        Number of spline baseline components.
    """

    def __init__(
        self,
        basis_set: np.ndarray,
        metabolite_names: list,
        n_baseline_components: int = 10,
    ):
        self.basis = basis_set.astype(np.complex64)
        self.metabolite_names = list(metabolite_names)
        self.n_met = len(metabolite_names)
        self.n_baseline = n_baseline_components
        self.n_points = basis_set.shape[1]

        self._build_design_matrix()

    def _build_design_matrix(self):
        """Build the full design matrix [metabolites | baseline]."""
        met_matrix = self.basis.real.T  # (N_points, N_met)

        baseline = np.zeros((self.n_points, self.n_baseline), dtype=np.float32)
        for i in range(self.n_baseline):
            x = np.linspace(0, 1, self.n_points)
            baseline[:, i] = x ** i  # polynomial baseline

        self.design_matrix = np.concatenate([met_matrix, baseline], axis=1)

    def fit(self, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a single spectrum.

        Parameters
        ----------
        spectrum : np.ndarray, shape (N_points,) complex

        Returns
        -------
        concentrations : np.ndarray, shape (N_metabolites,)
        fit_spectrum : np.ndarray, shape (N_points,) — the fitted spectrum
        """
        obs = spectrum.real.astype(np.float64)

        coeffs, _ = nnls(self.design_matrix, obs)

        concentrations = coeffs[:self.n_met]
        fit_spectrum = self.design_matrix @ coeffs

        return concentrations.astype(np.float32), fit_spectrum.astype(np.float32)

    def fit_batch(self, spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a batch of spectra.

        Parameters
        ----------
        spectra : np.ndarray, shape (N, N_points) complex

        Returns
        -------
        concentrations : np.ndarray, shape (N, N_metabolites)
        fit_spectra : np.ndarray, shape (N, N_points)
        """
        n = len(spectra)
        all_conc = np.zeros((n, self.n_met), dtype=np.float32)
        all_fits = np.zeros((n, self.n_points), dtype=np.float32)

        for i in range(n):
            conc, fit_spec = self.fit(spectra[i])
            all_conc[i] = conc
            all_fits[i] = fit_spec

        return all_conc, all_fits

    def predict_torch(self, signal_tensor: Tensor) -> Tensor:
        """
        PyTorch-compatible interface for the evaluation pipeline.

        Parameters
        ----------
        signal_tensor : Tensor, shape (B, 2, L) — real/imag channels

        Returns
        -------
        Tensor, shape (B, N_metabolites) — concentration estimates
        """
        spec_np = signal_tensor[:, 0].cpu().numpy() + 1j * signal_tensor[:, 1].cpu().numpy()
        concentrations, _ = self.fit_batch(spec_np)
        return torch.from_numpy(concentrations)


class DictionaryMatcherBaseline:
    """Wrapper that makes DictionaryMatcher compatible with the evaluation API."""

    def __init__(self, matcher: DictionaryMatcher):
        self.matcher = matcher
        self._is_eval = True

    def eval(self):
        self._is_eval = True
        return self

    def to(self, device):
        return self

    def encode(self, x: Tensor) -> Tensor:
        return self.matcher.predict_torch(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.matcher.predict_torch(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class BasisSetFitterBaseline:
    """Wrapper that makes BasisSetFitter compatible with the evaluation API."""

    def __init__(self, fitter: BasisSetFitter):
        self.fitter = fitter
        self._is_eval = True

    def eval(self):
        self._is_eval = True
        return self

    def to(self, device):
        return self

    def encode(self, x: Tensor) -> Tensor:
        return self.fitter.predict_torch(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.fitter.predict_torch(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
