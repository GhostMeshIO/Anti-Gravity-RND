#!/usr/bin/env python3
"""
sim3_quantum_metrology.py - Quantum Metrology with Cavity Optomechanics
=====================================================================
Science-grade simulation of Ramsey interferometry for detecting
mass shifts via a qubit coupled to a mechanical oscillator.

Hamiltonian (rotating frame, dispersive limit):
    H = omega_m a^dag a + chi Z a^dag a

Mass sensing mechanism:
    A mass perturbation delta_m shifts the oscillator frequency:
        delta_omega_m = -omega_m * delta_m / (2 m)
    In the dispersive optomechanical limit, the coupling depends on
    the oscillator frequency: chi = g_0^2 / omega_m. Therefore:
        delta_chi = -chi * delta_omega_m / omega_m = chi * delta_m / (2m)
    The qubit's Ramsey fringe depends on chi, so it shifts with mass.

    NOTE: A direct shift in omega_m does NOT change the Ramsey fringe
    for a Fock-diagonal oscillator state (thermal or any mixture of |n><n|).
    The omega_m dependence cancels because it shifts the energy of both
    qubit states equally. The chi-dependence mechanism is the correct
    transduction channel for dispersive mass sensing.

Thermal noise:
    At T = 10 mK and omega_m/2pi = 1 MHz: n_th ~ 208 phonons.
    Thermal occupation degrades sensitivity through increased phonon
    number fluctuations.

Known limits:
  SQL: delta_chi_min ~ 1/(sqrt(<n>) * T_int)
  HL:  delta_chi_min ~ 1/(<n> * T_int)

Physical constants and typical parameters:
  - chi ~ 2*pi*10 rad/s (state-of-art dispersive coupling)
  - omega_m ~ 2*pi*1 MHz (mechanical oscillator)
  - m ~ 1e-12 kg (nanomechanical resonator)
  - T ~ 10 mK (dilution refrigerator)
  - n_th ~ 208 at 1 MHz, 10 mK

Usage:
    python sim3_quantum_metrology.py
    python sim3_quantum_metrology.py --chi 628 --temperature 0.005
    python sim3_quantum_metrology.py --delta-m-signal 1e-15
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA 2018)
# ---------------------------------------------------------------------------
HBAR = 1.054571817e-34    # Reduced Planck constant [J*s]
K_B = 1.380649e-23        # Boltzmann constant [J/K]
C_LIGHT = 299792458.0     # Speed of light [m/s]

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from scipy.linalg import expm as _scipy_expm
    def _matrix_expm(A):
        return _scipy_expm(A)
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _matrix_expm_eig(A: np.ndarray) -> np.ndarray:
    """Fallback matrix exponential via eigendecomposition."""
    vals, vecs = np.linalg.eigh(A)
    return (vecs * np.exp(vals)[np.newaxis, :]) @ vecs.conj().T


def matrix_exponential(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential, preferring scipy."""
    if HAS_SCIPY:
        return _matrix_expm(A)
    return _matrix_expm_eig(A)


# ======================================================================
# 1. Build Optomechanical Hamiltonian (rotating frame)
# ======================================================================
def build_optomechanical_hamiltonian(
    omega_m: float,
    chi: float,
    N_max: int,
) -> np.ndarray:
    """
    Build the (2*N_max) x (2*N_max) optomechanical Hamiltonian in the
    qubit's rotating frame.

    The bare qubit frequency omega_q is omitted because Ramsey
    interferometry is performed in the rotating frame.

    H_rot = omega_m a^dag a + chi Z a^dag a

    Basis ordering: |q, n> -> index q * N_max + n
      q=0 (ground), q=1 (excited), n = 0..N_max-1

    Parameters
    ----------
    omega_m : float  - oscillator angular frequency [rad/s]
    chi : float      - dispersive optomechanical coupling [rad/s]
    N_max : int      - phonon Fock space truncation

    Returns
    -------
    H : np.ndarray, shape (2*N_max, 2*N_max), complex128
    """
    dim = 2 * N_max
    H = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(N_max):
        idx_g = n
        idx_e = N_max + n
        E_n = omega_m * n
        H[idx_g, idx_g] += E_n
        H[idx_e, idx_e] += E_n
        H[idx_g, idx_g] -= chi * n
        H[idx_e, idx_e] += chi * n
    return H


# ======================================================================
# 2. Thermal State Helpers
# ======================================================================
def thermal_occupation(omega_m: float, T: float) -> float:
    """
    Mean thermal phonon occupation: n_th = 1/(exp(hbar*omega_m/(k_B*T)) - 1).
    At T = 10 mK, omega_m/2pi = 1 MHz: n_th ~ 208.
    """
    if T <= 0:
        return 0.0
    x = HBAR * omega_m / (K_B * T)
    if x > 500:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def default_N_max(n_th: float, coverage: float = 0.999) -> int:
    """
    Auto-compute Fock space truncation to capture 'coverage' of the
    thermal distribution. P(n <= N_max) = 1 - lam^(N_max+1).
    """
    if n_th < 1e-10:
        return 5
    lam = n_th / (1.0 + n_th)
    if lam >= 1.0 or lam <= 0.0:
        return 1000
    n_max = int(np.ceil(np.log(1.0 - coverage) / np.log(lam))) - 1
    return max(n_max, 5)


def thermal_phonon_probabilities(n_th: float, N_max: int) -> np.ndarray:
    """
    Thermal phonon number probabilities p_n for n = 0..N_max-1.
    Vectorized for large N_max.
    """
    if n_th < 1e-15:
        p = np.zeros(N_max)
        p[0] = 1.0
        return p
    lam = n_th / (1.0 + n_th)
    n_arr = np.arange(N_max, dtype=np.float64)
    p = (1.0 - lam) * lam**n_arr
    p /= p.sum()
    return p


def thermal_state_oscillator(n_th: float, N_max: int) -> np.ndarray:
    """
    Thermal density matrix in the Fock basis: rho = sum_n p_n |n><n|.
    """
    p_n = thermal_phonon_probabilities(n_th, N_max)
    return np.diag(p_n).astype(np.complex128)


def prepare_ramsey_initial_state(
    omega_m: float,
    T: float,
    N_max: int,
) -> np.ndarray:
    """Prepare rho_0 = |+><+| (x) rho_thermal."""
    n_th = thermal_occupation(omega_m, T)
    rho_m = thermal_state_oscillator(n_th, N_max)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_plus = np.outer(plus, plus.conj())
    dim = 2 * N_max
    rho_0 = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            rho_0[i * N_max:(i + 1) * N_max, j * N_max:(j + 1) * N_max] += \
                rho_plus[i, j] * rho_m
    return rho_0


# ======================================================================
# 3. Ramsey Fringe (Exact, Analytical for Thermal States)
# ======================================================================
def ramsey_fringe_exact(
    omega_m: float,
    chi: float,
    wait_time: float,
    N_max: int,
    T: float = 0.01,
) -> float:
    """
    Compute P(1) for the Ramsey sequence with a thermal oscillator.

    Protocol (rotating frame):
        1. |+> (x) rho_thermal (first pi/2 already applied)
        2. Evolve under H_rot = omega_m a^dag a + chi Z a^dag a for tau
        3. Second pi/2 pulse
        4. P(1) = Tr[|e><e| x I_mech . rho_final]

    Analytical result (exact for Fock-diagonal states):
        P(1) = sum_n p_n sin^2(chi * n * tau)

    Key physics: P(1) is independent of omega_m for a Fock-diagonal state.
    The omega_m * n energy shifts |g,n> and |e,n> equally (no relative phase).
    Mass sensing works because chi depends on omega_m: chi = g_0^2/omega_m.
    """
    n_th = thermal_occupation(omega_m, T)
    p_n = thermal_phonon_probabilities(n_th, N_max)
    n_arr = np.arange(N_max, dtype=np.float64)
    return float(np.sum(p_n * np.sin(chi * n_arr * wait_time)**2))


def ramsey_fringe_scan(
    omega_m: float,
    chi: float,
    wait_time: float,
    N_max: int,
    T: float,
    chi_range: float,
    n_points: int = 200,
) -> Dict[str, Any]:
    """
    Scan chi values to produce the Ramsey fringe P(1) vs chi.
    The maximum slope occurs at the quadrature bias point P(1) ~ 0.5.
    """
    chi_values = np.linspace(max(chi - chi_range, 0), chi + chi_range, n_points)
    p1_values = np.zeros(n_points)

    for i, c in enumerate(chi_values):
        p1_values[i] = ramsey_fringe_exact(omega_m, c, wait_time, N_max, T)

    dp1_dchi = np.gradient(p1_values, chi_values)

    # Optimal operating point: maximum |dP(1)/d(chi)|
    # This is the QUADRATURE BIAS POINT where P(1) ~ 0.5.
    # Maximum Fisher information for a binary outcome is at P = 0.5,
    # NOT at 1/phi ~ 0.618.
    idx_max = np.argmax(np.abs(dp1_dchi))
    chi_opt = chi_values[idx_max]
    p1_opt = p1_values[idx_max]
    max_slope = abs(dp1_dchi[idx_max])

    contrast = (p1_values.max() - p1_values.min()) / max(p1_values.max() + p1_values.min(), 1e-30)

    return {
        'chi_values': chi_values.tolist(),
        'p1_values': p1_values.tolist(),
        'dp1_dchi_values': dp1_dchi.tolist(),
        'contrast': float(contrast),
        'optimal_point': {
            'chi': float(chi_opt),
            'p1': float(p1_opt),
            'slope': float(max_slope),
            'note': 'Quadrature bias point: maximum Fisher information at P(1) ~ 0.5',
        },
        'parameters': {
            'omega_m': omega_m,
            'chi_nominal': chi,
            'wait_time': wait_time,
            'N_max': N_max,
            'T': T,
        },
    }


# ======================================================================
# 4. Fisher Information
# ======================================================================
def compute_fisher_information(
    omega_m: float,
    chi: float,
    wait_time: float,
    N_max: int,
    T: float,
    chi_range: float,
    n_points: int = 500,
) -> Dict[str, Any]:
    """
    Classical Fisher information for estimating chi (and hence delta_m).

    FI(chi) = [dP(1)/d(chi)]^2 / [P(1) * (1 - P(1))]

    For delta_m: FI(delta_m) = FI(chi) * (chi / (2*m))^2

    Quantum Fisher information (unitary estimation):
        Generator G = Z a^dag a. For thermal state: <G>=0, <G^2>=<n^2>.
        F_Q = 4 * <n^2> * tau^2
    """
    chi_values = np.linspace(max(chi - chi_range, 0), chi + chi_range, n_points)
    p1_values = np.zeros(n_points)

    for i, c in enumerate(chi_values):
        p1_values[i] = ramsey_fringe_exact(omega_m, c, wait_time, N_max, T)

    dp1 = np.gradient(p1_values, chi_values)
    p_safe = np.clip(p1_values, 1e-15, 1.0 - 1e-15)
    fi_chi = dp1**2 / (p_safe * (1.0 - p_safe))

    # QFI
    n_th = thermal_occupation(omega_m, T)
    mean_n2 = 2.0 * n_th**2 + n_th
    F_Q = 4.0 * mean_n2 * wait_time**2

    # Find FI at the QUADRATURE POINT (P(1) ~ 0.5), not the global max.
    # The global max of FI diverges at P->0 or P->1 (Cramer-Rao breaks down),
    # but the physically meaningful operating point is P(1) ~ 0.5 where the
    # trade-off between slope and binomial variance is optimal.
    idx_quad = np.argmin(np.abs(p1_values - 0.5))
    fi_at_quad = float(fi_chi[idx_quad])
    chi_at_quad = float(chi_values[idx_quad])

    # Also report global max (for reference, but it's pathological)
    idx_max = np.argmax(fi_chi)

    return {
        'chi_values': chi_values.tolist(),
        'fi_chi_per_shot': fi_chi.tolist(),
        'fi_chi_max': float(fi_chi[idx_max]),       # pathological global max
        'chi_at_fi_max': float(chi_values[idx_max]),
        'p1_at_fi_max': float(p1_values[idx_max]),
        'fi_chi_at_quadrature': fi_at_quad,          # physically meaningful
        'chi_at_quadrature': chi_at_quad,
        'p1_at_quadrature': float(p1_values[idx_quad]),
        'p1_values': p1_values.tolist(),
        'quantum_fisher_per_shot': float(F_Q),
        'n_thermal': float(n_th),
        'mean_n': float(n_th),
        'mean_n2': float(mean_n2),
    }


# ======================================================================
# 5. Standard Quantum Limit and Heisenberg Limit
# ======================================================================
def sql_and_heisenberg_limit(
    n_phonon: float,
    T_int: float,
    n_th: float = 0.0,
) -> Dict[str, float]:
    """
    Analytical sensitivity limits for chi estimation.

    SQL (coherent state): delta_chi_min = sqrt(1 + 2*n_th) / (sqrt(n) * T)
    HL  (Fock state):     delta_chi_min = sqrt(1 + 2*n_th) / (n * T)

    Derivation:
    Phase accumulated per shot: phi = 2*chi*n*tau.
    For coherent state with <n> phonons, Var(n) = <n>, giving SQL.
    Thermal occupation adds Var(n) = n_th*(n_th+1), degrading by
    factor sqrt(1 + 2*n_th/<n>).
    """
    thermal_factor = np.sqrt(1.0 + 2.0 * n_th)

    if n_phonon > 0:
        sql = thermal_factor / (np.sqrt(n_phonon) * T_int)
        hl = thermal_factor / (n_phonon * T_int)
        sql_ideal = 1.0 / (np.sqrt(n_phonon) * T_int)
        hl_ideal = 1.0 / (n_phonon * T_int)
    else:
        sql = hl = sql_ideal = hl_ideal = float('inf')

    return {
        'sql_chi_min': sql,
        'hl_chi_min': hl,
        'sql_ideal': sql_ideal,
        'hl_ideal': hl_ideal,
        'thermal_factor': float(thermal_factor),
        'n_thermal': float(n_th),
    }


# ======================================================================
# 6. Mass Sensitivity Analysis
# ======================================================================
def mass_sensitivity_analysis(
    chi: float,
    omega_m: float,
    oscillator_mass: float,
    T: float,
    T_int: float,
    delta_m_signal: float,
    wait_time: float = 1e-6,
    N_max: int = 500,
) -> Dict[str, Any]:
    """
    Comprehensive mass shift sensitivity analysis.

    Mass -> omega_m shift -> chi shift -> Ramsey fringe shift.

    delta_chi = chi * delta_m / (2*m)
    delta_m = 2*m * delta_chi / chi
    """
    n_th = thermal_occupation(omega_m, T)
    n_phonon = max(n_th, 1.0)
    dm_from_dchi = 2.0 * oscillator_mass / chi

    limits = sql_and_heisenberg_limit(n_phonon, T_int, n_th)

    dm_sql = dm_from_dchi * limits['sql_chi_min']
    dm_hl = dm_from_dchi * limits['hl_chi_min']
    dm_sql_ideal = dm_from_dchi * limits['sql_ideal']

    cycle_time = 2.0 * wait_time + 1e-7
    n_shots = max(int(T_int / cycle_time), 1)

    snr_sql = abs(delta_m_signal) / dm_sql if dm_sql > 0 else float('inf')
    snr_hl = abs(delta_m_signal) / dm_hl if dm_hl > 0 else float('inf')

    gap_sql = -np.log10(max(snr_sql, 1e-300)) if snr_sql < 1 else 0.0
    gap_hl = -np.log10(max(snr_hl, 1e-300)) if snr_hl < 1 else 0.0

    # Exact Fisher info
    fi = compute_fisher_information(
        omega_m, chi, wait_time, N_max, T,
        chi_range=max(chi, 1.0), n_points=200)

    # Use FI at the quadrature point (P(1)~0.5), not the pathological global max.
    fi_operating = fi.get('fi_chi_at_quadrature', fi['fi_chi_max'])
    if fi_operating > 0 and n_shots > 0:
        dm_exact = dm_from_dchi / np.sqrt(n_shots * fi_operating)
        snr_exact = abs(delta_m_signal) / dm_exact if dm_exact > 0 else float('inf')
    else:
        dm_exact = float('inf')
        snr_exact = 0.0

    return {
        'thermal': {
            'n_th': float(n_th),
            'temperature_K': T,
            'omega_m_Hz': omega_m / (2 * np.pi),
        },
        'signal': {
            'delta_m_signal_kg': delta_m_signal,
            'delta_chi_signal_rad_s': chi * delta_m_signal / (2.0 * oscillator_mass),
        },
        'sensitivity_sql': {
            'delta_chi_min_rad_s': limits['sql_chi_min'],
            'delta_m_min_kg': dm_sql,
            'snr': float(snr_sql),
            'gap_oom': float(gap_sql),
            'detectable': bool(snr_sql >= 1.0),
        },
        'sensitivity_hl': {
            'delta_chi_min_rad_s': limits['hl_chi_min'],
            'delta_m_min_kg': dm_hl,
            'snr': float(snr_hl),
            'gap_oom': float(gap_hl),
            'detectable': bool(snr_hl >= 1.0),
        },
        'sensitivity_exact': {
            'fi_chi_max_per_shot': fi['fi_chi_max'],
            'delta_m_min_kg': dm_exact,
            'snr': float(snr_exact),
            'detectable': bool(snr_exact >= 1.0),
        },
        'fisher_info': fi,
        'conversion': {
            'dm_from_dchi': float(dm_from_dchi),
            'n_shots': n_shots,
            'cycle_time_s': cycle_time,
        },
        'overall_verdict': 'DETECTABLE' if snr_exact >= 1.0 else 'NOT DETECTABLE',
        'overall_gap_oom': float(gap_sql),
    }


# ======================================================================
# 7. Sensitivity vs Integration Time
# ======================================================================
def sensitivity_vs_integration_time(
    chi: float,
    omega_m: float,
    oscillator_mass: float,
    T: float,
    delta_m_signal: float,
    T_int_range: np.ndarray,
    N_max: int = 500,
    wait_time: float = 1e-6,
) -> Dict[str, Any]:
    """delta_m_min vs T_int: SQL (1/sqrt(T)) and HL (1/T) scaling."""
    n_th = thermal_occupation(omega_m, T)
    n_phonon = max(n_th, 1.0)
    dm_from_dchi = 2.0 * oscillator_mass / chi

    sql_dm = np.zeros_like(T_int_range, dtype=float)
    hl_dm = np.zeros_like(T_int_range, dtype=float)
    sql_ideal = np.zeros_like(T_int_range, dtype=float)

    for i, T_int in enumerate(T_int_range):
        lim = sql_and_heisenberg_limit(n_phonon, T_int, n_th)
        sql_dm[i] = dm_from_dchi * lim['sql_chi_min']
        hl_dm[i] = dm_from_dchi * lim['hl_chi_min']
        sql_ideal[i] = dm_from_dchi * lim['sql_ideal']

    return {
        'T_int_range': T_int_range.tolist(),
        'sql_dm_kg': sql_dm.tolist(),
        'hl_dm_kg': hl_dm.tolist(),
        'sql_dm_ideal_kg': sql_ideal.tolist(),
        'signal_dm_kg': delta_m_signal,
        'n_th': float(n_th),
        'n_phonon': float(n_phonon),
    }


# ======================================================================
# 8. Thermal Noise Analysis
# ======================================================================
def thermal_noise_analysis(
    chi: float,
    omega_m: float,
    oscillator_mass: float,
    T_int: float,
    delta_m_signal: float,
    T_range: np.ndarray,
    wait_time: float = 1e-6,
    N_max: int = 500,
) -> Dict[str, Any]:
    """
    Sensitivity vs temperature: thermal phonons degrade SNR.

    At each temperature, computes:
      1. SQL sensitivity (analytical approximation) — nearly T-independent
         for thermal probes because sqrt(1+2n_th)/sqrt(n_th) -> sqrt(2).
      2. Exact CRB sensitivity via Fisher information — correctly degrades
         with temperature because the Ramsey fringe washes out at high T.

    The SQL flatness is NOT a bug: for a thermal-state probe, the signal
    (proportional to <n>) and noise (proportional to sqrt(<n^2>-<n>^2))
    both grow with temperature, cancelling in the ratio.  However, the
    exact FI-based sensitivity does degrade because the fringe visibility
    (and hence the slope dP/dchi) decreases at higher T.
    """
    dm_from_dchi = 2.0 * oscillator_mass / chi
    sql_dm = np.zeros_like(T_range, dtype=float)
    crb_dm = np.zeros_like(T_range, dtype=float)
    n_th_arr = np.zeros_like(T_range, dtype=float)

    for i, T_val in enumerate(T_range):
        n_th = thermal_occupation(omega_m, T_val)
        n_th_arr[i] = n_th
        n_phonon = max(n_th, 1.0)

        # SQL (analytical)
        lim = sql_and_heisenberg_limit(n_phonon, T_int, n_th)
        sql_dm[i] = dm_from_dchi * lim['sql_chi_min']

        # Exact CRB via Fisher information at this temperature
        # Use temperature-specific N_max to capture thermal distribution
        N_T = max(N_max, default_N_max(n_th, 0.999))
        cycle_time = 2.0 * wait_time + 1e-7
        n_shots = max(int(T_int / cycle_time), 1)

        # Compute dP(1)/d(chi) at the operating point via finite difference
        eps = chi * 1e-6
        p1_nom = ramsey_fringe_exact(omega_m, chi, wait_time, N_T, T_val)
        p1_plus = ramsey_fringe_exact(omega_m, chi + eps, wait_time, N_T, T_val)
        dp1_dchi = (p1_plus - p1_nom) / eps

        # FI(chi) = (dP/dchi)^2 / [P(1-P)]
        p_clip = np.clip(p1_nom, 1e-15, 1.0 - 1e-15)
        fi_chi = dp1_dchi**2 / (p_clip * (1.0 - p_clip))

        if fi_chi > 0 and n_shots > 0:
            crb_dm[i] = dm_from_dchi / np.sqrt(n_shots * fi_chi)
        else:
            crb_dm[i] = float('inf')

    return {
        'T_range_K': T_range.tolist(),
        'n_th_arr': n_th_arr.tolist(),
        'sql_dm_kg': sql_dm.tolist(),
        'crb_dm_kg': crb_dm.tolist(),
        'signal_dm_kg': delta_m_signal,
    }


# ======================================================================
# 9. Optimal Operating Point Analysis
# ======================================================================
def optimal_operating_point_analysis(
    omega_m: float,
    chi: float,
    wait_time: float,
    N_max: int,
    T: float,
    chi_range: float,
    n_points: int = 500,
) -> Dict[str, Any]:
    """
    Find chi where Fisher information is maximized.

    For a binary outcome, FI = (dP/dtheta)^2 / [P(1-P)] is maximized
    at the steepest slope, which is the QUADRATURE BIAS POINT P(1)=0.5.

    Proof: by the Cramer-Rao bound, FI <= 4*P*(1-P) with equality when
    |dP/dtheta| = 2*sqrt(P*(1-P)), which is maximized at P = 0.5.

    This has NOTHING to do with 1/phi ~ 0.618.
    """
    fringe = ramsey_fringe_scan(
        omega_m, chi, wait_time, N_max, T,
        chi_range=chi_range, n_points=n_points)

    chi_vals = np.array(fringe['chi_values'])
    p1_vals = np.array(fringe['p1_values'])
    dp1_vals = np.array(fringe['dp1_dchi_values'])

    p_safe = np.clip(p1_vals, 1e-15, 1.0 - 1e-15)
    fi_vals = dp1_vals**2 / (p_safe * (1.0 - p_safe))

    # Report the quadrature point (P(1) ~ 0.5) as the optimal operating point.
    # The global max of FI diverges at P->0 or P->1 and is not physical.
    idx_quad = np.argmin(np.abs(p1_vals - 0.5))
    idx_max = np.argmax(fi_vals)

    return {
        'chi_values': chi_vals.tolist(),
        'p1_values': p1_vals.tolist(),
        'fi_values': fi_vals.tolist(),
        'optimal': {
            'chi': float(chi_vals[idx_quad]),
            'p1': float(p1_vals[idx_quad]),
            'fisher_info': float(fi_vals[idx_quad]),
            'is_quadrature_point': abs(p1_vals[idx_quad] - 0.5) < 0.05,
            'note': 'Maximum FI at quadrature point P(1)~0.5, NOT at 0.618',
        },
        'contrast': fringe['contrast'],
    }


# ======================================================================
# 10. Bayesian Mass Estimation
# ======================================================================
def bayesian_mass_estimation(
    true_delta_m: float = 1e-15,
    omega_m: float = 2 * np.pi * 1e6,
    chi: float = 2 * np.pi * 10.0,
    wait_time: float = 1e-6,
    N_max: int = 500,
    T: float = 0.01,
    oscillator_mass: float = 1e-12,
    n_measurements: int = 50,
    shots_per_measurement: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sequential Bayesian estimation of delta_m via Gaussian approximation.

    The mass shift changes chi -> chi*(1 + dm/(2m)). The Ramsey fringe
    P(1; dm) changes, and binomial measurements update the posterior.
    """
    rng = np.random.default_rng(seed)

    # Compute dP(1)/d(chi) at the nominal point
    eps_chi = chi * 1e-6
    p1_0 = ramsey_fringe_exact(omega_m, chi, wait_time, N_max, T)
    p1_plus = ramsey_fringe_exact(omega_m, chi + eps_chi, wait_time, N_max, T)
    dp1_dchi = (p1_plus - p1_0) / eps_chi

    # Chain rule: dP/d(dm) = dP/d(chi) * d(chi)/d(dm) = dP/d(chi) * chi/(2m)
    dm_conversion = chi / (2.0 * oscillator_mass)
    dp1_dm = dp1_dchi * dm_conversion

    # FI for delta_m per shot
    p1_clipped = np.clip(p1_0, 0.01, 0.99)
    fi_dm_per_shot = dp1_dm**2 / (p1_clipped * (1.0 - p1_clipped))

    # True P(1) at the signal
    chi_signal = chi * (1.0 + true_delta_m / (2.0 * oscillator_mass))
    p1_true = np.clip(
        ramsey_fringe_exact(omega_m, chi_signal, wait_time, N_max, T),
        0.01, 0.99)

    # Sequential update
    prior_mean = 0.0
    prior_std = max(abs(true_delta_m) * 10, 1e-14)

    means = [prior_mean]
    stds = [prior_std]
    m_grid = np.linspace(prior_mean - 5 * prior_std, prior_mean + 5 * prior_std, 200)
    posteriors = []

    for i in range(n_measurements):
        n_ones = rng.binomial(shots_per_measurement, p1_true)
        p1_measured = n_ones / shots_per_measurement

        # Linearized inversion: dm = (P1_meas - P1_0) / slope
        dm_meas = (p1_measured - p1_0) / dp1_dm if abs(dp1_dm) > 1e-50 else 0.0

        # Gaussian Bayesian update
        prior_prec = 1.0 / prior_std**2
        lik_prec = fi_dm_per_shot * shots_per_measurement
        post_prec = prior_prec + lik_prec
        post_std = 1.0 / np.sqrt(post_prec)
        post_mean = (prior_prec * prior_mean + lik_prec * dm_meas) / post_prec

        prior_mean = post_mean
        prior_std = post_std
        means.append(prior_mean)
        stds.append(prior_std)

        pdf = np.exp(-0.5 * ((m_grid - prior_mean) / prior_std)**2) / \
              (prior_std * np.sqrt(2 * np.pi))
        posteriors.append(pdf.tolist())

    error = abs(prior_mean - true_delta_m)
    snr = abs(true_delta_m) / prior_std if prior_std > 0 else 0.0

    return {
        'true_delta_m': true_delta_m,
        'p1_at_nominal': float(p1_0),
        'p1_at_signal': float(p1_true),
        'slope_dp1_dm': float(dp1_dm),
        'fi_dm_per_shot': float(fi_dm_per_shot),
        'posterior_means': [float(x) for x in means],
        'posterior_stds': [float(x) for x in stds],
        'm_grid': m_grid.tolist(),
        'posteriors': posteriors,
        'final_estimate': float(prior_mean),
        'final_uncertainty': float(prior_std),
        'error': float(error),
        'snr': float(snr),
        'detected': bool(snr >= 1.0),
    }


# ======================================================================
# 11. Feasibility Report
# ======================================================================
def feasibility_report(
    chi: float = 2 * np.pi * 10.0,
    omega_m: float = 2 * np.pi * 1e6,
    oscillator_mass: float = 1e-12,
    T: float = 0.01,
    T_int: float = 1.0,
    delta_m_signal: float = 1e-15,
    wait_time: float = 1e-6,
    N_max: int = 500,
) -> Dict[str, Any]:
    """Comprehensive feasibility assessment with SOTA comparison."""
    analysis = mass_sensitivity_analysis(
        chi=chi, omega_m=omega_m, oscillator_mass=oscillator_mass,
        T=T, T_int=T_int, delta_m_signal=delta_m_signal,
        wait_time=wait_time, N_max=N_max)

    sota_configs = {
        'Aspelmeyer_group': {
            'chi_Hz': 10.0, 'omega_m_Hz': 1e6, 'mass_kg': 1e-12, 'T_mK': 10,
        },
        'Teufel_NIST': {
            'chi_Hz': 5.0, 'omega_m_Hz': 10e6, 'mass_kg': 1e-15, 'T_mK': 20,
        },
        'ARCHIMEDES_proposal': {
            'chi_Hz': 0.001, 'omega_m_Hz': 1e3, 'mass_kg': 1e-9, 'T_mK': 100,
        },
    }

    sota_results = {}
    for name, p in sota_configs.items():
        chi_s = 2 * np.pi * p['chi_Hz']
        om_s = 2 * np.pi * p['omega_m_Hz']
        sa = mass_sensitivity_analysis(
            chi_s, om_s, p['mass_kg'], p['T_mK'] * 1e-3,
            T_int, delta_m_signal, N_max=N_max)
        sota_results[name] = {
            'params': p,
            'n_th': sa['thermal']['n_th'],
            'dm_min_sql_kg': sa['sensitivity_sql']['delta_m_min_kg'],
            'snr_sql': sa['sensitivity_sql']['snr'],
            'gap_oom': sa['sensitivity_sql']['gap_oom'],
            'detectable': sa['sensitivity_sql']['detectable'],
        }

    gap = analysis['overall_gap_oom']
    improvements = {}
    if gap > 0:
        improvements['chi_increase'] = 10.0**(gap / 2)
        improvements['T_int_increase'] = 10.0**gap
    else:
        improvements['chi_increase'] = 1.0
        improvements['T_int_increase'] = 1.0

    return {
        'primary': analysis,
        'sota': sota_results,
        'improvements': improvements,
        'verdict': analysis['overall_verdict'],
        'gap_summary': (f"NOT DETECTABLE: signal is {gap:.1f} OOM below SQL"
                        if gap > 0 else "DETECTABLE with current parameters"),
    }


# ======================================================================
# 12. Generate Plots
# ======================================================================
def generate_plots(
    fringe_data: Dict[str, Any],
    fisher_data: Dict[str, Any],
    sensitivity_time: Dict[str, Any],
    thermal_data: Dict[str, Any],
    bayesian_data: Optional[Dict[str, Any]],
    feasibility: Dict[str, Any],
    opt_point: Dict[str, Any],
    output_dir: str,
):
    """Generate all diagnostic plots."""
    if not HAS_PLOT:
        print("  [SKIP] matplotlib not available.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Ramsey Fringe ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    chi_v = np.array(fringe_data['chi_values']) / (2 * np.pi)
    p1 = np.array(fringe_data['p1_values'])
    ax.plot(chi_v, p1, 'b-', lw=1.5, label='$P(1)$')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='$P(1)=0.5$')
    chi_opt_hz = fringe_data['optimal_point']['chi'] / (2 * np.pi)
    ax.axvline(chi_opt_hz, color='r', ls='--', alpha=0.7,
               label=f'Optimal $\\chi/2\\pi$={chi_opt_hz:.1f} Hz')
    ax.set_xlabel('$\\chi / 2\\pi$ (Hz)')
    ax.set_ylabel('$P(1)$')
    ax.set_title(f'Optomechanical Ramsey Fringe (contrast={fringe_data["contrast"]:.3f})')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    dp1 = np.array(fringe_data['dp1_dchi_values'])
    ax2.plot(chi_v, np.abs(dp1), 'g-', lw=1.5)
    ax2.set_xlabel('$\\chi / 2\\pi$ (Hz)')
    ax2.set_ylabel('$|dP(1)/d\\chi|$')
    ax2.set_title('Fringe Slope (Proportional to Sensitivity)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'sim3_ramsey_fringe.png', dpi=150)
    plt.close()

    # --- 2. Fisher Information ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    fi = np.array(fisher_data['fi_chi_per_shot'])
    chi_fi = np.array(fisher_data['chi_values']) / (2 * np.pi)
    ax.semilogy(chi_fi, np.maximum(fi, 1e-20), 'b-', lw=1.0)
    ax.set_xlabel('$\\chi / 2\\pi$ (Hz)')
    ax.set_ylabel('FI($\\chi$) / shot')
    ax.set_title(f'Classical Fisher Information (QFI={fisher_data["quantum_fisher_per_shot"]:.2e})')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    p1_fi = np.array(fisher_data['p1_values'])
    ax2.plot(p1_fi, np.maximum(fi, 1e-20), 'r-', lw=1.0)
    ax2.axvline(0.5, color='gray', ls=':', label='$P(1)=0.5$ (quadrature)')
    ax2.set_xlabel('$P(1)$')
    ax2.set_ylabel('FI($\\chi$) / shot')
    ax2.set_title('FI vs $P(1)$ — Max at Quadrature, NOT $1/\\phi$')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'sim3_fisher_information.png', dpi=150)
    plt.close()

    # --- 3. Sensitivity vs Time ---
    fig, ax = plt.subplots(figsize=(8, 6))
    T_arr = np.array(sensitivity_time['T_int_range'])
    sql = np.array(sensitivity_time['sql_dm_kg'])
    hl = np.array(sensitivity_time['hl_dm_kg'])
    sql_id = np.array(sensitivity_time['sql_dm_ideal_kg'])
    sig = sensitivity_time['signal_dm_kg']

    v = sql > 0
    ax.loglog(T_arr[v], sql[v], 'b-', lw=2, label='SQL (thermal)')
    vi = sql_id > 0
    ax.loglog(T_arr[vi], sql_id[vi], 'b--', lw=1, alpha=0.5, label='SQL (ideal)')
    vh = hl > 0
    if np.any(vh):
        ax.loglog(T_arr[vh], hl[vh], 'r--', lw=2, label='Heisenberg limit')
    ax.axhline(sig, color='green', ls=':', lw=2, label=f'Signal={sig:.1e} kg')
    ax.set_xlabel('Integration Time $T$ (s)')
    ax.set_ylabel('$\\delta m_{min}$ (kg)')
    ax.set_title(f'Mass Sensitivity vs Time ($n_{{th}}$={sensitivity_time["n_th"]:.0f})')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(out / 'sim3_sensitivity_vs_time.png', dpi=150)
    plt.close()

    # --- 4. Sensitivity vs Temperature ---
    fig, ax1 = plt.subplots(figsize=(8, 6))
    T_mK = np.array(thermal_data['T_range_K']) * 1e3
    n_th_a = np.array(thermal_data['n_th_arr'])
    sql_T = np.array(thermal_data['sql_dm_kg'])
    sig_T = thermal_data['signal_dm_kg']

    # SQL (nearly T-independent for thermal probes — signal/noise cancel)
    vs = sql_T > 0
    ax1.semilogy(T_mK[vs], sql_T[vs], 'o-', color='steelblue', lw=1.5, ms=3,
                 label='SQL (analytical, ~T-indep.)')

    # Exact CRB via Fisher information (correctly degrades with T)
    if 'crb_dm_kg' in thermal_data:
        crb_T = np.array(thermal_data['crb_dm_kg'])
        vc = crb_T > 0
        ax1.semilogy(T_mK[vc], crb_T[vc], '^-', color='darkorange', lw=1.5, ms=3,
                     label='Exact CRB (Fisher info)')

    ax1.axhline(sig_T, color='green', ls=':', lw=2, label='Signal')
    ax1.set_xlabel('Temperature (mK)')
    ax1.set_ylabel('$\\delta m_{min}$ (kg)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.semilogy(T_mK, np.maximum(n_th_a, 1e-3), 's-', color='tab:red', lw=1.5, ms=3, alpha=0.7)
    ax2.set_ylabel('$n_{th}$', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title('Sensitivity vs Temperature — CRB degrades, SQL flat (thermal probe)')
    plt.tight_layout()
    plt.savefig(out / 'sim3_sensitivity_vs_temperature.png', dpi=150)
    plt.close()

    # --- 5. Bayesian ---
    if bayesian_data is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        means = np.array(bayesian_data['posterior_means'])
        stds = np.array(bayesian_data['posterior_stds'])
        tv = bayesian_data['true_delta_m']
        ax.plot(range(len(means)), means, 'b-', lw=1.5, label='Posterior mean')
        ax.fill_between(range(len(means)), means - stds, means + stds,
                        alpha=0.3, color='blue', label='±1σ')
        ax.axhline(tv, color='green', ls='--', lw=2, label=f'True={tv:.1e} kg')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('$\\delta m$ (kg)')
        ax.set_title(f'Bayesian Estimation (SNR={bayesian_data["snr"]:.2f})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        m_grid = np.array(bayesian_data['m_grid'])
        posts = bayesian_data['posteriors']
        n_snap = min(6, len(posts))
        idxs = np.linspace(0, len(posts) - 1, n_snap, dtype=int)
        cmap = plt.cm.viridis
        for j, idx in enumerate(idxs):
            ax2.plot(m_grid, posts[idx], color=cmap(j / max(n_snap - 1, 1)),
                     lw=1.0, alpha=0.7, label=f'Iter {idx}')
        ax2.axvline(tv, color='green', ls='--', lw=1.5)
        ax2.set_xlabel('$\\delta m$ (kg)')
        ax2.set_ylabel('Posterior density')
        ax2.set_title('Posterior Evolution')
        ax2.legend(loc='best', fontsize=7)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / 'sim3_bayesian_estimation.png', dpi=150)
        plt.close()

    # --- 6. Optimal Operating Point ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    chi_op = np.array(opt_point['chi_values']) / (2 * np.pi)
    p1_op = np.array(opt_point['p1_values'])
    fi_op = np.array(opt_point['fi_values'])

    axes[0].plot(chi_op, p1_op, 'b-', lw=1.5)
    axes[0].axhline(0.5, color='gray', ls=':', label='$P(1)=0.5$')
    axes[0].set_xlabel('$\\chi / 2\\pi$ (Hz)')
    axes[0].set_ylabel('$P(1)$')
    axes[0].set_title('Ramsey Fringe')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(p1_op, fi_op, 'r-', lw=1.5)
    axes[1].axvline(0.5, color='gray', ls=':', label='$P(1)=0.5$ (max FI)')
    axes[1].axvline(0.618, color='orange', ls='--', alpha=0.5,
                    label='$1/\\phi\\approx0.618$ (NOT special)')
    axes[1].set_xlabel('$P(1)$')
    axes[1].set_ylabel('FI / shot')
    axes[1].set_title('FI vs $P(1)$ — Max at Quadrature, NOT $1/\\phi$')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'sim3_optimal_operating_point.png', dpi=150)
    plt.close()

    # --- 7. Feasibility Gap ---
    fig, ax = plt.subplots(figsize=(10, 6))

    primary = feasibility['primary']
    sota = feasibility['sota']

    labels = ['Primary\nconfiguration']
    sql_vals = [primary['sensitivity_sql']['delta_m_min_kg']]
    signal_val = primary['signal']['delta_m_signal_kg']

    for name, data in sota.items():
        labels.append(name.replace('_', '\n'))
        sql_vals.append(data['dm_min_sql_kg'])

    x = np.arange(len(labels))
    width = 0.35

    sql_log = [max(np.log10(max(v, 1e-100)), -30) for v in sql_vals]
    signal_log = np.log10(max(signal_val, 1e-100))

    ax.bar(x - width / 2, sql_log, width, label='SQL $\\delta m_{min}$',
           color='steelblue', edgecolor='navy')
    ax.bar(x + width / 2, [signal_log] * len(labels), width,
           label=f'Signal={signal_val:.1e} kg',
           color='lightgreen', edgecolor='green', alpha=0.7)

    ax.set_ylabel('$\\log_{10}(\\delta m)$ (kg)')
    ax.set_title('Feasibility Gap Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    for i, sv in enumerate(sql_vals):
        g = primary['sensitivity_sql']['gap_oom'] if i == 0 else \
            list(sota.values())[i - 1]['gap_oom']
        txt = "DETECTABLE" if g <= 0 else f"NOT DETECTABLE\n({g:.1f} OOM gap)"
        clr = 'green' if g <= 0 else 'red'
        ax.annotate(txt, xy=(i, sql_log[i] + 0.5), ha='center', fontsize=7,
                    color=clr, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out / 'sim3_feasibility_gap.png', dpi=150)
    plt.close()

    print(f"  Plots saved to {out}/")


# ======================================================================
# 13. Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Quantum Metrology with Cavity Optomechanics — '
                    'Ramsey interferometry for mass shift detection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sensing mechanism:
  Mass shifts omega_m, which changes the dispersive coupling
  chi = g_0^2/omega_m. The qubit's Ramsey fringe depends on chi.

Examples:
  python sim3_quantum_metrology.py
  python sim3_quantum_metrology.py --chi 628 --temperature 0.005
  python sim3_quantum_metrology.py --delta-m-signal 1e-15 --T-int 1e4
        """)

    parser.add_argument('--chi', type=float, default=2 * np.pi * 10,
                        help='Dispersive coupling chi (rad/s). Default: 2*pi*10')
    parser.add_argument('--omega-m', type=float, default=2 * np.pi * 1e6,
                        help='Oscillator angular frequency (rad/s). Default: 2*pi*1e6')
    parser.add_argument('--N-max', type=int, default=0,
                        help='Phonon truncation (0=auto from n_th). Default: 0')
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='Temperature (K). Default: 0.01 (10 mK)')
    parser.add_argument('--wait-time', type=float, default=0.0,
                        help='Ramsey time tau (s). 0=auto: pi/(4*chi*n_th). Default: 0')
    parser.add_argument('--delta-m-signal', type=float, default=1e-15,
                        help='Mass shift to detect (kg). Default: 1e-15')
    parser.add_argument('--oscillator-mass', type=float, default=1e-12,
                        help='Oscillator mass (kg). Default: 1e-12')
    parser.add_argument('--T-int', type=float, default=1.0,
                        help='Integration time (s). Default: 1.0')
    parser.add_argument('--output-dir', type=str, default='sim3_results',
                        help='Output directory. Default: sim3_results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. Default: 42')
    parser.add_argument('--skip-bayesian', action='store_true',
                        help='Skip Bayesian estimation (faster)')

    args = parser.parse_args()
    np.random.seed(args.seed)

    # --- Derived quantities ---
    n_th = thermal_occupation(args.omega_m, args.temperature)
    n_phonon = max(n_th, 1.0)
    chi_hz = args.chi / (2 * np.pi)

    # Auto N_max
    if args.N_max <= 0:
        args.N_max = default_N_max(n_th, coverage=0.999)

    # Auto wait_time: quadrature point chi * n_th * tau = pi/4
    if args.wait_time <= 0:
        if n_th > 0 and args.chi > 0:
            args.wait_time = np.pi / (4.0 * args.chi * n_phonon)
        else:
            args.wait_time = 1e-6

    delta_chi_signal = args.chi * args.delta_m_signal / (2.0 * args.oscillator_mass)
    dm_from_dchi = 2.0 * args.oscillator_mass / args.chi

    print("=" * 72)
    print("QUANTUM METROLOGY WITH CAVITY OPTOMECHANICS")
    print("Ramsey Interferometry for Mass Shift Detection")
    print("=" * 72)
    print(f"  H (rotating frame): omega_m a^dag a + chi Z a^dag a")
    print(f"  Sensing chain: mass -> omega_m -> chi -> Ramsey fringe")
    print(f"  omega_m/2pi     = {args.omega_m / (2*np.pi):.3e} Hz")
    print(f"  chi/2pi         = {chi_hz:.3e} Hz")
    print(f"  Oscillator mass = {args.oscillator_mass:.3e} kg")
    print(f"  Temperature     = {args.temperature*1e3:.1f} mK")
    print(f"  n_th            = {n_th:.1f}")
    print(f"  N_max           = {args.N_max} (auto from n_th)")
    print(f"  tau             = {args.wait_time:.3e} s (auto: quadrature)")
    print(f"  T_int           = {args.T_int:.3e} s")
    print(f"  Signal dm       = {args.delta_m_signal:.3e} kg")
    print(f"  Signal dchi     = {delta_chi_signal:.3e} rad/s")
    print(f"  2m/chi          = {dm_from_dchi:.3e} kg/(rad/s)")
    print()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ---- 1. Ramsey fringe scan ----
    print("[1/8] Ramsey fringe scan...")
    scan_range = max(args.chi, 1.0)
    fringe_data = ramsey_fringe_scan(
        omega_m=args.omega_m, chi=args.chi, wait_time=args.wait_time,
        N_max=args.N_max, T=args.temperature, chi_range=scan_range, n_points=200)
    all_results['fringe'] = fringe_data
    print(f"  Contrast = {fringe_data['contrast']:.4f}")
    opt = fringe_data['optimal_point']
    print(f"  Optimal: chi/2pi={opt['chi']/(2*np.pi):.2f} Hz, P(1)={opt['p1']:.4f}")
    print(f"  {opt['note']}")

    # ---- 2. Fisher info ----
    print("\n[2/8] Fisher information...")
    fisher_data = compute_fisher_information(
        omega_m=args.omega_m, chi=args.chi, wait_time=args.wait_time,
        N_max=args.N_max, T=args.temperature, chi_range=scan_range, n_points=200)
    all_results['fisher'] = fisher_data
    print(f"  Max FI(chi)/shot = {fisher_data['fi_chi_max']:.3e}")
    print(f"  QFI/shot         = {fisher_data['quantum_fisher_per_shot']:.3e}")

    # ---- 3. SQL and HL ----
    print("\n[3/8] SQL and Heisenberg limits...")
    limits = sql_and_heisenberg_limit(n_phonon, args.T_int, n_th)
    all_results['limits'] = limits
    print(f"  SQL: dchi_min = {limits['sql_chi_min']:.3e} rad/s")
    print(f"  HL:  dchi_min = {limits['hl_chi_min']:.3e} rad/s")
    print(f"  Thermal degradation: {limits['thermal_factor']:.2f}x")

    # ---- 4. Mass sensitivity ----
    print("\n[4/8] Mass sensitivity analysis...")
    sensitivity = mass_sensitivity_analysis(
        chi=args.chi, omega_m=args.omega_m, oscillator_mass=args.oscillator_mass,
        T=args.temperature, T_int=args.T_int, delta_m_signal=args.delta_m_signal,
        wait_time=args.wait_time, N_max=args.N_max)
    all_results['sensitivity'] = sensitivity
    ss = sensitivity['sensitivity_sql']
    se = sensitivity['sensitivity_exact']
    print(f"  SQL:  dm_min={ss['delta_m_min_kg']:.3e} kg, SNR={ss['snr']:.2e}")
    print(f"  CRB:  dm_min={se['delta_m_min_kg']:.3e} kg, SNR={se['snr']:.2e}")
    print(f"  Verdict: {sensitivity['overall_verdict']}")

    if not ss['detectable']:
        print(f"\n  *** NOT DETECTABLE ***")
        print(f"  Signal is {ss['gap_oom']:.1f} OOM below the SQL noise floor.")

    # ---- 5. Sensitivity vs time ----
    print("\n[5/8] Sensitivity vs integration time...")
    T_int_range = np.logspace(-6, 8, 100)
    sens_time = sensitivity_vs_integration_time(
        chi=args.chi, omega_m=args.omega_m, oscillator_mass=args.oscillator_mass,
        T=args.temperature, delta_m_signal=args.delta_m_signal,
        T_int_range=T_int_range, N_max=args.N_max, wait_time=args.wait_time)
    all_results['sensitivity_vs_time'] = sens_time

    # ---- 6. Thermal noise ----
    print("\n[6/8] Thermal noise analysis...")
    T_range = np.linspace(0.001, 1.0, 100)
    therm_data = thermal_noise_analysis(
        chi=args.chi, omega_m=args.omega_m, oscillator_mass=args.oscillator_mass,
        T_int=args.T_int, delta_m_signal=args.delta_m_signal, T_range=T_range,
        wait_time=args.wait_time, N_max=args.N_max)
    all_results['thermal'] = therm_data

    # ---- 7. Optimal operating point ----
    print("\n[7/8] Optimal operating point...")
    opt_pt = optimal_operating_point_analysis(
        omega_m=args.omega_m, chi=args.chi, wait_time=args.wait_time,
        N_max=args.N_max, T=args.temperature, chi_range=scan_range, n_points=500)
    all_results['optimal_point'] = opt_pt
    print(f"  Optimal chi/2pi = {opt_pt['optimal']['chi']/(2*np.pi):.2f} Hz, "
          f"P(1)={opt_pt['optimal']['p1']:.4f}")
    print(f"  {opt_pt['optimal']['note']}")

    # ---- 8. Bayesian estimation ----
    bayesian_data = None
    if not args.skip_bayesian:
        print("\n[8/8] Bayesian mass estimation...")
        bayesian_data = bayesian_mass_estimation(
            true_delta_m=args.delta_m_signal,
            omega_m=args.omega_m, chi=args.chi,
            wait_time=args.wait_time, N_max=args.N_max,
            T=args.temperature, oscillator_mass=args.oscillator_mass,
            n_measurements=50, shots_per_measurement=1000, seed=args.seed)
        all_results['bayesian'] = bayesian_data
        print(f"  True: {bayesian_data['true_delta_m']:.3e} kg")
        print(f"  Est:  {bayesian_data['final_estimate']:.3e} kg")
        print(f"  Std:  {bayesian_data['final_uncertainty']:.3e} kg")
        print(f"  SNR={bayesian_data['snr']:.2f}, Detected: {bayesian_data['detected']}")
    else:
        print("\n[8/8] Bayesian estimation skipped.")

    # ================================================================
    # Feasibility Report
    # ================================================================
    print("\n" + "=" * 72)
    print("FEASIBILITY REPORT")
    print("=" * 72)

    report = feasibility_report(
        chi=args.chi, omega_m=args.omega_m, oscillator_mass=args.oscillator_mass,
        T=args.temperature, T_int=args.T_int, delta_m_signal=args.delta_m_signal,
        wait_time=args.wait_time, N_max=args.N_max)
    all_results['feasibility'] = report

    print(f"\n  VERDICT: {report['verdict']}")
    print(f"  {report['gap_summary']}")
    print(f"\n  Primary:")
    print(f"    SQL dm_min={report['primary']['sensitivity_sql']['delta_m_min_kg']:.3e} kg, "
          f"SNR={report['primary']['sensitivity_sql']['snr']:.2e}")

    for name, data in report['sota'].items():
        status = "DETECTABLE" if data['detectable'] else "NOT DETECTABLE"
        print(f"  {name}: n_th={data['n_th']:.0f}, "
              f"dm_min={data['dm_min_sql_kg']:.2e} kg, "
              f"gap={data['gap_oom']:.1f} OOM — {status}")

    if report['improvements']['chi_increase'] > 1:
        imp = report['improvements']
        print(f"\n  Required improvements:")
        print(f"    chi x{imp['chi_increase']:.1e}")
        print(f"    T_int x{imp['T_int_increase']:.1e}")

    # ---- Plots ----
    if HAS_PLOT:
        print("\nGenerating plots...")
        generate_plots(
            fringe_data=fringe_data, fisher_data=fisher_data,
            sensitivity_time=sens_time, thermal_data=therm_data,
            bayesian_data=bayesian_data, feasibility=report,
            opt_point=opt_pt, output_dir=args.output_dir)
    else:
        print("\nmatplotlib not available; skipping plots.")

    # ---- Save JSON ----
    def _ser(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_ser(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    with open(out / 'sim3_results.json', 'w') as f:
        json.dump(_ser(all_results), f, indent=2)
    print(f"\nResults saved to {out}/sim3_results.json")

    # ---- Final Summary ----
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"  Signal: delta_m = {args.delta_m_signal:.3e} kg")
    print(f"  SQL noise floor: dm_min = {sensitivity['sensitivity_sql']['delta_m_min_kg']:.3e} kg")
    print(f"  Exact CRB: dm_min = {sensitivity['sensitivity_exact']['delta_m_min_kg']:.3e} kg")

    if not sensitivity['sensitivity_sql']['detectable']:
        g = sensitivity['sensitivity_sql']['gap_oom']
        print(f"\n  *** NOT DETECTABLE ***")
        print(f"  Signal is {g:.1f} orders of magnitude below the SQL noise floor.")
        print(f"  chi/2pi = {chi_hz:.1e} Hz transduces dm={args.delta_m_signal:.1e} kg into")
        print(f"  dchi/2pi = {delta_chi_signal/(2*np.pi):.3e} Hz, too small vs thermal")
        print(f"  fluctuations (n_th = {n_th:.0f}).")
    else:
        print(f"\n  *** DETECTABLE ***")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()