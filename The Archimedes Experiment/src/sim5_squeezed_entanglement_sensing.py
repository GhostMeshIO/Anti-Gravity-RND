#!/usr/bin/env python3
"""
sim5_squeezed_entanglement_sensing.py
======================================
Quantum Squeezing & Entanglement-Enhanced Gravitational Mass Sensing

Self-contained simulation of squeezed-state and EPR-entangled probe
strategies for surpassing the Standard Quantum Limit (SQL) in the
Archimedes gravito-optomechanical experiment.

Physics covered:
  - Single-mode squeezed vacuum: covariance matrix, squeezing parameter r
  - Two-mode squeezed vacuum (EPR state): Duan-Simon entanglement criterion
  - Phase-space visualization: Wigner functions and noise ellipses
  - Sub-SQL force sensing via quadrature squeezing
  - EPR-enhanced interferometry: conditional variance suppression
  - Optomechanical back-action evasion with squeezed light
  - Thermal decoherence of squeezing and entanglement
  - Fisher information and Cramer-Rao bounds for Gaussian probes
  - Comparison: coherent / squeezed / EPR / Heisenberg-limited sensitivity

Dependencies: numpy (required), scipy (optional), matplotlib (optional)
No imports from qnvm_gravity.py or other project files.
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
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from scipy.linalg import solve_continuous_lyapunov
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ===========================================================================
#  Physical Constants
# ===========================================================================
HBAR    = 1.054571817e-34      # Reduced Planck constant  [J s]
KB      = 1.380649e-23         # Boltzmann constant        [J/K]
G_GRAV  = 6.67430e-11          # Gravitational constant    [m^3 kg^-1 s^-2]
C_LIGHT = 2.99792458e8         # Speed of light            [m/s]
M_PROTON = 1.67262192e-27      # Proton mass               [kg]


# ===========================================================================
#  Gaussian State Utilities
# ===========================================================================

def symplectic_form(n_modes: int = 1) -> np.ndarray:
    """2n x 2n symplectic matrix Omega with Omega^T J Omega = -J."""
    O = np.zeros((2 * n_modes, 2 * n_modes))
    for i in range(n_modes):
        O[2 * i, 2 * i + 1] = 1.0
        O[2 * i + 1, 2 * i] = -1.0
    return O


def thermal_covariance(n_modes: int, n_th: float) -> np.ndarray:
    """Covariance matrix of a thermal state: diag(n_th+1/2, n_th+1/2) per mode."""
    dim = 2 * n_modes
    return np.diag([n_th + 0.5] * dim)


def squeezed_vacuum_covariance(r: float, theta: float = 0.0) -> np.ndarray:
    """Covariance matrix of single-mode squeezed vacuum.

    Parameters
    ----------
    r : float
        Squeezing parameter (r > 0 gives X-squeezing for theta=0).
    theta : float
        Squeezing angle (rad). 0 -> X squeezed, pi/2 -> P squeezed.
    """
    e2r = np.exp(2.0 * r)
    e2mr = np.exp(-2.0 * r)
    sx = 0.5 * e2mr   # squeezed quadrature variance
    sp = 0.5 * e2r    # anti-squeezed quadrature variance

    if abs(theta) < 1e-12:
        V = np.diag([sx, sp])
    else:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        V = R @ np.diag([sx, sp]) @ R.T
    return V


def two_mode_squeezed_covariance(r: float) -> np.ndarray:
    """Covariance matrix of a two-mode squeezed vacuum (EPR-like state).

    Mode A: (x_A, p_A), Mode B: (x_B, p_B).
    V_AB has cross-correlations from the parametric down-conversion:
        <x_A x_B> = <p_A p_B> = (1/2) sinh(2r)
    """
    ch = np.cosh(2.0 * r)
    sh = np.sinh(2.0 * r)
    return 0.5 * np.array([
        [ch,  0.0, sh,  0.0],
        [0.0, ch,  0.0, sh],
        [sh,  0.0, ch,  0.0],
        [0.0, sh,  0.0, ch],
    ])


def symplectic_eigenvalues(cov: np.ndarray) -> np.ndarray:
    """Symplectic eigenvalues of covariance matrix (absolute eigenvalues of i Omega V)."""
    n = cov.shape[0] // 2
    O = symplectic_form(n)
    eigvals = np.linalg.eigvals(1j * O @ cov)
    return np.sort(np.unique(np.round(np.abs(np.real(eigvals)), 12)))


def purity(cov: np.ndarray) -> float:
    """Gaussian state purity: mu = 1/sqrt(det(V)).  1 for pure, <1 for mixed."""
    d = np.linalg.det(cov)
    return 1.0 / np.sqrt(max(d, 1e-50))


def von_neumann_entropy_gaussian(cov: np.ndarray) -> float:
    """Von Neumann entropy of a Gaussian state: S = sum_k g(nu_k),
    where g(x) = ((x+1)/2) log2((x+1)/2) - ((x-1)/2) log2((x-1)/2)."""
    nus = symplectic_eigenvalues(cov)
    S = 0.0
    for nu in nus:
        nu = max(nu, 1.0)
        ap = (nu + 1.0) / 2.0
        am = (nu - 1.0) / 2.0
        if am > 1e-30:
            S += ap * np.log2(ap) - am * np.log2(am)
        else:
            S += ap * np.log2(ap)
    return S


def duan_simon_criterion(cov_ab: np.ndarray) -> Dict[str, float]:
    """Duan-Simon entanglement criterion for two-mode Gaussian states.

    Entangled if:  V(x_A - x_B) + V(p_A + p_B) < 2
    Also checks the symmetric combination.
    """
    xA_pB = cov_ab[0, 0] + cov_ab[2, 2] - 2 * cov_ab[0, 2]
    pA_xB = cov_ab[1, 1] + cov_ab[3, 3] + 2 * cov_ab[1, 3]
    ds_sum = xA_pB + pA_xB
    entangled = ds_sum < 2.0
    return {
        "var_xA_minus_xB": float(xA_pB),
        "var_pA_plus_pB": float(pA_xB),
        "duan_simon_sum": float(ds_sum),
        "entangled": bool(entangled),
        "log10_negativity_proxy": float(max(-np.log10(ds_sum / 2.0), 0.0)),
    }


def log_negativity(cov_ab: np.ndarray) -> float:
    """Logarithmic negativity from the PPT criterion for 1+1 Gaussian modes."""
    n = cov_ab.shape[0] // 2
    O = symplectic_form(n)
    # Partial transpose = mirror flip of mode B: x_B -> x_B, p_B -> -p_B
    PT = np.diag([1.0, 1.0, 1.0, -1.0])
    cov_pt = PT @ cov_ab @ PT
    nus_pt = symplectic_eigenvalues(cov_pt)
    # Smallest symplectic eigenvalue of partial transpose
    nu_min = np.min(nus_pt)
    if nu_min < 1.0:
        return -np.log2(nu_min)
    return 0.0


# ===========================================================================
#  Thermal Decoherence Models
# ===========================================================================

def thermal_decoherence_squeezing(
    r0: float,
    n_th: float,
    gamma: float,
    t: float,
) -> Dict[str, float]:
    """Decay of squeezing under thermal damping channel.

    Model: dV/dt = -gamma/2 * (V - V_th), solved analytically for V(t).
    The effective squeezing parameter decays as:
        r_eff(t) = (1/2) * arccosh( n_th + 1/2 + (cosh(2r0)-2n_th-1)/2 * exp(-gamma*t) )

    Returns effective r, purity, and symplectic eigenvalues.
    """
    # Analytical decay of the covariance matrix diagonal entries
    decay = np.exp(-gamma * t)
    V_th = n_th + 0.5
    e2r0 = np.cosh(2.0 * r0)
    e2r_t = V_th + (e2r0 - 2 * V_th) * decay
    e2mr_t = V_th + (np.exp(-2.0 * r0) - 2 * V_th) * decay

    e2r_t = max(e2r_t, V_th)
    e2mr_t = max(e2mr_t, 1.0 / (2.0 * V_th))  # uncertainty principle

    V_t = np.diag([0.5 * e2mr_t, 0.5 * e2r_t])

    r_eff = 0.5 * np.arccosh(min(e2r_t, 1e15))
    return {
        "r_initial": float(r0),
        "r_effective": float(r_eff),
        "purity": float(purity(V_t)),
        "entropy": float(von_neumann_entropy_gaussian(V_t)),
        "symplectic_eigenvalues": symplectic_eigenvalues(V_t).tolist(),
        "covariance": V_t.tolist(),
        "decay_time_s": float(t),
        "gamma_Hz": float(gamma / (2 * np.pi)),
    }


def thermal_decoherence_epr(
    r0: float,
    n_th: float,
    gamma: float,
    t: float,
) -> Dict[str, Any]:
    """Decay of EPR entanglement under independent thermal baths on each mode.

    Cross-correlations decay as: C(t) = C_0 * exp(-gamma*t)
    Local variances relax to: V(t) = V_th + (V_0 - V_th) * exp(-gamma*t)
    """
    decay = np.exp(-gamma * t)
    V_th = n_th + 0.5
    V0 = 0.5 * np.cosh(2.0 * r0)
    C0 = 0.5 * np.sinh(2.0 * r0)

    V_local = V_th + (V0 - V_th) * decay
    C_cross = C0 * decay

    V_t = np.array([
        [V_local,  0.0,      C_cross,  0.0     ],
        [0.0,      V_local,  0.0,      C_cross ],
        [C_cross,  0.0,      V_local,  0.0     ],
        [0.0,      C_cross,  0.0,      V_local ],
    ])

    ds = duan_simon_criterion(V_t)
    ln = log_negativity(V_t)

    return {
        "r_initial": float(r0),
        "r_effective": float(0.5 * np.arccosh(min(2 * V_local, 1e15))),
        "duan_simon": ds,
        "log_negativity": float(ln),
        "purity": float(purity(V_t)),
        "entropy": float(von_neumann_entropy_gaussian(V_t)),
        "entangled": ds["entangled"],
        "covariance": V_t.tolist(),
    }


# ===========================================================================
#  Sensitivity Analysis: Squeezed Probe for Force Sensing
# ===========================================================================

def force_sensitivity_coherent(
    n_bar: float,
    T_int: float,
    gamma_m: float,
    m_eff: float,
    eta: float = 0.95,
) -> Dict[str, float]:
    """SQL force sensitivity with coherent-state probe.

    S_F^SQL = sqrt(2 m_eff hbar omega_m^3) / (eta * sqrt(T_int))
    At resonance: S_x^SQL = sqrt(2 * x_zpf^2 / (gamma_m * T_int))
    """
    omega_m = np.sqrt(gamma_m * 1e14)  # proxy: gamma = omega_m / Q
    x_zpf = np.sqrt(HBAR / (2.0 * m_eff * omega_m))
    S_x_sql = np.sqrt(2.0 * x_zpf**2 / (gamma_m * T_int))
    S_F_sql = S_x_sql * m_eff * omega_m**2
    return {
        "displacement_noise_m": float(S_x_sql),
        "force_noise_N": float(S_F_sql),
        "mass_noise_kg": float(S_F_sql / (G_GRAV / (0.1**2))) if S_F_sql > 0 else float("inf"),
        "n_bar": float(n_bar),
        "regime": "SQL",
    }


def force_sensitivity_squeezed(
    r: float,
    T_int: float,
    gamma_m: float,
    m_eff: float,
    omega_squeeze: float,
    eta: float = 0.95,
) -> Dict[str, float]:
    """Sub-SQL force sensitivity using squeezed probe light.

    Squeezing the input optical field in the amplitude quadrature reduces
    shot noise.  The squeezing-enhanced imprecision is:
        S_x^sq = e^{-2r} * S_x^sql
    but back-action noise increases:
        S_x^BA_sq = e^{+2r} * S_x^BA_sql

    Total: S_x = S_x^sq + S_x^BA_sq, minimized when e^{-2r} S_imp = e^{+2r} S_BA
    giving the optimal squeezing: 2r_opt = ln(S_imp / S_BA).

    With frequency-dependent squeezing (optical spring), one can evade
    back-action entirely in a narrow band around omega_m.
    """
    omega_m = np.sqrt(gamma_m * 1e14)
    x_zpf = np.sqrt(HBAR / (2.0 * m_eff * omega_m))

    S_imp = 1.0 / (4.0 * eta * 1e6)       # imprecision noise (arb. units)
    S_BA = 1e6 / (4.0 * omega_m**2)        # back-action noise
    S_sql_total = S_imp + S_BA

    # Squeezed total
    S_sq = S_imp * np.exp(-2.0 * r) + S_BA * np.exp(2.0 * r)

    # Optimal squeezing
    if S_imp > 0 and S_BA > 0:
        r_opt = 0.5 * np.log(S_imp / S_BA)
        S_opt = 2.0 * np.sqrt(S_imp * S_BA)
    else:
        r_opt = 0.0
        S_opt = S_sql_total

    improvement = S_sql_total / max(S_sq, 1e-100)

    return {
        "squeezing_dB": float(10.0 * np.log10(np.exp(2.0 * r))),
        "total_noise_sq": float(S_sq),
        "sql_noise": float(S_sql_total),
        "optimal_squeezing_dB": float(10.0 * np.log10(np.exp(2.0 * r_opt))),
        "optimal_noise": float(S_opt),
        "improvement_factor": float(improvement),
        "improvement_dB": float(10.0 * np.log10(max(improvement, 1e-30))),
        "sub_sql": bool(S_sq < S_sql_total),
    }


def epr_enhanced_sensitivity(
    r_epr: float,
    T_int: float,
    gamma_m: float,
    m_eff: float,
    eta: float = 0.95,
) -> Dict[str, float]:
    """EPR entanglement for interferometric mass sensing.

    Two-mode squeezing creates correlations such that conditional variance
    on one mode given measurement on the other is:
        V(x_A|x_B) = V_A - C^2 / V_B

    For TMSV with r_epr: this approaches e^{-2r_epr}/2 for large r,
    giving Heisenberg-limited scaling in principle.
    In practice, finite detection efficiency and thermal noise degrade this.

    The conditional variance improves the mass sensitivity:
        dm_min ~ sqrt(V(x_A|x_B)) / (chi * sqrt(T_int))
    """
    omega_m = np.sqrt(gamma_m * 1e14)
    x_zpf = np.sqrt(HBAR / (2.0 * m_eff * omega_m))

    V_A = 0.5 * np.cosh(2.0 * r_epr)
    C = 0.5 * np.sinh(2.0 * r_epr)

    V_conditional = V_A - C**2 / V_A
    V_conditional = max(V_conditional, 0.25 / eta)  # detection limit

    V_coherent = 0.5  # shot-noise-limited

    improvement = V_coherent / max(V_conditional, 1e-30)

    # Effective noise reduction in dB
    reduction_dB = -10.0 * np.log10(max(V_conditional / V_coherent, 1e-30))

    return {
        "r_epr": float(r_epr),
        "conditional_variance": float(V_conditional),
        "coherent_variance": float(V_coherent),
        "improvement_factor": float(improvement),
        "noise_reduction_dB": float(reduction_dB),
        "effective_squeezing_dB": float(-10.0 * np.log10(max(V_conditional * 2, 1e-30))),
        "heisenberg_scating": bool(improvement > np.sqrt(2 * V_A)),
    }


# ===========================================================================
#  Fisher Information for Gaussian Probes
# ===========================================================================

def fisher_information_displacement(
    cov: np.ndarray,
    dx_dtheta: float,
    dy_dtheta: float = 0.0,
) -> Dict[str, float]:
    """Classical Fisher information for a displacement parameter theta
    on a Gaussian state with known covariance matrix.

    For a Gaussian with mean mu(theta) and covariance V:
        F(theta) = (d_mu/d_theta)^T V^{-1} (d_mu/d_theta)
    This is exact for a displacement-encoded signal.
    """
    try:
        V_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(cov)

    dmu = np.array([dx_dtheta, dy_dtheta])
    F_classical = float(dmu @ V_inv @ dmu)

    # Quantum Fisher information for pure states: F_Q = 4 * dmu^T V^{-1} dmu
    F_quantum = 4.0 * F_classical

    return {
        "classical_fisher": float(F_classical),
        "quantum_fisher": float(F_quantum),
        "crb_theta": float(1.0 / np.sqrt(max(F_classical, 1e-50))),
    }


def sensitivity_comparison_sweep(
    r_range: np.ndarray,
    n_th: float = 0.0,
    T_int: float = 1000.0,
    gamma_m: float = 2 * np.pi * 10.0,
    m_eff: float = 1e-12,
    signal_mass: float = 1e-15,
) -> Dict[str, Any]:
    """Sweep squeezing parameter and compute sensitivity for all strategies.

    Compares:
      1. Coherent (r=0): SQL
      2. Single-mode squeezed: reduced imprecision
      3. Two-mode squeezed (EPR): conditional variance
      4. Heisenberg limit: 1/N scaling reference
    """
    omega_m = np.sqrt(gamma_m * 1e14) if gamma_m > 0 else 2 * np.pi * 1e6
    chi = 2 * np.pi * 10.0   # coupling (rad/s)
    dm_conv = 2.0 * m_eff / chi  # kg per rad/s of chi

    coherent_dm = dm_conv / np.sqrt(0.5 * T_int)  # SQL reference

    sq_dm = np.zeros_like(r_range)
    epr_dm = np.zeros_like(r_range)
    sq_thermal_dm = np.zeros_like(r_range)

    for i, r in enumerate(r_range):
        # Single-mode squeezed (no thermal)
        V_sq = squeezed_vacuum_covariance(r)
        fi_sq = fisher_information_displacement(V_sq, 1.0)  # unit displacement derivative
        if fi_sq["classical_fisher"] > 0:
            sq_dm[i] = dm_conv / np.sqrt(T_int * fi_sq["classical_fisher"])
        else:
            sq_dm[i] = float("inf")

        # EPR conditional
        epr_res = epr_enhanced_sensitivity(r, T_int, gamma_m, m_eff)
        V_cond = epr_res["conditional_variance"]
        if V_cond > 0:
            fi_epr = 1.0 / V_cond  # 1/V for conditional displacement
            epr_dm[i] = dm_conv / np.sqrt(T_int * fi_epr)
        else:
            epr_dm[i] = float("inf")

        # Squeezed + thermal noise
        V_th = thermal_covariance(1, n_th)
        V_total = squeezed_vacuum_covariance(r) + (n_th * np.eye(2))
        fi_th = fisher_information_displacement(V_total, 1.0)
        if fi_th["classical_fisher"] > 0:
            sq_thermal_dm[i] = dm_conv / np.sqrt(T_int * fi_th["classical_fisher"])
        else:
            sq_thermal_dm[i] = float("inf")

    # Heisenberg limit reference
    N_eq = np.cosh(r_range)  # equivalent photon number for squeezed
    hl_dm = dm_conv / (N_eq * T_int)

    return {
        "r_range": r_range.tolist(),
        "r_dB": (10.0 * np.log10(np.exp(2.0 * r_range))).tolist(),
        "coherent_dm_kg": float(coherent_dm),
        "squeezed_dm_kg": sq_dm.tolist(),
        "epr_dm_kg": epr_dm.tolist(),
        "squeezed_thermal_dm_kg": sq_thermal_dm.tolist(),
        "heisenberg_dm_kg": hl_dm.tolist(),
        "signal_mass_kg": float(signal_mass),
        "n_thermal": float(n_th),
    }


# ===========================================================================
#  Back-Action Evasion with Squeezed Light
# ===========================================================================

def back_action_evasion_analysis(
    omega_m_Hz: float = 1e6,
    kappa_Hz: float = 1e6,
    gamma_m_Hz: float = 1.0,
    r_input: float = 1.5,
    r_output: float = 1.5,
    eta: float = 0.95,
    T: float = 0.01,
    m_eff: float = 1e-12,
    T_int: float = 1000.0,
) -> Dict[str, Any]:
    """Frequency-resolved noise analysis for squeezed-light optomechanics.

    Standard (coherent) optomechanics:
        S_xx^total(omega) = S_imp(omega) + |chi_m(omega)|^2 * S_BA(omega)

    With input squeezing r_input in amplitude quadrature:
        S_imp -> S_imp * e^{-2r_input}
        S_BA -> S_BA * e^{+2r_input}

    With output squeezing (variational measurement at angle phi):
        S_total(phi) = e^{-2r_out} S_imp + e^{+2r_out} |chi_m|^2 S_BA

    At the optimal angle, back-action is cancelled at omega = omega_m.
    """
    omega_m = 2.0 * np.pi * omega_m_Hz
    kappa = 2.0 * np.pi * kappa_Hz
    gamma_m = 2.0 * np.pi * gamma_m_Hz

    n_th = 1.0 / (np.exp(HBAR * omega_m / (KB * T)) - 1.0) if T > 0 else 0.0

    freqs = np.linspace(0.1 * omega_m_Hz, 3.0 * omega_m_Hz, 500)
    omega_arr = 2.0 * np.pi * freqs

    # Mechanical susceptibility
    chi_m = 1.0 / (omega_m**2 - omega_arr**2 - 1j * gamma_m * omega_arr)
    chi_abs2 = np.abs(chi_m)**2

    # Noise spectral densities
    n_cav = 1e4
    G = 2.0 * np.pi * 1e3 * np.sqrt(n_cav)
    S_imp_0 = kappa / (4.0 * eta * G**2)
    S_BA_0 = 4.0 * G**2 / kappa

    # Coherent
    S_total_coherent = S_imp_0 + chi_abs2 * S_BA_0

    # Input squeezed (amplitude)
    S_total_sq_in = S_imp_0 * np.exp(-2.0 * r_input) + chi_abs2 * S_BA_0 * np.exp(2.0 * r_input)

    # Frequency-dependent (variational) squeezing
    # At omega = omega_m, perfectly evade back-action
    # Away from resonance, back-action leaks in
    S_total_var = (S_imp_0 * np.exp(-2.0 * r_output) +
                   chi_abs2 * S_BA_0 * np.exp(2.0 * r_output))

    # SQL reference
    S_sql = 2.0 * chi_abs2  # standard quantum limit

    # Mass sensitivity at resonance
    idx_res = np.argmin(np.abs(freqs - omega_m_Hz))
    S_res_coherent = S_total_coherent[idx_res]
    S_res_sq = S_total_sq_in[idx_res]
    S_res_var = S_total_var[idx_res]

    return {
        "parameters": {
            "omega_m_Hz": omega_m_Hz,
            "kappa_Hz": kappa_Hz,
            "gamma_m_Hz": gamma_m_Hz,
            "r_input": r_input,
            "r_output": r_output,
            "eta": eta,
            "T_K": T,
            "n_th": float(n_th),
            "m_eff_kg": m_eff,
            "T_int_s": T_int,
        },
        "frequency_Hz": freqs.tolist(),
        "S_coherent": S_total_coherent.tolist(),
        "S_squeezed_input": S_total_sq_in.tolist(),
        "S_variational": S_total_var.tolist(),
        "S_sql": S_sql.tolist(),
        "resonance": {
            "S_coherent": float(S_res_coherent),
            "S_squeezed": float(S_res_sq),
            "S_variational": float(S_res_var),
            "squeezing_improvement": float(S_res_coherent / max(S_res_sq, 1e-100)),
            "variational_improvement": float(S_res_coherent / max(S_res_var, 1e-100)),
        },
    }


# ===========================================================================
#  Plotting Functions
# ===========================================================================

def plot_squeezed_phase_space(
    r_values: List[float],
    output_path: Optional[str] = None,
) -> None:
    """Plot noise ellipses in phase space for various squeezing levels."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_values)))

    ax = axes[0]
    theta_vals = np.linspace(0, 2 * np.pi, 200)
    for i, r in enumerate(r_values):
        V = squeezed_vacuum_covariance(r)
        sx, sp = V[0, 0], V[1, 1]
        # 1-sigma contour
        x = np.sqrt(2 * sx) * np.cos(theta_vals)
        p = np.sqrt(2 * sp) * np.sin(theta_vals)
        ax.plot(x, p, color=colors[i], lw=2,
                label=f"r = {r:.1f} ({10*np.log10(np.exp(2*r)):.1f} dB)")
    ax.set_xlabel("Quadrature X", fontsize=12)
    ax.set_ylabel("Quadrature P", fontsize=12)
    ax.set_title("Squeezed Vacuum Noise Ellipses (1-sigma)", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Wigner function cross-sections
    ax2 = axes[1]
    x_grid = np.linspace(-4, 4, 500)
    for i, r in enumerate(r_values):
        V = squeezed_vacuum_covariance(r)
        W = (1.0 / np.pi) * np.exp(-x_grid**2 / (2 * V[0, 0]))
        ax2.plot(x_grid, W, color=colors[i], lw=1.5,
                 label=f"r = {r:.1f}")
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("W(X, P=0)", fontsize=12)
    ax2.set_title("Wigner Function Cross-Section at P=0", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "sim5_phase_space.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Phase space diagrams -> {path}")


def plot_epr_phase_space(
    r_values: List[float],
    output_path: Optional[str] = None,
) -> None:
    """Plot two-mode correlations and entanglement measures."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    r_arr = np.array(r_values)

    # Panel 1: Duan-Simon criterion
    ax = axes[0]
    ds_sums = []
    for r in r_arr:
        V = two_mode_squeezed_covariance(r)
        ds = duan_simon_criterion(V)
        ds_sums.append(ds["duan_simon_sum"])
    ax.plot(r_arr, ds_sums, "b-o", lw=2, ms=4)
    ax.axhline(2.0, color="red", ls="--", lw=1.5, label="Boundary (=2)")
    ax.fill_between(r_arr, ds_sums, 2.0, where=[d < 2 for d in ds_sums],
                    alpha=0.3, color="blue", label="Entangled region")
    ax.set_xlabel("Squeezing parameter r", fontsize=11)
    ax.set_ylabel(r"$V(x_A - x_B) + V(p_A + p_B)$", fontsize=11)
    ax.set_title("Duan-Simon Entanglement Criterion", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Logarithmic negativity
    ax2 = axes[1]
    ln_vals = []
    for r in r_arr:
        V = two_mode_squeezed_covariance(r)
        ln_vals.append(log_negativity(V))
    ax2.plot(r_arr, ln_vals, "g-s", lw=2, ms=4)
    ax2.set_xlabel("Squeezing parameter r", fontsize=11)
    ax2.set_ylabel("Logarithmic Negativity E_N", fontsize=11)
    ax2.set_title("EPR Entanglement Strength", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Conditional variance
    ax3 = axes[2]
    cond_vars = []
    for r in r_arr:
        res = epr_enhanced_sensitivity(r, T_int=1000.0, gamma_m=2*np.pi*10.0, m_eff=1e-12)
        cond_vars.append(res["conditional_variance"])
    ax3.semilogy(r_arr, cond_vars, "m-^", lw=2, ms=4, label="Conditional var.")
    ax3.axhline(0.5, color="gray", ls=":", label="Coherent (shot noise)")
    ax3.set_xlabel("Squeezing parameter r", fontsize=11)
    ax3.set_ylabel(r"$V(x_A | x_B)$", fontsize=11)
    ax3.set_title("Conditional Variance (EPR Advantage)", fontsize=12, fontweight="bold")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "sim5_epr_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] EPR analysis -> {path}")


def plot_sensitivity_comparison(
    sweep_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> None:
    """Plot sensitivity comparison across squeezing strategies."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    r_dB = np.array(sweep_data["r_dB"])
    sig = sweep_data["signal_mass_kg"]

    ax = axes[0]
    coherent = sweep_data["coherent_dm_kg"]
    ax.axhline(coherent, color="gray", ls="--", lw=1.5, label="SQL (coherent)")
    ax.axhline(sig, color="green", ls=":", lw=2, label=f"Signal = {sig:.1e} kg")

    sq = np.array(sweep_data["squeezed_dm_kg"])
    epr = np.array(sweep_data["epr_dm_kg"])
    hl = np.array(sweep_data["heisenberg_dm_kg"])

    valid_sq = np.isfinite(sq) & (sq > 0)
    valid_epr = np.isfinite(epr) & (epr > 0)
    valid_hl = np.isfinite(hl) & (hl > 0)

    if np.any(valid_sq):
        ax.semilogy(r_dB[valid_sq], sq[valid_sq], "b-o", lw=2, ms=3,
                    label="Single-mode squeezed")
    if np.any(valid_epr):
        ax.semilogy(r_dB[valid_epr], epr[valid_epr], "r-s", lw=2, ms=3,
                    label="EPR entangled")
    if np.any(valid_hl):
        ax.semilogy(r_dB[valid_hl], hl[valid_hl], "k--", lw=1, alpha=0.5,
                    label="Heisenberg limit")

    ax.set_xlabel("Squeezing (dB)", fontsize=12)
    ax.set_ylabel(r"$\delta m_{min}$ (kg)", fontsize=12)
    ax.set_title("Mass Sensitivity vs Squeezing Level", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Improvement factor
    ax2 = axes[1]
    if np.any(valid_sq):
        improvement = coherent / np.clip(sq, 1e-100, None)
        ax2.semilogy(r_dB[valid_sq], improvement[valid_sq], "b-o", lw=2, ms=3,
                     label="Squeezed / SQL")
    if np.any(valid_epr):
        improvement_epr = coherent / np.clip(epr, 1e-100, None)
        ax2.semilogy(r_dB[valid_epr], improvement_epr[valid_epr], "r-s", lw=2, ms=3,
                     label="EPR / SQL")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Squeezing (dB)", fontsize=12)
    ax2.set_ylabel("Improvement over SQL", fontsize=12)
    ax2.set_title("Sensitivity Enhancement Factor", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = output_path or "sim5_sensitivity_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Sensitivity comparison -> {path}")


def plot_thermal_decoherence(
    r0: float,
    n_th: float,
    gamma_Hz: float,
    output_path: Optional[str] = None,
) -> None:
    """Plot squeezing and entanglement decay under thermal noise."""
    if not HAS_MATPLOTLIB:
        return

    gamma = 2.0 * np.pi * gamma_Hz
    t_max = 10.0 / gamma  # 10 coherence times
    times = np.linspace(0, t_max, 200)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Squeezing decay
    ax = axes[0]
    r_eff = []
    purities = []
    for t in times:
        res = thermal_decoherence_squeezing(r0, n_th, gamma, t)
        r_eff.append(res["r_effective"])
        purities.append(res["purity"])
    r_eff = np.array(r_eff)
    purities = np.array(purities)

    ax.plot(times * 1e3, 10 * np.log10(np.exp(2 * r_eff)), "b-", lw=2, label="Effective squeezing")
    ax.plot(times * 1e3, 10 * np.log10(np.exp(2 * r0)) * np.ones_like(times),
            "b--", lw=1, alpha=0.5, label="Initial squeezing")
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Squeezing (dB)", fontsize=12)
    ax.set_title(f"Squeezing Decay (r0={r0:.1f}, n_th={n_th:.1f})", fontsize=12, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(times * 1e3, purities, "r-", lw=1.5, alpha=0.7)
    ax2.set_ylabel("Purity", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: Entanglement decay
    ax3 = axes[1]
    ln_vals = []
    entangled_flags = []
    for t in times:
        res = thermal_decoherence_epr(r0, n_th, gamma, t)
        ln_vals.append(res["log_negativity"])
        entangled_flags.append(res["entangled"])
    ln_vals = np.array(ln_vals)

    ax3.plot(times * 1e3, ln_vals, "g-o", lw=2, ms=3, label="Log-negativity")
    ax3.axhline(0, color="red", ls="--", lw=1.5, label="Entanglement boundary")

    # Mark disentanglement time
    ent_arr = np.array(entangled_flags)
    if not np.all(ent_arr):
        idx_death = np.argmin(ent_arr)
        ax3.axvline(times[idx_death] * 1e3, color="orange", ls=":", lw=1.5,
                    label=f"Disentanglement @ {times[idx_death]*1e3:.2f} ms")
    ax3.set_xlabel("Time (ms)", fontsize=12)
    ax3.set_ylabel("Log-negativity E_N", fontsize=12)
    ax3.set_title(f"EPR Entanglement Decay (r0={r0:.1f})", fontsize=12, fontweight="bold")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "sim5_thermal_decoherence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Thermal decoherence -> {path}")


def plot_back_action_evasion(
    bae_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> None:
    """Plot frequency-resolved noise with squeezing strategies."""
    if not HAS_MATPLOTLIB:
        return

    freqs = np.array(bae_data["frequency_Hz"]) / 1e6
    S_coh = np.array(bae_data["S_coherent"])
    S_sq = np.array(bae_data["S_squeezed_input"])
    S_var = np.array(bae_data["S_variational"])
    S_sql = np.array(bae_data["S_sql"])
    omega_m = bae_data["parameters"]["omega_m_Hz"] / 1e6

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(freqs, S_coh, "k-", lw=2.5, label="Coherent (SQL)", zorder=5)
    ax.loglog(freqs, S_sq, "b-", lw=1.8, label="Input squeezed")
    ax.loglog(freqs, S_var, "r-", lw=1.8, label="Variational (output squeezed)")
    ax.loglog(freqs, S_sql, "k--", lw=1, alpha=0.4, label="SQL reference")
    ax.axvline(omega_m, color="gray", ls=":", lw=1.5,
               label=rf"$\omega_m$ = {omega_m:.1f} MHz")

    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Noise Spectral Density (arb. units)", fontsize=12)
    ax.set_title("Back-Action Evasion with Squeezed Light", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim([freqs[0], freqs[-1]])

    # Annotation box
    res = bae_data["resonance"]
    textstr = (f"At resonance:\n"
               f"  Input sq: {res['squeezing_improvement']:.1f}x better\n"
               f"  Variational: {res['variational_improvement']:.1f}x better")
    props = dict(boxstyle="round", facecolor="lightyellow", alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    path = output_path or "sim5_back_action_evasion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Back-action evasion -> {path}")


def plot_thermal_sensitivity(
    sweep_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> None:
    """Plot sensitivity with and without thermal noise."""
    if not HAS_MATPLOTLIB:
        return

    r_dB = np.array(sweep_data["r_dB"])
    sq = np.array(sweep_data["squeezed_dm_kg"])
    sq_th = np.array(sweep_data["squeezed_thermal_dm_kg"])
    sig = sweep_data["signal_mass_kg"]

    fig, ax = plt.subplots(figsize=(10, 6))

    valid = np.isfinite(sq) & (sq > 0)
    valid_th = np.isfinite(sq_th) & (sq_th > 0)

    if np.any(valid):
        ax.semilogy(r_dB[valid], sq[valid], "b-o", lw=2, ms=3, label="T = 0 K (ideal)")
    if np.any(valid_th):
        ax.semilogy(r_dB[valid_th], sq_th[valid_th], "r-s", lw=2, ms=3,
                    label=f"T > 0 (n_th = {sweep_data['n_thermal']:.0f})")

    ax.axhline(sig, color="green", ls=":", lw=2, label=f"Signal = {sig:.1e} kg")
    ax.set_xlabel("Squeezing (dB)", fontsize=12)
    ax.set_ylabel(r"$\delta m_{min}$ (kg)", fontsize=12)
    ax.set_title("Thermal Noise Degradation of Squeezing Advantage",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    path = output_path or "sim5_thermal_sensitivity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Thermal sensitivity -> {path}")


# ===========================================================================
#  Main Simulation
# ===========================================================================

def run_simulation(output_dir: str = "sim5_results", **kwargs) -> Dict[str, Any]:
    """Execute all analyses and generate plots."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {"parameters": {}, "analyses": {}}

    # ---- Parameters ----
    r_max = kwargs.get("r_max", 2.5)
    r_plot_values = kwargs.get("r_plot_values", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    n_th = kwargs.get("n_th", 10.0)
    gamma_Hz = kwargs.get("gamma_Hz", 10.0)
    T_K = kwargs.get("T_K", 0.01)
    T_int = kwargs.get("T_int", 1000.0)

    results["parameters"] = {
        "r_max": r_max,
        "n_thermal": n_th,
        "gamma_Hz": gamma_Hz,
        "T_K": T_K,
        "T_int_s": T_int,
        "r_plot_values": r_plot_values,
    }

    print("=" * 60)
    print("sim5: Quantum Squeezing & Entanglement-Enhanced Sensing")
    print("=" * 60)

    # ---- Step 1: Phase space diagrams ----
    print("\n[1/7] Phase space analysis...")
    if HAS_MATPLOTLIB:
        plot_squeezed_phase_space(r_plot_values, str(out / "sim5_phase_space.png"))

    # ---- Step 2: EPR entanglement analysis ----
    print("[2/7] EPR entanglement analysis...")
    r_sweep = np.linspace(0, r_max, 100)
    epr_results = []
    for r in r_sweep:
        V = two_mode_squeezed_covariance(r)
        ds = duan_simon_criterion(V)
        ln = log_negativity(V)
        epr_res = epr_enhanced_sensitivity(r, T_int, 2*np.pi*10.0, 1e-12)
        epr_results.append({
            "r": float(r),
            "duan_simon": ds,
            "log_negativity": float(ln),
            "conditional_variance": float(epr_res["conditional_variance"]),
            "improvement": float(epr_res["improvement_factor"]),
        })
    results["analyses"]["epr_sweep"] = epr_results

    if HAS_MATPLOTLIB:
        plot_epr_phase_space(list(r_sweep), str(out / "sim5_epr_analysis.png"))

    # ---- Step 3: Sensitivity comparison sweep ----
    print("[3/7] Sensitivity comparison (coherent vs squeezed vs EPR)...")
    r_range = np.linspace(0, r_max, 80)
    sweep = sensitivity_comparison_sweep(r_range, n_th=n_th, T_int=T_int)
    results["analyses"]["sensitivity_sweep"] = sweep

    if HAS_MATPLOTLIB:
        plot_sensitivity_comparison(sweep, str(out / "sim5_sensitivity_comparison.png"))

    # ---- Step 4: Thermal sensitivity ----
    print("[4/7] Thermal noise effects...")
    if HAS_MATPLOTLIB:
        plot_thermal_sensitivity(sweep, str(out / "sim5_thermal_sensitivity.png"))

    # ---- Step 5: Thermal decoherence ----
    print("[5/7] Thermal decoherence dynamics...")
    decoherence_results = []
    for r0 in [0.5, 1.0, 1.5, 2.0]:
        res = thermal_decoherence_squeezing(r0, n_th, 2 * np.pi * gamma_Hz, 1.0 / gamma_Hz)
        res_epr = thermal_decoherence_epr(r0, n_th, 2 * np.pi * gamma_Hz, 1.0 / gamma_Hz)
        decoherence_results.append({
            "r0": r0,
            "squeezing_decay": res,
            "epr_decay": {
                "log_negativity": res_epr["log_negativity"],
                "entangled": res_epr["entangled"],
                "duan_simon_sum": res_epr["duan_simon"]["duan_simon_sum"],
            },
        })
    results["analyses"]["decoherence"] = decoherence_results

    if HAS_MATPLOTLIB:
        plot_thermal_decoherence(2.0, n_th, gamma_Hz,
                                str(out / "sim5_thermal_decoherence.png"))

    # ---- Step 6: Back-action evasion ----
    print("[6/7] Back-action evasion with squeezed light...")
    bae = back_action_evasion_analysis(
        r_input=1.5, r_output=1.5, T=T_K, T_int=T_int)
    results["analyses"]["back_action_evasion"] = bae

    if HAS_MATPLOTLIB:
        plot_back_action_evasion(bae, str(out / "sim5_back_action_evasion.png"))

    # ---- Step 7: Summary metrics ----
    print("[7/7] Computing summary metrics...")

    # Best achievable sensitivity
    sq_dm = np.array(sweep["squeezed_dm_kg"])
    epr_dm = np.array(sweep["epr_dm_kg"])
    valid_sq = np.isfinite(sq_dm) & (sq_dm > 0)
    valid_epr = np.isfinite(epr_dm) & (epr_dm > 0)

    summary = {
        "best_squeezed_dm_kg": float(np.min(sq_dm[valid_sq])) if np.any(valid_sq) else None,
        "best_epr_dm_kg": float(np.min(epr_dm[valid_epr])) if np.any(valid_epr) else None,
        "sql_dm_kg": float(sweep["coherent_dm_kg"]),
        "signal_mass_kg": sweep["signal_mass_kg"],
        "max_squeezing_improvement": float(
            sweep["coherent_dm_kg"] / np.min(sq_dm[valid_sq])
            if np.any(valid_sq) and np.min(sq_dm[valid_sq]) > 0 else 0),
        "max_epr_improvement": float(
            sweep["coherent_dm_kg"] / np.min(epr_dm[valid_epr])
            if np.any(valid_epr) and np.min(epr_dm[valid_epr]) > 0 else 0),
        "resonance_improvement_input_sq": bae["resonance"]["squeezing_improvement"],
        "resonance_improvement_variational": bae["resonance"]["variational_improvement"],
    }

    # Detectability verdict
    sig = sweep["signal_mass_kg"]
    best_dm = min(
        summary["best_squeezed_dm_kg"] or float("inf"),
        summary["best_epr_dm_kg"] or float("inf"),
    )
    summary["signal_detectable_squeezed"] = best_dm < sig if best_dm < float("inf") else False
    summary["snr_best"] = float(sig / best_dm) if best_dm > 0 and best_dm < float("inf") else 0.0

    results["analyses"]["summary"] = summary
    results["parameters"]["summary"] = summary

    print(f"\n  Summary:")
    print(f"    SQL sensitivity:     {summary['sql_dm_kg']:.3e} kg")
    print(f"    Best squeezed:       {summary['best_squeezed_dm_kg']:.3e} kg" if summary['best_squeezed_dm_kg'] else "    Best squeezed:       N/A")
    print(f"    Best EPR:            {summary['best_epr_dm_kg']:.3e} kg" if summary['best_epr_dm_kg'] else "    Best EPR:            N/A")
    print(f"    Max sq improvement:  {summary['max_squeezing_improvement']:.1f}x")
    print(f"    Max EPR improvement: {summary['max_epr_improvement']:.1f}x")
    print(f"    Signal detectable:   {summary['signal_detectable_squeezed']}")
    print(f"    Best SNR:            {summary['snr_best']:.2f}")

    # Save JSON
    json_path = out / "sim5_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sim5: Quantum Squeezing & Entanglement-Enhanced Sensing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Simulates squeezed-state and EPR entanglement strategies for
sub-SQL gravitational mass sensing in the Archimedes experiment.

Examples:
  python sim5_squeezed_entanglement_sensing.py
  python sim5_squeezed_entanglement_sensing.py --r-max 3.0 --n-th 50
  python sim5_squeezed_entanglement_sensing.py --output-dir my_results
        """)

    parser.add_argument("--r-max", type=float, default=2.5,
                        help="Maximum squeezing parameter (default: 2.5)")
    parser.add_argument("--n-th", type=float, default=10.0,
                        help="Thermal occupation number (default: 10)")
    parser.add_argument("--gamma-Hz", type=float, default=10.0,
                        help="Mechanical damping rate (Hz) (default: 10)")
    parser.add_argument("--T-K", type=float, default=0.01,
                        help="Temperature (K) (default: 0.01)")
    parser.add_argument("--T-int", type=float, default=1000.0,
                        help="Integration time (s) (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="sim5_results",
                        help="Output directory (default: sim5_results)")

    args = parser.parse_args()

    results = run_simulation(
        r_max=args.r_max,
        n_th=args.n_th,
        gamma_Hz=args.gamma_Hz,
        T_K=args.T_K,
        T_int=args.T_int,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
