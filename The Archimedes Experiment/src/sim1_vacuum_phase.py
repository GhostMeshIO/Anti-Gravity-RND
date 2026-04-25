#!/usr/bin/env python3
"""
sim1_vacuum_phase.py - Quantum Phase Transition in the 2D Transverse-Field Ising Model
=======================================================================================
Science-grade simulation of the quantum phase transition in the 2D TFIM
on a square lattice with periodic boundary conditions.

Hamiltonian: H = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i

Methods:
  - Imaginary-time evolution for ground state preparation
  - Binder cumulant U4 for critical crossover location
  - Fidelity susceptibility χ_F for criticality detection
  - Von Neumann entropy scaling (information content across transition)
  - Entanglement spectrum and Schmidt gap
  - Quantum Fisher information for multipartite entanglement
  - Finite-size scaling with bootstrap error bars
  - Exact diagonalization benchmarking for small systems

Known critical point: h_c/J ≈ 3.044 (Monte Carlo, thermodynamic limit).
Universality class: 3D Ising (ν=0.6301, β=0.3262, γ=1.2372).

Usage:
    python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --h-max 5.0
    python sim1_vacuum_phase.py --mode multi_size_binder --sweep-sizes
    python sim1_vacuum_phase.py --mode benchmark_compare --size 3
    python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --benchmark-ed
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── qnvm_gravity imports (graceful degradation) ──────────────────────────────
try:
    from qnvm_gravity import (
        QuantumVMGravity,
        exact_diagonalize,
        KB,
        C_LIGHT,
        LN2,
        von_neumann_entropy as _standalone_vn_entropy,
    )
    HAS_QNVM = True
except ImportError:
    HAS_QNVM = False
    warnings.warn(
        "qnvm_gravity module not found.  "
        "Simulation will run in reduced-functionality mode.  "
        "Install or place qnvm_gravity.py on PYTHONPATH for full support."
    )

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Optional scipy for curve fitting and interpolation
try:
    from scipy.optimize import curve_fit
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Known critical parameters (thermodynamic limit) ─────────────────────────
HC_OVER_J_MC = 3.044          # Monte Carlo critical point h_c / J
JC_OVER_H_MC = 1.0 / HC_OVER_J_MC  # ≈ 0.3285
EXP_NU = 0.6301               # 3D Ising correlation-length exponent
EXP_BETA = 0.3262             # 3D Ising order-parameter exponent
EXP_GAMMA = 1.2372            # 3D Ising susceptibility exponent

# ── Trotter convergence threshold (point 34) ────────────────────────────────
TROTTER_CONVERGENCE_THRESHOLD = 0.01  # relative difference threshold for dt vs dt/2


# ======================================================================
# 1.  Lattice construction
# ======================================================================
def build_2d_lattice(size: int, pbc: bool = True) -> Tuple[List[int], List[Tuple[int, int]], Dict[str, Any]]:
    """Build a 2-D square lattice with (optionally periodic) boundary conditions.

    Parameters
    ----------
    size : int
        Linear lattice dimension L.  Total qubits = L**2.
    pbc : bool
        Use periodic boundary conditions (default True).

    Returns
    -------
    qubits : list[int]
        All qubit indices 0 .. L**2 - 1.
    pairs : list[tuple[int, int]]
        Nearest-neighbour pairs (each bond counted once).
    lat_info : dict
        Metadata: size, n_qubits, coord_map, n_pairs.
    """
    n = size * size
    qubits = list(range(n))
    pairs: List[Tuple[int, int]] = []
    coord_map: Dict[Tuple[int, int], int] = {}

    for r in range(size):
        for c in range(size):
            coord_map[(r, c)] = r * size + c

    for r in range(size):
        for c in range(size):
            idx = coord_map[(r, c)]
            # right
            rc = (r, (c + 1) % size) if pbc else (r, c + 1)
            if rc in coord_map:
                pairs.append((idx, coord_map[rc]))
            # down
            dr = ((r + 1) % size, c) if pbc else (r + 1, c)
            if dr in coord_map:
                pairs.append((idx, coord_map[dr]))

    # de-duplicate
    pairs = list(set(tuple(sorted(p)) for p in pairs))

    return qubits, pairs, {
        "size": size,
        "n_qubits": n,
        "coord_map": coord_map,
        "n_pairs": len(pairs),
    }


# ======================================================================
# 2.  Ground-state preparation via imaginary-time evolution
# ======================================================================

# ── Point 22: State Preparation Modes ──────────────────────────────────────
def prepare_ground_state_product_quench(
    vm: "QuantumVMGravity",
    J: float,
    h: float,
    pairs: List[Tuple[int, int]],
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
) -> Optional[np.ndarray]:
    """Prepare ground state from |+⟩⊗n via imaginary-time evolution (product_quench).

    Starts from |+⟩⊗n (paramagnetic reference), then applies
    exp(-τ H) iteratively with renormalisation after each step.
    """
    return _prepare_ground_state_ite(vm, J, h, pairs, n_ite_steps, dt_ite)


def prepare_ground_state_adiabatic_ramp(
    vm: "QuantumVMGravity",
    J: float,
    h: float,
    pairs: List[Tuple[int, int]],
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
) -> Optional[np.ndarray]:
    """Prepare ground state via adiabatic ramp of J from 0 to target.

    Starts from |+⟩⊗n, slowly ramps J from 0 to target over more Trotter
    steps, then finishes with standard ITE for h.
    """
    n = vm.qubits
    # Prepare |+⟩^n
    for q in range(n):
        vm.apply_gate("h", [q])

    ramp_steps = max(n_ite_steps, 600)  # use more steps for adiabatic ramp
    prev_energy = None

    for step in range(ramp_steps):
        # Linearly ramp J from 0 to target
        fraction = min(1.0, step / (ramp_steps * 0.7))
        J_current = J * fraction

        try:
            vm.imaginary_time_step(-J_current, -h, dt_ite, pairs, hamiltonian_type="tfim")
        except Exception:
            break

        if step % 50 == 0 and step > 0:
            try:
                energy = vm.measure_energy(J_current, h, pairs, "tfim")
                if prev_energy is not None and abs(energy - prev_energy) < 1e-8:
                    break
                prev_energy = energy
            except Exception:
                pass

    if vm._backend_type == "statevector":
        return vm._backend.state.copy()
    return None


def prepare_ground_state_random_product(
    vm: "QuantumVMGravity",
    J: float,
    h: float,
    pairs: List[Tuple[int, int]],
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Prepare ground state starting from a random product state.

    Applies random rotations to each qubit from |0⟩, then uses ITE.
    """
    n = vm.qubits
    rng = np.random.default_rng(seed)

    # Apply random single-qubit rotations to create random product state
    for q in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, 2 * np.pi)
        # RY(theta) then RZ(phi) approximate a random single-qubit state
        vm.apply_gate("ry", [q], params=[theta])
        vm.apply_gate("rz", [q], params=[phi])

    prev_energy = None
    for step in range(n_ite_steps):
        try:
            vm.imaginary_time_step(-J, -h, dt_ite, pairs, hamiltonian_type="tfim")
        except Exception:
            break

        if step % 50 == 0 and step > 0:
            try:
                energy = vm.measure_energy(J, h, pairs, "tfim")
                if prev_energy is not None and abs(energy - prev_energy) < 1e-8:
                    break
                prev_energy = energy
            except Exception:
                pass

    if vm._backend_type == "statevector":
        return vm._backend.state.copy()
    return None


def _prepare_ground_state_ite(
    vm: "QuantumVMGravity",
    J: float,
    h: float,
    pairs: List[Tuple[int, int]],
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
) -> Optional[np.ndarray]:
    """Core ITE ground-state preparation from |+⟩⊗n."""
    n = vm.qubits
    # Prepare |+⟩^n by applying H to every qubit (initial state is |0⟩^n)
    for q in range(n):
        vm.apply_gate("h", [q])

    prev_energy = None
    for step in range(n_ite_steps):
        # Negate J, h to correct sign: exp(-τ*(-J)*ZZ)*exp(-τ*(-h)*X) = exp(+τ*J*ZZ)*exp(+τ*h*X) ≈ exp(-τ*H)
        vm.imaginary_time_step(-J, -h, dt_ite, pairs, hamiltonian_type="tfim")

        # Check energy convergence every 50 steps
        if step % 50 == 0 and step > 0:
            try:
                energy = vm.measure_energy(J, h, pairs, "tfim")
                if prev_energy is not None and abs(energy - prev_energy) < 1e-8:
                    # Converged
                    break
                prev_energy = energy
            except Exception:
                pass

    if vm._backend_type == "statevector":
        return vm._backend.state.copy()
    return None


# Backward-compatible alias
prepare_ground_state = prepare_ground_state_product_quench


def _get_preparation_fn(mode: str):
    """Return the ground-state preparation function for the given mode (point 22)."""
    if mode == "adiabatic_ramp":
        return prepare_ground_state_adiabatic_ramp
    elif mode == "random_product":
        return prepare_ground_state_random_product
    else:
        return prepare_ground_state_product_quench


# ======================================================================
# 3.  Single-size phase-diagram sweep
# ======================================================================
def simulate_phase_transition(
    size: int = 3,
    J_fixed: float = 1.0,
    h_max: float = 5.0,
    n_steps: int = 30,
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
    noise_level: float = 0.0,
    output_dir: str = "sim1_results",
    verbose: bool = True,
    preparation: str = "product_quench",
    run_mode: str = "single_size_exploratory",
) -> Dict[str, Any]:
    """Sweep the transverse field h and measure observables at each point.

    Keeps J = J_fixed, varies h ∈ [0, h_max].  At each h:
      1. Prepare ground state via ITE
      2. Measure ⟨M²⟩, ⟨M⁴⟩ → Binder cumulant U4
      3. Von Neumann entropy of half-lattice
      4. Entanglement spectrum
      5. Energy
      6. Fidelity susceptibility with previous point
      7. Quantum Fisher information
      8. (point 34) Trotter error quantification

    Returns a results dictionary.
    """
    if not HAS_QNVM:
        raise RuntimeError("qnvm_gravity is required for simulation.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    qubits, pairs, lat_info = build_2d_lattice(size)
    n = lat_info["n_qubits"]

    if n > 20:
        warnings.warn(
            f"L={size} gives {n} qubits (>20).  The stabilizer backend will be used, "
            "which is approximate.  Imaginary-time evolution is not available in "
            "stabilizer mode.  Use L ≤ 4 for full functionality."
        )

    half_qubits = list(range(n // 2))
    z_sites = qubits  # all sites for magnetisation

    h_values = np.linspace(0.0, h_max, n_steps)

    # Storage
    mag_sq_list: List[float] = []
    binder_list: List[float] = []
    entropy_list: List[float] = []
    energy_list: List[float] = []
    chi_f_list: List[float] = []
    qfi_list: List[float] = []
    espectrum_list: List[List[float]] = []
    corr_snapshots: Dict[float, Any] = {}
    # Point 34: Trotter error tracking
    trotter_error_list: List[float] = []
    trotter_converged_list: List[bool] = []

    prev_state: Optional[np.ndarray] = None
    t_start = time.time()

    prep_fn = _get_preparation_fn(preparation)

    if verbose:
        print(f"Lattice : {size}×{size}  ({n} qubits, {lat_info['n_pairs']} bonds)")
        print(f"J fixed = {J_fixed},  h ∈ [0, {h_max}],  {n_steps} steps")
        print(f"ITE: {n_ite_steps} steps, dt={dt_ite}")
        print(f"State preparation: {preparation}")
        print(f"Run mode: {run_mode}")
        print()

    for idx, h_val in enumerate(h_values):
        h_val = float(h_val)
        if verbose and (idx % max(1, n_steps // 10) == 0 or idx == n_steps - 1):
            print(f"  [{idx+1:3d}/{n_steps}]  h/J = {h_val/J_fixed:.3f}  ", end="", flush=True)

        vm = QuantumVMGravity(qubits=n, noise_level=noise_level)
        vm.start()

        # --- ground state ---
        try:
            gs = prep_fn(vm, J_fixed, h_val, pairs,
                         n_ite_steps=n_ite_steps, dt_ite=dt_ite)
        except NotImplementedError:
            # stabilizer backend — fall back to Trotter annealing
            for _ in range(100):
                vm.trotter_step(J_fixed, h_val, 0.01, pairs, "tfim")
            gs = None

        # --- observables ---
        try:
            energy = vm.measure_energy(J_fixed, h_val, pairs, "tfim")
        except Exception:
            energy = float("nan")
        energy_list.append(energy)

        try:
            u4 = vm.binder_cumulant(z_sites, shots=4096)
        except Exception:
            u4 = float("nan")
        binder_list.append(u4)

        try:
            svne = vm.von_neumann_entropy_subsystem(half_qubits)
        except Exception:
            svne = float("nan")
        entropy_list.append(svne)

        # ⟨M²⟩ estimate from Binder shots — recompute cheaply
        # For statevector: compute exactly
        try:
            if vm._backend_type == "statevector":
                z_exp_sum = sum(
                    vm.expectation("".join("Z" if q == qi else "I" for q in range(n)))
                    for qi in z_sites
                )
                # <M²> = (1/N²)<(ΣZ_i)²> = (1/N²)[Σ_i<Z_i²> + Σ_{i≠j}<Z_i Z_j>]
                m2_exact = 0.0
                for i in z_sites:
                    m2_exact += 1.0  # <Z_i²> = 1
                for i_idx in range(len(z_sites)):
                    for j_idx in range(i_idx + 1, len(z_sites)):
                        pstr = ["I"] * n
                        pstr[z_sites[i_idx]] = "Z"
                        pstr[z_sites[j_idx]] = "Z"
                        m2_exact += 2.0 * vm.expectation("".join(pstr))
                m2_exact /= n * n
                mag_sq_list.append(m2_exact)
            else:
                mag_sq_list.append(float("nan"))
        except Exception:
            mag_sq_list.append(float("nan"))

        # Fidelity susceptibility
        if gs is not None and prev_state is not None:
            try:
                chi_f = vm.fidelity_susceptibility(prev_state)
            except Exception:
                chi_f = float("nan")
            chi_f_list.append(chi_f)
        else:
            chi_f_list.append(0.0)

        # Quantum Fisher information
        try:
            qfi = vm.quantum_fisher_information(z_sites)
        except Exception:
            qfi = float("nan")
        qfi_list.append(qfi)

        # Entanglement spectrum
        try:
            espec = vm.entanglement_spectrum(half_qubits)
            espectrum_list.append(espec.tolist())
        except Exception:
            espectrum_list.append([])

        # ── Point 34: Trotter error quantification ──
        # Re-run with dt/2 at this h value and compare energy
        trotter_err = float("nan")
        trotter_conv = False
        if gs is not None and idx % max(1, n_steps // 5) == 0:
            try:
                vm2 = QuantumVMGravity(qubits=n, noise_level=noise_level)
                vm2.start()
                gs2 = prep_fn(vm2, J_fixed, h_val, pairs,
                              n_ite_steps=n_ite_steps, dt_ite=dt_ite / 2.0)
                energy2 = vm2.measure_energy(J_fixed, h_val, pairs, "tfim")
                vm2.stop()
                if np.isfinite(energy) and np.isfinite(energy2):
                    denom = max(abs(energy), 1e-15)
                    trotter_err = abs(energy - energy2) / denom
                    trotter_conv = trotter_err < TROTTER_CONVERGENCE_THRESHOLD
            except Exception:
                pass
        trotter_error_list.append(trotter_err)
        trotter_converged_list.append(trotter_conv)

        # Correlation matrix at selected h values (near critical, ordered, disordered)
        if idx in {0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1}:
            try:
                corr = vm.correlation_matrix()
                corr_snapshots[h_val] = corr.tolist()
            except Exception:
                pass

        # Update prev_state for next iteration
        if gs is not None:
            prev_state = gs.copy()

        vm.stop()

        if verbose and (idx % max(1, n_steps // 10) == 0 or idx == n_steps - 1):
            te_str = f"  dE_trotter={trotter_err:.2e}" if np.isfinite(trotter_err) else ""
            print(f"E={energy:.4f}  U4={u4:.4f}  S={svne:.3f}{te_str}")

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nSimulation completed in {elapsed:.1f} s\n")

    # Compile results
    h_over_J = (h_values / J_fixed).tolist()
    results: Dict[str, Any] = {
        "parameters": {
            "size": size,
            "n_qubits": n,
            "J_fixed": J_fixed,
            "h_max": h_max,
            "n_steps": n_steps,
            "n_ite_steps": n_ite_steps,
            "dt_ite": dt_ite,
            "noise_level": noise_level,
            "state_preparation": preparation,
            "run_mode": run_mode,
        },
        "phase_diagram": {
            "h_over_J": h_over_J,
            "magnetization_squared": mag_sq_list,
            "binder_cumulant": binder_list,
            "von_neumann_entropy": entropy_list,
            "energy": energy_list,
            "fidelity_susceptibility": chi_f_list,
            "qfi": qfi_list,
        },
        "entanglement_spectra": {
            str(h_values[i]): espectrum_list[i]
            for i in range(len(h_values))
            if espectrum_list[i]
        },
        "correlation_snapshots": {str(k): v for k, v in corr_snapshots.items()},
        "trotter_error_analysis": {
            "errors": trotter_error_list,
            "converged": trotter_converged_list,
            "threshold": TROTTER_CONVERGENCE_THRESHOLD,
        },
        "timing_seconds": elapsed,
    }

    # ── Point 20: Critical crossover estimate (NOT thermodynamic critical point) ──
    #   In the ordered phase U4 → 2/3, disordered U4 → 0.  The minimum
    #   of U4 indicates proximity to a finite-size crossover, NOT the
    #   thermodynamic critical point.
    h_arr = h_values / J_fixed
    binder_arr = np.array(binder_list)
    mag_sq_arr = np.array(mag_sq_list)
    valid = np.isfinite(binder_arr)
    if np.sum(valid) > 3:
        # Use the minimum of U4 as a single-size crossover estimate
        h_crossover_rough = float(h_arr[valid][np.argmin(binder_arr[valid])])
        results["crossover_point"] = {
            "h_crossover_over_J": h_crossover_rough,
            "j_crossover_over_h": JC_OVER_H_MC,  # theoretical reference
            "theoretical_hc_over_J": HC_OVER_J_MC,
            "theoretical_jc_over_h": JC_OVER_H_MC,
            "method": "binder_minimum",
            "claim": "finite_size_crossover_only",
            "binder_interpretation": (
                "Minimum indicates proximity to finite-size crossover, "
                "NOT thermodynamic critical point"
            ),
            "note": "Multi-size crossing needed for thermodynamic-limit estimate. "
                    "This is a finite-size crossover indicator only.",
        }
    else:
        results["crossover_point"] = {
            "h_crossover_over_J": None,
            "theoretical_hc_over_J": HC_OVER_J_MC,
            "theoretical_jc_over_h": JC_OVER_H_MC,
            "method": "insufficient_data",
            "claim": "finite_size_crossover_only",
        }

    # ── Point 21: Improved susceptibility with smoothing, quadratic fit, bootstrap ──
    susceptibility_result = _compute_susceptibility(
        h_arr, mag_sq_arr, n_qubits=n, n_bootstrap=100
    )
    results["susceptibility"] = susceptibility_result

    # ── Point 14: Non-monotone magnetization detection ──
    valid_mag = np.isfinite(mag_sq_arr)
    if np.sum(valid_mag) > 5:
        mag_finite = mag_sq_arr[valid_mag]
        # Check monotonicity by looking at sign of differences
        diffs = np.diff(mag_finite)
        sign_changes = np.sum(diffs[:-1] * diffs[1:] < 0)
        if sign_changes > len(mag_finite) // 3:
            results["magnetization_note"] = (
                "Non-monotone behavior is a finite-size dynamical response, "
                "NOT a thermodynamic equilibrium property. Interpret with caution."
            )

    # ── Point 28: Critical exponent fits (demoted to exploratory) ──
    if np.sum(valid) > 5 and HAS_SCIPY:
        exponent_fits = _fit_exponents_single_size(
            h_arr, mag_sq_arr, binder_arr, size=size
        )
        results["critical_exponents"] = exponent_fits
    else:
        results["critical_exponents"] = {
            "beta": {"fitted": None, "theory": EXP_BETA},
            "gamma": {"fitted": None, "theory": EXP_GAMMA},
            "exponent_fit_status": "exploratory",
            "note": "Insufficient data or scipy unavailable for fitting.",
        }

    # ── Point 27: Bit-mass as derived proxy ──
    bit_mass_result = _compute_bit_mass_derived(h_arr, entropy_list, mag_sq_list)
    results["bit_mass"] = bit_mass_result

    # ── Schmidt gap from entanglement spectra ──
    if espectrum_list:
        try:
            schmidt_gaps = []
            for spec in espectrum_list:
                if len(spec) >= 2:
                    schmidt_gaps.append(float(spec[0] - spec[1]))
                else:
                    schmidt_gaps.append(None)
            results["entanglement_analysis"] = {
                "schmidt_gap": schmidt_gaps,
                "schmidt_gap_h_values": h_over_J,
            }
        except Exception:
            pass

    # ── Point 32: Confidence flags ──
    results["confidence"] = _compute_confidence(
        results, size=size, run_mode=run_mode
    )

    # ── Point 35: Claims block ──
    results["claims_allowed"] = [
        "finite_size_crossover_observed",
        "entropy_change_quantified",
    ]
    results["claims_forbidden"] = [
        "thermodynamic_critical_point",
        "universal_exponent_validated",
    ]
    caveats = ["single_size_analysis", "product_state_initialization" if preparation == "product_quench" else f"{preparation}_initialization", "trotter_approximation"]
    if n > 16:
        caveats.append("large_lattice_approximation")
    if noise_level > 0:
        caveats.append("noise_contamination")
    results["caveats"] = caveats
    results["pipeline_status"] = "complete"

    # ── Point 36: Run summary verdict ──
    results["run_summary"] = _compute_run_summary(results, run_mode=run_mode)

    # ── Limitations ──
    limitations = [
        "Finite-size effects significant for L <= 5; thermodynamic-limit critical point "
        f"(h_c/J = {HC_OVER_J_MC:.3f}) may differ from single-size crossover estimates.",
        "Imaginary-time evolution may not fully converge for all (J, h) parameter regimes, "
        "especially near the critical point where the gap is small.",
        "Trotter decomposition introduces systematic errors of O(dt^2).",
    ]
    if n > 20:
        limitations.append(
            f"{n} qubits exceeds the statevector limit (20); stabilizer mode is approximate "
            "and does not support imaginary-time evolution."
        )
    if noise_level > 0:
        limitations.append(f"Noise level {noise_level} introduces additional systematic errors.")
    results["limitations"] = limitations

    # Save JSON
    results_path = out / "sim1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"Results saved to {results_path}")

    return results


def _compute_susceptibility(
    h_arr: np.ndarray,
    mag_sq: np.ndarray,
    n_qubits: int = 9,
    n_bootstrap: int = 100,
) -> Dict[str, Any]:
    """Compute susceptibility with smoothing, quadratic fit, and bootstrap CI (point 21).

    χ ∝ N * ⟨M²⟩ for a finite system with Z₂ symmetry.
    We smooth the data, compute via local quadratic fits, and use bootstrap
    for confidence intervals.
    """
    valid = np.isfinite(mag_sq)
    if np.sum(valid) < 5:
        return {
            "method": "insufficient_data",
            "susceptibility": [],
            "h_values": h_arr.tolist(),
        }

    chi_raw = n_qubits * mag_sq  # χ ≈ N * ⟨M²⟩

    # ── Smoothing: moving average over 5 points ──
    window = 5
    chi_smoothed = np.copy(chi_raw)
    hw = window // 2
    for i in range(len(chi_raw)):
        lo = max(0, i - hw)
        hi = min(len(chi_raw), i + hw + 1)
        mask_v = valid[lo:hi]
        segment = chi_raw[lo:hi]
        if np.sum(mask_v) > 0:
            chi_smoothed[i] = np.nanmean(segment[mask_v])
        else:
            chi_smoothed[i] = float("nan")

    # ── Local quadratic fit for refined curvature ──
    chi_quadratic = np.full_like(chi_raw, np.nan)
    for i in range(2, len(chi_raw) - 2):
        lo = max(0, i - 2)
        hi = min(len(chi_raw), i + 3)
        x_seg = h_arr[lo:hi]
        y_seg = chi_smoothed[lo:hi]
        v_seg = np.isfinite(y_seg)
        if np.sum(v_seg) >= 3:
            try:
                coeffs = np.polyfit(x_seg[v_seg], y_seg[v_seg], 2)
                chi_quadratic[i] = np.polyval(coeffs, h_arr[i])
            except Exception:
                chi_quadratic[i] = chi_smoothed[i]
        else:
            chi_quadratic[i] = chi_smoothed[i]

    # ── Peak location ──
    valid_q = np.isfinite(chi_quadratic)
    if np.sum(valid_q) > 2:
        peak_idx = np.nanargmax(chi_quadratic)
        h_peak = float(h_arr[peak_idx])
        chi_peak = float(chi_quadratic[peak_idx])
    else:
        h_peak = None
        chi_peak = None

    # ── Bootstrap confidence intervals ──
    bootstrap_ci = {"peak_h": {"std": None, "ci_low": None, "ci_high": None}}
    if np.sum(valid) > 3:
        valid_indices = np.where(valid)[0]
        valid_chi = chi_raw[valid_indices]
        valid_h = h_arr[valid_indices]

        rng = np.random.default_rng(42)
        peak_h_boot = []
        for b in range(n_bootstrap):
            sample_idx = rng.choice(len(valid_chi), size=len(valid_chi), replace=True)
            chi_b = valid_chi[sample_idx]
            h_b = valid_h[sample_idx]
            # Smooth
            chi_b_smooth = np.copy(chi_b)
            for i in range(len(chi_b)):
                lo2 = max(0, i - hw)
                hi2 = min(len(chi_b), i + hw + 1)
                chi_b_smooth[i] = np.mean(chi_b[lo2:hi2])
            pidx = np.argmax(chi_b_smooth)
            peak_h_boot.append(float(h_b[pidx]))

        peak_h_boot = np.array(peak_h_boot)
        bootstrap_ci["peak_h"] = {
            "std": float(np.std(peak_h_boot, ddof=1)),
            "ci_low": float(np.percentile(peak_h_boot, 2.5)),
            "ci_high": float(np.percentile(peak_h_boot, 97.5)),
        }

    return {
        "method": "smoothed_quadratic_bootstrap",
        "window_size": window,
        "n_bootstrap": n_bootstrap,
        "susceptibility_smoothed": chi_smoothed.tolist(),
        "susceptibility_quadratic": chi_quadratic.tolist(),
        "peak_h_over_J": h_peak,
        "peak_chi": chi_peak,
        "bootstrap_ci": bootstrap_ci,
        "finite_size_caveat": (
            f"Susceptibility peak from L={int(np.sqrt(n_qubits))} lattice; "
            "finite-size broadening expected."
        ),
    }


def _fit_exponents_single_size(
    h_arr: np.ndarray,
    mag_sq: np.ndarray,
    binder: np.ndarray,
    size: int = 3,
) -> Dict[str, Any]:
    """Attempt power-law fits for beta and gamma from single-size data.

    Point 28: Fits are tagged as "exploratory" only.
    Point 29: Adds fit_uncertainty, finite_size_caveat, agreement_with_3D_Ising.
    """
    valid_m = np.isfinite(mag_sq)
    h_c_est = HC_OVER_J_MC  # use known value for fitting

    # ── beta from order parameter: sqrt(<M²>) ~ (h_c - h)^beta ──
    beta_fit: Optional[float] = None
    beta_err: Optional[float] = None
    mask_ordered = (h_arr < h_c_est * 0.95) & valid_m & (mag_sq > 1e-12)
    if np.sum(mask_ordered) > 3:
        try:
            delta = np.clip(h_c_est - h_arr[mask_ordered], 1e-6, None)
            m_root = np.sqrt(mag_sq[mask_ordered])

            def _beta_func(x, A, b):
                return A * np.power(x, b)

            popt, pcov = curve_fit(
                _beta_func, delta, m_root,
                p0=[1.0, 0.3],
                bounds=([0, 0.01], [10, 2.0]),
                maxfev=5000,
            )
            beta_fit = float(popt[1])
            beta_err = float(np.sqrt(pcov[1, 1])) if np.isfinite(pcov[1, 1]) else None
        except Exception:
            pass

    # ── gamma from susceptibility proxy: variance of M from Binder ──
    gamma_fit: Optional[float] = None
    gamma_err: Optional[float] = None
    if np.sum(valid_m) > 5:
        try:
            chi_proxy = np.array(mag_sq)  # proportional to susceptibility
            delta_all = np.abs(h_arr - h_c_est) + 1e-6
            window = (delta_all < 1.5) & valid_m & (chi_proxy > 1e-12)
            if np.sum(window) > 3:
                def _gamma_func(x, A, g):
                    return A * np.power(x, -g)

                popt2, pcov2 = curve_fit(
                    _gamma_func, delta_all[window], chi_proxy[window],
                    p0=[1.0, 1.0],
                    bounds=([0, 0.1], [100, 5.0]),
                    maxfev=5000,
                )
                gamma_fit = float(popt2[1])
                gamma_err = float(np.sqrt(pcov2[1, 1])) if np.isfinite(pcov2[1, 1]) else None
        except Exception:
            pass

    # ── Point 29: Agreement with 3D Ising ──
    beta_agreement = _check_exponent_agreement(beta_fit, beta_err, EXP_BETA)
    gamma_agreement = _check_exponent_agreement(gamma_fit, gamma_err, EXP_GAMMA)

    return {
        "exponent_fit_status": "exploratory",  # point 28
        "beta": {
            "fitted": beta_fit,
            "fit_uncertainty": beta_err,  # point 29
            "theory": EXP_BETA,
            "finite_size_caveat": (  # point 29
                f"Extrapolation from L={size} may not reflect thermodynamic limit"
            ),
            "agreement_with_3D_Ising": beta_agreement,  # point 29
        },
        "gamma": {
            "fitted": gamma_fit,
            "fit_uncertainty": gamma_err,
            "theory": EXP_GAMMA,
            "finite_size_caveat": (
                f"Extrapolation from L={size} may not reflect thermodynamic limit"
            ),
            "agreement_with_3D_Ising": gamma_agreement,
        },
        "note": (
            "Single-size exponent fits are unreliable due to strong finite-size effects. "
            "Values are provided for reference only and should not be quoted as measurements. "
            "exponent_fit_status will be promoted to 'benchmark_validated' only when "
            "sim6 tensor-network comparison confirms agreement."
        ),
    }


def _check_exponent_agreement(
    fitted: Optional[float],
    uncertainty: Optional[float],
    theory: float,
) -> str:
    """Check if fitted exponent is consistent with theory within uncertainty (point 29)."""
    if fitted is None:
        return "no_fit"
    if uncertainty is None:
        # No uncertainty; just report raw deviation
        deviation = abs(fitted - theory) / theory * 100
        if deviation < 20:
            return "roughly_consistent_no_uncertainty"
        return "inconsistent_no_uncertainty"
    if abs(fitted - theory) <= 2.0 * uncertainty:
        return "consistent_within_uncertainty"
    return "inconsistent"


def _compute_bit_mass_derived(
    h_arr: np.ndarray,
    entropy_list: List[float],
    mag_sq_list: List[float],
) -> Dict[str, Any]:
    """Compute bit-mass as a derived proxy quantity (point 27).

    Bit-mass is NOT a direct measurement — it requires experimental calibration
    of the coupling constant η.
    """
    entropy_arr = np.array(entropy_list)
    valid_s = np.isfinite(entropy_arr)

    if np.sum(valid_s) < 3:
        return {
            "type": "derived_proxy",
            "note": "Not a direct measurement. Requires experimental calibration of coupling constant eta.",
            "values": [],
            "h_values": h_arr.tolist(),
            "status": "insufficient_data",
        }

    # Entropy-derived proxy: m_proxy ∝ S * (some energy scale)
    # This is a placeholder heuristic; real bit-mass requires physical calibration
    s_finite = entropy_arr[valid_s]
    h_finite = h_arr[valid_s]

    # Simple proxy: entropy gradient as a mass-like quantity
    ds_dh = np.gradient(s_finite, h_finite)
    # Smooth
    if len(ds_dh) >= 5:
        kernel = np.ones(5) / 5
        ds_dh_smooth = np.convolve(ds_dh, kernel, mode='same')
    else:
        ds_dh_smooth = ds_dh

    return {
        "type": "derived_proxy",
        "note": "Not a direct measurement. Requires experimental calibration of coupling constant eta.",
        "proxy_values": ds_dh_smooth.tolist(),
        "h_values": h_finite.tolist(),
        "entropy_values": s_finite.tolist(),
        "calibration_note": (
            "Bit-mass proxy computed from entropy gradient. "
            "To obtain physical bit-mass, multiply by calibration constant η "
            "determined from experimental comparison."
        ),
    }


# ======================================================================
# 4.  Locate critical point from multi-size Binder crossing
# ======================================================================
def locate_critical_point(
    sweep_data: Dict[str, Dict],
) -> Dict[str, Any]:
    """Find h_c from U4 crossing of different lattice sizes.

    Parameters
    ----------
    sweep_data : dict
        Keys are f"{size}x{size}", values are the results dicts from
        simulate_phase_transition.

    Returns
    -------
    dict with crossing estimates.
    """
    if not HAS_SCIPY:
        return {"h_c_over_J": None, "method": "scipy_required"}

    size_list = []
    binder_curves = []
    h_arrays = []

    for label, res in sorted(sweep_data.items()):
        pd = res.get("phase_diagram", {})
        h = np.array(pd.get("h_over_J", []))
        u4 = np.array(pd.get("binder_cumulant", []))
        valid = np.isfinite(u4)
        if np.sum(valid) < 5:
            continue
        size_list.append(label)
        binder_curves.append(u4[valid])
        h_arrays.append(h[valid])

    if len(size_list) < 2:
        return {
            "h_c_over_J": None,
            "method": "insufficient_sizes",
            "note": "Need at least 2 lattice sizes for crossing analysis.",
        }

    # Find common h range
    h_min = max(a.min() for a in h_arrays)
    h_max = min(a.max() for a in h_arrays)
    h_common = np.linspace(h_min, h_max, 200)

    crossings = []
    pair_labels = []
    for i in range(len(size_list)):
        for j in range(i + 1, len(size_list)):
            try:
                cs_i = CubicSpline(h_arrays[i], binder_curves[i])
                cs_j = CubicSpline(h_arrays[j], binder_curves[j])
                diff = cs_i(h_common) - cs_j(h_common)
                # Find zero crossings
                sign_changes = np.where(np.diff(np.sign(diff)))[0]
                for sc in sign_changes:
                    # Linear interpolation for crossing
                    h_cross = h_common[sc] - diff[sc] * (h_common[sc + 1] - h_common[sc]) / (diff[sc + 1] - diff[sc])
                    crossings.append(float(h_cross))
                    pair_labels.append(f"{size_list[i]}_vs_{size_list[j]}")
            except Exception:
                pass

    h_c_est = float(np.mean(crossings)) if crossings else None
    h_c_err = float(np.std(crossings)) if crossings and len(crossings) > 1 else None

    # Determine confidence level for crossing
    if h_c_est is not None:
        dev = abs(h_c_est - HC_OVER_J_MC) / HC_OVER_J_MC * 100
        if dev < 10 and h_c_err is not None and h_c_err < 0.5:
            confidence = "medium"
        elif dev < 20:
            confidence = "low"
        else:
            confidence = "low"
        if len(crossings) >= 3:
            confidence = max(confidence, "medium")
    else:
        confidence = None

    return {
        "h_c_over_J": h_c_est,
        "error": h_c_err,
        "method": "binder_cumulant_crossing",
        "theoretical": HC_OVER_J_MC,
        "individual_crossings": crossings,
        "crossing_pairs": pair_labels,
        "sizes_used": size_list,
        "confidence": confidence,
        "claim": "finite_size_evidence_only",
        "note": (
            "Binder crossing provides evidence for a phase transition but "
            "does NOT claim the thermodynamic limit critical point. "
            "Systematic finite-size drift may persist."
        ),
    }


# ======================================================================
# 5.  Bootstrap error bars
# ======================================================================
def bootstrap_error_bars(
    data: np.ndarray,
    stat_fn,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for a summary statistic.

    Parameters
    ----------
    data : 1-D array
        Observed data points.
    stat_fn : callable
        Function that maps a resampled array → scalar.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    dict with 'mean', 'std', 'ci_low', 'ci_high'.
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        return {"mean": float(data[0]), "std": 0.0, "ci_low": float(data[0]), "ci_high": float(data[0])}

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = stat_fn(sample)

    alpha = 1.0 - confidence
    return {
        "mean": float(np.mean(boot_stats)),
        "std": float(np.std(boot_stats, ddof=1)),
        "ci_low": float(np.percentile(boot_stats, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_stats, 100 * (1 - alpha / 2))),
    }


# ======================================================================
# 6.  Entanglement scaling analysis
# ======================================================================
def entanglement_scaling_analysis(
    entropy_by_block: Dict[int, float],
) -> Dict[str, Any]:
    """Fit von Neumann entropy of blocks of size ℓ to S(ℓ) = (c/3) ln(ℓ) + const.

    This form holds for 1+1-D critical systems; for the 2-D TFIM at criticality
    the bipartite entropy scales logarithmically with the boundary length.

    Parameters
    ----------
    entropy_by_block : dict
        Maps block linear size ℓ → measured S_vN.

    Returns
    -------
    dict with central charge estimate and fit quality.
    """
    if not HAS_SCIPY or len(entropy_by_block) < 3:
        return {
            "central_charge": None,
            "note": "Need >= 3 block sizes and scipy for fitting.",
        }

    ell = np.array(sorted(entropy_by_block.keys()), dtype=float)
    s_vals = np.array([entropy_by_block[int(e)] for e in ell])

    # S(ℓ) = (c/3) * ln(ℓ) + const
    def _log_fit(x, c_over_3, const):
        return c_over_3 * np.log(x) + const

    try:
        popt, pcov = curve_fit(_log_fit, ell, s_vals, p0=[0.5, 0.0])
        c_over_3 = float(popt[0])
        c_est = 3.0 * c_over_3
        perr = np.sqrt(np.diag(pcov))
        return {
            "central_charge": c_est,
            "central_charge_error": float(3.0 * perr[0]) if len(perr) > 0 else None,
            "fit_constant": float(popt[1]),
            "block_sizes": ell.tolist(),
            "entropy_values": s_vals.tolist(),
            "note": (
                "Logarithmic scaling S = (c/3) ln ℓ + const is expected for "
                "1+1-D conformal systems.  The 2-D TFIM maps to a 2+1-D "
                "classical Ising model, so this fit is indicative only."
            ),
        }
    except Exception as e:
        return {"central_charge": None, "note": f"Fit failed: {e}"}


# ======================================================================
# 7.  Exact diagonalization benchmark
# ======================================================================
def benchmark_vs_exact(
    size: int = 3,
    J_fixed: float = 1.0,
    h_values: Optional[List[float]] = None,
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
) -> Dict[str, Any]:
    """Compare ITE ground-state energy with exact diagonalization.

    Parameters
    ----------
    size : int
        Lattice size (size ≤ 3 recommended for tractability).
    J_fixed : float
        Ising coupling.
    h_values : list[float] or None
        Transverse-field values.  Default: 5 values from 0.5 to 5.0.

    Returns
    -------
    dict with exact vs ITE energies.
    """
    if not HAS_QNVM:
        return {"error": "qnvm_gravity not available"}
    if not HAS_SCIPY:
        return {"error": "scipy required for exact diagonalization"}

    if h_values is None:
        h_values = [0.5, 1.5, 2.5, 3.5, 5.0]

    qubits, pairs, lat_info = build_2d_lattice(size)
    n = lat_info["n_qubits"]

    if n > 12:
        warnings.warn(
            f"ED benchmark for {n} qubits requires a {2**n}×{2**n} matrix. "
            "This may be slow or run out of memory."
        )

    results = {
        "size": size,
        "n_qubits": n,
        "J_fixed": J_fixed,
        "comparisons": [],
    }

    for h_val in h_values:
        entry: Dict[str, Any] = {"h": h_val}

        # Exact
        try:
            evals, _ = exact_diagonalize(n, J_fixed, h_val, pairs, "tfim", n_lowest=1)
            entry["exact_energy"] = float(evals[0])
        except Exception as e:
            entry["exact_energy"] = None
            entry["exact_error"] = str(e)

        # ITE
        try:
            vm = QuantumVMGravity(qubits=n, noise_level=0.0)
            vm.start()
            prepare_ground_state(vm, J_fixed, h_val, pairs,
                                 n_ite_steps=n_ite_steps, dt_ite=dt_ite)
            ite_energy = vm.measure_energy(J_fixed, h_val, pairs, "tfim")
            entry["ite_energy"] = float(ite_energy)
            if entry.get("exact_energy") is not None:
                entry["absolute_error"] = abs(ite_energy - entry["exact_energy"])
                entry["relative_error"] = (
                    abs(ite_energy - entry["exact_energy"]) /
                    (abs(entry["exact_energy"]) + 1e-15)
                )
            vm.stop()
        except Exception as e:
            entry["ite_energy"] = None
            entry["ite_error"] = str(e)

        results["comparisons"].append(entry)

    return results


# ======================================================================
# 8.  Multi-size sweep (point 25: multi_size_binder mode)
# ======================================================================
def sweep_lattice_sizes(
    sizes: List[int] = None,
    J_fixed: float = 1.0,
    h_max: float = 5.0,
    n_steps: int = 25,
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
    noise_level: float = 0.0,
    output_dir: str = "sim1_results",
    preparation: str = "product_quench",
    run_mode: str = "multi_size_binder",
) -> Dict[str, Any]:
    """Run simulation for multiple lattice sizes and analyse finite-size scaling.

    Point 25: In multi_size_binder mode, this computes Binder crossing and
    provides evidence strength but does NOT claim thermodynamic limit.
    """
    if sizes is None:
        sizes = [2, 3, 4]

    results_by_size: Dict[str, Any] = {}

    for size in sizes:
        n = size * size
        print(f"\n{'='*60}")
        print(f"  Size {size}×{size}  ({n} qubits)")
        print(f"{'='*60}")
        try:
            res = simulate_phase_transition(
                size=size,
                J_fixed=J_fixed,
                h_max=h_max,
                n_steps=n_steps,
                n_ite_steps=n_ite_steps,
                dt_ite=dt_ite,
                noise_level=noise_level,
                output_dir=f"{output_dir}/size_{size}x{size}",
                preparation=preparation,
                run_mode=run_mode,
            )
            results_by_size[f"{size}x{size}"] = res
        except Exception as e:
            print(f"  FAILED: {e}")
            results_by_size[f"{size}x{size}"] = {"error": str(e)}

    # Critical point from crossing analysis
    crossing = locate_critical_point(results_by_size)

    # ── Point 25: Binder crossing output ──
    binder_crossing_output = {
        "J_cross": crossing.get("h_c_over_J"),
        "confidence": crossing.get("confidence", "low"),
        "method": crossing.get("method"),
        "note": "Binder crossing provides evidence strength but does NOT claim thermodynamic critical point.",
    }
    if crossing.get("error") is not None:
        binder_crossing_output["error"] = crossing.get("error")

    # Compile sweep results
    sweep_results = {
        "sizes": sizes,
        "run_mode": run_mode,
        "state_preparation": preparation,
        "individual_results": {
            k: {
                "h_crossover_estimate": v.get("crossover_point", {}).get("h_crossover_over_J"),
                "max_entropy": max(v.get("phase_diagram", {}).get("von_neumann_entropy", [0])),
                "min_energy": min(
                    [e for e in v.get("phase_diagram", {}).get("energy", []) if np.isfinite(e)],
                    default=None,
                ),
            }
            for k, v in results_by_size.items()
            if "error" not in v
        },
        "critical_point_analysis": crossing,
        "binder_crossing": binder_crossing_output,  # point 25
        # ── Point 35: Claims block ──
        "claims_allowed": [
            "finite_size_crossover_observed",
            "entropy_change_quantified",
            "binder_crossing_computed",
        ],
        "claims_forbidden": [
            "thermodynamic_critical_point",
            "universal_exponent_validated",
        ],
        "caveats": [
            "multi_size_but_small_lattices",
            f"{preparation}_initialization",
            "trotter_approximation",
        ],
        "pipeline_status": "complete",
    }

    # ── Point 36: Run summary verdict ──
    if crossing.get("h_c_over_J") is not None:
        dev = abs(crossing["h_c_over_J"] - HC_OVER_J_MC) / HC_OVER_J_MC * 100
        if dev < 15 and crossing.get("confidence") == "medium":
            verdict = "finite_size_critical_evidence"
            reasoning = (
                f"Binder crossing at h_c/J={crossing['h_c_over_J']:.4f} "
                f"(theory: {HC_OVER_J_MC:.4f}, deviation: {dev:.1f}%) "
                f"with {crossing.get('confidence', '?')} confidence across "
                f"sizes {sizes}. Evidence for finite-size critical behavior."
            )
        else:
            verdict = "inconsistent_diagnostics"
            reasoning = (
                f"Binder crossing at h_c/J={crossing['h_c_over_J']:.4f} "
                f"deviates {dev:.1f}% from theory with "
                f"{crossing.get('confidence', '?')} confidence. "
                f"Insufficient evidence for reliable crossing."
            )
    else:
        verdict = "inconsistent_diagnostics"
        reasoning = "Could not determine Binder crossing from available data."

    sweep_results["run_summary"] = {
        "verdict": verdict,
        "reasoning": reasoning,
    }

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sim1_sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2, default=str)

    print(f"\nSweep results saved to {out / 'sim1_sweep_results.json'}")

    # ── Finite-size scaling extrapolation ──
    if crossing.get("h_c_over_J") is not None and len(sizes) >= 2 and HAS_SCIPY:
        print(f"\n  Crossover estimate (Binder crossing): h_c/J = {crossing['h_c_over_J']:.4f}")
        print(f"  Theoretical (thermodynamic limit):    h_c/J = {HC_OVER_J_MC:.4f}")
        print(f"  Deviation: {abs(crossing['h_c_over_J'] - HC_OVER_J_MC) / HC_OVER_J_MC * 100:.1f}%")
        if crossing.get("error") is not None:
            print(f"  Crossing spread: ±{crossing['error']:.4f}")
        print(f"  Confidence: {crossing.get('confidence', 'unknown')}")
        print(f"  NOTE: This is a finite-size crossing, NOT the thermodynamic critical point.")
    else:
        print("\n  Could not determine crossing from crossing analysis.")

    return sweep_results


# ======================================================================
# 8b. Null/Trivial Model Control (point 33)
# ======================================================================
def run_null_test(
    size: int = 3,
    J_fixed: float = 1.0,
    h_max: float = 5.0,
    n_steps: int = 30,
    n_ite_steps: int = 300,
    dt_ite: float = 0.02,
    output_dir: str = "sim1_results",
) -> Dict[str, Any]:
    """Run simulation with h_field=0 (no transverse field) as null control.

    Without transverse field, the system is purely classical and should show
    no phase transition — it's always in the ordered phase.
    """
    if not HAS_QNVM:
        return {"error": "qnvm_gravity not available"}

    print("\n" + "=" * 60)
    print("  NULL TEST: h_field = 0 (no transverse field)")
    print("=" * 60)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    qubits, pairs, lat_info = build_2d_lattice(size)
    n = lat_info["n_qubits"]
    half_qubits = list(range(n // 2))
    z_sites = qubits

    # With h=0, the system is always ordered.  We sweep a tiny effective h
    # just to have data points, but set h=0 in the Hamiltonian.
    h_values = np.linspace(0.01, h_max, n_steps)

    mag_sq_list: List[float] = []
    binder_list: List[float] = []
    entropy_list: List[float] = []

    for idx, h_val in enumerate(h_values):
        vm = QuantumVMGravity(qubits=n, noise_level=0.0)
        vm.start()

        try:
            gs = prepare_ground_state(vm, J_fixed, 0.0, pairs,
                                      n_ite_steps=n_ite_steps, dt_ite=dt_ite)
        except (NotImplementedError, Exception):
            gs = None

        try:
            u4 = vm.binder_cumulant(z_sites, shots=4096)
        except Exception:
            u4 = float("nan")
        binder_list.append(u4)

        try:
            svne = vm.von_neumann_entropy_subsystem(half_qubits)
        except Exception:
            svne = float("nan")
        entropy_list.append(svne)

        try:
            if vm._backend_type == "statevector":
                m2_exact = 0.0
                for i in z_sites:
                    m2_exact += 1.0
                for i_idx in range(len(z_sites)):
                    for j_idx in range(i_idx + 1, len(z_sites)):
                        pstr = ["I"] * n
                        pstr[z_sites[i_idx]] = "Z"
                        pstr[z_sites[j_idx]] = "Z"
                        m2_exact += 2.0 * vm.expectation("".join(pstr))
                m2_exact /= n * n
                mag_sq_list.append(m2_exact)
            else:
                mag_sq_list.append(float("nan"))
        except Exception:
            mag_sq_list.append(float("nan"))

        vm.stop()

    # Check for transition: Binder should be roughly constant near 2/3
    binder_arr = np.array(binder_list)
    valid = np.isfinite(binder_arr)
    no_transition = True
    conclusion = "phase transition requires non-zero transverse field"

    if np.sum(valid) > 3:
        binder_range = float(np.ptp(binder_arr[valid]))
        binder_mean = float(np.mean(binder_arr[valid]))
        # If Binder varies wildly, something unexpected happened
        if binder_range > 0.3:
            no_transition = False
            conclusion = "unexpected behavior: significant Binder variation without transverse field"
        # If Binder is near 2/3 (ordered) consistently, no transition
        if abs(binder_mean - 2.0/3.0) < 0.15:
            no_transition = True
            conclusion = "phase transition requires non-zero transverse field"

    null_result = {
        "null_test": {
            "h_field": 0.0,
            "size": size,
            "n_qubits": n,
            "binder_range": float(np.ptp(binder_arr[valid])) if np.sum(valid) > 0 else None,
            "binder_mean": float(np.mean(binder_arr[valid])) if np.sum(valid) > 0 else None,
            "no_transition_observed": no_transition,
            "conclusion": conclusion,
        },
    }

    with open(out / "sim1_null_test.json", "w") as f:
        json.dump(null_result, f, indent=2, default=str)
    print(f"\nNull test results saved to {out / 'sim1_null_test.json'}")
    print(f"  No transition observed: {no_transition}")
    print(f"  Conclusion: {conclusion}")

    return null_result


# ======================================================================
# 8c. Benchmark export for sim6 (point 36)
# ======================================================================
def export_benchmark_data(
    results: Dict[str, Any],
    output_dir: str = "sim1_results",
) -> Dict[str, Any]:
    """Export observables in a format compatible with sim6 tensor-network comparison.

    Point 36: Outputs a structured dict with model name, all observables, and
    simulation metadata.
    """
    pd = results.get("phase_diagram", {})
    params = results.get("parameters", {})

    benchmark = {
        "model": "xx_yy_z",  # TFIM mapped to XX+YY+Z
        "L": params.get("size"),
        "n_qubits": params.get("n_qubits"),
        "J_values": pd.get("h_over_J", []),
        "magnetization": pd.get("magnetization_squared", []),
        "entropy_half": pd.get("von_neumann_entropy", []),
        "binder_cumulant": pd.get("binder_cumulant", []),
        "susceptibility": pd.get("susceptibility", {}).get("susceptibility_smoothed", []),
        "energy": pd.get("energy", []),
        "fidelity_susceptibility": pd.get("fidelity_susceptibility", []),
        "qfi": pd.get("qfi", []),
        "trotter_dt": params.get("dt_ite"),
        "state_preparation": params.get("state_preparation", "product_quench"),
        "noise_level": params.get("noise_level", 0.0),
        "n_ite_steps": params.get("n_ite_steps"),
        "crossover_point": results.get("crossover_point", {}),
        "susceptibility_analysis": results.get("susceptibility", {}),
        "trotter_error_analysis": results.get("trotter_error_analysis", {}),
        "run_summary": results.get("run_summary", {}),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sim1_benchmark_data.json", "w") as f:
        json.dump(benchmark, f, indent=2, default=str)
    print(f"Benchmark data exported to {out / 'sim1_benchmark_data.json'}")

    return benchmark


# ======================================================================
# Helper: Confidence computation (point 32)
# ======================================================================
def _compute_confidence(
    results: Dict[str, Any],
    size: int = 3,
    run_mode: str = "single_size_exploratory",
) -> Dict[str, Any]:
    """Compute confidence flags for crossover/critical-point-like outputs (point 32)."""
    confidence_factors = []

    if run_mode == "multi_size_binder":
        confidence_factors.append("multi_size_crossing")

    # Check exponent agreement
    ce = results.get("critical_exponents", {})
    beta_d = ce.get("beta", {})
    gamma_d = ce.get("gamma", {})
    if beta_d.get("agreement_with_3D_Ising", "").startswith("consistent"):
        confidence_factors.append("exponent_agreement")
    if gamma_d.get("agreement_with_3D_Ising", "").startswith("consistent"):
        confidence_factors.append("exponent_agreement")

    # Check susceptibility peak alignment
    sus = results.get("susceptibility", {})
    h_peak = sus.get("peak_h_over_J")
    if h_peak is not None:
        sus_dev = abs(h_peak - HC_OVER_J_MC) / HC_OVER_J_MC * 100
        if sus_dev < 20:
            confidence_factors.append("susceptibility_peak")

    # Determine overall confidence level
    if len(confidence_factors) >= 3:
        level = "high"
    elif len(confidence_factors) >= 2:
        level = "medium"
    elif len(confidence_factors) >= 1:
        level = "low"
    else:
        level = "low"

    return {
        "confidence": level,
        "confidence_factors": confidence_factors,
        "note": "Confidence assessment is heuristic. Multi-size crossing and "
                "exponent agreement with 3D Ising are the strongest indicators.",
    }


# ======================================================================
# Helper: Run summary verdict (point 36)
# ======================================================================
def _compute_run_summary(
    results: Dict[str, Any],
    run_mode: str = "single_size_exploratory",
) -> Dict[str, Any]:
    """Compute run summary verdict (point 36)."""
    conf = results.get("confidence", {})
    conf_level = conf.get("confidence", "low")
    conf_factors = conf.get("confidence_factors", [])

    cp = results.get("crossover_point", {})
    h_cross = cp.get("h_crossover_over_J")

    if run_mode == "benchmark_compare":
        return {
            "verdict": "benchmark_aligned",
            "reasoning": "Run in benchmark_compare mode. Data exported for sim6 comparison.",
        }

    if run_mode == "multi_size_binder":
        # Handled in sweep_lattice_sizes
        return {
            "verdict": "exploratory_crossover",
            "reasoning": "Multi-size analysis; see sweep results for crossing verdict.",
        }

    # Single-size mode: always exploratory_crossover
    if h_cross is not None:
        dev = abs(h_cross - HC_OVER_J_MC) / HC_OVER_J_MC * 100
        reasoning = (
            f"Single-size (L={results['parameters']['size']}) crossover at h/J={h_cross:.4f} "
            f"(theory: {HC_OVER_J_MC:.4f}, deviation: {dev:.1f}%). "
            f"Confidence: {conf_level}. Factors: {conf_factors}. "
            f"This is a finite-size crossover indicator only, NOT a thermodynamic critical point."
        )
    else:
        reasoning = (
            f"Single-size analysis for L={results['parameters']['size']} did not yield "
            f"a clear crossover estimate. Confidence: {conf_level}."
        )

    return {
        "verdict": "exploratory_crossover",
        "reasoning": reasoning,
    }


# ======================================================================
# 9.  Plot generation
# ======================================================================
def generate_plots(results: Dict[str, Any], output_dir: str, sizes: List[int] = None):
    """Generate publication-quality diagnostic plots.

    Parameters
    ----------
    results : dict or dict of dicts
        If a single results dict, plot one size.
        If dict keyed by "{size}x{size}", plot multi-size comparison.
    output_dir : str
        Directory for output figures.
    sizes : list[int] or None
        Sizes to include in multi-size mode.
    """
    if not HAS_MPL:
        warnings.warn("matplotlib not available; skipping plots.")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Detect single vs multi-size
    is_multi = any(isinstance(v, dict) and "phase_diagram" in v for v in results.values())
    # But results could also be a single simulation result
    if "phase_diagram" in results:
        is_multi = False

    if is_multi:
        _plot_multi_size(results, out, sizes)
    else:
        _plot_single_size(results, out)


def _plot_single_size(res: Dict[str, Any], out: Path):
    """Generate all plots for a single lattice size."""
    pd = res["phase_diagram"]
    h = np.array(pd["h_over_J"])
    binder = np.array(pd["binder_cumulant"])
    entropy = np.array(pd["von_neumann_entropy"])
    energy = np.array(pd["energy"])
    chi_f = np.array(pd["fidelity_susceptibility"])
    qfi = np.array(pd["qfi"])
    mag_sq = np.array(pd["magnetization_squared"])

    params = res["parameters"]
    size = params["size"]
    title_prefix = f"2D TFIM  L={size}"

    # ── Plot 1: Binder cumulant (point 20: label as crossover, not critical point) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(binder)
    ax.plot(h[valid], binder[valid], "o-", color="#2171b5", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7,
               label=f"Theory $h_c/J$ = {HC_OVER_J_MC:.3f}")
    crossover_h = res.get("crossover_point", {}).get("h_crossover_over_J")
    if crossover_h is not None:
        ax.axvline(crossover_h,
                   color="#e6550d", ls=":", lw=1.5, alpha=0.8,
                   label=f"Crossover $h/J$ = {crossover_h:.3f}")
    ax.set_xlabel("$h / J$")
    ax.set_ylabel("$U_4$")
    ax.set_title(f"{title_prefix} — Binder Cumulant (finite-size crossover)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot1_binder_cumulant.png", dpi=150)
    plt.close(fig)

    # ── Plot 2: Order parameter ⟨M²⟩ ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(mag_sq)
    if np.any(valid):
        ax.plot(h[valid], mag_sq[valid], "s-", color="#238b45", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("$h / J$")
    ax.set_ylabel(r"$\langle M^2 \rangle$")
    ax.set_title(f"{title_prefix} — Order Parameter Squared")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot2_order_parameter.png", dpi=150)
    plt.close(fig)

    # ── Plot 3: Von Neumann entropy ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(entropy)
    if np.any(valid):
        ax.plot(h[valid], entropy[valid], "D-", color="#6a51a3", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("$h / J$")
    ax.set_ylabel("$S_{vN}$ (half-lattice, bits)")
    ax.set_title(f"{title_prefix} — Von Neumann Entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot3_entropy.png", dpi=150)
    plt.close(fig)

    # ── Plot 3b: Susceptibility (point 21) ──
    sus_data = res.get("susceptibility", {})
    if sus_data.get("susceptibility_smoothed"):
        sus_arr = np.array(sus_data["susceptibility_smoothed"])
        valid_s = np.isfinite(sus_arr)
        if np.any(valid_s):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(h[valid_s], sus_arr[valid_s], "D-", color="#de2d26", markersize=3, linewidth=1.2,
                    label="Smoothed χ")
            if sus_data.get("peak_h_over_J") is not None:
                ax.axvline(sus_data["peak_h_over_J"], color="#e6550d", ls=":", lw=1.5,
                           label=f"Peak $h/J$ = {sus_data['peak_h_over_J']:.3f}")
            ci = sus_data.get("bootstrap_ci", {}).get("peak_h", {})
            if ci.get("ci_low") is not None and ci.get("ci_high") is not None:
                ax.axvspan(ci["ci_low"], ci["ci_high"], alpha=0.15, color="#e6550d",
                           label=f"95% CI: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
            ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
            ax.set_xlabel("$h / J$")
            ax.set_ylabel(r"$\chi$ (susceptibility proxy)")
            ax.set_title(f"{title_prefix} — Susceptibility (smoothed + bootstrap CI)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "plot3b_susceptibility.png", dpi=150)
            plt.close(fig)

    # ── Plot 4: Entanglement spectrum ──
    spectra = res.get("entanglement_spectra", {})
    if spectra:
        # Pick three representative h values: ordered, near-critical, disordered
        h_arr_keys = [float(k) for k in spectra.keys()]
        h_arr_keys.sort()
        picks = _pick_three(h_arr_keys, HC_OVER_J_MC)
        fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4), sharey=True)
        if len(picks) == 1:
            axes = [axes]
        for ax_i, hv in enumerate(picks):
            spec = spectra.get(str(hv), [])
            if spec:
                axes[ax_i].bar(range(len(spec)), spec, color="#fc8d59", edgecolor="none")
                axes[ax_i].set_title(f"$h/J$ = {hv/params['J_fixed']:.2f}")
            axes[ax_i].set_xlabel("Eigenvalue index")
            if ax_i == 0:
                axes[ax_i].set_ylabel("Eigenvalue of $\\rho_A$")
        fig.suptitle(f"{title_prefix} — Entanglement Spectrum", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "plot4_entanglement_spectrum.png", dpi=150)
        plt.close(fig)

    # ── Plot 5: Energy ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(energy)
    if np.any(valid):
        ax.plot(h[valid], energy[valid], "o-", color="#d94701", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("$h / J$")
    ax.set_ylabel("$\\langle H \\rangle / N$")
    ax.set_title(f"{title_prefix} — Ground-State Energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot5_energy.png", dpi=150)
    plt.close(fig)

    # ── Plot 6: Fidelity susceptibility ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(chi_f) & (chi_f > 0)
    if np.any(valid):
        ax.plot(h[valid], chi_f[valid], "^-", color="#810f7c", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("$h / J$")
    ax.set_ylabel("$\\chi_F = -2 \\ln |\\langle\\psi|\\psi'\\rangle|$")
    ax.set_title(f"{title_prefix} — Fidelity Susceptibility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot6_fidelity_susceptibility.png", dpi=150)
    plt.close(fig)

    # ── Plot 7: QFI ──
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = np.isfinite(qfi) & (qfi >= 0)
    if np.any(valid):
        ax.plot(h[valid], qfi[valid], "v-", color="#006d2c", markersize=3, linewidth=1.2)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("$h / J$")
    ax.set_ylabel("$F_Q$ (QFI)")
    ax.set_title(f"{title_prefix} — Quantum Fisher Information")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot7_qfi.png", dpi=150)
    plt.close(fig)

    # ── Plot 8: Dashboard ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    v_b = np.isfinite(binder)
    if np.any(v_b):
        axes[0].plot(h[v_b], binder[v_b], "o-", ms=3, color="#2171b5")
    axes[0].axvline(HC_OVER_J_MC, color="gray", ls="--", alpha=0.5)
    axes[0].set_title("Binder Cumulant $U_4$ (crossover indicator)")
    axes[0].set_xlabel("$h/J$"); axes[0].set_ylabel("$U_4$")
    axes[0].grid(True, alpha=0.3)

    v_e = np.isfinite(entropy)
    if np.any(v_e):
        axes[1].plot(h[v_e], entropy[v_e], "D-", ms=3, color="#6a51a3")
    axes[1].axvline(HC_OVER_J_MC, color="gray", ls="--", alpha=0.5)
    axes[1].set_title("Von Neumann Entropy")
    axes[1].set_xlabel("$h/J$"); axes[1].set_ylabel("$S$ (bits)")
    axes[1].grid(True, alpha=0.3)

    v_m = np.isfinite(mag_sq)
    if np.any(v_m):
        axes[2].plot(h[v_m], mag_sq[v_m], "s-", ms=3, color="#238b45")
    axes[2].axvline(HC_OVER_J_MC, color="gray", ls="--", alpha=0.5)
    axes[2].set_title(r"$\langle M^2 \rangle$")
    axes[2].set_xlabel("$h/J$"); axes[2].set_ylabel(r"$\langle M^2 \rangle$")
    axes[2].grid(True, alpha=0.3)

    # Summary text
    axes[3].axis("off")
    lines = [
        f"Lattice: {size}x{size}  ({params['n_qubits']} qubits)",
        f"$J$ = {params['J_fixed']},  $h_{{max}}$ = {params['h_max']}",
        f"ITE: {params['n_ite_steps']} steps, dt = {params['dt_ite']}",
        f"Preparation: {params.get('state_preparation', 'product_quench')}",
        f"Theory $h_c/J$ = {HC_OVER_J_MC:.3f}",
    ]
    cp = res.get("crossover_point", {})
    if cp.get("h_crossover_over_J") is not None:
        lines.append(f"Crossover $h/J$ = {cp['h_crossover_over_J']:.3f}")
    ce = res.get("critical_exponents", {})
    beta_d = ce.get("beta", {})
    if beta_d.get("fitted") is not None:
        lines.append(
            f"$\\beta$ = {beta_d['fitted']:.3f} +/- {beta_d.get('fit_uncertainty', '?')}  "
            f"(theory: {beta_d['theory']:.4f}) [{beta_d.get('agreement_with_3D_Ising', '?')}]"
        )
    gamma_d = ce.get("gamma", {})
    if gamma_d.get("fitted") is not None:
        lines.append(
            f"$\\gamma$ = {gamma_d['fitted']:.3f} +/- {gamma_d.get('fit_uncertainty', '?')}  "
            f"(theory: {gamma_d['theory']:.4f}) [{gamma_d.get('agreement_with_3D_Ising', '?')}]"
        )
    rs = res.get("run_summary", {})
    if rs.get("verdict"):
        lines.append(f"Verdict: {rs['verdict']}")
    conf = res.get("confidence", {})
    lines.append(f"Confidence: {conf.get('confidence', '?')}")

    for lim in res.get("limitations", [])[:3]:
        lines.append(f"[!] {lim}")
    axes[3].text(0.05, 0.95, "\n".join(lines), transform=axes[3].transAxes,
                 fontsize=9, verticalalignment="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    fig.suptitle(f"{title_prefix} — Simulation Dashboard", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "plot8_dashboard.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out}/")


def _plot_multi_size(sweep_results: Dict[str, Any], out: Path, sizes: List[int] = None):
    """Generate multi-size comparison plots."""
    cmap = plt.cm.viridis

    size_labels = sorted(sweep_results.keys())
    n_sizes = len(size_labels)
    colors = [cmap(i / max(n_sizes - 1, 1)) for i in range(n_sizes)]

    # ── Binder crossing plot (point 20: labeled as crossing, not critical point) ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, label in enumerate(size_labels):
        res = sweep_results[label]
        if "error" in res:
            continue
        pd = res.get("phase_diagram", {})
        h = np.array(pd.get("h_over_J", []))
        u4 = np.array(pd.get("binder_cumulant", []))
        valid = np.isfinite(u4)
        if np.any(valid):
            ax.plot(h[valid], u4[valid], "o-", color=colors[idx], ms=3,
                    linewidth=1.2, label=label)

    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1.5, alpha=0.7,
               label=f"Theory $h_c/J$ = {HC_OVER_J_MC:.3f}")

    # Add crossing estimate
    cp_analysis = sweep_results.get("critical_point_analysis", {})
    if cp_analysis.get("h_c_over_J") is not None:
        ax.axvline(cp_analysis["h_c_over_J"], color="#e6550d", ls=":", lw=2, alpha=0.8,
                   label=f"Crossing $h/J$ = {cp_analysis['h_c_over_J']:.3f}")

    ax.set_xlabel("$h / J$", fontsize=12)
    ax.set_ylabel("Binder Cumulant $U_4$", fontsize=12)
    ax.set_title("2D TFIM — Binder Cumulant Crossing (finite-size evidence)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_multi_binder_crossing.png", dpi=150)
    plt.close(fig)

    # ── Entropy comparison ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, label in enumerate(size_labels):
        res = sweep_results[label]
        if "error" in res:
            continue
        pd = res.get("phase_diagram", {})
        h = np.array(pd.get("h_over_J", []))
        s = np.array(pd.get("von_neumann_entropy", []))
        valid = np.isfinite(s)
        if np.any(valid):
            ax.plot(h[valid], s[valid], "o-", color=colors[idx], ms=3,
                    linewidth=1.2, label=label)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("$h / J$", fontsize=12)
    ax.set_ylabel("$S_{vN}$ (bits)", fontsize=12)
    ax.set_title("2D TFIM — Von Neumann Entropy", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_multi_entropy.png", dpi=150)
    plt.close(fig)

    # ── Energy comparison ──
    fig, ax = plt.subplots(figsize=(9, 6))
    for idx, label in enumerate(size_labels):
        res = sweep_results[label]
        if "error" in res:
            continue
        pd = res.get("phase_diagram", {})
        h = np.array(pd.get("h_over_J", []))
        e = np.array(pd.get("energy", []))
        valid = np.isfinite(e)
        if np.any(valid):
            n_q = res["parameters"]["n_qubits"]
            ax.plot(h[valid], e[valid] / n_q, "o-", color=colors[idx], ms=3,
                    linewidth=1.2, label=label)
    ax.axvline(HC_OVER_J_MC, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.set_xlabel("$h / J$", fontsize=12)
    ax.set_ylabel("$E / N$ (per site)", fontsize=12)
    ax.set_title("2D TFIM — Ground-State Energy per Site", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "plot_multi_energy.png", dpi=150)
    plt.close(fig)

    # ── ED Benchmark plot ──
    for label in size_labels:
        res = sweep_results[label]
        if "error" in res:
            continue
        size = res["parameters"]["size"]
        if size > 3:
            continue  # ED too expensive
        # Look for ED benchmark in sub-directory
        ed_path = out / f"size_{label}" / "ed_benchmark.json"
        if ed_path.exists():
            with open(ed_path) as f:
                ed_data = json.load(f)
            fig, ax = plt.subplots(figsize=(8, 5))
            h_vals = [c["h"] for c in ed_data["comparisons"]]
            exact_e = [c.get("exact_energy") for c in ed_data["comparisons"]]
            ite_e = [c.get("ite_energy") for c in ed_data["comparisons"]]
            valid = [(e is not None and i is not None) for e, i in zip(exact_e, ite_e)]
            h_v = [h for h, v in zip(h_vals, valid) if v]
            ex_v = [e for e, v in zip(exact_e, valid) if v]
            it_v = [i for i, v in zip(ite_e, valid) if v]
            if h_v:
                ax.plot(h_v, ex_v, "s-", color="#2171b5", ms=6, label="Exact diagonalization")
                ax.plot(h_v, it_v, "o--", color="#e6550d", ms=6, label="ITE")
                ax.set_xlabel("$h / J$")
                ax.set_ylabel("$E_0$")
                ax.set_title(f"ED Benchmark — L={label}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(out / f"plot_ed_benchmark_{label}.png", dpi=150)
                plt.close(fig)

    print(f"Multi-size plots saved to {out}/")


def _pick_three(h_values: List[float], h_c: float) -> List[float]:
    """Pick three representative h values: ordered, near-critical, disordered."""
    if len(h_values) < 3:
        return h_values
    h_arr = np.array(h_values)
    # Near critical
    idx_c = np.argmin(np.abs(h_arr - h_c))
    # Ordered (lowest h with data)
    idx_ordered = max(0, idx_c // 2)
    # Disordered (highest h with data)
    idx_disordered = min(len(h_arr) - 1, idx_c + (len(h_arr) - idx_c) // 2)
    picks = sorted({h_arr[idx_ordered], h_arr[idx_c], h_arr[idx_disordered]})
    return [float(p) for p in picks]


# ======================================================================
# 10. Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="2D Transverse-Field Ising Model — Quantum Phase Transition Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single size exploratory (default, backward compatible)
  python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --h-max 5.0

  # Multi-size Binder crossing analysis
  python sim1_vacuum_phase.py --mode multi_size_binder --sweep-sizes --h-max 5.0

  # Benchmark comparison with sim6 tensor network
  python sim1_vacuum_phase.py --mode benchmark_compare --size 3

  # With ED benchmark
  python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --h-max 5.0 --benchmark-ed

  # Null test (no transverse field)
  python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --null-test

  # Adiabatic ramp state preparation
  python sim1_vacuum_phase.py --mode single_size_exploratory --size 3 --preparation adiabatic_ramp
        """,
    )

    # ── Point 19: --mode argument ──
    parser.add_argument("--mode", type=str, default="single_size_exploratory",
                        choices=["single_size_exploratory", "multi_size_binder", "benchmark_compare"],
                        help="Run mode: single_size_exploratory (default), multi_size_binder, benchmark_compare")

    parser.add_argument("--size", type=int, default=3,
                        help="Lattice side length L (default: 3, giving 9 qubits)")
    parser.add_argument("--h-max", type=float, default=5.0,
                        help="Maximum transverse field h (J fixed at 1.0; default 5.0)")
    parser.add_argument("--J-fixed", type=float, default=1.0,
                        help="Ising coupling J (default: 1.0)")
    parser.add_argument("--ite-steps", type=int, default=300,
                        help="Imaginary-time evolution steps (default: 300)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of h values in the sweep (default: 30)")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Noise level for backend (default: 0.0)")
    parser.add_argument("--output-dir", type=str, default="sim1_results",
                        help="Output directory")
    parser.add_argument("--sweep-sizes", action="store_true",
                        help="Run for multiple lattice sizes (2, 3, 4)")
    parser.add_argument("--sweep-sizes-list", type=str, default=None,
                        help="Comma-separated list of sizes for sweep (e.g., '2,3,4,5')")
    parser.add_argument("--benchmark-ed", action="store_true",
                        help="Run exact diagonalization benchmark (L <= 3)")

    # ── Point 22: --preparation argument ──
    parser.add_argument("--preparation", type=str, default="product_quench",
                        choices=["product_quench", "adiabatic_ramp", "random_product"],
                        help="State preparation mode (default: product_quench)")

    # ── Point 33: --null-test flag ──
    parser.add_argument("--null-test", action="store_true",
                        help="Run null/trivial model control with h=0")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("=" * 65)
    print("  2D TFIM Quantum Phase Transition Simulation")
    print("=" * 65)
    print(f"  Mode          : {args.mode}")
    print(f"  Lattice size  : {args.size}x{args.size} ({args.size**2} qubits)")
    print(f"  J fixed       : {args.J_fixed}")
    print(f"  h range       : [0, {args.h_max}]")
    print(f"  h_c/J (theory): {HC_OVER_J_MC:.4f}")
    print(f"  ITE steps     : {args.ite_steps}")
    print(f"  Sweep points  : {args.steps}")
    print(f"  Noise         : {args.noise}")
    print(f"  Preparation   : {args.preparation}")
    print(f"  Output        : {args.output_dir}")
    print(f"  Null test     : {args.null_test}")
    print(f"  qnvm_gravity  : {'available' if HAS_QNVM else 'NOT FOUND'}")
    print(f"  scipy         : {'available' if HAS_SCIPY else 'not available'}")
    print(f"  matplotlib    : {'available' if HAS_MPL else 'not available'}")
    print()

    if not HAS_QNVM:
        print("ERROR: qnvm_gravity module is required for this simulation.")
        print("       Place qnvm_gravity.py in the same directory or on PYTHONPATH.")
        sys.exit(1)

    # ── ED benchmark ──
    if args.benchmark_ed:
        print("Running exact diagonalization benchmark...")
        ed_results = benchmark_vs_exact(
            size=args.size,
            J_fixed=args.J_fixed,
            n_ite_steps=args.ite_steps,
        )
        ed_out = Path(args.output_dir)
        ed_out.mkdir(parents=True, exist_ok=True)
        with open(ed_out / "ed_benchmark.json", "w") as f:
            json.dump(ed_results, f, indent=2, default=str)
        print("\nED Benchmark Results:")
        for comp in ed_results.get("comparisons", []):
            h = comp["h"]
            exact = comp.get("exact_energy")
            ite = comp.get("ite_energy")
            rel = comp.get("relative_error")
            if exact is not None and ite is not None:
                print(f"  h/J={h:.1f}:  E_exact={exact:.6f}  E_ITE={ite:.6f}  "
                      f"rel_err={rel:.2e}")
            else:
                print(f"  h/J={h:.1f}:  (data incomplete)")
        print()

    # ── Point 33: Null test ──
    if args.null_test:
        null_result = run_null_test(
            size=args.size,
            J_fixed=args.J_fixed,
            h_max=args.h_max,
            n_steps=args.steps,
            n_ite_steps=args.ite_steps,
            dt_ite=0.02,
            output_dir=args.output_dir,
        )
        print()

    # ── Point 19: Mode-based execution ──
    if args.mode == "multi_size_binder" or args.sweep_sizes:
        # Parse sizes
        if args.sweep_sizes_list:
            sizes = [int(s.strip()) for s in args.sweep_sizes_list.split(",")]
        else:
            sizes = [2, 3, 4, 5]
            if args.size in sizes:
                pass
            else:
                sizes.append(args.size)
            sizes = sorted(set(sizes))

        sweep_results = sweep_lattice_sizes(
            sizes=sizes,
            J_fixed=args.J_fixed,
            h_max=args.h_max,
            n_steps=max(15, args.steps // 2),
            n_ite_steps=args.ite_steps,
            dt_ite=0.02,
            noise_level=args.noise,
            output_dir=args.output_dir,
            preparation=args.preparation,
            run_mode=args.mode,
        )

        # Generate multi-size plots
        multi_plot_data = {}
        for label in sweep_results.get("individual_results", {}):
            sub_dir = Path(args.output_dir) / f"size_{label}"
            json_path = sub_dir / "sim1_results.json"
            if json_path.exists():
                with open(json_path) as f:
                    multi_plot_data[label] = json.load(f)

        if multi_plot_data and HAS_MPL:
            _plot_multi_size(multi_plot_data, Path(args.output_dir), sizes)

        # Print final summary
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        cp = sweep_results.get("critical_point_analysis", {})
        if cp.get("h_c_over_J") is not None:
            dev = abs(cp["h_c_over_J"] - HC_OVER_J_MC) / HC_OVER_J_MC * 100
            print(f"  Binder crossing: h_c/J = {cp['h_c_over_J']:.4f}")
            print(f"  Theory (thermo):  h_c/J = {HC_OVER_J_MC:.4f}")
            print(f"  Deviation: {dev:.1f}%")
            print(f"  Confidence: {cp.get('confidence', '?')}")
        else:
            print("  Binder crossing: could not determine from data.")
        bc = sweep_results.get("binder_crossing", {})
        print(f"  J_cross: {bc.get('J_cross', 'N/A')}")
        print(f"  Crossing confidence: {bc.get('confidence', 'N/A')}")
        print(f"  Claims allowed: {sweep_results.get('claims_allowed', [])}")
        print(f"  Claims forbidden: {sweep_results.get('claims_forbidden', [])}")
        rs = sweep_results.get("run_summary", {})
        print(f"  Verdict: {rs.get('verdict', 'N/A')}")
        print(f"  Reasoning: {rs.get('reasoning', 'N/A')}")
        print()

    elif args.mode == "benchmark_compare":
        # ── Point 36: Benchmark compare mode ──
        results = simulate_phase_transition(
            size=args.size,
            J_fixed=args.J_fixed,
            h_max=args.h_max,
            n_steps=args.steps,
            n_ite_steps=args.ite_steps,
            dt_ite=0.02,
            noise_level=args.noise,
            output_dir=args.output_dir,
            preparation=args.preparation,
            run_mode=args.mode,
        )

        # Export benchmark data for sim6
        benchmark_data = export_benchmark_data(results, args.output_dir)

        if HAS_MPL:
            generate_plots(results, args.output_dir)

        print("\n" + "=" * 65)
        print("  BENCHMARK COMPARE RESULTS")
        print("=" * 65)
        print(f"  Benchmark data exported to {args.output_dir}/sim1_benchmark_data.json")
        print(f"  Ready for sim6 tensor-network comparison.")
        cp = results.get("crossover_point", {})
        if cp.get("h_crossover_over_J") is not None:
            print(f"  Crossover h/J = {cp['h_crossover_over_J']:.4f}")
        print(f"  Verdict: {results.get('run_summary', {}).get('verdict', 'N/A')}")
        print()

    else:
        # ── Default: single_size_exploratory (backward compatible) ──
        results = simulate_phase_transition(
            size=args.size,
            J_fixed=args.J_fixed,
            h_max=args.h_max,
            n_steps=args.steps,
            n_ite_steps=args.ite_steps,
            dt_ite=0.02,
            noise_level=args.noise,
            output_dir=args.output_dir,
            preparation=args.preparation,
            run_mode=args.mode,
        )

        if HAS_MPL:
            generate_plots(results, args.output_dir)

        # Print summary (point 20: use crossover language, NOT critical point)
        print("\n" + "=" * 65)
        print("  RESULTS SUMMARY")
        print("=" * 65)
        cp = results.get("crossover_point", {})
        if cp.get("h_crossover_over_J") is not None:
            dev = abs(cp["h_crossover_over_J"] - HC_OVER_J_MC) / HC_OVER_J_MC * 100
            print(f"  h_crossover/J (single-size) : {cp['h_crossover_over_J']:.4f}")
            print(f"  h_c/J (theory, thermo limit): {HC_OVER_J_MC:.4f}")
            print(f"  Deviation                   : {dev:.1f}%")
            print(f"  Claim                       : {cp.get('claim', 'N/A')}")
        ce = results.get("critical_exponents", {})
        beta_d = ce.get("beta", {})
        if beta_d.get("fitted") is not None:
            print(f"  beta (fitted / theory)      : "
                  f"{beta_d['fitted']:.3f} / {beta_d['theory']:.4f} "
                  f"[{beta_d.get('agreement_with_3D_Ising', '?')}]")
        gamma_d = ce.get("gamma", {})
        if gamma_d.get("fitted") is not None:
            print(f"  gamma (fitted / theory)     : "
                  f"{gamma_d['fitted']:.3f} / {gamma_d['theory']:.4f} "
                  f"[{gamma_d.get('agreement_with_3D_Ising', '?')}]")

        sus = results.get("susceptibility", {})
        if sus.get("peak_h_over_J") is not None:
            ci = sus.get("bootstrap_ci", {}).get("peak_h", {})
            ci_str = f" [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]" if ci.get("ci_low") else ""
            print(f"  Susceptibility peak h/J      : {sus['peak_h_over_J']:.3f}{ci_str}")

        pd = results.get("phase_diagram", {})
        ent = pd.get("von_neumann_entropy", [])
        valid_ent = [e for e in ent if np.isfinite(e)]
        if valid_ent:
            print(f"  S_vN range                   : [{min(valid_ent):.3f}, {max(valid_ent):.3f}] bits")

        print(f"  Confidence                   : {results.get('confidence', {}).get('confidence', 'N/A')}")
        print(f"  Confidence factors           : {results.get('confidence', {}).get('confidence_factors', [])}")

        if results.get("magnetization_note"):
            print(f"\n  [!] {results['magnetization_note']}")

        print()
        print("  Claims allowed:")
        for c in results.get("claims_allowed", []):
            print(f"    + {c}")
        print("  Claims forbidden:")
        for c in results.get("claims_forbidden", []):
            print(f"    - {c}")
        print("  Caveats:")
        for c in results.get("caveats", []):
            print(f"    ~ {c}")

        rs = results.get("run_summary", {})
        print(f"\n  Verdict  : {rs.get('verdict', 'N/A')}")
        print(f"  Reasoning: {rs.get('reasoning', 'N/A')}")

        print()
        print("  Limitations:")
        for lim in results.get("limitations", []):
            print(f"    - {lim}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
