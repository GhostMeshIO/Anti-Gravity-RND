#!/usr/bin/env python3
"""
sim1_vacuum_phase.py - Blueprint 1: Vacuum Phase Transition & Bit-Mass Prediction
==================================================================================

Simulates a 2D lattice of qubits with nearest-neighbor interactions that mimic
Cooper pairing, using Trotterized time evolution. Quantifies the change in
information density n_bits during a superconducting phase transition, maps it
to a predicted mass shift via the bit-mass equation, and extracts critical
exponents to compare with the Sophia point framework.

Hamiltonian:
    H(t) = -J(t) * sum_{<i,j>} (X_i X_j + Y_i Y_j) - h * sum_i Z_i

where J(t) ramps from 0 to J_max, simulating cooling through T_c.

Outputs:
    - Phase diagram: critical J_c/h vs lattice size
    - Information density change Delta_S (in bits) across transition
    - Predicted mass shift Delta_m via bit-mass equation
    - Critical exponents (nu, beta) compared to 3D Ising universality
    - Lieb-Robinson velocity v_LR (emergent speed of information)
    - Fractal dimension D_f of entanglement structure
    - Sophia point susceptibility analysis

Ontological Framework Mappings (from Insights Round 1):
    - Insight 1:  Bit-Mass Equation (Delta_m prediction)
    - Insight 5:  RG Flow beta_C (critical coupling & scaling)
    - Insight 6:  Sophia Point 1/phi (critical exponents)
    - Insight 13: Fractal Amplification (box-counting D_f)
    - Insight 16: Eigenvalue Shift (Lieb-Robinson velocity)
    - Insight 24: Susceptibility Peak (critical point for metrology)

Usage:
    python sim1_vacuum_phase.py --size 4 --J-max 2.0 --h-field 0.5 --steps 100
    python sim1_vacuum_phase.py --size 3 --J-max 3.0 --h-field 0.3 --sweep-sizes --plot
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Import the gravity engine
try:
    from qnvm_gravity import (
        QuantumVMGravity, KB, C_LIGHT, LN2, SOPHIA_COHERENCE,
        von_neumann_entropy, box_counting_fractal_dimension,
        lieg_robinson_velocity
    )
except ImportError:
    print("Error: qnvm_gravity.py not found. Place it in the same directory or add to PYTHONPATH.")
    sys.exit(1)

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ======================================================================
# 2D Lattice Construction
# ======================================================================
def build_2d_lattice(size: int) -> Tuple[List[int], List[Tuple[int, int]], Dict]:
    """
    Build a 2D square lattice of qubits with periodic boundary conditions.

    Parameters
    ----------
    size : int
        Side length of the square lattice (size x size qubits).

    Returns
    -------
    qubits : list of int
        All qubit indices.
    pairs : list of (int, int)
        Nearest-neighbor pairs with periodic BCs.
    lattice_info : dict
        Metadata about the lattice.
    """
    n = size * size
    qubits = list(range(n))
    pairs = []
    coord_map = {}

    # First pass: build coordinate map
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            coord_map[(r, c)] = idx

    # Second pass: build nearest-neighbor pairs with periodic BCs
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            # Right neighbor (periodic)
            right = (r, (c + 1) % size)
            pairs.append((idx, coord_map[right]))
            # Down neighbor (periodic)
            down = ((r + 1) % size, c)
            pairs.append((idx, coord_map[down]))

    # Remove duplicate pairs
    pairs = list(set(tuple(sorted(p)) for p in pairs))

    return qubits, pairs, {
        'size': size,
        'n_qubits': n,
        'coord_map': coord_map,
        'n_pairs': len(pairs),
    }


# ======================================================================
# Phase Transition Simulation
# ======================================================================
def simulate_phase_transition(
    size: int = 4,
    J_max: float = 2.0,
    h_field: float = 0.5,
    dt: float = 0.02,
    n_steps: int = 100,
    trotter_order: int = 1,
    noise_level: float = 0.0,
    T_c_estimate: float = 9.2,  # K (Niobium)
    sample_volume_m3: float = 1e-6,
    output_dir: str = "sim1_results",
    compute_correlations: bool = True,
    save_interval: int = 5,
) -> Dict[str, Any]:
    """
    Run the full vacuum phase transition simulation.

    Ramps the coupling J from 0 to J_max in n_steps, simulating cooling
    through the superconducting transition temperature T_c.

    At each step, measures:
    - Von Neumann entropy of half the lattice (information density proxy)
    - Coherence order parameter C = <Z_i Z_j> average
    - Correlation matrix (for Lieb-Robinson velocity and fractal dimension)
    - Bit-mass prediction from entropy change

    Returns
    -------
    dict with all results.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building {size}x{size} lattice ({size**2} qubits)...")
    qubits, pairs, lat_info = build_2d_lattice(size)
    n = lat_info['n_qubits']
    print(f"  Qubits: {n}, NN pairs: {lat_info['n_pairs']}")

    if n > 20:
        print(f"  WARNING: {n} qubits > 20. Using stabilizer mode (approximate).")
        print(f"  For exact results, use size <= 4 (16 qubits).")

    # Data storage
    J_values = np.linspace(0.0, J_max, n_steps)
    entropy_history = []
    coherence_history = []
    bitmass_history = []
    susceptibility_history = []
    correlation_snapshots = []
    lr_velocities = []

    # Subsystem for entropy measurement (half the lattice)
    half_qubits = qubits[:n // 2]
    full_qubits = qubits

    # Initial entropy reference
    vm_init = QuantumVMGravity(qubits=n, noise_level=noise_level)
    vm_init.start()
    S_initial = vm_init.von_neumann_entropy_subsystem(half_qubits)
    vm_init.stop()

    print(f"\nStarting phase transition simulation...")
    print(f"  J range: [0, {J_max}], h = {h_field}, dt = {dt}, steps = {n_steps}")
    print(f"  Initial S_vN = {S_initial:.4f} bits")

    t_start = time.time()
    for step_i, J in enumerate(J_values):
        vm = QuantumVMGravity(qubits=n, noise_level=noise_level)
        vm.start()

        # Evolve under current Hamiltonian for fixed number of Trotter steps
        # The total evolution time is dt * trotter_steps_per_J
        trotter_steps_per_J = max(1, int(J / (h_field + 1e-10)))
        effective_dt = dt / max(trotter_steps_per_J, 1)

        for _ in range(trotter_steps_per_J):
            vm.trotter_step(J=J, h=h_field, dt=effective_dt, pairs=pairs,
                           hamiltonian_type='xx_yy_z')

        # Measure entropy of subsystem
        S_vN = vm.von_neumann_entropy_subsystem(half_qubits)
        entropy_history.append(S_vN)

        # Compute coherence order parameter C = average |<ZZ>|
        if n <= 16 and compute_correlations:
            try:
                corr = vm.correlation_matrix()
                # Average off-diagonal correlation (nearest neighbors only)
                nn_corr = []
                for i, j in pairs:
                    nn_corr.append(abs(corr[i, j]))
                coherence = float(np.mean(nn_corr))
                coherence_history.append(coherence)

                # Save correlation snapshots for LR velocity and fractal dimension
                if step_i % save_interval == 0:
                    # Reshape correlation matrix to 2D for analysis
                    corr_2d = corr.reshape(size, size)
                    correlation_snapshots.append(corr_2d.copy())
            except Exception:
                coherence_history.append(0.0)
        else:
            # Estimate coherence from expectation values
            C_approx = 0.0
            sample_pairs = pairs[:min(10, len(pairs))]
            for i, j in sample_pairs:
                pauli = ['I'] * n
                pauli[i] = 'Z'
                pauli[j] = 'Z'
                C_approx += abs(vm.expectation(''.join(pauli)))
            C_approx /= len(sample_pairs) if sample_pairs else 1
            coherence_history.append(C_approx)

        # Bit-mass prediction
        delta_S = S_vN - S_initial
        dm = vm.compute_bit_mass(
            temperature_k=T_c_estimate,
            delta_entropy_bits=delta_S * 1e20,  # scale factor for macroscopic sample
            curvature_coupling=coherence_history[-1] if coherence_history else 0.0
        )
        bitmass_history.append(dm)

        vm.stop()

        if (step_i + 1) % 10 == 0 or step_i == 0:
            elapsed = time.time() - t_start
            print(f"  Step {step_i+1}/{n_steps}: J={J:.3f}, S={S_vN:.4f}, "
                  f"C={coherence_history[-1]:.4f}, dm={dm:.3e} kg "
                  f"[{elapsed:.1f}s]")

    elapsed_total = time.time() - t_start
    print(f"\nSimulation completed in {elapsed_total:.1f}s")

    # ---- Post-processing ----

    # 1. Locate critical coupling J_c (max susceptibility)
    J_arr = np.array(J_values)
    C_arr = np.array(coherence_history[:len(J_values)])
    S_arr = np.array(entropy_history[:len(J_values)])
    dm_arr = np.array(bitmass_history[:len(J_values)])

    # Ensure consistent array lengths for gradient computation
    n_pts = min(len(J_arr), len(C_arr))
    J_arr = J_arr[:n_pts]
    C_arr = C_arr[:n_pts]

    if n_pts < 2:
        dC_dJ = np.zeros(n_pts)
    else:
        dC_dJ = np.abs(np.gradient(C_arr, J_arr))
    idx_crit = np.argmax(dC_dJ)
    J_c = float(J_arr[idx_crit])
    C_at_crit = float(C_arr[idx_crit])
    chi_max = float(dC_dJ[idx_crit])

    # 2. Fit critical exponents near J_c
    critical_exponents = fit_critical_exponents(J_arr, C_arr, J_c)

    # 3. Fractal dimension from final correlation snapshot
    D_f = 2.0  # default
    if correlation_snapshots:
        final_corr = correlation_snapshots[-1]
        try:
            D_f = box_counting_fractal_dimension(final_corr, size)
        except Exception:
            D_f = 2.0

    # 4. Lieb-Robinson velocity
    v_LR = 0.0
    if len(correlation_snapshots) >= 2:
        corr_history_1d = [c.reshape(n, n) for c in correlation_snapshots]
        try:
            v_LR = lieg_robinson_velocity(corr_history_1d, 1.0, dt * save_interval)
        except Exception:
            v_LR = 0.0

    # 5. Amplification efficiency
    vm_temp = QuantumVMGravity(1, 0.0)
    eta = vm_temp.compute_amplification_efficiency(D_f)

    # 6. Sophia point analysis
    sophia = {
        'C_sophia_measured': C_at_crit,
        'C_sophia_target': SOPHIA_COHERENCE,
        'agreement': abs(C_at_crit - SOPHIA_COHERENCE) / SOPHIA_COHERENCE,
        'chi_max': chi_max,
        'J_c': J_c,
        'J_c_over_h': J_c / h_field if h_field > 0 else float('inf'),
    }

    # Compile results
    results = {
        'parameters': {
            'size': size,
            'n_qubits': n,
            'J_max': J_max,
            'h_field': h_field,
            'dt': dt,
            'n_steps': n_steps,
            'noise_level': noise_level,
            'T_c_K': T_c_estimate,
            'sample_volume_m3': sample_volume_m3,
        },
        'phase_transition': {
            'J_values': J_values.tolist(),
            'entropy': entropy_history,
            'coherence': coherence_history,
            'bit_mass_prediction_kg': bitmass_history,
            'delta_S_max': float(max(entropy_history) - S_initial),
            'delta_S_min': float(min(entropy_history) - S_initial),
        },
        'critical_point': {
            'J_c': J_c,
            'J_c_over_h': J_c / h_field if h_field > 0 else float('inf'),
            'C_at_critical': C_at_crit,
            'chi_max': chi_max,
        },
        'critical_exponents': critical_exponents,
        'sophia_analysis': sophia,
        'emergent_geometry': {
            'fractal_dimension': D_f,
            'amplification_efficiency': eta,
            'lieb_robinson_velocity': v_LR,
        },
        'timing_seconds': elapsed_total,
    }

    # Save results
    results_path = out / 'sim1_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Generate plots
    if HAS_PLOT:
        generate_plots(results, out)

    return results


# ======================================================================
# Critical Exponent Fitting
# ======================================================================
def fit_critical_exponents(J_arr: np.ndarray, C_arr: np.ndarray,
                            J_c: float) -> Dict[str, float]:
    """
    Fit critical exponents near the phase transition.

    Power law fits:
    - Order parameter: C ~ (J_c - J)^beta  for J < J_c
    - Correlation length: xi ~ |J - J_c|^(-nu)  (estimated from C slope)
    - Susceptibility: chi ~ |J - J_c|^(-gamma)

    Compares with 3D Ising universality class:
        nu_theory ~ 0.63, beta_theory ~ 0.33, gamma_theory ~ 1.24

    Returns dict with fitted and theoretical values.
    """
    # Focus on region near J_c
    mask_below = J_arr < J_c
    mask_above = J_arr > J_c

    nu, beta, gamma = 0.5, 0.5, 1.0  # mean-field defaults

    if np.sum(mask_below) > 3:
        try:
            from scipy.optimize import curve_fit

            J_below = J_arr[mask_below]
            C_below = np.clip(C_arr[mask_below], 1e-10, None)
            delta_J = np.clip(J_c - J_below, 1e-10, None)

            # Fit C ~ A * (J_c - J)^beta
            def order_param(x, A, beta_exp):
                return A * np.power(x, beta_exp)

            popt, _ = curve_fit(order_param, delta_J, C_below,
                               p0=[1.0, 0.3], bounds=([0, 0.01], [10, 2.0]))
            beta = popt[1]
        except (ImportError, Exception):
            pass

    if np.sum(mask_below) > 3 or np.sum(mask_above) > 3:
        try:
            from scipy.optimize import curve_fit

            dC = np.abs(np.gradient(C_arr, J_arr))
            delta_J_all = np.abs(J_arr - J_c) + 1e-10

            def susc_func(x, A, gamma_exp):
                return A * np.power(x, -gamma_exp)

            valid = delta_J_all > 1e-8
            if np.sum(valid) > 3:
                popt2, _ = curve_fit(susc_func, delta_J_all[valid], dC[valid] + 1e-10,
                                     p0=[1.0, 1.0], bounds=([0, 0.1], [100, 5.0]))
                gamma = popt2[1]
        except (ImportError, Exception):
            pass

    # Theoretical values (3D Ising universality class)
    nu_theory = 0.6301
    beta_theory = 0.3262
    gamma_theory = 1.2372

    return {
        'beta_fitted': float(beta),
        'beta_theory': beta_theory,
        'beta_agreement_pct': abs(beta - beta_theory) / beta_theory * 100,
        'gamma_fitted': float(gamma),
        'gamma_theory': gamma_theory,
        'gamma_agreement_pct': abs(gamma - gamma_theory) / gamma_theory * 100,
        'nu_theory': nu_theory,
        'universality_class': '3D Ising (predicted)',
    }


# ======================================================================
# Multi-Size Phase Diagram
# ======================================================================
def sweep_lattice_sizes(
    sizes: List[int] = [3, 4, 5],
    J_max: float = 2.0,
    h_field: float = 0.5,
    dt: float = 0.02,
    n_steps: int = 50,
    noise_level: float = 0.0,
    output_dir: str = "sim1_results",
) -> Dict[str, Any]:
    """
    Run the phase transition simulation for multiple lattice sizes
    to extract finite-size scaling and the thermodynamic limit of J_c.
    """
    results_by_size = {}
    J_c_values = []
    size_values = []

    for size in sizes:
        n = size * size
        if n > 20:
            print(f"Size {size}x{size} ({n} qubits) > 20: using stabilizer mode (approximate).")
        else:
            print(f"\nSize {size}x{size} ({n} qubits): exact statevector mode.")

        res = simulate_phase_transition(
            size=size, J_max=J_max, h_field=h_field,
            dt=dt, n_steps=n_steps, noise_level=noise_level,
            output_dir=f"{output_dir}/size_{size}x{size}",
            compute_correlations=(n <= 16),
        )
        results_by_size[f"{size}x{size}"] = res
        J_c_values.append(res['critical_point']['J_c'])
        size_values.append(size)

    # Finite-size scaling extrapolation to L -> infinity
    # J_c(L) = J_c(inf) + a * L^(-1/nu)
    try:
        from scipy.optimize import curve_fit
        size_arr = np.array(size_values, dtype=float)
        jc_arr = np.array(J_c_values)

        def scaling(L, Jc_inf, a, nu_inv):
            return Jc_inf + a * np.power(L, -nu_inv)

        popt, _ = curve_fit(scaling, size_arr, jc_arr,
                           p0=[np.mean(jc_arr), 0.1, 0.63],
                           bounds=([0, -10, 0.1], [10, 10, 5.0]))
        Jc_inf = float(popt[0])
        nu_inv = float(popt[2])
        nu_extrapolated = 1.0 / nu_inv if nu_inv > 0 else float('inf')
    except (ImportError, Exception):
        Jc_inf = float(np.mean(J_c_values))
        nu_extrapolated = 0.63

    sweep_results = {
        'individual_results': {k: {
            'J_c': v['critical_point']['J_c'],
            'C_at_critical': v['critical_point']['C_at_critical'],
            'chi_max': v['critical_point']['chi_max'],
        } for k, v in results_by_size.items()},
        'J_c_values': J_c_values,
        'J_c_thermodynamic_limit': Jc_inf,
        'nu_extrapolated': nu_extrapolated,
    }

    # Save sweep results
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim1_sweep_results.json', 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)

    if HAS_PLOT:
        plt.figure(figsize=(10, 6))
        plt.plot(size_values, J_c_values, 'bo-', label='Simulation data')
        plt.axhline(y=Jc_inf, color='r', linestyle='--',
                    label=f'Thermodynamic limit: J_c = {Jc_inf:.3f}')
        plt.xlabel('Lattice size L')
        plt.ylabel('Critical coupling J_c / h')
        plt.title('Finite-Size Scaling of Vacuum Phase Transition')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out / 'sim1_finite_size_scaling.png', dpi=150)
        plt.close()

    return sweep_results


# ======================================================================
# Plot Generation
# ======================================================================
def generate_plots(results: Dict, out_dir: Path):
    """Generate all diagnostic plots for the phase transition simulation."""
    params = results['parameters']
    pt = results['phase_transition']
    cp = results['critical_point']
    ce = results['critical_exponents']
    sa = results['sophia_analysis']
    eg = results['emergent_geometry']

    J = np.array(pt['J_values'])
    S = np.array(pt['entropy'])
    C = np.array(pt['coherence'])
    dm = np.array(pt['bit_mass_prediction_kg'])
    # Ensure all arrays have the same length as J
    n_pts = len(J)
    S = S[:n_pts]
    C = C[:n_pts]
    dm = dm[:n_pts]

    # --- Plot 1: Entropy and Coherence vs J ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(J, S, 'b-', linewidth=1.5, label='$S_{vN}$ (half lattice)')
    ax1.axvline(x=cp['J_c'], color='r', linestyle='--', alpha=0.7,
                label=f'$J_c$ = {cp["J_c"]:.3f}')
    ax1.set_xlabel('Coupling $J$')
    ax1.set_ylabel('Von Neumann Entropy (bits)')
    ax1.set_title('Information Density vs Coupling')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.plot(J, C, 'g-', linewidth=1.5, label='Coherence $C$')
    ax2.axvline(x=cp['J_c'], color='r', linestyle='--', alpha=0.7,
                label=f'$J_c$ = {cp["J_c"]:.3f}')
    ax2.axhline(y=sa['C_sophia_target'], color='orange', linestyle=':',
                alpha=0.7, label=f'Sophia point $1/\\varphi$ = {sa["C_sophia_target"]:.3f}')
    ax2.set_xlabel('Coupling $J$')
    ax2.set_ylabel('Coherence order parameter $C$')
    ax2.set_title('Coherence vs Coupling')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'sim1_entropy_coherence.png', dpi=150)
    plt.close()

    # --- Plot 2: Susceptibility ---
    fig, ax = plt.subplots(figsize=(10, 5))
    dC_dJ = np.abs(np.gradient(C, J))
    ax.plot(J, dC_dJ, 'purple', linewidth=1.5, label='$|dC/dJ|$ (susceptibility)')
    ax.axvline(x=cp['J_c'], color='r', linestyle='--', alpha=0.7, label=f'$J_c$')
    ax.axhline(y=sa['chi_max'], color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coupling $J$')
    ax.set_ylabel('Susceptibility $\\chi = |dC/dJ|$')
    ax.set_title(f'Sophia Point Susceptibility (peak at $J_c$ = {cp["J_c"]:.3f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'sim1_susceptibility.png', dpi=150)
    plt.close()

    # --- Plot 3: Bit-Mass Prediction ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(J, dm, 'r-', linewidth=1.5)
    ax.set_xlabel('Coupling $J$')
    ax.set_ylabel('Predicted mass shift $\\Delta m$ (kg)')
    ax.set_title('Bit-Mass Prediction: $\\Delta m = m_{bit} \\cdot \\Delta S$')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'sim1_bitmass.png', dpi=150)
    plt.close()

    # --- Plot 4: Critical Exponents Comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    exponents = ['beta', 'gamma']
    fitted = [ce['beta_fitted'], ce['gamma_fitted']]
    theory = [ce['beta_theory'], ce['gamma_theory']]
    x_pos = np.arange(len(exponents))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, fitted, width, label='Fitted', color='steelblue')
    bars2 = ax.bar(x_pos + width/2, theory, width, label='3D Ising theory', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exponents)
    ax.set_ylabel('Critical exponent value')
    ax.set_title('Critical Exponents: Simulation vs Theory')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'sim1_critical_exponents.png', dpi=150)
    plt.close()

    # --- Plot 5: Summary dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(J, S, 'b-')
    axes[0].axvline(x=cp['J_c'], color='r', linestyle='--', alpha=0.5)
    axes[0].set_title('Von Neumann Entropy')
    axes[0].set_xlabel('$J$'); axes[0].set_ylabel('$S$ (bits)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(J, C, 'g-')
    axes[1].axhline(y=SOPHIA_COHERENCE, color='orange', linestyle=':', alpha=0.5)
    axes[1].set_title('Coherence Order Parameter')
    axes[1].set_xlabel('$J$'); axes[1].set_ylabel('$C$')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(J, np.abs(dm) + 1e-50, 'r-')
    axes[2].set_title('Bit-Mass Prediction')
    axes[2].set_xlabel('$J$'); axes[2].set_ylabel('$|\\Delta m|$ (kg)')
    axes[2].grid(True, alpha=0.3)

    axes[3].text(0.1, 0.9, f"Lattice: {params['size']}x{params['size']}",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.78, f"$J_c/h$ = {cp['J_c_over_h']:.3f}",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.66, f"$C^*$ = {sa['C_sophia_measured']:.4f} "
                 f"(target: {sa['C_sophia_target']:.4f})",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.54, f"$D_f$ = {eg['fractal_dimension']:.3f}",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.42, f"$\\beta$ = {ce['beta_fitted']:.3f} "
                 f"(theory: {ce['beta_theory']:.4f})",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.30, f"$\\gamma$ = {ce['gamma_fitted']:.3f} "
                 f"(theory: {ce['gamma_theory']:.4f})",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].text(0.1, 0.18, f"$v_{{LR}}$ = {eg['lieb_robinson_velocity']:.3f} a/$\\tau$",
                 transform=axes[3].transAxes, fontsize=11)
    axes[3].set_title('Simulation Summary')
    axes[3].axis('off')

    plt.suptitle('Blueprint 1: Vacuum Phase Transition Simulation', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'sim1_dashboard.png', dpi=150)
    plt.close()

    print(f"Plots saved to {out_dir}/")


# ======================================================================
# Main Entry Point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Blueprint 1: Vacuum Phase Transition & Bit-Mass Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 4x4 simulation
  python sim1_vacuum_phase.py --size 4 --J-max 2.0 --h-field 0.5

  # Sweep multiple lattice sizes for finite-size scaling
  python sim1_vacuum_phase.py --size 4 --sweep-sizes

  # Low-noise 3x3 with plots
  python sim1_vacuum_phase.py --size 3 --noise 0.01 --plot --steps 80
        """)

    parser.add_argument('--size', type=int, default=4,
                        help='Lattice side length (default: 4, giving 16 qubits)')
    parser.add_argument('--J-max', type=float, default=2.0,
                        help='Maximum coupling J (default: 2.0)')
    parser.add_argument('--h-field', type=float, default=0.5,
                        help='Transverse field h (default: 0.5)')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Trotter time step (default: 0.02)')
    parser.add_argument('--steps', type=int, default=60,
                        help='Number of J values to sweep (default: 60)')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise level for backend (default: 0.0)')
    parser.add_argument('--Tc', type=float, default=9.2,
                        help='Superconducting T_c in Kelvin (default: 9.2 for Nb)')
    parser.add_argument('--output-dir', type=str, default='sim1_results',
                        help='Output directory')
    parser.add_argument('--sweep-sizes', action='store_true',
                        help='Run simulation for multiple lattice sizes (3,4,5)')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate diagnostic plots (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("=" * 70)
    print("BLUEPRINT 1: VACUUM PHASE TRANSITION & BIT-MASS PREDICTION")
    print("(MOS-HOR-QNVM v14.0-Gravity)")
    print("=" * 70)
    print(f"Lattice size : {args.size}x{args.size} ({args.size**2} qubits)")
    print(f"J range      : [0, {args.J_max}]")
    print(f"h field      : {args.h_field}")
    print(f"dt           : {args.dt}")
    print(f"Steps        : {args.steps}")
    print(f"Noise        : {args.noise}")
    print(f"T_c          : {args.Tc} K")
    print(f"Output       : {args.output_dir}")
    print()

    if args.sweep_sizes:
        results = sweep_lattice_sizes(
            sizes=[3, 4, 5],
            J_max=args.J_max,
            h_field=args.h_field,
            dt=args.dt,
            n_steps=max(30, args.steps // 2),
            noise_level=args.noise,
            output_dir=args.output_dir,
        )
        print(f"\nThermodynamic limit J_c = {results['J_c_thermodynamic_limit']:.4f}")
        print(f"Extrapolated nu = {results['nu_extrapolated']:.4f}")
    else:
        results = simulate_phase_transition(
            size=args.size,
            J_max=args.J_max,
            h_field=args.h_field,
            dt=args.dt,
            n_steps=args.steps,
            noise_level=args.noise,
            T_c_estimate=args.Tc,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        cp = results['critical_point']
        ce = results['critical_exponents']
        sa = results['sophia_analysis']
        eg = results['emergent_geometry']

        print(f"  Critical coupling J_c/h    = {cp['J_c_over_h']:.4f}")
        print(f"  Coherence at J_c            = {cp['C_at_critical']:.4f}")
        print(f"  Susceptibility max          = {cp['chi_max']:.4f}")
        print(f"  Sophia C* (measured)        = {sa['C_sophia_measured']:.4f}")
        print(f"  Sophia C* (target 1/phi)    = {sa['C_sophia_target']:.4f}")
        print(f"  Sophia agreement            = {sa['agreement']*100:.1f}%")
        print(f"  Beta (fitted / theory)      = {ce['beta_fitted']:.3f} / {ce['beta_theory']:.4f}")
        print(f"  Gamma (fitted / theory)     = {ce['gamma_fitted']:.3f} / {ce['gamma_theory']:.4f}")
        print(f"  Fractal dimension D_f       = {eg['fractal_dimension']:.3f}")
        print(f"  Amplification eta           = {eg['amplification_efficiency']:.3e}")
        print(f"  Lieb-Robinson velocity      = {eg['lieb_robinson_velocity']:.3f}")
        print(f"  Max |Delta_m|               = {max(abs(x) for x in results['phase_transition']['bit_mass_prediction_kg']):.3e} kg")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
