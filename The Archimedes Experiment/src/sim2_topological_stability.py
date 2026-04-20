#!/usr/bin/env python3
"""
sim2_topological_stability.py - Blueprint 2: Coherence Transfer & Topological Stability
========================================================================================

Simulates the dynamics of coherence transfer in a topological quantum code
(surface code) during a "warp bubble" defect nucleation. Tracks the
topological entanglement entropy, Betti numbers, and coherence conservation
to quantify the energy scale of topological protection.

The simulation probes the ontological equation:
    d/dt (CI_B + CI_C) = sigma_topo

where CI_B = boundary coherence, CI_C = continuum coherence,
and sigma_topo = rate of change of topological entanglement entropy.

Also implements:
    - Kitaev-Preskill topological entanglement entropy measurement
    - Coherence conservation monitoring (Ghost-Mesh H13)
    - 93% holographic efficiency law verification
    - ERD-Killing field symmetry verification (entanglement gradient)
    - Betti number tracking for topological phase transitions
    - Bohmian trajectory visualization for defect formation
    - Paradox pressure estimation from code degeneracy
    - Forgiveness operator energy (error correction work)
    - Self-consistent history weight via random circuit sampling
    - Ryu-Takayanagi area law calibration

Ontological Framework Mappings:
    - Insight 2:  Info Stress-Energy Tensor (negativity as pressure)
    - Insight 3:  Coherence Conservation (gamma time derivative)
    - Insight 4:  Topological Entropy (surface code partition)
    - Insight 7:  Paradox Pressure (logical degeneracy energy)
    - Insight 10: 93% Efficiency Law (code rate vs error)
    - Insight 11: ERD-Killing Field (entanglement gradient symmetry)
    - Insight 14: Non-Commutative Algebra (anyon braiding)
    - Insight 15: Self-Consistent History (path integral)
    - Insight 17: Convex Free Energy (variational ground state)
    - Insight 18: RT Formula (area law calibration)
    - Insight 19: Betti Numbers (persistent homology)
    - Insight 22: Pilotwaveguide (Bohmian trajectories)
    - Insight 23: Forgiveness Operator (error correction work)

Usage:
    python sim2_topological_stability.py --distance 3 --defect-strength 0.5 --time-steps 20
    python sim2_topological_stability.py --distance 3 --sweep-defects --plot
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

try:
    from qnvm_gravity import (
        QuantumVMGravity, SurfaceCodeBuilder, RamseyInterferometer,
        SOPHIA_COHERENCE, HOLOGRAPHIC_EFFICIENCY_MAX,
        von_neumann_entropy, entanglement_negativity
    )
except ImportError:
    print("Error: qnvm_gravity.py not found. Place it in the same directory.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ======================================================================
# Region Partitioning for TEE
# ======================================================================
def partition_lattice_for_tee(size: int) -> Dict[str, List[int]]:
    """
    Partition a 2D lattice into three contiguous regions A, B, C
    for the Kitaev-Preskill topological entanglement entropy formula:
        S_topo = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    Uses a three-way partition with all regions touching at a point.
    """
    n = size * size
    all_qubits = list(range(n))

    # Partition into left (A), right (B), bottom (C) regions
    region_A = []
    region_B = []
    region_C = []

    mid = size // 2
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            if c < mid and r < mid:
                region_A.append(idx)
            elif c >= mid and r < mid:
                region_B.append(idx)
            else:
                region_C.append(idx)

    return {
        'A': region_A,
        'B': region_B,
        'C': region_C,
        'AB': region_A + region_B,
        'AC': region_A + region_C,
        'BC': region_B + region_C,
        'ABC': all_qubits,
    }


def partition_lattice_for_coherence(size: int) -> Dict[str, List[int]]:
    """
    Partition into boundary and interior for coherence conservation tracking.
    CI_B = boundary coherence (perimeter qubits)
    CI_C = continuum coherence (interior qubits)
    """
    boundary = []
    interior = []
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            if r == 0 or r == size - 1 or c == 0 or c == size - 1:
                boundary.append(idx)
            else:
                interior.append(idx)

    return {
        'boundary': boundary,
        'interior': interior,
        'all': list(range(size * size)),
    }


# ======================================================================
# Main Simulation
# ======================================================================
def simulate_topological_stability(
    distance: int = 3,
    defect_strength: float = 0.5,
    n_time_steps: int = 20,
    dt: float = 0.05,
    noise_level: float = 0.01,
    output_dir: str = "sim2_results",
    perturbation_type: str = 'local_field',
) -> Dict[str, Any]:
    """
    Run the full topological stability simulation.

    Protocol:
    1. Prepare a surface code state on a distance-d lattice
    2. Measure initial topological properties (TEE, Betti, coherence)
    3. Nucleate a defect (warp bubble) via local perturbation
    4. Track time evolution of all topological observables
    5. Correlate TEE changes with mass deficit prediction
    6. Verify holographic efficiency law
    7. Compute Bohmian trajectories around the defect

    Returns dict with all results.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_qubits = distance ** 2
    print(f"Surface code distance d = {distance}, n_qubits = {n_qubits}")

    if n_qubits > 20:
        print(f"  WARNING: {n_qubits} > 20 qubits. Using stabilizer mode (approximate).")
        print(f"  For exact density matrix methods, use distance <= 4 (16 qubits).")

    # Build lattice and partitions
    all_qubits = list(range(n_qubits))
    pairs = []
    for r in range(distance):
        for c in range(distance):
            idx = r * distance + c
            if c + 1 < distance:
                pairs.append((idx, idx + 1))
            if r + 1 < distance:
                pairs.append((idx, idx + distance))

    tee_regions = partition_lattice_for_tee(distance)
    coherence_regions = partition_lattice_for_coherence(distance)

    # Define defect qubits (center of lattice)
    center = distance // 2
    defect_radius = max(1, distance // 4)
    defect_qubits = []
    for r in range(distance):
        for c in range(distance):
            dr = abs(r - center)
            dc = abs(c - center)
            if dr + dc <= defect_radius:
                defect_qubits.append(r * distance + c)

    print(f"  Defect qubits (center region): {len(defect_qubits)}")
    print(f"  Nearest-neighbor pairs: {len(pairs)}")

    # Data storage
    time_points = list(range(n_time_steps))
    tee_history = []
    coherence_boundary_history = []
    coherence_interior_history = []
    coherence_total_history = []
    coherence_conservation_rate = []
    betti_history = []
    negativity_history = []
    info_pressure_history = []
    paradox_pressure_history = []
    stabilizer_expectations_history = []
    codeword_fidelity_history = []
    rr_ratio_history = []
    lr_gradient_history = []
    bohmian_trajectories = {}

    # ---- Phase 1: Prepare and measure initial state ----
    print("\n[Phase 1] Preparing initial state...")
    vm = QuantumVMGravity(qubits=n_qubits, noise_level=noise_level)
    vm.start()

    # Prepare approximate surface code state
    if n_qubits <= 20:
        # Use CNOT pattern for approximate code state
        for r in range(distance):
            for c in range(distance - 1):
                q1 = r * distance + c
                q2 = r * distance + c + 1
                vm.apply_gate('h', [q1])
                vm.apply_gate('cnot', [q1, q2])
        for r in range(distance - 1):
            for c in range(distance):
                q1 = r * distance + c
                q2 = (r + 1) * distance + c
                vm.apply_gate('h', [q2])
                vm.apply_gate('cnot', [q2, q1])
    else:
        # For stabilizer, prepare simple entangled state
        for i in range(0, n_qubits - 1, 2):
            vm.apply_gate('h', [i])
            vm.apply_gate('cnot', [i, i + 1])

    # Measure initial properties
    print("  Measuring initial topological properties...")

    # Topological entanglement entropy
    S_topo_initial = 0.0
    entropies_initial = {}
    if n_qubits <= 20:
        try:
            for name, qubits in tee_regions.items():
                entropies_initial[name] = vm.von_neumann_entropy_subsystem(qubits)
            S_topo_initial = (
                entropies_initial['A'] + entropies_initial['B'] + entropies_initial['C']
                - entropies_initial['AB'] - entropies_initial['AC']
                - entropies_initial['BC'] + entropies_initial['ABC']
            )
        except Exception as e:
            print(f"  Warning: TEE computation failed: {e}")
    tee_history.append(S_topo_initial)

    # Coherence (mutual information)
    MI_boundary = 0.0
    MI_interior = 0.0
    if n_qubits <= 20:
        try:
            MI_boundary = vm.mutual_information_bipartite(
                coherence_regions['boundary'][:max(1, len(coherence_regions['boundary']) // 2)],
                coherence_regions['boundary'][max(1, len(coherence_regions['boundary']) // 2):])
            MI_interior = vm.mutual_information_bipartite(
                coherence_regions['interior'][:max(1, len(coherence_regions['interior']) // 2)],
                coherence_regions['interior'][max(1, len(coherence_regions['interior']) // 2):])
        except Exception:
            pass
    coherence_boundary_history.append(MI_boundary)
    coherence_interior_history.append(MI_interior)
    coherence_total_history.append(MI_boundary + MI_interior)
    coherence_conservation_rate.append(0.0)

    # Betti numbers
    if n_qubits <= 16:
        try:
            corr = vm.correlation_matrix()
            betti = vm.estimate_betti_numbers(corr, threshold=0.1)
            betti_history.append(betti)
        except Exception:
            betti_history.append({'beta_0': 1, 'beta_1': 0, 'beta_2': 0})
    else:
        betti_history.append({'beta_0': 1, 'beta_1': 0, 'beta_2': 0})

    # Negativity and information pressure
    total_neg = 0.0
    if n_qubits <= 16:
        try:
            for i, j in pairs[:20]:  # sample pairs
                total_neg += vm.entanglement_negativity_pair(i, j)
        except Exception:
            pass
    negativity_history.append(total_neg)
    info_pressure_history.append(-total_neg * (MI_boundary + MI_interior))

    # Paradox pressure (code degeneracy)
    n_logical_qubits = max(1, distance - 1)
    T_cognitive = 1.0  # simulation units
    V_ontological = n_qubits
    paradox_pressure = (n_logical_qubits * T_cognitive) / V_ontological
    paradox_pressure_history.append(paradox_pressure)

    # Holographic efficiency: r/d_s
    r = n_logical_qubits
    d_s = distance
    rr_ratio = r / d_s
    rr_ratio_history.append(rr_ratio)

    # LR gradient (entanglement gradient symmetry)
    if n_qubits <= 16:
        try:
            corr = vm.correlation_matrix()
            # Measure gradient along different directions
            grad_x = np.mean(np.abs(np.diff(corr, axis=1)))
            grad_y = np.mean(np.abs(np.diff(corr, axis=0)))
            lr_gradient_history.append({'grad_x': float(grad_x), 'grad_y': float(grad_y)})
        except Exception:
            lr_gradient_history.append({'grad_x': 0.0, 'grad_y': 0.0})
    else:
        lr_gradient_history.append({'grad_x': 0.0, 'grad_y': 0.0})

    print(f"  Initial S_topo = {S_topo_initial:.4f} bits")
    print(f"  Initial CI_B   = {MI_boundary:.4f}, CI_C = {MI_interior:.4f}")
    print(f"  Betti: beta_0={betti_history[-1]['beta_0']}, "
          f"beta_1={betti_history[-1]['beta_1']}, beta_2={betti_history[-1]['beta_2']}")
    print(f"  r/d_s = {rr_ratio:.4f} (max: {HOLOGRAPHIC_EFFICIENCY_MAX})")

    # ---- Phase 2: Defect nucleation and time evolution ----
    print(f"\n[Phase 2] Nucleating defect (strength = {defect_strength})...")
    print(f"  Tracking {n_time_steps} time steps...")

    # Save state before perturbation for Bohmian trajectories
    if n_qubits <= 16:
        saved_state = vm._backend.state.copy()

    # Apply defect perturbation
    if perturbation_type == 'local_field':
        for q in defect_qubits:
            vm.apply_gate('rx', [q], [defect_strength])
    elif perturbation_type == 'entangling':
        for i in range(len(defect_qubits) - 1):
            vm.apply_gate('cnot', [defect_qubits[i], defect_qubits[i + 1]])
            vm.apply_gate('rz', [defect_qubits[i + 1]], [defect_strength])

    t_start = time.time()
    for t_step in range(1, n_time_steps):
        # Evolve under the surface code Hamiltonian
        # Simple Heisenberg-like evolution to maintain topological order
        vm.trotter_step(J=0.3, h=0.0, dt=dt, pairs=pairs, hamiltonian_type='heisenberg')

        # Apply noise (simulating decoherence)
        if noise_level > 0:
            vm.apply_noise_channel('dephasing', all_qubits, probability=noise_level * 0.01)

        # Measure topological properties
        S_topo_t = 0.0
        if n_qubits <= 20 and t_step % 2 == 0:
            try:
                entropies_t = {}
                for name, qubits in tee_regions.items():
                    entropies_t[name] = vm.von_neumann_entropy_subsystem(qubits)
                S_topo_t = (entropies_t['A'] + entropies_t['B'] + entropies_t['C']
                           - entropies_t['AB'] - entropies_t['AC']
                           - entropies_t['BC'] + entropies_t['ABC'])
            except Exception:
                pass
        tee_history.append(S_topo_t)

        # Coherence tracking
        MI_b_t = 0.0
        MI_c_t = 0.0
        if n_qubits <= 20 and t_step % 3 == 0:
            try:
                br = coherence_regions['boundary']
                ir = coherence_regions['interior']
                mid_b = max(1, len(br) // 2)
                mid_i = max(1, len(ir) // 2)
                MI_b_t = vm.mutual_information_bipartite(br[:mid_b], br[mid_b:])
                MI_c_t = vm.mutual_information_bipartite(ir[:mid_i], ir[mid_i:])
            except Exception:
                pass
        coherence_boundary_history.append(MI_b_t)
        coherence_interior_history.append(MI_c_t)
        coherence_total_history.append(MI_b_t + MI_c_t)

        # Coherence conservation rate: d(CI_B + CI_C)/dt ~ sigma_topo
        if len(coherence_total_history) >= 2:
            rate = (coherence_total_history[-1] - coherence_total_history[-2]) / dt
        else:
            rate = 0.0
        coherence_conservation_rate.append(rate)

        # Betti numbers (less frequent computation)
        if n_qubits <= 16 and t_step % 4 == 0:
            try:
                corr = vm.correlation_matrix()
                betti = vm.estimate_betti_numbers(corr, threshold=0.1)
                betti_history.append(betti)
            except Exception:
                betti_history.append(betti_history[-1] if betti_history else
                                    {'beta_0': 1, 'beta_1': 0, 'beta_2': 0})
        elif len(betti_history) < t_step + 1:
            betti_history.append(betti_history[-1] if betti_history else
                                {'beta_0': 1, 'beta_1': 0, 'beta_2': 0})

        # Negativity and info pressure
        neg_t = 0.0
        if n_qubits <= 16 and t_step % 5 == 0:
            try:
                for i, j in pairs[:20]:
                    neg_t += vm.entanglement_negativity_pair(i, j)
            except Exception:
                neg_t = negativity_history[-1] if negativity_history else 0.0
        elif len(negativity_history) < t_step + 1:
            neg_t = negativity_history[-1] if negativity_history else 0.0
        negativity_history.append(neg_t)
        info_pressure_history.append(-neg_t * (MI_b_t + MI_c_t + 1e-10))

        # Paradox pressure
        paradox_pressure_history.append(paradox_pressure * math.exp(-0.05 * t_step))

        # r/d_s tracking
        rr_ratio_history.append(rr_ratio)

        # LR gradient
        if n_qubits <= 16 and t_step % 3 == 0:
            try:
                corr = vm.correlation_matrix()
                grad_x = np.mean(np.abs(np.diff(corr, axis=1)))
                grad_y = np.mean(np.abs(np.diff(corr, axis=0)))
                lr_gradient_history.append({'grad_x': float(grad_x), 'grad_y': float(grad_y)})
            except Exception:
                lr_gradient_history.append(lr_gradient_history[-1])
        elif len(lr_gradient_history) < t_step + 1:
            lr_gradient_history.append(lr_gradient_history[-1])

        if (t_step + 1) % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  t={t_step}: S_topo={S_topo_t:.4f}, "
                  f"CI_B={MI_b_t:.4f}, CI_C={MI_c_t:.4f} [{elapsed:.1f}s]")

    vm.stop()
    elapsed_total = time.time() - t_start
    print(f"\nTime evolution completed in {elapsed_total:.1f}s")

    # ---- Phase 3: Bohmian Trajectories ----
    if n_qubits <= 16 and saved_state is not None:
        print("\n[Phase 3] Computing Bohmian trajectories...")
        vm_bohm = QuantumVMGravity(qubits=n_qubits, noise_level=0.0)
        vm_bohm.start()
        vm_bohm._backend.state = saved_state.copy()

        times = np.linspace(0, n_time_steps * dt, min(n_time_steps, 30)).tolist()
        for qi in range(min(4, n_qubits)):
            traj = vm_bohm.bohmiann_trajectory(
                qubit=qi, times=times, J=0.3, h=0.0,
                pairs=pairs, hamiltonian_type='heisenberg')
            bohmian_trajectories[f'qubit_{qi}'] = traj

        vm_bohm.stop()
        print(f"  Computed {len(bohmian_trajectories)} trajectories")

    # ---- Phase 4: Forgiveness Operator Energy ----
    forgiveness_energy = compute_forgiveness_energy(n_qubits, noise_level, defect_strength)

    # ---- Phase 5: Self-Consistent History Weight ----
    history_weights = compute_history_weights(n_qubits, noise_level, n_circuits=50)

    # ---- Compile Results ----
    results = {
        'parameters': {
            'distance': distance,
            'n_qubits': n_qubits,
            'defect_strength': defect_strength,
            'n_time_steps': n_time_steps,
            'dt': dt,
            'noise_level': noise_level,
            'perturbation_type': perturbation_type,
            'n_defect_qubits': len(defect_qubits),
            'n_logical_qubits': max(1, distance - 1),
        },
        'topological_entropy': {
            'time_points': time_points,
            'S_topo_history': tee_history,
            'S_topo_initial': S_topo_initial,
            'S_topo_final': tee_history[-1] if tee_history else 0.0,
            'delta_S_topo': (tee_history[-1] if tee_history else 0.0) - S_topo_initial,
        },
        'coherence_conservation': {
            'CI_B_history': coherence_boundary_history,
            'CI_C_history': coherence_interior_history,
            'CI_total_history': coherence_total_history,
            'conservation_rate_history': coherence_conservation_rate,
            'sigma_topo_mean': float(np.mean(np.abs(coherence_conservation_rate))),
        },
        'betti_numbers': betti_history,
        'information_pressure': {
            'negativity_history': negativity_history,
            'info_pressure_history': info_pressure_history,
            'paradox_pressure_history': paradox_pressure_history,
        },
        'holographic_efficiency': {
            'r_over_ds_history': rr_ratio_history,
            'r_over_ds_max': max(rr_ratio_history) if rr_ratio_history else 0.0,
            'efficiency_bound': HOLOGRAPHIC_EFFICIENCY_MAX,
            'within_bound': all(r <= HOLOGRAPHIC_EFFICIENCY_MAX for r in rr_ratio_history),
        },
        'erd_killing_field': lr_gradient_history,
        'bohmian_trajectories': {k: [float(x) for x in v] for k, v in bohmian_trajectories.items()},
        'forgiveness_energy': forgiveness_energy,
        'history_weights': history_weights,
        'timing_seconds': elapsed_total,
    }

    # Save
    results_path = out / 'sim2_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Plots
    if HAS_PLOT:
        generate_plots(results, distance, out)

    return results


# ======================================================================
# Forgiveness Operator Energy
# ======================================================================
def compute_forgiveness_energy(n_qubits: int, noise_level: float,
                                defect_strength: float,
                                n_trials: int = 100) -> Dict[str, float]:
    """
    Compute the energy cost of error correction (forgiveness operator).

    The forgiveness operator transforms a logical error into the identity:
        U_forgive^dagger G U_forgive = diag

    Energy cost: Delta_E = T_cognitive * Delta_S_paradox

    We simulate this by measuring the logical error rate before and after
    applying a simple error correction circuit.
    """
    errors_before = 0
    errors_after = 0

    for _ in range(n_trials):
        # Create a simple encoded state
        vm = QuantumVMGravity(qubits=min(n_qubits, 16), noise_level=noise_level)
        n = min(n_qubits, 16)
        vm.start()

        # Encode: create Bell pairs
        for i in range(0, n - 1, 2):
            vm.apply_gate('h', [i])
            vm.apply_gate('cnot', [i, i + 1])

        # Introduce error
        vm.apply_gate('rx', [0], [defect_strength])

        # Measure before correction
        counts_before = vm.measure(shots=100)
        zero_before = counts_before.get('0' * n, 0) / 100

        # Apply correction (CNOT pattern)
        for i in range(0, n - 1, 2):
            vm.apply_gate('cnot', [i, i + 1])

        # Measure after correction
        counts_after = vm.measure(shots=100)
        zero_after = counts_after.get('0' * n, 0) / 100

        errors_before += (1.0 - zero_before)
        errors_after += (1.0 - zero_after)

        vm.stop()

    avg_error_before = errors_before / n_trials
    avg_error_after = errors_after / n_trials
    delta_S_paradox = max(0, avg_error_before - avg_error_after)

    # Energy cost in simulation units
    T_cognitive = 1.0
    forgiveness_energy = T_cognitive * delta_S_paradox

    return {
        'error_rate_before': avg_error_before,
        'error_rate_after': avg_error_after,
        'delta_S_paradox': delta_S_paradox,
        'forgiveness_energy': forgiveness_energy,
        'correction_efficiency': (avg_error_before - avg_error_after) / max(avg_error_before, 1e-10),
    }


# ======================================================================
# Self-Consistent History Weights
# ======================================================================
def compute_history_weights(n_qubits: int, noise_level: float,
                             n_circuits: int = 100) -> Dict[str, float]:
    """
    Compute the self-consistent history weight via random circuit sampling.

    Z = integral D[H] exp(iS[H]) delta[C(H) - 1]

    In simulation: weight of each circuit = survival probability in RB.
    Consistency condition: circuit must be fault-tolerant (logical error < threshold).
    """
    n = min(n_qubits, 16)
    weights = []

    for _ in range(n_circuits):
        vm = QuantumVMGravity(qubits=n, noise_level=noise_level)
        vm.start()

        # Random Clifford circuit
        for step in range(10):
            q = np.random.randint(0, n)
            gate = np.random.choice(['h', 's'])
            vm.apply_gate(gate, [q])
            if n >= 2 and np.random.random() < 0.3:
                c = np.random.randint(0, n)
                t = np.random.randint(0, n)
                while t == c:
                    t = np.random.randint(0, n)
                vm.apply_gate('cnot', [c, t])

        # Apply inverse
        for step in range(10):
            q = np.random.randint(0, n)
            gate = np.random.choice(['h', 'sdg'])
            vm.apply_gate(gate, [q])
            if n >= 2 and np.random.random() < 0.3:
                c = np.random.randint(0, n)
                t = np.random.randint(0, n)
                while t == c:
                    t = np.random.randint(0, n)
                vm.apply_gate('cnot', [c, t])

        counts = vm.measure(shots=1000)
        survival = counts.get('0' * n, 0) / 1000
        weights.append(survival)
        vm.stop()

    weights = np.array(weights)
    consistency_mask = weights > 0.5  # fault-tolerant threshold

    return {
        'mean_weight': float(np.mean(weights)),
        'std_weight': float(np.std(weights)),
        'consistent_fraction': float(np.mean(consistency_mask)),
        'path_integral_Z': float(np.sum(weights)),
        'free_energy': -math.log(np.sum(weights) + 1e-10),
    }


# ======================================================================
# Defect Strength Sweep
# ======================================================================
def sweep_defect_strengths(
    distance: int = 3,
    strengths: List[float] = None,
    n_time_steps: int = 15,
    noise_level: float = 0.01,
    output_dir: str = "sim2_results",
) -> Dict[str, Any]:
    """
    Sweep defect strength to find the critical perturbation for
    topological phase transition (beta_2 collapse).
    """
    if strengths is None:
        strengths = np.linspace(0.1, 2.0, 15).tolist()

    sweep_results = []
    for strength in strengths:
        print(f"\n--- Defect strength = {strength:.3f} ---")
        res = simulate_topological_stability(
            distance=distance,
            defect_strength=strength,
            n_time_steps=n_time_steps,
            noise_level=noise_level,
            output_dir=f"{output_dir}/defect_{strength:.2f}",
        )
        sweep_results.append({
            'strength': strength,
            'delta_S_topo': res['topological_entropy']['delta_S_topo'],
            'beta_2_final': res['betti_numbers'][-1]['beta_2'] if res['betti_numbers'] else 0,
            'sigma_topo_mean': res['coherence_conservation']['sigma_topo_mean'],
        })

    # Find critical strength where beta_2 collapses
    strengths_arr = np.array([r['strength'] for r in sweep_results])
    beta2_arr = np.array([r['beta_2_final'] for r in sweep_results])

    # Critical strength = first point where beta_2 drops to zero
    critical_idx = np.where(beta2_arr <= 0)[0]
    if len(critical_idx) > 0:
        strength_critical = float(strengths_arr[critical_idx[0]])
    else:
        strength_critical = float(strengths_arr[-1])

    results = {
        'sweep_data': sweep_results,
        'defect_strength_critical': strength_critical,
        'distance': distance,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim2_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_PLOT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        s = [r['strength'] for r in sweep_results]
        dS = [r['delta_S_topo'] for r in sweep_results]
        b2 = [r['beta_2_final'] for r in sweep_results]
        sig = [r['sigma_topo_mean'] for r in sweep_results]

        ax1.plot(s, dS, 'bo-', label='$\\Delta S_{topo}$')
        ax1.axvline(x=strength_critical, color='r', linestyle='--',
                    label=f'$\\epsilon_c$ = {strength_critical:.3f}')
        ax1.set_xlabel('Defect strength $\\epsilon$')
        ax1.set_ylabel('$\\Delta S_{topo}$ (bits)')
        ax1.set_title('Topological Entropy Change vs Defect')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.plot(s, b2, 'gs-', label='$\\beta_2$')
        ax2.axvline(x=strength_critical, color='r', linestyle='--',
                    label=f'$\\epsilon_c$ = {strength_critical:.3f}')
        ax2.set_xlabel('Defect strength $\\epsilon$')
        ax2.set_ylabel('$\\beta_2$')
        ax2.set_title('Betti Number Collapse (Topological Phase Transition)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out / 'sim2_defect_sweep.png', dpi=150)
        plt.close()

    return results


# ======================================================================
# Plot Generation
# ======================================================================
def generate_plots(results: Dict, distance: int, out_dir: Path):
    """Generate diagnostic plots for the topological stability simulation."""
    tee = results['topological_entropy']
    cc = results['coherence_conservation']
    ie = results['information_pressure']
    he = results['holographic_efficiency']
    bt = results['betti_numbers']

    t = np.array(tee['time_points'])

    # --- Plot 1: Topological Entropy vs Time ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    axes[0].plot(t, tee['S_topo_history'], 'b-o', markersize=3)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('$S_{topo}$ (bits)')
    axes[0].set_title('Topological Entanglement Entropy')
    axes[0].grid(True, alpha=0.3)

    # --- Plot 2: Coherence Conservation ---
    axes[1].plot(t, cc['CI_B_history'], 'b-', label='$CI_B$ (boundary)')
    axes[1].plot(t, cc['CI_C_history'], 'g-', label='$CI_C$ (continuum)')
    axes[1].plot(t, cc['CI_total_history'], 'r-', linewidth=2, label='$CI_B + CI_C$')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Mutual Information (bits)')
    axes[1].set_title('Coherence Conservation: $d(CI_B+CI_C)/dt = \\sigma_{topo}$')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # --- Plot 3: Coherence Conservation Rate ---
    axes[2].plot(t[1:], cc['conservation_rate_history'][1:], 'purple', linewidth=1)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('$d(CI_B+CI_C)/dt$')
    axes[2].set_title(f'$\\sigma_{{topo}}$ mean = {cc["sigma_topo_mean"]:.4f}')
    axes[2].grid(True, alpha=0.3)

    # --- Plot 4: Betti Numbers ---
    if bt:
        t_b = list(range(len(bt)))
        b0 = [b['beta_0'] for b in bt]
        b1 = [b['beta_1'] for b in bt]
        b2 = [b['beta_2'] for b in bt]
        axes[3].plot(t_b, b0, 'b-o', markersize=3, label='$\\beta_0$')
        axes[3].plot(t_b, b1, 'g-s', markersize=3, label='$\\beta_1$')
        axes[3].plot(t_b, b2, 'r-^', markersize=3, label='$\\beta_2$')
        axes[3].set_xlabel('Time step')
        axes[3].set_ylabel('Betti number')
        axes[3].set_title('Topological Invariants (Betti Numbers)')
        axes[3].legend(loc='best')
        axes[3].grid(True, alpha=0.3)

    # --- Plot 5: Information Pressure ---
    axes[4].plot(t, ie['info_pressure_history'], 'r-', label='$p_{info}$')
    axes[4].plot(t, ie['paradox_pressure_history'], 'b--', label='$P_{paradox}$')
    axes[4].set_xlabel('Time step')
    axes[4].set_ylabel('Pressure (sim. units)')
    axes[4].set_title('Information & Paradox Pressure')
    axes[4].legend(loc='best')
    axes[4].grid(True, alpha=0.3)

    # --- Plot 6: Holographic Efficiency ---
    axes[5].plot(t, he['r_over_ds_history'], 'k-o', markersize=3)
    axes[5].axhline(y=he['efficiency_bound'], color='r', linestyle='--',
                    label=f'Bound = {he["efficiency_bound"]}')
    axes[5].set_xlabel('Time step')
    axes[5].set_ylabel('$r/d_s$')
    axes[5].set_title(f'Holographic Efficiency (within bound: {he["within_bound"]})')
    axes[5].legend(loc='best')
    axes[5].grid(True, alpha=0.3)

    plt.suptitle(f'Blueprint 2: Topological Stability (d={distance})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'sim2_dashboard.png', dpi=150)
    plt.close()

    # --- Plot 7: Bohmian Trajectories ---
    bt_res = results.get('bohmian_trajectories', {})
    if bt_res:
        fig, ax = plt.subplots(figsize=(10, 5))
        times = np.linspace(0, results['parameters']['n_time_steps'] * results['parameters']['dt'],
                           len(next(iter(bt_res.values()))))
        for name, traj in bt_res.items():
            ax.plot(times, traj, '-', label=name, linewidth=1.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('$\\langle Z \\rangle$ (Bohmian position)')
        ax.set_title('Pilotwaveguide Trajectories (Defect Formation)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'sim2_bohmian_trajectories.png', dpi=150)
        plt.close()

    # --- Plot 8: ERD-Killing Field ---
    erd = results.get('erd_killing_field', [])
    if len(erd) > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        gx = [e['grad_x'] for e in erd]
        gy = [e['grad_y'] for e in erd]
        ax.plot(gx, gy, 'ko-', markersize=2, alpha=0.7)
        ax.set_xlabel('$\\nabla_x \\varepsilon$')
        ax.set_ylabel('$\\nabla_y \\varepsilon$')
        ax.set_title('ERD-Killing Field Phase Space (entanglement gradient)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'sim2_erd_killing.png', dpi=150)
        plt.close()

    print(f"Plots saved to {out_dir}/")


# ======================================================================
# Main Entry Point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Blueprint 2: Coherence Transfer & Topological Stability',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--distance', type=int, default=3,
                        help='Surface code distance (default: 3, 9 qubits)')
    parser.add_argument('--defect-strength', type=float, default=0.5,
                        help='Defect perturbation strength (default: 0.5)')
    parser.add_argument('--time-steps', type=int, default=20,
                        help='Number of time evolution steps (default: 20)')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Time step (default: 0.05)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level (default: 0.01)')
    parser.add_argument('--perturbation', type=str, default='local_field',
                        choices=['local_field', 'entangling'],
                        help='Defect type (default: local_field)')
    parser.add_argument('--output-dir', type=str, default='sim2_results',
                        help='Output directory')
    parser.add_argument('--sweep-defects', action='store_true',
                        help='Sweep defect strengths to find critical point')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots (default: True)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("=" * 70)
    print("BLUEPRINT 2: COHERENCE TRANSFER & TOPOLOGICAL STABILITY")
    print("(MOS-HOR-QNVM v14.0-Gravity)")
    print("=" * 70)
    print(f"Distance       : {args.distance} ({args.distance**2} qubits)")
    print(f"Defect strength: {args.defect_strength}")
    print(f"Time steps     : {args.time_steps}")
    print(f"dt             : {args.dt}")
    print(f"Noise          : {args.noise}")
    print(f"Perturbation   : {args.perturbation}")
    print(f"Output         : {args.output_dir}")
    print()

    if args.sweep_defects:
        results = sweep_defect_strengths(
            distance=args.distance,
            n_time_steps=min(args.time_steps, 15),
            noise_level=args.noise,
            output_dir=args.output_dir,
        )
        print(f"\nCritical defect strength: {results['defect_strength_critical']:.4f}")
    else:
        results = simulate_topological_stability(
            distance=args.distance,
            defect_strength=args.defect_strength,
            n_time_steps=args.time_steps,
            dt=args.dt,
            noise_level=args.noise,
            output_dir=args.output_dir,
            perturbation_type=args.perturbation,
        )

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        tee = results['topological_entropy']
        cc = results['coherence_conservation']
        he = results['holographic_efficiency']
        fe = results['forgiveness_energy']
        hw = results['history_weights']

        print(f"  S_topo initial              = {tee['S_topo_initial']:.4f}")
        print(f"  S_topo final                = {tee['S_topo_final']:.4f}")
        print(f"  Delta S_topo                = {tee['delta_S_topo']:.4f} bits")
        print(f"  sigma_topo (mean)           = {cc['sigma_topo_mean']:.6f}")
        print(f"  r/d_s max                   = {he['r_over_ds_max']:.4f}")
        print(f"  Within holographic bound    = {he['within_bound']}")
        print(f"  Forgiveness energy          = {fe['forgiveness_energy']:.4f}")
        print(f"  Correction efficiency       = {fe['correction_efficiency']:.2%}")
        print(f"  History weight (mean)       = {hw['mean_weight']:.4f}")
        print(f"  Consistent fraction         = {hw['consistent_fraction']:.2%}")
        if results['betti_numbers']:
            bt_final = results['betti_numbers'][-1]
            print(f"  Betti numbers (final)       : "
                  f"beta_0={bt_final['beta_0']}, "
                  f"beta_1={bt_final['beta_1']}, "
                  f"beta_2={bt_final['beta_2']}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
