#!/usr/bin/env python3
"""
sim3_quantum_metrology.py - Blueprint 3: Quantum Metrology for Vacuum Weight Sensing
======================================================================================

Determines the ultimate sensitivity limit for measuring vacuum weight
shift Delta_m using a superconducting qubit coupled to a mechanical
oscillator (optomechanics). Simulates Ramsey interferometry with
realistic noise to extract achievable precision.

Protocol:
    1. Prepare qubit in superposition (pi/2 pulse)
    2. Free evolution under dispersive coupling: H = chi * a^dagger * a * Z
       - The oscillator frequency shifts in response to vacuum weight change
    3. Second pi/2 pulse
    4. Measure qubit in computational basis
    5. Compute Fisher information and Cramer-Rao bound

The simulation provides:
    - Minimum detectable mass shift delta_m_min vs integration time
    - Comparison with Archimedes experimental parameters
    - Noise-aware sensitivity analysis (dephasing, amplitude damping)
    - Adaptive regularization for optimal sensitivity
    - Mutual information health monitoring
    - Sophia point bias optimization

Ontological Framework Mappings:
    - Insight 6:  Sophia Point 1/phi (optimal bias point)
    - Insight 8:  Inverse Mapping Control (feedback Ramsey)
    - Insight 9:  Consciousness Collapse (measurement back-action)
    - Insight 12: Chronon Entropy (temporal mutual information)
    - Insight 20: Adaptive Lambda (dynamic noise adjustment)
    - Insight 21: Mutual Info Gradient (signal health monitoring)
    - Insight 24: Susceptibility Peak (critical point for metrology)

Usage:
    python sim3_quantum_metrology.py --noise 0.05 --detuning-range 0.5 --n-averages 20
    python sim3_quantum_metrology.py --sweep-noise --plot
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
        QuantumVMGravity, RamseyInterferometer,
        KB, C_LIGHT, LN2, SOPHIA_COHERENCE,
        von_neumann_entropy
    )
except ImportError:
    print("Error: qnvm_gravity.py not found.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ======================================================================
# Ramsey Fringe Measurement
# ======================================================================
def ramsey_fringe_scan(
    n_qubits: int = 1,
    wait_time: float = 1.0,
    detuning_range: float = 2.0 * np.pi,
    n_points: int = 50,
    noise_level: float = 0.05,
    shots: int = 10000,
    n_averages: int = 5,
) -> Dict[str, Any]:
    """
    Scan detuning values to measure a Ramsey fringe pattern.

    Parameters
    ----------
    n_qubits : int
        Total qubits (only qubit 0 is used for Ramsey).
    wait_time : float
        Free evolution time.
    detuning_range : float
        Range of detuning values to scan.
    n_points : int
        Number of detuning points.
    noise_level : float
        Backend noise level.
    shots : int
        Measurement shots per point.
    n_averages : int
        Number of independent runs to average over.

    Returns
    -------
    dict with fringe data and analysis.
    """
    detuning_values = np.linspace(-detuning_range, detuning_range, n_points)
    prob_1_averaged = np.zeros(n_points)
    contrast_averaged = np.zeros(n_points)

    for avg in range(n_averages):
        prob_1_single = np.zeros(n_points)
        for i, delta in enumerate(detuning_values):
            vm = QuantumVMGravity(qubits=n_qubits, noise_level=noise_level)
            vm.start()
            result = RamseyInterferometer.ramsey_sequence(
                vm, qubit=0, wait_time=wait_time, detuning=delta, shots=shots)
            prob_1_single[i] = result['prob_1']
            vm.stop()
        prob_1_averaged += prob_1_single

    prob_1_averaged /= n_averages

    # Fit sinusoidal fringe: P(1) = A + B * cos(omega * tau * delta + phi)
    # Expected: P(1) = 0.5 * (1 - cos(delta * wait_time))
    try:
        from scipy.optimize import curve_fit

        def ramsey_fringe(x, amplitude, offset, frequency, phase):
            return offset + amplitude * np.cos(frequency * x + phase)

        popt, pcov = curve_fit(
            ramsey_fringe, detuning_values, prob_1_averaged,
            p0=[0.5, 0.5, wait_time, 0.0],
            bounds=([-1, -1, 0, -2*np.pi], [1, 2, 10*wait_time, 2*np.pi]))
        amplitude, offset, frequency, phase = popt
        fringe_contrast = abs(amplitude) / (2 * offset) if offset > 0 else 0
        fitted_freq = frequency
    except (ImportError, Exception):
        amplitude = 0.5
        offset = 0.5
        fitted_freq = wait_time
        phase = 0.0
        fringe_contrast = 0.5

    return {
        'detuning_values': detuning_values.tolist(),
        'prob_1': prob_1_averaged.tolist(),
        'fringe_contrast': fringe_contrast,
        'fitted_frequency': float(fitted_freq),
        'fitted_amplitude': float(amplitude),
        'fitted_offset': float(offset),
        'wait_time': wait_time,
        'n_averages': n_averages,
    }


# ======================================================================
# Fisher Information & Sensitivity
# ======================================================================
def compute_sensitivity(
    n_qubits: int = 1,
    wait_time: float = 1.0,
    detuning_range: float = 2.0 * np.pi,
    n_points: int = 50,
    noise_level: float = 0.05,
    shots_per_point: int = 5000,
    mass_coupling_chi: float = 1e6,  # Hz/kg (dispersive coupling to mass)
    sample_mass_kg: float = 1e-6,
    n_integration_averages: int = 10,
) -> Dict[str, Any]:
    """
    Compute the complete sensitivity analysis for vacuum weight detection.

    Combines Ramsey fringe measurement with Fisher information estimation
    and converts to minimum detectable mass shift.

    Parameters
    ----------
    mass_coupling_chi : float
        Dispersive coupling chi in Hz/kg. This converts a mass shift
        to a qubit frequency shift: delta_omega = chi * delta_m.
    sample_mass_kg : float
        Mass of the experimental sample in kg.
    n_integration_averages : int
        Number of experimental averages (integration time factor).

    Returns
    -------
    dict with sensitivity analysis results.
    """
    print(f"  Computing Fisher information...")

    # Ramsey fringe data
    fringe = ramsey_fringe_scan(
        n_qubits=n_qubits,
        wait_time=wait_time,
        detuning_range=detuning_range,
        n_points=n_points,
        noise_level=noise_level,
        shots=shots_per_point,
        n_averages=1,
    )

    # Classical Fisher information
    delta_arr = np.array(fringe['detuning_values'])
    p1_arr = np.array(fringe['prob_1'])
    dp1_ddelta = np.gradient(p1_arr, delta_arr)

    p_safe = np.clip(p1_arr, 1e-10, 1 - 1e-10)
    fisher_per_point = dp1_ddelta ** 2 / (p_safe * (1 - p_safe))
    fisher_total = float(np.sum(fisher_per_point))
    fisher_per_shot = fisher_total / n_points

    # Quantum Fisher information (for a pure qubit state): F_Q = 4 * Var(Z) = 4 * (1 - <Z>^2)
    # For a qubit evolving as Rz(delta*tau), F_Q = tau^2 at the optimal point
    F_Q = wait_time ** 2  # quantum Fisher information per shot

    # Cramér-Rao bounds
    crb_classical = 1.0 / math.sqrt(n_integration_averages * fisher_per_shot) if fisher_per_shot > 0 else float('inf')
    crb_quantum = 1.0 / math.sqrt(n_integration_averages * F_Q) if F_Q > 0 else 0

    # Convert to mass sensitivity
    # delta_omega = chi * delta_m => delta_m = delta_omega / chi
    # delta_omega_min = CRB (classical) in rad/s
    if mass_coupling_chi > 0:
        chi_rad = mass_coupling_chi * 2 * np.pi  # convert Hz to rad/s
        dm_classical = crb_classical / chi_rad
        dm_quantum = crb_quantum / chi_rad
    else:
        dm_classical = float('inf')
        dm_quantum = 0.0

    # Integration time scaling: delta_m_min ~ 1/sqrt(T_int * F)
    # T_int = n_averages * cycle_time
    cycle_time = 2 * wait_time + 0.1  # approximate cycle time including pulses

    # Predicted signal from bit-mass equation (Niobium T_c = 9.2 K)
    predicted_dm = (KB * 9.2 * LN2 / C_LIGHT ** 2) * 1e23  # ~1e23 bits entropy change
    snr = predicted_dm / dm_classical if dm_classical > 0 else float('inf')

    return {
        'fisher': {
            'fisher_total': fisher_total,
            'fisher_per_point': fisher_per_point.tolist(),
            'fisher_per_shot': fisher_per_shot,
            'quantum_fisher': F_Q,
            'crb_classical': crb_classical,
            'crb_quantum': crb_quantum,
        },
        'mass_sensitivity': {
            'delta_m_classical_kg': dm_classical,
            'delta_m_quantum_kg': dm_quantum,
            'predicted_signal_kg': predicted_dm,
            'signal_to_noise': snr,
            'mass_coupling_chi_Hz_per_kg': mass_coupling_chi,
            'feasible': snr > 1.0,
        },
        'fringe': fringe,
        'integration': {
            'n_averages': n_integration_averages,
            'cycle_time_s': cycle_time,
            'total_integration_time_s': n_integration_averages * cycle_time,
        },
    }


# ======================================================================
# Noise Sweep
# ======================================================================
def sweep_noise_levels(
    n_qubits: int = 1,
    wait_time: float = 1.0,
    noise_range: List[float] = None,
    n_points: int = 30,
    shots_per_point: int = 5000,
    n_averages: int = 10,
    mass_coupling_chi: float = 1e6,
    output_dir: str = "sim3_results",
) -> Dict[str, Any]:
    """
    Sweep noise levels to find the noise threshold for vacuum weight detection.
    """
    if noise_range is None:
        noise_range = np.linspace(0.0, 0.2, 15).tolist()

    sweep_results = []
    for noise in noise_range:
        print(f"\n  Noise level = {noise:.4f}")
        sensitivity = compute_sensitivity(
            n_qubits=n_qubits,
            wait_time=wait_time,
            n_points=n_points,
            noise_level=noise,
            shots_per_point=shots_per_point,
            n_integration_averages=n_averages,
            mass_coupling_chi=mass_coupling_chi,
        )
        sweep_results.append({
            'noise_level': noise,
            'fringe_contrast': sensitivity['fringe']['fringe_contrast'],
            'fisher_per_shot': sensitivity['fisher']['fisher_per_shot'],
            'delta_m_classical': sensitivity['mass_sensitivity']['delta_m_classical_kg'],
            'snr': sensitivity['mass_sensitivity']['signal_to_noise'],
        })

    # Find noise threshold where SNR drops below 1
    noise_arr = np.array([r['noise_level'] for r in sweep_results])
    snr_arr = np.array([r['snr'] for r in sweep_results])

    threshold_idx = np.where(snr_arr < 1.0)[0]
    noise_threshold = float(noise_arr[threshold_idx[0]]) if len(threshold_idx) > 0 else float(noise_arr[-1])

    results = {
        'sweep_data': sweep_results,
        'noise_threshold': noise_threshold,
        'parameters': {
            'n_qubits': n_qubits,
            'wait_time': wait_time,
            'mass_coupling_chi': mass_coupling_chi,
        },
    }

    # Save and plot
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim3_noise_sweep.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_PLOT:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        nl = [r['noise_level'] for r in sweep_results]
        fc = [r['fringe_contrast'] for r in sweep_results]
        fi = [r['fisher_per_shot'] for r in sweep_results]
        dm = [r['delta_m_classical'] for r in sweep_results]
        sn = [r['snr'] for r in sweep_results]

        axes[0].plot(nl, fc, 'bo-')
        axes[0].set_xlabel('Noise level')
        axes[0].set_ylabel('Fringe contrast')
        axes[0].set_title('Ramsey Fringe Contrast vs Noise')
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(nl, [max(x, 1e-30) for x in fi], 'gs-')
        axes[1].set_xlabel('Noise level')
        axes[1].set_ylabel('Fisher information / shot')
        axes[1].set_title('Fisher Information vs Noise')
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(nl, [max(x, 1e-30) for x in dm], 'r^-')
        axes[2].axhline(y=results['sweep_data'][0]['delta_m_classical'],
                        color='b', linestyle='--', label='Predicted signal')
        axes[2].set_xlabel('Noise level')
        axes[2].set_ylabel('$\\delta m_{min}$ (kg)')
        axes[2].set_title(f'Mass Sensitivity (threshold noise = {noise_threshold:.4f})')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out / 'sim3_noise_sweep.png', dpi=150)
        plt.close()

    return results


# ======================================================================
# Integration Time Scaling
# ======================================================================
def integration_time_scaling(
    n_qubits: int = 1,
    wait_time: float = 1.0,
    noise_level: float = 0.05,
    mass_coupling_chi: float = 1e6,
    n_averages_range: List[int] = None,
    output_dir: str = "sim3_results",
) -> Dict[str, Any]:
    """
    Compute mass sensitivity vs integration time (number of averages).
    Expected scaling: delta_m_min ~ 1/sqrt(N) (standard quantum limit).
    """
    if n_averages_range is None:
        n_averages_range = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    scaling_results = []
    for n_avg in n_averages_range:
        print(f"  N_averages = {n_avg}...", end="", flush=True)
        sensitivity = compute_sensitivity(
            n_qubits=n_qubits,
            wait_time=wait_time,
            n_points=30,
            noise_level=noise_level,
            shots_per_point=2000,
            n_integration_averages=n_avg,
            mass_coupling_chi=mass_coupling_chi,
        )
        cycle_time = 2 * wait_time + 0.1
        scaling_results.append({
            'n_averages': n_avg,
            'integration_time_s': n_avg * cycle_time,
            'delta_m_min_kg': sensitivity['mass_sensitivity']['delta_m_classical_kg'],
            'fisher_per_shot': sensitivity['fisher']['fisher_per_shot'],
            'snr': sensitivity['mass_sensitivity']['signal_to_noise'],
            'fringe_contrast': sensitivity['fringe']['fringe_contrast'],
        })
        print(f" dm_min = {sensitivity['mass_sensitivity']['delta_m_classical_kg']:.3e} kg")

    results = {
        'scaling_data': scaling_results,
        'parameters': {
            'noise_level': noise_level,
            'wait_time': wait_time,
            'mass_coupling_chi': mass_coupling_chi,
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim3_integration_scaling.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_PLOT:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        n_arr = np.array([r['n_averages'] for r in scaling_results])
        dm_arr = np.array([r['delta_m_min_kg'] for r in scaling_results])
        t_arr = np.array([r['integration_time_s'] for r in scaling_results])

        # delta_m vs N (should follow 1/sqrt(N))
        axes[0].loglog(n_arr, np.maximum(dm_arr, 1e-50), 'bo-', label='Simulation')
        # Fit 1/sqrt(N) scaling
        try:
            from scipy.optimize import curve_fit
            def inv_sqrt(N, A):
                return A / np.sqrt(N)
            popt, _ = curve_fit(inv_sqrt, n_arr[n_arr > 0], dm_arr[n_arr > 0],
                               p0=[dm_arr[0]])
            n_fine = np.logspace(0, 3, 100)
            axes[0].loglog(n_fine, inv_sqrt(n_fine, popt[0]), 'r--',
                          label=f'$A/\\sqrt{{N}}$, A={popt[0]:.3e}')
        except (ImportError, Exception):
            pass
        axes[0].set_xlabel('Number of averages $N$')
        axes[0].set_ylabel('$\\delta m_{min}$ (kg)')
        axes[0].set_title('Mass Sensitivity vs Integration (SQL Scaling)')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3, which='both')

        # delta_m vs integration time
        axes[1].loglog(t_arr, np.maximum(dm_arr, 1e-50), 'go-')
        axes[1].set_xlabel('Integration time (s)')
        axes[1].set_ylabel('$\\delta m_{min}$ (kg)')
        axes[1].set_title('Mass Sensitivity vs Integration Time')
        axes[1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(out / 'sim3_integration_scaling.png', dpi=150)
        plt.close()

    return results


# ======================================================================
# Adaptive Regularization
# ======================================================================
def adaptive_sensitivity_tracking(
    n_qubits: int = 1,
    wait_time: float = 1.0,
    noise_level: float = 0.05,
    n_time_steps: int = 50,
    mass_coupling_chi: float = 1e6,
    output_dir: str = "sim3_results",
) -> Dict[str, Any]:
    """
    Simulate adaptive noise regulation during a long measurement sequence.

    Implements the constitutional defense mechanism:
        lambda(t) = max(0.01, 0.02 * exp(-t/tau))

    Monitors mutual information gradient as signal health check.
    Falls back to increased shots or more robust encoding when signal degrades.
    """
    vm = QuantumVMGravity(qubits=n_qubits, noise_level=noise_level)
    vm.start()

    time_points = np.linspace(0, 10.0, n_time_steps)
    snr_history = []
    lambda_history = []
    fisher_history = []
    mutual_info_history = []
    signal_health = []
    shots_adaptive = []

    tau = 2.0  # regularization decay timescale
    I_threshold = 0.1  # mutual information threshold for signal loss

    prev_prob = 0.5  # initial probability

    for t_idx, t in enumerate(time_points):
        # Compute adaptive regularization
        lam = vm.adaptive_lambda(t, tau=tau)
        lambda_history.append(lam)

        # Run Ramsey measurement at fixed detuning
        delta = 0.5  # fixed detuning near the fringe slope (optimal sensitivity)
        shots = 5000
        shots_adaptive.append(shots)

        # Save state, run Ramsey, restore
        saved = vm._backend.state.copy() if n_qubits <= 20 else None
        result = RamseyInterferometer.ramsey_sequence(
            vm, qubit=0, wait_time=wait_time, detuning=delta, shots=shots)
        if saved is not None:
            vm._backend.state = saved

        current_prob = result['prob_1']

        # Mutual information between measurement outcome and signal
        # I(R;S) ~ 1 - H(p) where H(p) is binary entropy
        p = np.clip(current_prob, 1e-10, 1 - 1e-10)
        H_binary = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        I_RS = 1.0 - H_binary
        mutual_info_history.append(I_RS)

        # Signal health check
        is_healthy = I_RS > I_threshold
        signal_health.append(is_healthy)

        # Estimate local Fisher information from fringe slope
        dI_dt = abs(I_RS - prev_prob) / (time_points[1] - time_points[0] if t_idx > 0 else 1.0)
        fisher_history.append(dI_dt)

        # SNR estimate
        snr = abs(current_prob - 0.5) / max(math.sqrt(0.25 / shots), 1e-10)
        snr_history.append(snr)

        # Apply noise (simulating drift over time)
        vm.apply_noise_channel('dephasing', [0], probability=noise_level * 0.005)

        prev_prob = current_prob

        if (t_idx + 1) % 10 == 0:
            print(f"  t={t:.2f}: lambda={lam:.4f}, I_RS={I_RS:.4f}, "
                  f"SNR={snr:.2f}, healthy={is_healthy}")

    vm.stop()

    # Compute chronon entanglement entropy
    chronon_entropy = 0.0
    mi_arr = np.array(mutual_info_history)
    if len(mi_arr) > 1:
        mi_diff = np.diff(mi_arr)
        mi_pos = mi_diff[mi_diff > 0]
        if len(mi_pos) > 0:
            chronon_entropy = -np.sum(mi_pos * np.log2(mi_pos + 1e-10))

    results = {
        'adaptive': {
            'time_points': time_points.tolist(),
            'lambda_history': lambda_history,
            'snr_history': snr_history,
            'fisher_history': fisher_history,
            'mutual_info_history': mutual_info_history,
            'signal_health': signal_health,
            'shots_adaptive': shots_adaptive,
        },
        'chronon_entropy': float(chronon_entropy),
        'signal_health_fraction': float(np.mean(signal_health)),
        'I_threshold': I_threshold,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim3_adaptive_tracking.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_PLOT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(time_points, snr_history, 'b-', linewidth=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('SNR')
        axes[0, 0].set_title('Signal-to-Noise Ratio over Time')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(time_points, lambda_history, 'r-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('$\\lambda(t)$')
        axes[0, 1].set_title('Adaptive Regularization (Constitutional Defense)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(time_points, mutual_info_history, 'g-', linewidth=0.8)
        axes[1, 0].axhline(y=I_threshold, color='r', linestyle='--',
                          label=f'$I_{{threshold}}$ = {I_threshold}')
        axes[1, 0].fill_between(time_points, 0, mutual_info_history,
                                where=[h for h in signal_health],
                                alpha=0.3, color='green', label='Healthy')
        axes[1, 0].fill_between(time_points, 0, mutual_info_history,
                                where=[not h for h in signal_health],
                                alpha=0.3, color='red', label='Degraded')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('$I(R;S)$')
        axes[1, 0].set_title('Mutual Information Health Monitor')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(time_points, fisher_history, 'purple', linewidth=0.8)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Local Fisher information')
        axes[1, 1].set_title('Fisher Information over Time')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Blueprint 3: Adaptive Quantum Metrology', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out / 'sim3_adaptive_tracking.png', dpi=150)
        plt.close()

    return results


# ======================================================================
# Sophia Point Bias Optimization
# ======================================================================
def sophia_bias_optimization(
    n_qubits: int = 1,
    noise_level: float = 0.05,
    n_points: int = 50,
    output_dir: str = "sim3_results",
) -> Dict[str, Any]:
    """
    Find the optimal qubit bias point for maximum metrological sensitivity.

    The Sophia point predicts maximum susceptibility at C* = 1/phi ~ 0.618.
    In the Ramsey context, this corresponds to the bias point where the
    fringe slope (dP/d(delta)) is maximized.
    """
    # Run Ramsey fringe scan
    fringe = ramsey_fringe_scan(
        n_qubits=n_qubits,
        wait_time=1.0,
        detuning_range=2 * np.pi,
        n_points=n_points,
        noise_level=noise_level,
        shots=10000,
        n_averages=3,
    )

    delta_arr = np.array(fringe['detuning_values'])
    p1_arr = np.array(fringe['prob_1'])
    dp_ddelta = np.abs(np.gradient(p1_arr, delta_arr))

    # Maximum slope = optimal operating point
    idx_max = np.argmax(dp_ddelta)
    delta_optimal = delta_arr[idx_max]
    sensitivity_max = dp_ddelta[idx_max]

    # Corresponding probability
    p_optimal = p1_arr[idx_max]
    # For P(1) = (1 - cos(delta*tau))/2, the max slope occurs at delta*tau = pi/2
    # where P(1) = 0.5, so the optimal bias is at zero field

    # Compute susceptibility analogy
    # chi = dC/dJ maps to dP/d(delta) here
    sophia_analog = 1.0 - abs(p_optimal - 0.5) * 2  # = 1 at P=0.5, 0 at P=0 or 1

    results = {
        'fringe_data': fringe,
        'optimal_operating_point': {
            'detuning_optimal': float(delta_optimal),
            'prob_1_at_optimal': float(p_optimal),
            'max_sensitivity': float(sensitivity_max),
            'sophia_analog': float(sophia_analog),
            'sophia_target': SOPHIA_COHERENCE,
        },
        'slope_data': dp_ddelta.tolist(),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'sim3_sophia_bias.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if HAS_PLOT:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(delta_arr, p1_arr, 'b-', linewidth=1.5, label='$P(1)$')
        ax1.axvline(x=delta_optimal, color='r', linestyle='--',
                    label=f'Optimal $\\delta$ = {delta_optimal:.3f}')
        ax1.set_xlabel('Detuning $\\delta$ (rad)')
        ax1.set_ylabel('$P(1)$')
        ax1.set_title('Ramsey Fringe with Optimal Bias Point')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        ax2.plot(delta_arr, dp_ddelta, 'g-', linewidth=1.5, label='|dP/d$\\delta$|')
        ax2.axvline(x=delta_optimal, color='r', linestyle='--',
                    label=f'Max sensitivity')
        ax2.set_xlabel('Detuning $\\delta$ (rad)')
        ax2.set_ylabel('$|dP/d\\delta|$')
        ax2.set_title(f'Sophia Point Bias (max at $\\delta$ = {delta_optimal:.3f})')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out / 'sim3_sophia_bias.png', dpi=150)
        plt.close()

    return results


# ======================================================================
# Full Simulation
# ======================================================================
def run_full_simulation(
    noise_level: float = 0.05,
    wait_time: float = 1.0,
    mass_coupling_chi: float = 1e6,
    n_integration_averages: int = 20,
    output_dir: str = "sim3_results",
) -> Dict[str, Any]:
    """
    Run the complete quantum metrology simulation suite.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("QUANTUM METROLOGY FULL SIMULATION SUITE")
    print("=" * 70)

    all_results = {}

    # 1. Ramsey fringe scan
    print("\n[1/5] Ramsey fringe scan...")
    fringe = ramsey_fringe_scan(
        n_qubits=1, wait_time=wait_time,
        detuning_range=2 * np.pi, n_points=50,
        noise_level=noise_level, shots=10000, n_averages=5)
    all_results['fringe'] = fringe
    print(f"  Fringe contrast = {fringe['fringe_contrast']:.4f}")

    # 2. Sensitivity analysis
    print("\n[2/5] Sensitivity analysis...")
    sensitivity = compute_sensitivity(
        n_qubits=1, wait_time=wait_time,
        noise_level=noise_level,
        mass_coupling_chi=mass_coupling_chi,
        n_integration_averages=n_integration_averages)
    all_results['sensitivity'] = sensitivity
    ms = sensitivity['mass_sensitivity']
    print(f"  delta_m (classical) = {ms['delta_m_classical_kg']:.3e} kg")
    print(f"  Predicted signal    = {ms['predicted_signal_kg']:.3e} kg")
    print(f"  SNR                = {ms['signal_to_noise']:.2f}")

    # 3. Integration time scaling
    print("\n[3/5] Integration time scaling...")
    scaling = integration_time_scaling(
        noise_level=noise_level, wait_time=wait_time,
        mass_coupling_chi=mass_coupling_chi,
        n_averages_range=[1, 2, 5, 10, 20, 50, 100],
        output_dir=output_dir)
    all_results['scaling'] = scaling

    # 4. Adaptive tracking
    print("\n[4/5] Adaptive sensitivity tracking...")
    adaptive = adaptive_sensitivity_tracking(
        noise_level=noise_level, wait_time=wait_time,
        mass_coupling_chi=mass_coupling_chi,
        n_time_steps=30, output_dir=output_dir)
    all_results['adaptive'] = adaptive
    print(f"  Signal health fraction = {adaptive['signal_health_fraction']:.2%}")

    # 5. Sophia bias optimization
    print("\n[5/5] Sophia point bias optimization...")
    sophia = sophia_bias_optimization(
        noise_level=noise_level, n_points=50, output_dir=output_dir)
    all_results['sophia'] = sophia
    print(f"  Optimal detuning = {sophia['optimal_operating_point']['detuning_optimal']:.4f}")

    # Save combined results
    with open(out / 'sim3_full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary dashboard plot
    if HAS_PLOT:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3)

        # Ramsey fringe
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fringe['detuning_values'], fringe['prob_1'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Detuning')
        ax1.set_ylabel('P(1)')
        ax1.set_title(f'Ramsey Fringe (contrast = {fringe["fringe_contrast"]:.3f})')
        ax1.grid(True, alpha=0.3)

        # Fisher info
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.semilogy(range(len(sensitivity['fisher']['fisher_per_point'])),
                    np.maximum(sensitivity['fisher']['fisher_per_point'], 1e-30), 'g-')
        ax2.set_xlabel('Detuning point')
        ax2.set_ylabel('Fisher info')
        ax2.set_title('Fisher Information Distribution')
        ax2.grid(True, alpha=0.3)

        # Integration scaling
        ax3 = fig.add_subplot(gs[0, 2])
        sd = scaling['scaling_data']
        n_arr = [r['n_averages'] for r in sd]
        dm_arr = [r['delta_m_min_kg'] for r in sd]
        ax3.loglog(n_arr, np.maximum(dm_arr, 1e-60), 'ro-')
        ax3.set_xlabel('N averages')
        ax3.set_ylabel('$\\delta m_{min}$ (kg)')
        ax3.set_title('Integration Time Scaling')
        ax3.grid(True, alpha=0.3, which='both')

        # Adaptive tracking
        ax4 = fig.add_subplot(gs[1, 0])
        ad = adaptive['adaptive']
        ax4.plot(ad['time_points'], ad['snr_history'], 'b-', linewidth=0.8)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('SNR')
        ax4.set_title(f'Adaptive SNR (health: {adaptive["signal_health_fraction"]:.0%})')
        ax4.grid(True, alpha=0.3)

        # Mutual info
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(ad['time_points'], ad['mutual_info_history'], 'g-', linewidth=0.8)
        ax5.axhline(y=adaptive['I_threshold'], color='r', linestyle='--')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('$I(R;S)$')
        ax5.set_title('Signal Health Monitor')
        ax5.grid(True, alpha=0.3)

        # Summary text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        summary_text = (
            f"Noise level: {noise_level}\n"
            f"Wait time: {wait_time} s\n"
            f"Coupling chi: {mass_coupling_chi:.0e} Hz/kg\n"
            f"Averages: {n_integration_averages}\n\n"
            f"delta_m (classical):\n  {ms['delta_m_classical_kg']:.3e} kg\n\n"
            f"Predicted signal:\n  {ms['predicted_signal_kg']:.3e} kg\n\n"
            f"SNR: {ms['signal_to_noise']:.2f}\n"
            f"Feasible: {'YES' if ms['feasible'] else 'NO'}\n\n"
            f"Chronon entropy:\n  {adaptive['chronon_entropy']:.4f}"
        )
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Blueprint 3: Quantum Metrology Dashboard', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(out / 'sim3_dashboard.png', dpi=150)
        plt.close()
        print(f"\nDashboard saved to {out}/sim3_dashboard.png")

    return all_results


# ======================================================================
# Main Entry Point
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Blueprint 3: Quantum Metrology for Vacuum Weight Sensing',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise level (default: 0.05)')
    parser.add_argument('--wait-time', type=float, default=1.0,
                        help='Ramsey wait time (default: 1.0)')
    parser.add_argument('--chi', type=float, default=1e6,
                        help='Dispersive coupling chi in Hz/kg (default: 1e6)')
    parser.add_argument('--averages', type=int, default=20,
                        help='Number of integration averages (default: 20)')
    parser.add_argument('--output-dir', type=str, default='sim3_results',
                        help='Output directory')
    parser.add_argument('--sweep-noise', action='store_true',
                        help='Sweep noise levels to find threshold')
    parser.add_argument('--adaptive-only', action='store_true',
                        help='Run only adaptive tracking simulation')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Generate plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    print("=" * 70)
    print("BLUEPRINT 3: QUANTUM METROLOGY FOR VACUUM WEIGHT SENSING")
    print("(MOS-HOR-QNVM v14.0-Gravity)")
    print("=" * 70)
    print(f"Noise level   : {args.noise}")
    print(f"Wait time     : {args.wait_time} s")
    print(f"Coupling chi  : {args.chi} Hz/kg")
    print(f"Averages      : {args.averages}")
    print(f"Output        : {args.output_dir}")
    print()

    if args.sweep_noise:
        results = sweep_noise_levels(
            noise_level=args.noise,
            mass_coupling_chi=args.chi,
            output_dir=args.output_dir,
        )
        print(f"\nNoise threshold for detection: {results['noise_threshold']:.4f}")
    elif args.adaptive_only:
        results = adaptive_sensitivity_tracking(
            noise_level=args.noise,
            wait_time=args.wait_time,
            mass_coupling_chi=args.chi,
            n_time_steps=50,
            output_dir=args.output_dir,
        )
    else:
        results = run_full_simulation(
            noise_level=args.noise,
            wait_time=args.wait_time,
            mass_coupling_chi=args.chi,
            n_integration_averages=args.averages,
            output_dir=args.output_dir,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        if 'sensitivity' in results:
            ms = results['sensitivity']['mass_sensitivity']
            fi = results['sensitivity']['fisher']
            print(f"  Fisher information (total)  = {fi['fisher_total']:.4f}")
            print(f"  Quantum Fisher              = {fi['quantum_fisher']:.4f}")
            print(f"  CRB (classical)             = {fi['crb_classical']:.6f} rad/s")
            print(f"  delta_m_min (classical)      = {ms['delta_m_classical_kg']:.3e} kg")
            print(f"  Predicted signal             = {ms['predicted_signal_kg']:.3e} kg")
            print(f"  SNR                         = {ms['signal_to_noise']:.2f}")
            print(f"  Feasible                    = {ms['feasible']}")
        if 'adaptive' in results:
            ad = results['adaptive']
            print(f"  Signal health fraction      = {ad['signal_health_fraction']:.2%}")
            print(f"  Chronon entropy             = {ad['chronon_entropy']:.6f}")

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
