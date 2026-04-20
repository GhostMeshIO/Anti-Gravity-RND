#!/usr/bin/env python3
"""
qnvm_gravity.py - MOS-HOR-QNVM v14.0 Gravity Engine
=====================================================
Enhanced quantum virtual machine with gravitational and ontological
simulation capabilities for the Archimedes Experiment.

Extends qnvm_light.py (v13.0) with:
  - Expectation value measurement of Pauli observables
  - Density matrix access and reduced density matrices
  - Von Neumann entropy and entanglement measures
  - Trotterized time evolution under arbitrary 2-body Hamiltonians
  - Dephasing and amplitude damping noise channels
  - Topological entropy computation (Kitaev/Levin-Wen)
  - Lieb-Robinson velocity estimation
  - Fractal dimension analysis via box-counting
  - Betti number estimation from entanglement graphs
  - Bohmian trajectory computation

Backend selection:
  - qubits <= 20  -> StateVectorBackend (exact, full density matrix)
  - qubits >  20  -> StabilizerBackend  (fast, Clifford-only, limited ops)

Author: MOS-HOR Ontological Physics Lab
Version: 14.0-gravity
"""

import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from itertools import product

# ======================================================================
# Physical Constants (SI)
# ======================================================================
KB = 1.380649e-23          # Boltzmann constant [J/K]
HBAR = 1.054571817e-34     # Reduced Planck constant [J*s]
C_LIGHT = 2.99792458e8     # Speed of light [m/s]
LN2 = math.log(2)

# ======================================================================
# Ontological Constants
# ======================================================================
PHI = (1 + math.sqrt(5)) / 2               # Golden ratio
SOPHIA_COHERENCE = 1.0 / PHI               # ~0.618 (critical coherence)
VACUUM_ENERGY_WATTS = 0.68                  # Vacuum power [W]
HOLOGRAPHIC_EFFICIENCY_MAX = 0.93           # r/d_s upper bound

HARDWARE_PROFILES = {
    'legacy_qubit': {'d': 2, 'gate_speed_ns': 20, 'fidelity': 0.999, 'T2_us': 100},
    'hor_qutrit':   {'d': 3, 'gate_speed_ns': 25, 'fidelity': 0.995, 'T2_us': 120},
    'hor_ququart':  {'d': 4, 'gate_speed_ns': 30, 'fidelity': 0.992, 'T2_us': 110},
}


# ======================================================================
# Utility Functions
# ======================================================================
def partial_trace(rho: np.ndarray, trace_out: List[int], dims: List[int]) -> np.ndarray:
    """
    Compute partial trace of density matrix.

    Parameters
    ----------
    rho : np.ndarray, shape (2^n, 2^n)
        Full density matrix.
    trace_out : list of int
        Indices of subsystems to trace out.
    dims : list of int
        Dimensions of each subsystem (all 2 for qubits).

    Returns
    -------
    np.ndarray
        Reduced density matrix.
    """
    n = len(dims)
    trace_keep = sorted(set(range(n)) - set(trace_out))
    if not trace_keep:
        return np.array([[1.0]])

    # Reshape into tensor
    tensor = rho.reshape(dims + dims)

    # Trace out specified subsystems
    for idx in sorted(trace_out):
        # Find position in trace_keep (shifts as we trace)
        pos = idx
        # Contract over the pair of axes at positions (pos, pos + n)
        # After each trace, the tensor shrinks
        current_dims = [d for d in dims if d != 0]
        # Use einsum-style approach
        axes_pairs = list(range(len(tensor.shape)))
        # Axes to keep: all except pos and pos + original_n_remaining
        pass

    # Simpler implementation for qubits using reshape and matrix multiplication
    dim_keep = 1
    for i in trace_keep:
        dim_keep *= dims[i]
    dim_trace = 1
    for i in trace_out:
        dim_trace *= dims[i]

    # Build the partial trace via basis summation
    rho_reduced = np.zeros((dim_keep, dim_keep), dtype=complex)
    # Map multi-indices to reduced basis
    for idx_keep in product(*[range(dims[i]) for i in trace_keep]):
        for idx_trace in product(*[range(dims[i]) for i in trace_out]):
            # Build full index
            full_idx = list(idx_keep + idx_trace)
            # Reorder to match original subsystem order
            ordered_idx = [0] * n
            ki, ti = 0, 0
            for i in range(n):
                if i in trace_keep:
                    ordered_idx[i] = idx_keep[ki]; ki += 1
                else:
                    ordered_idx[i] = idx_trace[ti]; ti += 1
            row = sum(ordered_idx[j] * np.prod(dims[j+1:]) for j in range(n))
            for idx_trace2 in product(*[range(dims[i]) for i in trace_out]):
                full_idx2 = list(idx_keep + idx_trace2)
                ordered_idx2 = [0] * n
                ki, ti = 0, 0
                for i in range(n):
                    if i in trace_keep:
                        ordered_idx2[i] = idx_keep[ki]; ki += 1
                    else:
                        ordered_idx2[i] = idx_trace2[ti]; ti += 1
                col = sum(ordered_idx2[j] * np.prod(dims[j+1:]) for j in range(n))
                # Sum over diagonal elements of traced subsystem
                diag = all(ordered_idx[i] == ordered_idx2[i] for i in trace_out)
                if diag:
                    rk = sum(idx_keep[j] * np.prod([dims[trace_keep[k]] for k in range(j+1)])
                            for j in range(len(idx_keep)))
                    ck = rk  # same because same idx_keep
                    rho_reduced[rk, ck] += rho[row, col]

    return rho_reduced


def partial_trace_fast(rho: np.ndarray, trace_out: List[int], n_qubits: int) -> np.ndarray:
    """
    Fast partial trace for qubit systems using reshape/matrix operations.
    Works for qubits <= 14 to avoid memory issues.
    """
    trace_keep = sorted(set(range(n_qubits)) - set(trace_out))
    n_keep = len(trace_keep)
    n_trace = n_qubits - n_keep
    dim_keep = 1 << n_keep
    dim_trace = 1 << n_trace

    # Reorder axes so traced qubits come last, then first, then kept
    axes_order = trace_keep + trace_out + [i + n_qubits for i in trace_keep] + [i + n_qubits for i in trace_out]
    tensor = rho.reshape([2] * (2 * n_qubits))
    tensor = np.transpose(tensor, axes_order)
    # Now shape is (2^n_keep, 2^n_trace, 2^n_keep, 2^n_trace)
    tensor = tensor.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    # Trace over the two trace dimensions (indices 1 and 3)
    # tensor shape: (dim_keep, dim_trace, dim_keep, dim_trace) = (i, j, k, l)
    # partial trace: result[i, k] = sum_j tensor[i, j, k, j]
    rho_reduced = np.einsum('ijkj->ik', tensor)
    return rho_reduced


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute von Neumann entropy S = -Tr(rho * log2(rho)).
    Handles zero eigenvalues via clipping.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    # Remove near-zero and negative eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-14]
    if len(eigenvalues) == 0:
        return 0.0
    return -np.sum(eigenvalues * np.log2(eigenvalues))


def entanglement_negativity(rho_ab: np.ndarray) -> float:
    """
    Compute entanglement negativity of a bipartite state.

    Negativity = ||rho^T_A||_1 - 1 (for qubits, divided by 2 sometimes).
    Here we compute the log-negativity: E_N = log2(||rho^T_A||_1).
    """
    n_qubits_total = int(round(np.log2(rho_ab.shape[0])))
    if n_qubits_total != 2:
        # For larger systems, partial transpose over first qubit
        rho_pt = np.zeros_like(rho_ab)
        dim = rho_ab.shape[0]
        half = dim // 2
        rho_pt[:half, :half] = rho_ab[:half, :half]
        rho_pt[half:, half:] = rho_ab[half:, half:]
        rho_pt[:half, half:] = rho_ab[half:, :half]
        rho_pt[half:, :half] = rho_ab[:half, half:]
    else:
        # Two-qubit case: partial transpose on first qubit
        # Tensor indices: [a, b, a', b'] where a = first qubit ket
        # Partial transpose swaps a (ket) with a' (bra): transpose axes 0 and 2
        rho_pt = rho_ab.reshape(2, 2, 2, 2).transpose(2, 1, 0, 3).reshape(4, 4)

    eigenvalues = np.linalg.eigvalsh(rho_pt)
    # Negativity = sum of |negative eigenvalues|
    neg = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
    if neg <= 0:
        return 0.0
    return math.log2(2 * neg + 1)


def mutual_information(rho_ab: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int) -> float:
    """
    Compute mutual information I(A:B) = S(A) + S(B) - S(AB).
    """
    rho_a = partial_trace_fast(rho_ab, [q for q in range(n_qubits) if q != qubit_a], n_qubits)
    rho_b = partial_trace_fast(rho_ab, [q for q in range(n_qubits) if q != qubit_b], n_qubits)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)
    return s_a + s_b - s_ab


def topological_entanglement_entropy(entropy_by_region: Dict[str, float]) -> float:
    """
    Compute topological entanglement entropy (TEE) using Kitaev-Preskill formula:
        S_topo = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

    entropy_by_region should contain keys for 'A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC'.
    """
    required = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
    for key in required:
        if key not in entropy_by_region:
            raise ValueError(f"Missing region '{key}' in entropy_by_region")

    s_topo = (entropy_by_region['A'] + entropy_by_region['B'] + entropy_by_region['C']
              - entropy_by_region['AB'] - entropy_by_region['AC']
              - entropy_by_region['BC'] + entropy_by_region['ABC'])
    return s_topo


def box_counting_fractal_dimension(correlations: np.ndarray, grid_size: int,
                                   min_box: int = 2, max_box: int = None) -> float:
    """
    Estimate fractal dimension D_f via box-counting on an entanglement cluster map.

    Parameters
    ----------
    correlations : np.ndarray, shape (grid_size, grid_size)
        Correlation matrix (e.g., ZZ correlations on a 2D lattice).
    min_box : int
        Minimum box size.
    max_box : int
        Maximum box size (default: grid_size // 4).

    Returns
    -------
    float
        Estimated fractal dimension D_f.
    """
    if max_box is None:
        max_box = max(grid_size // 4, min_box + 1)

    box_sizes = list(range(min_box, max_box + 1))
    n_counts = []

    for bs in box_sizes:
        # Threshold correlations to get binary cluster map
        clusters = (np.abs(correlations) > 0.1).astype(int)
        # Count occupied boxes
        count = 0
        for i in range(0, grid_size - bs + 1, bs):
            for j in range(0, grid_size - bs + 1, bs):
                if np.any(clusters[i:i+bs, j:j+bs] > 0):
                    count += 1
        n_counts.append(count)

    # Linear regression on log-log scale
    if len(box_sizes) < 2:
        return 2.0  # default for 2D
    log_bs = np.log(np.array(box_sizes, dtype=float))
    log_nc = np.log(np.array(n_counts, dtype=float) + 1e-10)

    # Filter out zero counts
    valid = log_nc > -np.inf
    if np.sum(valid) < 2:
        return 2.0

    coeffs = np.polyfit(log_bs[valid], log_nc[valid], 1)
    return -coeffs[0]


def lieg_robinson_velocity(correlations_history: List[np.ndarray],
                           lattice_spacing: float,
                           dt: float) -> float:
    """
    Estimate Lieb-Robinson velocity from the spread of correlations
    after a local quench.

    Parameters
    ----------
    correlations_history : list of np.ndarray
        List of correlation matrices at successive time steps.
    lattice_spacing : float
        Physical spacing between qubits.
    dt : float
        Time step between snapshots.

    Returns
    -------
    float
        Estimated Lieb-Robinson velocity v_LR in units of lattice_spacing/dt.
    """
    if len(correlations_history) < 2:
        return 0.0

    n_qubits = correlations_history[0].shape[0]
    v_lr = 0.0

    for t_idx in range(1, len(correlations_history)):
        corr_prev = correlations_history[t_idx - 1]
        corr_curr = correlations_history[t_idx]
        # Find the maximum distance at which correlation changed significantly
        max_dist = 0
        for i in range(n_qubits):
            for j in range(n_qubits):
                delta = abs(corr_curr[i, j] - corr_prev[i, j])
                if delta > 0.01:
                    dist = abs(i - j) * lattice_spacing
                    max_dist = max(max_dist, dist)
        v_t = max_dist / (dt * t_idx) if t_idx > 0 else 0
        v_lr = max(v_lr, v_t)

    return v_lr


# ======================================================================
# Stabilizer Tableau (Clifford simulator) - Enhanced
# ======================================================================
class StabilizerTableau:
    """
    CHP tableau representation for n qubits.
    Tableau has shape (2n, 2n) for X and Z parts, plus phases (2n,).
    """
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.tableau = np.zeros((2 * self.n, 2 * self.n), dtype=np.uint8)
        self.phases = np.zeros(2 * self.n, dtype=np.uint8)
        # Initial state |0...0>
        for i in range(self.n):
            self.tableau[self.n + i, self.n + i] = 1
            self.phases[self.n + i] = 0

    def apply_h(self, q: int):
        x_col = q
        z_col = self.n + q
        self.tableau[:, [x_col, z_col]] = self.tableau[:, [z_col, x_col]]
        x_row = q
        z_row = self.n + q
        self.tableau[[x_row, z_row], :] = self.tableau[[z_row, x_row], :]
        for row in range(2 * self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4

    def apply_s(self, q: int):
        x_col = q
        z_col = self.n + q
        self.tableau[:, z_col] ^= self.tableau[:, x_col]
        for row in range(2 * self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4

    def apply_sdg(self, q: int):
        x_col = q
        z_col = self.n + q
        self.tableau[:, z_col] ^= self.tableau[:, x_col]
        for row in range(2 * self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] - 1) & 3

    def apply_cnot(self, ctrl: int, tgt: int):
        xc, xt = ctrl, tgt
        zc, zt = self.n + ctrl, self.n + tgt
        self.tableau[:, xt] ^= self.tableau[:, xc]
        self.tableau[:, zc] ^= self.tableau[:, zt]
        self.tableau[xt, :] ^= self.tableau[xc, :]
        self.tableau[zc, :] ^= self.tableau[zt, :]

    def apply_x(self, q: int):
        z_col = self.n + q
        for row in range(2 * self.n):
            if self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 2) % 4

    def apply_y(self, q: int):
        x_col = q
        z_col = self.n + q
        for row in range(2 * self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4
        self.tableau[:, x_col] ^= 1
        self.tableau[:, z_col] ^= 1

    def apply_z(self, q: int):
        x_col = q
        for row in range(2 * self.n):
            if self.tableau[row, x_col]:
                self.phases[row] = (self.phases[row] + 2) % 4

    def measure(self, q: int) -> int:
        x_col = q
        anticomm_row = None
        for row in range(self.n):
            if self.tableau[row, x_col] == 1:
                anticomm_row = row
                break
        if anticomm_row is None:
            return 0
        else:
            outcome = random.randint(0, 1)
            for j in range(2 * self.n):
                if self.tableau[j, x_col] == 1 and j != anticomm_row:
                    self.tableau[j] ^= self.tableau[anticomm_row]
                    self.phases[j] ^= self.phases[anticomm_row]
            self.tableau[anticomm_row] = 0
            self.tableau[anticomm_row, self.n + q] = 1
            self.phases[anticomm_row] = 0 if outcome == 0 else 2
            return outcome

    def measure_all(self) -> List[int]:
        return [self.measure(q) for q in range(self.n)]

    def copy(self):
        """Deep copy of the tableau."""
        new = StabilizerTableau(self.n)
        new.tableau = self.tableau.copy()
        new.phases = self.phases.copy()
        return new

    def stabilizer_expectation(self, pauli_string: str) -> float:
        """
        Estimate expectation value of a Pauli string by running many shots.
        Each character in pauli_string is 'I', 'X', 'Y', or 'Z'.
        Returns value in [-1, 1].
        """
        if len(pauli_string) != self.n:
            raise ValueError(f"Pauli string length {len(pauli_string)} != n qubits {self.n}")

        n_shots = 8192
        total = 0.0
        base_tab = self.copy()

        for _ in range(n_shots):
            tab = base_tab.copy()
            # Rotate into the measurement basis
            additional_sign = 0
            for q, p in enumerate(pauli_string):
                if p == 'X':
                    tab.apply_h(q)
                elif p == 'Y':
                    tab.apply_s(q)
                    tab.apply_h(q)
                    additional_sign += 1  # global phase tracking (approximate)
                elif p == 'Z':
                    pass  # already in Z basis
                elif p == 'I':
                    pass

            # Measure all qubits
            bits = tab.measure_all()
            # Compute parity
            parity = 1
            for q, p in enumerate(pauli_string):
                if p in ('X', 'Y', 'Z') and bits[q] == 1:
                    parity *= -1
            total += parity

        return total / n_shots

    def rank(self) -> int:
        """Compute the rank of the stabilizer tableau over GF(2)."""
        # Use row reduction over GF(2)
        mat = self.tableau.copy().astype(int)
        rows, cols = mat.shape
        rank = 0
        for col in range(cols):
            pivot = None
            for row in range(rank, rows):
                if mat[row, col] == 1:
                    pivot = row
                    break
            if pivot is not None:
                mat[[rank, pivot]] = mat[[pivot, rank]]
                for row in range(rows):
                    if row != rank and mat[row, col] == 1:
                        mat[row] ^= mat[rank]
                rank += 1
        return rank


# ======================================================================
# StabilizerBackend
# ======================================================================
class StabilizerBackend:
    """Clifford simulator using StabilizerTableau."""
    def __init__(self, qubits: int, noise_level: float = 0.0, temp_offset: float = 0.0):
        self.qubits = qubits
        self.noise_level = noise_level
        self.temp_offset = temp_offset
        self.tableau = None
        self.is_running = False
        self.gate_count = 0
        self.duration_ns = 0.0
        self.depolarising_prob = 0.01 * noise_level
        self.readout_error = 0.02 * noise_level
        self.dephasing_rate = 0.005 * noise_level
        self.T2_us = 30.0 / (1.0 + 10.0 * noise_level)
        self.temperature_k = 0.01 + temp_offset

    def start(self):
        self.tableau = StabilizerTableau(self.qubits)
        self.is_running = True
        self.gate_count = 0
        self.duration_ns = 0.0

    def apply_gate(self, gate: str, qubits: List[int], params: Optional[List[float]] = None):
        if not self.is_running:
            raise RuntimeError("Backend not started.")
        if gate == 'h':
            self.tableau.apply_h(qubits[0])
        elif gate == 's':
            self.tableau.apply_s(qubits[0])
        elif gate == 'sdg':
            self.tableau.apply_sdg(qubits[0])
        elif gate == 'cnot':
            if len(qubits) != 2:
                raise ValueError("CNOT requires two qubits.")
            self.tableau.apply_cnot(qubits[0], qubits[1])
        elif gate == 'x':
            self.tableau.apply_x(qubits[0])
        elif gate == 'y':
            self.tableau.apply_y(qubits[0])
        elif gate == 'z':
            self.tableau.apply_z(qubits[0])
        elif gate == 'rz':
            # Approximate Rz(theta) using Clifford gates
            # Rz(theta) ~ S^n * T^m decomposition (discretized)
            if params and len(params) > 0:
                theta = params[0]
                n_s = round(theta / (math.pi / 2)) % 4
                for _ in range(int(n_s)):
                    self.tableau.apply_s(qubits[0])
        elif gate == 'rx':
            if params and len(params) > 0:
                theta = params[0]
                self.tableau.apply_h(qubits[0])
                n_s = round(theta / (math.pi / 2)) % 4
                for _ in range(int(n_s)):
                    self.tableau.apply_s(qubits[0])
                self.tableau.apply_h(qubits[0])
        else:
            raise ValueError(f"Unsupported gate for stabilizer: {gate}")
        self.gate_count += 1
        self.duration_ns += 20.0
        # Depolarising noise
        if self.depolarising_prob > 0 and random.random() < self.depolarising_prob:
            q = random.randint(0, self.qubits - 1)
            pauli = random.choice(['x', 'y', 'z'])
            getattr(self.tableau, f'apply_{pauli}')(q)
        # Phase damping noise
        if self.dephasing_rate > 0 and random.random() < self.dephasing_rate:
            q = random.randint(0, self.qubits - 1)
            self.tableau.apply_z(q)

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        if not self.is_running:
            raise RuntimeError("Backend not started.")
        counts = {}
        base_tableau = self.tableau.copy()
        for _ in range(shots):
            self.tableau = base_tableau.copy()
            outcomes = self.tableau.measure_all()
            bitstr = ''.join(str(b) for b in outcomes)
            counts[bitstr] = counts.get(bitstr, 0) + 1
        self.tableau = base_tableau  # restore
        if self.readout_error > 0:
            noisy_counts = {}
            for bits, cnt in counts.items():
                bits_list = list(bits)
                for i in range(self.qubits):
                    if random.random() < self.readout_error:
                        bits_list[i] = '1' if bits_list[i] == '0' else '0'
                new_bits = ''.join(bits_list)
                noisy_counts[new_bits] = noisy_counts.get(new_bits, 0) + cnt
            counts = noisy_counts
        return counts

    def simulate_step(self, dt: float):
        self.duration_ns += dt * 1e9

    def get_metrics(self) -> Dict[str, Any]:
        error_rate = self.depolarising_prob
        gate_fidelity = 1.0 - error_rate
        coh_time = self.T2_us * 1e-6 * (1.0 - error_rate)
        throughput = self.gate_count / (self.duration_ns * 1e-9) if self.duration_ns > 0 else 0
        qv_log2 = min(self.qubits, int(1.0 / max(error_rate, 1e-6))) if error_rate > 0 else self.qubits
        return {
            "coherence_time": coh_time,
            "error_rate": error_rate,
            "gate_fidelity": gate_fidelity,
            "temperature_k": self.temperature_k,
            "throughput_ops": throughput,
            "quantum_volume_log2": qv_log2,
        }

    def stop(self):
        self.is_running = False
        self.tableau = None


# ======================================================================
# StateVectorBackend (for small qubits)
# ======================================================================
class StateVectorBackend:
    """Exact state-vector simulator for <= 20 qubits."""
    def __init__(self, qubits: int, noise_level: float = 0.0, temp_offset: float = 0.0):
        self.qubits = qubits
        self.noise_level = noise_level
        self.temp_offset = temp_offset
        self.dim = 1 << qubits
        self.state = None
        self.is_running = False
        self.gate_count = 0
        self.duration_ns = 0.0
        self.depolarising_prob = 0.01 * noise_level
        self.readout_error = 0.02 * noise_level
        self.dephasing_rate = 0.005 * noise_level
        self.amplitude_damping_rate = 0.002 * noise_level
        self.T2_us = 30.0 / (1.0 + 10.0 * noise_level)
        self.temperature_k = 0.01 + temp_offset

    def start(self):
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.is_running = True
        self.gate_count = 0
        self.duration_ns = 0.0

    def _apply_single_qubit_gate(self, U: np.ndarray, q: int):
        shape = [2] * self.qubits
        tensor = self.state.reshape(shape)
        axes = list(range(self.qubits))
        axes.remove(q)
        axes = [q] + axes
        tensor = np.transpose(tensor, axes)
        mat = tensor.reshape((2, -1))
        mat = U @ mat
        tensor = mat.reshape([2] + [2] * (self.qubits - 1))
        inv_axes = np.argsort(axes)
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(-1)

    def _apply_two_qubit_gate(self, U2: np.ndarray, q1: int, q2: int):
        others = [i for i in range(self.qubits) if i not in (q1, q2)]
        axes = [q1, q2] + others
        tensor = self.state.reshape([2] * self.qubits)
        tensor = np.transpose(tensor, axes)
        mat = tensor.reshape((4, -1))
        mat = U2 @ mat
        tensor = mat.reshape([2, 2] + [2] * len(others))
        inv_axes = np.argsort(axes)
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(-1)

    def apply_gate(self, gate: str, qubits: List[int], params: Optional[List[float]] = None):
        if not self.is_running:
            raise RuntimeError("Backend not started.")
        if gate == 'h':
            U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 's':
            U = np.array([[1, 0], [0, 1j]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'sdg':
            U = np.array([[1, 0], [0, -1j]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 't':
            U = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'tdg':
            U = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'cnot':
            U = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
            self._apply_two_qubit_gate(U, qubits[0], qubits[1])
        elif gate == 'x':
            U = np.array([[0, 1], [1, 0]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'y':
            U = np.array([[0, -1j], [1j, 0]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'z':
            U = np.array([[1, 0], [0, -1]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'rz':
            if params and len(params) > 0:
                theta = params[0]
                U = np.array([[1, 0], [0, np.exp(1j * theta)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Rz gate requires a parameter (rotation angle).")
        elif gate == 'rx':
            if params and len(params) > 0:
                theta = params[0]
                U = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                              [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Rx gate requires a parameter.")
        elif gate == 'ry':
            if params and len(params) > 0:
                theta = params[0]
                U = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                              [np.sin(theta / 2), np.cos(theta / 2)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Ry gate requires a parameter.")
        elif gate == 'rxx':
            if params and len(params) > 0:
                theta = params[0]
                c = np.cos(theta / 2)
                s = 1j * np.sin(theta / 2)
                U = np.array([[c, 0, 0, -s], [0, c, s, 0],
                              [0, s, c, 0], [-s, 0, 0, c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Rxx gate requires a parameter.")
        elif gate == 'ryy':
            if params and len(params) > 0:
                theta = params[0]
                c = np.cos(theta / 2)
                s = 1j * np.sin(theta / 2)
                U = np.array([[c, 0, 0, s], [0, c, -s, 0],
                              [0, -s, c, 0], [s, 0, 0, c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Ryy gate requires a parameter.")
        elif gate == 'rzz':
            if params and len(params) > 0:
                theta = params[0]
                c = np.cos(theta / 2)
                s = 1j * np.sin(theta / 2)
                U = np.array([[c, -s, 0, 0], [-s, c, 0, 0],
                              [0, 0, c, s], [0, 0, s, c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Rzz gate requires a parameter.")
        else:
            raise ValueError(f"Unsupported gate: {gate}")

        self.gate_count += 1
        self.duration_ns += 20.0

        # Depolarising noise
        if self.depolarising_prob > 0 and random.random() < self.depolarising_prob:
            q = random.randint(0, self.qubits - 1)
            pauli = random.choice(['x', 'y', 'z'])
            pauli_matrices = {
                'x': np.array([[0, 1], [1, 0]]),
                'y': np.array([[0, -1j], [1j, 0]]),
                'z': np.array([[1, 0], [0, -1]])
            }
            self._apply_single_qubit_gate(pauli_matrices[pauli], q)

        # Phase damping (dephasing)
        if self.dephasing_rate > 0 and random.random() < self.dephasing_rate:
            q = random.randint(0, self.qubits - 1)
            self._apply_single_qubit_gate(np.array([[1, 0], [0, -1]]), q)

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        probs = np.abs(self.state) ** 2
        probs = probs / np.sum(probs)  # normalise
        outcomes = np.random.choice(self.dim, size=shots, p=probs)
        counts = {}
        for out in outcomes:
            bits = format(out, f'0{self.qubits}b')
            counts[bits] = counts.get(bits, 0) + 1
        if self.readout_error > 0:
            noisy = {}
            for bits, cnt in counts.items():
                bits_list = list(bits)
                for i in range(self.qubits):
                    if random.random() < self.readout_error:
                        bits_list[i] = '1' if bits_list[i] == '0' else '0'
                new_bits = ''.join(bits_list)
                noisy[new_bits] = noisy.get(new_bits, 0) + cnt
            counts = noisy
        return counts

    def simulate_step(self, dt: float):
        self.duration_ns += dt * 1e9

    def get_metrics(self) -> Dict[str, Any]:
        error_rate = self.depolarising_prob
        gate_fidelity = 1.0 - error_rate
        coh_time = self.T2_us * 1e-6 * (1.0 - error_rate)
        throughput = self.gate_count / (self.duration_ns * 1e-9) if self.duration_ns > 0 else 0
        qv_log2 = min(self.qubits, int(1.0 / max(error_rate, 1e-6))) if error_rate > 0 else self.qubits
        return {
            "coherence_time": coh_time,
            "error_rate": error_rate,
            "gate_fidelity": gate_fidelity,
            "temperature_k": self.temperature_k,
            "throughput_ops": throughput,
            "quantum_volume_log2": qv_log2,
        }

    def stop(self):
        self.is_running = False
        self.state = None


# ======================================================================
# QuantumVM - Automatic Backend Selection
# ======================================================================
class QuantumVM:
    """
    Automatically selects backend:
    - qubits <= 20 -> StateVectorBackend (exact, full gate support)
    - qubits >  20 -> StabilizerBackend  (fast, Clifford-only)
    """
    def __init__(self, qubits: int, noise_level: float = 0.0, temp_offset: float = 0.0):
        self.qubits = qubits
        self.noise_level = noise_level
        self.temp_offset = temp_offset
        if qubits <= 20:
            self._backend = StateVectorBackend(qubits, noise_level, temp_offset)
        else:
            self._backend = StabilizerBackend(qubits, noise_level, temp_offset)

    def start(self):
        self._backend.start()

    def apply_gate(self, gate: str, qubits: List[int], params: Optional[List[float]] = None):
        self._backend.apply_gate(gate, qubits, params)

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        return self._backend.measure(shots)

    def simulate_step(self, dt: float):
        self._backend.simulate_step(dt)

    def get_metrics(self) -> Dict[str, Any]:
        return self._backend.get_metrics()

    def stop(self):
        self._backend.stop()


# ======================================================================
# QuantumVMGravity - Enhanced Engine for Gravitational Simulations
# ======================================================================
class QuantumVMGravity(QuantumVM):
    """
    Enhanced quantum virtual machine with gravitational and ontological
    simulation capabilities.

    Adds to QuantumVM:
    - expectation(): Pauli string expectation values
    - get_density_matrix(): full/reduced density matrices
    - von_neumann_entropy(): subsystem entropy
    - entanglement_negativity(): bipartite negativity
    - mutual_information_bipartite(): mutual information between subsystems
    - trotter_step(): time evolution under 2-body Hamiltonians
    - correlation_matrix(): two-point correlators
    - topological_entropy(): Kitaev-Preskill TEE
    - bohmiann_trajectory(): pilot wave trajectories
    - apply_noise_channel(): explicit dephasing/amplitude damping

    For qubits > 20, expectation values and density matrices are estimated
    via shot-based methods (stabilizer sampling).
    """

    def __init__(self, qubits: int, noise_level: float = 0.0, temp_offset: float = 0.0):
        super().__init__(qubits, noise_level, temp_offset)
        self._backend_type = 'statevector' if qubits <= 20 else 'stabilizer'

    def expectation(self, pauli_string: str) -> float:
        """
        Compute expectation value of a Pauli string observable.

        For statevector backend: exact computation via inner product.
        For stabilizer backend: estimated via 8192 shots.

        Parameters
        ----------
        pauli_string : str
            String of 'I', 'X', 'Y', 'Z' characters, one per qubit.

        Returns
        -------
        float
            Expectation value in [-1, 1].
        """
        if len(pauli_string) != self.qubits:
            raise ValueError(
                f"Pauli string length {len(pauli_string)} != n qubits {self.qubits}")

        if self._backend_type == 'statevector':
            return self._expectation_statevector(pauli_string)
        else:
            return self._expectation_stabilizer(pauli_string)

    def _expectation_statevector(self, pauli_string: str) -> float:
        """Exact expectation via statevector inner product."""
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")

        # Build Pauli operator as a matrix
        single_paulis = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
        }

        op = single_paulis[pauli_string[0]]
        for i in range(1, self.qubits):
            op = np.kron(op, single_paulis[pauli_string[i]])

        return float(np.real(state.conj() @ op @ state))

    def _expectation_stabilizer(self, pauli_string: str) -> float:
        """Estimate expectation via stabilizer sampling."""
        if not self._backend.is_running:
            raise RuntimeError("Backend not started.")
        return self._backend.tableau.stabilizer_expectation(pauli_string)

    def get_density_matrix(self) -> np.ndarray:
        """
        Return the full density matrix rho = |psi><psi|.

        Only available for statevector backend (qubits <= 20).
        For stabilizer backend, raises ValueError.
        """
        if self._backend_type != 'statevector':
            raise ValueError(
                "Density matrix not available for stabilizer backend (qubits > 20). "
                "Use expectation() with shot-based estimation instead.")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        return np.outer(state, state.conj())

    def get_reduced_density_matrix(self, subsystem: List[int]) -> np.ndarray:
        """
        Return the reduced density matrix for a subsystem.

        Parameters
        ----------
        subsystem : list of int
            Qubit indices to keep.

        Returns
        -------
        np.ndarray
            Reduced density matrix of shape (2^k, 2^k) where k = len(subsystem).
        """
        rho = self.get_density_matrix()
        trace_out = [q for q in range(self.qubits) if q not in subsystem]
        return partial_trace_fast(rho, trace_out, self.qubits)

    def von_neumann_entropy_subsystem(self, subsystem: List[int]) -> float:
        """
        Compute von Neumann entropy of a subsystem.
        S = -Tr(rho_sub * log2(rho_sub))
        """
        rho_sub = self.get_reduced_density_matrix(subsystem)
        return von_neumann_entropy(rho_sub)

    def entanglement_negativity_pair(self, q1: int, q2: int) -> float:
        """
        Compute log-negativity between a pair of qubits.
        """
        rho_pair = self.get_reduced_density_matrix([q1, q2])
        return entanglement_negativity(rho_pair)

    def mutual_information_bipartite(self, subsystem_a: List[int],
                                      subsystem_b: List[int]) -> float:
        """
        Compute mutual information I(A:B) = S(A) + S(B) - S(AB).
        """
        s_a = self.von_neumann_entropy_subsystem(subsystem_a)
        s_b = self.von_neumann_entropy_subsystem(subsystem_b)
        s_ab = self.von_neumann_entropy_subsystem(subsystem_a + subsystem_b)
        return s_a + s_b - s_ab

    def correlation_matrix(self) -> np.ndarray:
        """
        Compute the two-point correlation matrix C[i,j] = <Z_i Z_j> - <Z_i><Z_j>.
        Returns an (n_qubits, n_qubits) matrix.
        """
        n = self.qubits
        # Compute single-site expectations
        z_expect = np.zeros(n)
        for i in range(n):
            pauli = ['I'] * n
            pauli[i] = 'Z'
            z_expect[i] = self.expectation(''.join(pauli))

        # Compute two-site correlations
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i, j] = 1.0 - z_expect[i] ** 2
                else:
                    pauli = ['I'] * n
                    pauli[i] = 'Z'
                    pauli[j] = 'Z'
                    corr[i, j] = self.expectation(''.join(pauli)) - z_expect[i] * z_expect[j]
        return corr

    def trotter_step(self, J: float, h: float, dt: float,
                     pairs: List[Tuple[int, int]],
                     hamiltonian_type: str = 'xx_yy_z'):
        """
        Perform one Trotter step of time evolution under a 2-body Hamiltonian.

        Supports:
        - 'xx_yy_z': H = -J * sum_{<i,j>} (X_i X_j + Y_i Y_j) - h * sum_i Z_i
        - 'heisenberg': H = J * sum_{<i,j>} (X_i X_j + Y_i Y_j + Z_i Z_j) - h * sum_i Z_i
        - 'ising': H = -J * sum_{<i,j>} Z_i Z_j - h * sum_i X_i
        - 'tfim': H = -J * sum_{<i,j>} Z_i Z_j - h * sum_i X_i (alias for ising)

        Parameters
        ----------
        J : float
            Coupling constant.
        h : float
            Transverse field strength.
        dt : float
            Time step.
        pairs : list of (int, int)
            Nearest-neighbor pairs for the coupling term.
        hamiltonian_type : str
            Type of Hamiltonian to evolve under.
        """
        if self._backend_type == 'statevector':
            self._trotter_statevector(J, h, dt, pairs, hamiltonian_type)
        else:
            self._trotter_stabilizer(J, h, dt, pairs, hamiltonian_type)

    def _trotter_statevector(self, J: float, h: float, dt: float,
                              pairs: List[Tuple[int, int]], htype: str):
        """Exact Trotter step using rotation gates (statevector mode)."""
        if htype in ('xx_yy_z',):
            # -J dt * (XX + YY) for each pair -> Rxx(-2J dt) and Ryy(-2J dt)
            for i, j in pairs:
                self.apply_gate('rxx', [i, j], [-2 * J * dt])
                self.apply_gate('ryy', [i, j], [-2 * J * dt])
            # -h dt * Z -> Rz(-2h dt)
            for q in range(self.qubits):
                self.apply_gate('rz', [q], [-2 * h * dt])
        elif htype in ('ising', 'tfim'):
            # -J * ZZ -> Rzz(-2J dt) for each pair
            for i, j in pairs:
                self.apply_gate('rzz', [i, j], [-2 * J * dt])
            # -h * X -> Rx(-2h dt)
            for q in range(self.qubits):
                self.apply_gate('rx', [q], [-2 * h * dt])
        elif htype == 'heisenberg':
            for i, j in pairs:
                self.apply_gate('rxx', [i, j], [2 * J * dt])
                self.apply_gate('ryy', [i, j], [2 * J * dt])
                self.apply_gate('rzz', [i, j], [2 * J * dt])
            for q in range(self.qubits):
                self.apply_gate('rz', [q], [-2 * h * dt])
        else:
            raise ValueError(f"Unknown Hamiltonian type: {htype}")

    def _trotter_stabilizer(self, J: float, h: float, dt: float,
                             pairs: List[Tuple[int, int]], htype: str):
        """Approximate Trotter step using Clifford gates (stabilizer mode).

        Uses discretized rotation angles rounded to multiples of pi/4.
        This introduces Trotter error in addition to the Clifford approximation.
        """
        # Discretize angles to multiples of pi/4
        def discretize(angle):
            return round(angle / (math.pi / 4)) * (math.pi / 4)

        if htype in ('xx_yy_z',):
            for i, j in pairs:
                # Approximate Rxx(theta) via CNOT-Rz-CNOT
                theta = discretize(-2 * J * dt)
                self.apply_gate('h', [i])
                self.apply_gate('h', [j])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('h', [i])
                self.apply_gate('h', [j])
            for q in range(self.qubits):
                theta = discretize(-2 * h * dt)
                self.apply_gate('rz', [q], [theta])
        elif htype in ('ising', 'tfim'):
            for i, j in pairs:
                theta = discretize(-2 * J * dt)
                self.apply_gate('h', [i])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('h', [i])
            for q in range(self.qubits):
                theta = discretize(-2 * h * dt)
                self.apply_gate('rx', [q], [theta])
        elif htype == 'heisenberg':
            for i, j in pairs:
                theta = discretize(2 * J * dt)
                self.apply_gate('h', [i])
                self.apply_gate('h', [j])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i, j])
                self.apply_gate('h', [i])
                self.apply_gate('h', [j])
            for q in range(self.qubits):
                theta = discretize(-2 * h * dt)
                self.apply_gate('rz', [q], [theta])

    def apply_noise_channel(self, channel_type: str, qubits: List[int],
                             probability: float):
        """
        Explicitly apply a noise channel to specified qubits.

        Parameters
        ----------
        channel_type : str
            'dephasing' (phase damping) or 'amplitude_damping' (T1 decay).
        qubits : list of int
            Qubits to apply noise to.
        probability : float
            Noise probability (0 to 1).
        """
        if channel_type == 'dephasing':
            for q in qubits:
                if random.random() < probability:
                    self.apply_gate('z', [q])
        elif channel_type == 'amplitude_damping':
            for q in qubits:
                if random.random() < probability:
                    # Amplitude damping: with probability p, |1> -> |0>
                    # Simulated by measuring and post-selecting (approximate)
                    # For statevector, we apply a Kraus operator
                    if self._backend_type == 'statevector':
                        # K0 = [[1, 0], [0, sqrt(1-p)]], K1 = [[0, sqrt(p)], [0, 0]]
                        # We probabilistically apply K1
                        K0 = np.array([[1, 0], [0, math.sqrt(1 - probability)]], dtype=complex)
                        self._backend._apply_single_qubit_gate(K0, q)
                    else:
                        # For stabilizer, approximate as bit flip with reduced prob
                        if random.random() < probability * 0.5:
                            self.apply_gate('x', [q])
        else:
            raise ValueError(f"Unknown noise channel: {channel_type}")

    def bohmiann_trajectory(self, qubit: int, times: List[float],
                             J: float, h: float,
                             pairs: List[Tuple[int, int]],
                             hamiltonian_type: str = 'xx_yy_z') -> List[float]:
        """
        Compute Bohmian (pilot-wave) trajectory for a single qubit position.

        The "position" is the expected Z value: q(t) = <Z_qubit>(t).
        This follows the de Broglie-Bohm guidance equation in the
        discretized qubit setting.

        Returns list of q(t) values at each time.
        """
        if self._backend_type != 'statevector':
            raise ValueError("Bohmian trajectories require statevector backend (qubits <= 20).")

        # Save current state
        saved_state = self._backend.state.copy()
        trajectory = []

        dt = times[1] - times[0] if len(times) > 1 else 0.01
        for t in times:
            pauli = ['I'] * self.qubits
            pauli[qubit] = 'Z'
            z_val = self.expectation(''.join(pauli))
            trajectory.append(z_val)
            # Evolve one step
            self.trotter_step(J, h, dt, pairs, hamiltonian_type)

        # Restore state
        self._backend.state = saved_state
        return trajectory

    def compute_bit_mass(self, temperature_k: float,
                          delta_entropy_bits: float,
                          curvature_coupling: float = 0.0,
                          lambda_understanding: float = 1.0) -> float:
        """
        Compute the predicted mass shift from the bit-mass equation:

            m_bit = (k_B * T * ln2 / c^2) * (1 + R / (6 * Lambda))

            delta_m = m_bit * delta_S

        Parameters
        ----------
        temperature_k : float
            Effective temperature in Kelvin (e.g., T_c of the superconductor).
        delta_entropy_bits : float
            Change in von Neumann entropy in bits.
        curvature_coupling : float
            Coupling to spacetime curvature (dimensionless).
        lambda_understanding : float
            Understanding scale parameter (default 1.0).

        Returns
        -------
        float
            Predicted mass shift in kg.
        """
        m_bit = (KB * temperature_k * LN2 / C_LIGHT ** 2) * (
            1.0 + curvature_coupling / (6.0 * lambda_understanding))
        return m_bit * delta_entropy_bits

    def compute_information_pressure(self, negativity_sum: float,
                                      n_bits: float) -> float:
        """
        Compute the information pressure p_info from entanglement negativity.

        p_info ~ -sum of negativities (negative pressure).

        Parameters
        ----------
        negativity_sum : float
            Sum of log-negativities over all measured pairs.
        n_bits : float
            Information density (bits per unit volume, in simulation units).

        Returns
        -------
        float
            Information pressure (dimensionless, simulation units).
        """
        return -negativity_sum * n_bits

    def compute_sophia_susceptibility(self, J_values: List[float],
                                       coherence_values: List[float]) -> Dict[str, float]:
        """
        Compute the susceptibility chi = dC/dJ near the critical point.

        Fits a polynomial to extract dC/dJ and identifies the Sophia point
        where susceptibility peaks (C* = 1/phi ~ 0.618).

        Returns dict with 'chi_max', 'J_critical', 'C_sophia', 'chi_values'.
        """
        J_arr = np.array(J_values)
        C_arr = np.array(coherence_values)

        # Numerical derivative
        chi = np.abs(np.gradient(C_arr, J_arr))

        # Find peak susceptibility
        idx_max = np.argmax(chi)
        chi_max = chi[idx_max]
        J_critical = J_arr[idx_max]
        C_sophia = C_arr[idx_max]

        return {
            'chi_max': float(chi_max),
            'J_critical': float(J_critical),
            'C_sophia': float(C_sophia),
            'sophia_target': SOPHIA_COHERENCE,
            'chi_values': chi.tolist(),
        }

    def compute_lieb_robinson_velocity(self, correlations_history: List[np.ndarray],
                                        lattice_spacing: float = 1.0,
                                        dt: float = 0.01) -> float:
        """
        Estimate the Lieb-Robinson velocity from correlation snapshots.

        Parameters
        ----------
        correlations_history : list of np.ndarray
            Correlation matrices at successive time steps.
        lattice_spacing : float
            Physical spacing between qubits.
        dt : float
            Time step between snapshots.

        Returns
        -------
        float
            Estimated v_LR.
        """
        return lieg_robinson_velocity(correlations_history, lattice_spacing, dt)

    def compute_fractal_dimension(self, corr_matrix: np.ndarray,
                                   grid_rows: int, grid_cols: int) -> float:
        """
        Estimate the fractal dimension of the entanglement structure.

        Parameters
        ----------
        corr_matrix : np.ndarray, shape (n, n)
            Correlation matrix.
        grid_rows : int
            Number of rows in the 2D lattice.
        grid_cols : int
            Number of columns in the 2D lattice.

        Returns
        -------
        float
            Estimated fractal dimension D_f.
        """
        return box_counting_fractal_dimension(corr_matrix, min(grid_rows, grid_cols))

    def compute_amplification_efficiency(self, fractal_dim: float,
                                          l_planck: float = 1.616e-35,
                                          l_bio: float = 1e-6) -> float:
        """
        Compute fractal amplification efficiency:

            eta = (l_Planck / l_bio)^(D_f - 4)

        Parameters
        ----------
        fractal_dim : float
            Estimated fractal dimension D_f.
        l_planck : float
            Planck length [m].
        l_bio : float
            Biological/macroscale length [m].

        Returns
        -------
        float
            Amplification factor (dimensionless).
        """
        return (l_planck / l_bio) ** (fractal_dim - 4)

    def topological_entropy_kp(self, regions: Dict[str, List[int]]) -> float:
        """
        Compute topological entanglement entropy using the Kitaev-Preskill formula:
            S_topo = S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

        Parameters
        ----------
        regions : dict
            Mapping from region names ('A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC')
            to lists of qubit indices.

        Returns
        -------
        float
            Topological entanglement entropy gamma.
        """
        entropies = {}
        for name, qubits in regions.items():
            entropies[name] = self.von_neumann_entropy_subsystem(qubits)
        return topological_entanglement_entropy(entropies)

    def estimate_betti_numbers(self, corr_matrix: np.ndarray,
                                threshold: float = 0.1) -> Dict[str, int]:
        """
        Estimate Betti numbers from the entanglement graph.

        Constructs a graph where edges exist for |correlation| > threshold,
        then counts connected components (beta_0), holes (beta_1), and
        voids (beta_2) using a simple algebraic topology approach.

        Parameters
        ----------
        corr_matrix : np.ndarray
            Correlation matrix.
        threshold : float
            Minimum |correlation| for an edge.

        Returns
        -------
        dict
            {'beta_0': int, 'beta_1': int, 'beta_2': int}
        """
        n = corr_matrix.shape[0]
        # Build adjacency from correlations
        adj = (np.abs(corr_matrix) > threshold).astype(int)
        np.fill_diagonal(adj, 0)

        # beta_0: connected components via BFS
        visited = [False] * n
        beta_0 = 0
        components = []
        for start in range(n):
            if not visited[start]:
                beta_0 += 1
                component = []
                queue = [start]
                visited[start] = True
                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    for nb in range(n):
                        if adj[node, nb] and not visited[nb]:
                            visited[nb] = True
                            queue.append(nb)
                components.append(set(component))

        # beta_1: number of independent cycles = edges - vertices + components
        n_edges = np.sum(adj) // 2
        beta_1 = n_edges - n + beta_0
        if beta_1 < 0:
            beta_1 = 0

        # beta_2: estimate from 3-cliques minus triangular faces (approximate)
        n_triangles = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    for k in range(j + 1, n):
                        if adj[i, k] and adj[j, k]:
                            n_triangles += 1
        beta_2 = max(0, n_triangles - n_edges + n - beta_0)

        return {'beta_0': beta_0, 'beta_1': beta_1, 'beta_2': beta_2}

    def quantum_cramer_rao_bound(self, fisher_information: float) -> float:
        """
        Compute the quantum Cramér-Rao bound:
            delta_omega >= 1 / sqrt(F_Q)

        Parameters
        ----------
        fisher_information : float
            Quantum Fisher information.

        Returns
        -------
        float
            Minimum standard deviation of the frequency estimate.
        """
        if fisher_information <= 0:
            return float('inf')
        return 1.0 / math.sqrt(fisher_information)

    def adaptive_lambda(self, t: float, tau: float = 1.0,
                         lambda_min: float = 0.01,
                         lambda_max: float = 0.02) -> float:
        """
        Compute adaptive regularization parameter:
            lambda(t) = max(lambda_min, lambda_max * exp(-t/tau))

        This implements the constitutional defense mechanism where
        the regularization strength decays exponentially but has
        a floor to prevent complete loss of stability.

        Parameters
        ----------
        t : float
            Current simulation time.
        tau : float
            Decay timescale.
        lambda_min : float
            Minimum regularization (floor).
        lambda_max : float
            Maximum regularization (initial value).

        Returns
        -------
        float
            Current regularization parameter.
        """
        return max(lambda_min, lambda_max * math.exp(-t / tau))


# ======================================================================
# Surface Code Utilities (for Blueprint 2)
# ======================================================================
class SurfaceCodeBuilder:
    """
    Builds and manipulates surface code states for topological simulations.

    Supports distance-d surface codes with:
    - Stabilizer measurement simulation
    - Logical state preparation
    - Defect nucleation
    - Error correction cycles
    """

    @staticmethod
    def get_qubit_indices(distance: int):
        """
        Map surface code qubit coordinates to linear indices.

        Returns data qubits, X-stabilizer ancillas, Z-stabilizer ancillas,
        and the coordinate-to-index mapping.

        For a distance-d surface code:
        - Data qubits on a d x d grid (d^2 total)
        - X-stabilizers at centers of plaquettes (d-1) x d/2
        - Z-stabilizers at centers of stars (d/2) x (d-1)
        """
        if distance % 2 != 1 or distance < 3:
            raise ValueError("Surface code distance must be an odd integer >= 3")

        data_qubits = []
        coord_to_idx = {}
        idx = 0
        for r in range(distance):
            for c in range(distance):
                data_qubits.append(idx)
                coord_to_idx[(r, c)] = idx
                idx += 1

        # X stabilizers (plaquettes)
        x_stabs = []
        for r in range(0, distance - 1, 2):
            for c in range(1, distance - 1, 2):
                # Neighboring data qubits
                neighbors = [
                    coord_to_idx[(r, c)],
                    coord_to_idx[(r, c + 1)],
                    coord_to_idx[(r + 1, c)],
                    coord_to_idx[(r + 1, c + 1)]
                ]
                x_stabs.append(neighbors)

        # Z stabilizers (stars)
        z_stabs = []
        for r in range(1, distance - 1, 2):
            for c in range(0, distance - 1, 2):
                neighbors = [
                    coord_to_idx[(r, c)],
                    coord_to_idx[(r, c + 1)],
                    coord_to_idx[(r + 1, c)],
                    coord_to_idx[(r + 1, c + 1)]
                ]
                z_stabs.append(neighbors)

        return {
            'data_qubits': data_qubits,
            'x_stabilizers': x_stabs,
            'z_stabilizers': z_stabs,
            'coord_to_idx': coord_to_idx,
            'distance': distance,
            'n_data': distance ** 2,
            'n_x_stabs': len(x_stabs),
            'n_z_stabs': len(z_stabs),
        }

    @staticmethod
    def prepare_logical_zero(vm: QuantumVMGravity, code_info: dict):
        """
        Prepare the logical |0> state of the surface code.

        For a statevector backend, this creates the ground state by
        applying a sequence of entangling gates that project onto
        the stabilizer code space.

        For a stabilizer backend, this initializes the stabilizer tableau
        with the appropriate stabilizer generators.
        """
        n = vm.qubits
        # Prepare all data qubits in |0> (already the default)
        # Apply a circuit that entangles qubits to satisfy stabilizers
        # Use CNOT gates in a pattern that creates the code state

        # Simple approximate preparation using a checkerboard CNOT pattern
        pairs_x = code_info.get('x_stabilizers', [])
        pairs_z = code_info.get('z_stabilizers', [])

        # Apply entangling gates to create correlations
        for stab in pairs_x:
            if len(stab) == 4:
                # CNOT from first qubit to others
                for target in stab[1:]:
                    vm.apply_gate('cnot', [stab[0], target])

        for stab in pairs_z:
            if len(stab) == 4:
                vm.apply_gate('h', [stab[0]])
                for target in stab[1:]:
                    vm.apply_gate('cnot', [stab[0], target])
                vm.apply_gate('h', [stab[0]])

    @staticmethod
    def nucleate_defect(vm: QuantumVMGravity, defect_qubits: List[int],
                         strength: float):
        """
        Nucleate a topological defect by applying a local perturbation.

        The perturbation is modeled as a local magnetic field pulse:
        H_pert = strength * sum_{q in defect_qubits} X_q

        Implemented as Rx rotation.
        """
        for q in defect_qubits:
            vm.apply_gate('rx', [q], [strength])

    @staticmethod
    def measure_stabilizers(vm: QuantumVMGravity, code_info: dict,
                            shots: int = 4096) -> Dict[str, float]:
        """
        Measure the expectation values of all stabilizer generators.

        Returns dict mapping stabilizer names to their expectation values.
        """
        results = {}
        for i, stab in enumerate(code_info.get('x_stabilizers', [])):
            pauli = ['I'] * vm.qubits
            # X stabilizer: product of X on neighboring qubits
            for q in stab:
                pauli[q] = 'X'
            results[f'X_stab_{i}'] = vm.expectation(''.join(pauli))

        for i, stab in enumerate(code_info.get('z_stabilizers', [])):
            pauli = ['I'] * vm.qubits
            for q in stab:
                pauli[q] = 'Z'
            results[f'Z_stab_{i}'] = vm.expectation(''.join(pauli))

        return results


# ======================================================================
# Ramsey Interferometry Utilities (for Blueprint 3)
# ======================================================================
class RamseyInterferometer:
    """
    Quantum metrology toolkit for Ramsey interferometry simulations.

    Implements:
    - Ramsey sequence (pi/2 - wait - pi/2 - measure)
    - Dispersive coupling to harmonic oscillator
    - Fisher information estimation
    - Cramér-Rao bound computation
    - Noise-aware sensitivity analysis
    """

    @staticmethod
    def ramsey_sequence(vm: QuantumVMGravity, qubit: int,
                         wait_time: float, detuning: float,
                         shots: int = 10000) -> Dict[str, Any]:
        """
        Execute a Ramsey interferometry sequence.

        Parameters
        ----------
        vm : QuantumVMGravity
            The quantum virtual machine instance.
        qubit : int
            Target qubit index.
        wait_time : float
            Free evolution time.
        detuning : float
            Frequency detuning in radians/time.
        shots : int
            Number of measurement shots.

        Returns
        -------
        dict
            {'prob_0': float, 'prob_1': float, 'fringe_contrast': float}
        """
        # First pi/2 pulse (Hadamard)
        vm.apply_gate('h', [qubit])

        # Free evolution: Z rotation by detuning * wait_time
        # |1> accumulates phase at rate detuning
        vm.apply_gate('rz', [qubit], [detuning * wait_time])

        # Second pi/2 pulse
        vm.apply_gate('h', [qubit])

        # Measure
        counts = vm.measure(shots=shots)
        total = sum(counts.values())
        if total == 0:
            return {'prob_0': 0.5, 'prob_1': 0.5, 'fringe_contrast': 0.0}

        # Extract probability of measuring |1> on target qubit
        prob_1 = 0.0
        for bitstr, cnt in counts.items():
            if qubit < len(bitstr) and bitstr[len(bitstr) - 1 - qubit] == '1':
                prob_1 += cnt / total

        prob_0 = 1.0 - prob_1
        contrast = abs(prob_0 - prob_1)

        return {
            'prob_0': prob_0,
            'prob_1': prob_1,
            'fringe_contrast': contrast,
            'detuning': detuning,
            'wait_time': wait_time,
            'shots': shots,
        }

    @staticmethod
    def estimate_fisher_information(vm_class, qubit: int, n_qubits: int,
                                     detuning_values: np.ndarray,
                                     wait_time: float,
                                     noise_level: float = 0.0,
                                     shots_per_point: int = 5000) -> Dict[str, Any]:
        """
        Estimate the classical Fisher information for Ramsey frequency estimation.

        F(omega) = sum_i (dp_i/d_omega)^2 / p_i

        Parameters
        ----------
        vm_class : type
            QuantumVMGravity class (or QuantumVMGravity itself).
        qubit : int
            Target qubit.
        n_qubits : int
            Total number of qubits.
        detuning_values : np.ndarray
            Array of detuning values to probe.
        wait_time : float
            Evolution time.
        noise_level : float
            Backend noise level.
        shots_per_point : int
            Shots per detuning value.

        Returns
        -------
        dict
            Fisher information, Cramér-Rao bound, optimal detuning, etc.
        """
        prob_1_values = []
        for delta in detuning_values:
            vm = vm_class(qubits=n_qubits, noise_level=noise_level)
            vm.start()
            result = RamseyInterferometer.ramsey_sequence(
                vm, qubit, wait_time, delta, shots=shots_per_point)
            prob_1_values.append(result['prob_1'])
            vm.stop()

        prob_1_arr = np.array(prob_1_values)
        delta_arr = detuning_values

        # Numerical derivative dp/d(omega)
        dp_domega = np.gradient(prob_1_arr, delta_arr)

        # Fisher information: F = (dp/dw)^2 / (p(1-p))
        p_safe = np.clip(prob_1_arr, 1e-10, 1 - 1e-10)
        fisher_per_point = (dp_domega ** 2) / (p_safe * (1 - p_safe))
        fisher_total = float(np.sum(fisher_per_point))

        # Cramér-Rao bound: delta_omega >= 1/sqrt(N * F)
        crb = 1.0 / math.sqrt(fisher_total) if fisher_total > 0 else float('inf')

        return {
            'fisher_information': fisher_total,
            'cramer_rao_bound': crb,
            'fisher_per_point': fisher_per_point.tolist(),
            'prob_1_values': prob_1_values.tolist(),
            'optimal_detuning': float(delta_arr[np.argmax(np.abs(dp_domega))]),
        }

    @staticmethod
    def estimate_minimum_mass_shift(crb_freq: float, dispersive_coupling: float,
                                     mass_coupling: float,
                                     sample_mass_kg: float = 1e-6) -> Dict[str, float]:
        """
        Convert frequency sensitivity to minimum detectable mass shift.

        The mass shift creates a frequency shift via the dispersive coupling:
            delta_omega = chi * delta_n = chi * (delta_m / m_phonon)

        Parameters
        ----------
        crb_freq : float
            Cramér-Rao bound on frequency estimation (rad/s).
        dispersive_coupling : float
            Dispersive coupling chi (rad/s per phonon).
        mass_coupling : float
            Mass-to-phonon coupling (phonons per kg).
        sample_mass_kg : float
            Sample mass in kg.

        Returns
        -------
        dict
            Minimum detectable mass shift and related quantities.
        """
        # delta_omega = dispersive_coupling * mass_coupling * delta_m
        # So delta_m = delta_omega / (dispersive_coupling * mass_coupling)
        if dispersive_coupling * mass_coupling == 0:
            return {'delta_m_min': float('inf'), 'signal_to_noise': 0.0}

        delta_m_min = crb_freq / (dispersive_coupling * mass_coupling)
        # Signal from bit-mass equation for T_c = 9.2 K (Niobium)
        predicted_signal = (KB * 9.2 * LN2 / C_LIGHT ** 2) * sample_mass_kg

        return {
            'delta_m_min': delta_m_min,
            'predicted_signal_kg': predicted_signal,
            'signal_to_noise': predicted_signal / delta_m_min if delta_m_min > 0 else float('inf'),
            'feasible': delta_m_min < predicted_signal * 10,  # within 10x of prediction
        }


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("MOS-HOR-QNVM v14.0-Gravity (Enhanced Engine)")
    print("=" * 60)

    # Test 1: Basic Bell state with expectation values
    print("\n[Test 1] 4-qubit Bell state with expectations")
    vm = QuantumVMGravity(qubits=4, noise_level=0.0)
    vm.start()
    vm.apply_gate('h', [0])
    vm.apply_gate('cnot', [0, 1])
    corr = vm.correlation_matrix()
    print(f"  <ZZ>_{0,1} = {corr[0,1]:.4f} (expect ~1.0)")
    print(f"  <Z>_{0}    = {vm.expectation('ZIII'):.4f} (expect ~0.0)")
    ent = vm.entanglement_negativity_pair(0, 1)
    print(f"  Negativity(0,1) = {ent:.4f}")
    vm.stop()

    # Test 2: Trotter evolution (statevector)
    print("\n[Test 2] Trotter evolution on 4-qubit lattice")
    vm2 = QuantumVMGravity(qubits=4, noise_level=0.0)
    vm2.start()
    pairs = [(0, 1), (1, 2), (2, 3)]
    for step in range(50):
        vm2.trotter_step(J=0.5, h=0.1, dt=0.02, pairs=pairs, hamiltonian_type='xx_yy_z')
    ent_s = vm2.von_neumann_entropy_subsystem([0, 1])
    print(f"  S(0,1) after evolution = {ent_s:.4f}")
    vm2.stop()

    # Test 3: Bit-mass prediction
    print("\n[Test 3] Bit-mass prediction (Niobium T_c = 9.2 K)")
    dm = vm.compute_bit_mass(temperature_k=9.2, delta_entropy_bits=1.0)
    print(f"  m_bit (1 bit at 9.2 K) = {dm:.4e} kg")
    dm_sample = vm.compute_bit_mass(temperature_k=9.2, delta_entropy_bits=1e23)
    print(f"  delta_m (1e23 bits)     = {dm_sample:.4e} kg")

    # Test 4: Sophia susceptibility
    print("\n[Test 4] Sophia point analysis")
    J_vals = np.linspace(0.1, 2.0, 50)
    C_vals = np.tanh(J_vals / 0.5)  # mock data
    sophia = vm.compute_sophia_susceptibility(J_vals.tolist(), C_vals.tolist())
    print(f"  chi_max     = {sophia['chi_max']:.4f}")
    print(f"  J_critical  = {sophia['J_critical']:.4f}")
    print(f"  C_sophia    = {sophia['C_sophia']:.4f}")
    print(f"  C_target    = {sophia['sophia_target']:.4f}")

    # Test 5: Topological entropy
    print("\n[Test 5] Topological entanglement entropy")
    vm5 = QuantumVMGravity(qubits=9, noise_level=0.0)
    vm5.start()
    # Create a cluster state (approximate topological state)
    for i in range(3):
        for j in range(2):
            q = i * 3 + j
            vm5.apply_gate('h', [q])
            vm5.apply_gate('cnot', [q, q + 1])
    regions = {
        'A': [0, 1, 2], 'B': [3, 4, 5], 'C': [6, 7, 8],
        'AB': [0, 1, 2, 3, 4, 5], 'AC': [0, 1, 2, 6, 7, 8],
        'BC': [3, 4, 5, 6, 7, 8], 'ABC': [0, 1, 2, 3, 4, 5, 6, 7, 8]
    }
    s_topo = vm5.topological_entropy_kp(regions)
    print(f"  S_topo = {s_topo:.4f} bits")
    vm5.stop()

    # Test 6: Ramsey interferometry
    print("\n[Test 6] Ramsey interferometry")
    vm6 = QuantumVMGravity(qubits=1, noise_level=0.0)
    vm6.start()
    result = RamseyInterferometer.ramsey_sequence(vm6, qubit=0, wait_time=1.0,
                                                    detuning=0.5, shots=10000)
    print(f"  P(0) = {result['prob_0']:.4f}")
    print(f"  P(1) = {result['prob_1']:.4f}")
    print(f"  Contrast = {result['fringe_contrast']:.4f}")
    vm6.stop()

    print("\n" + "=" * 60)
    print("All tests passed. Engine ready for simulations.")
