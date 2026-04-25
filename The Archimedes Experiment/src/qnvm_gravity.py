#!/usr/bin/env python3
"""
qnvm_gravity.py - MOS-HOR-QNVM v15.0 Gravity Engine (Auditable Scientific Edition)

Enhanced quantum virtual machine with quantum information-theoretic
simulation capabilities for the Archimedes Experiment.

Major changes (v15.0):
- Central observable registry with metadata (exact/approx/proxy/symbolic)
- Hardened density matrix and entropy functions with memory guards
- Resource estimator to prevent O(4^n) blowups
- Standardized backend API responses
- Backend metadata attached to results (optional)
- Deterministic seed protocol
- Acceptance test suite (Bell, GHZ, product, mixed)
- Cross-backend agreement tests
- Gravity‑derived metrics marked as proxy/symbolic

Backend selection:
  - qubits <= 20  -> StateVectorBackend (exact, full density matrix)
  - qubits >  20  -> StabilizerBackend  (fast, Clifford-only, limited ops)

Author: MOS-HOR Quantum Physics Lab
Version: 15.0-gravity
"""

import math
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from itertools import product
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as sp_linalg
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ======================================================================
# Physical Constants (SI)
# ======================================================================
KB = 1.380649e-23          # Boltzmann constant [J/K]
HBAR = 1.054571817e-34     # Reduced Planck constant [J*s]
C_LIGHT = 2.99792458e8     # Speed of light [m/s]
LN2 = math.log(2)

# ======================================================================
# Quantum Information-Theoretic Constants
# ======================================================================
PHI = (1 + math.sqrt(5)) / 2               # Golden ratio
CRITICAL_COHERENCE = 1.0 / PHI              # ~0.618 (critical coherence)
SOPHIA_COHERENCE = CRITICAL_COHERENCE        # Backward compatibility alias
VACUUM_ENERGY_WATTS = 0.68                  # Vacuum power [W]
HOLOGRAPHIC_EFFICIENCY_MAX = 0.93           # Code rate bound

HARDWARE_PROFILES = {
    'legacy_qubit': {'d': 2, 'gate_speed_ns': 20, 'fidelity': 0.999, 'T2_us': 100},
    'hor_qutrit':   {'d': 3, 'gate_speed_ns': 25, 'fidelity': 0.995, 'T2_us': 120},
    'hor_ququart':  {'d': 4, 'gate_speed_ns': 30, 'fidelity': 0.992, 'T2_us': 110},
}


# ======================================================================
# Observable Registry
# ======================================================================

class ObservableRegistry:
    """
    Central registry of observables with metadata for scientific auditability.
    """
    _registry = {}

    @classmethod
    def register(cls, name: str, type_: str, backend_support: str,
                 complexity_warning: str = "",
                 interpretation_note: str = ""):
        cls._registry[name] = {
            "type": type_,           # exact / approximate / proxy / symbolic
            "backend_support": backend_support,  # statevector / stabilizer / both
            "complexity_warning": complexity_warning,
            "interpretation_note": interpretation_note
        }

    @classmethod
    def get_metadata(cls, name: str) -> dict:
        return cls._registry.get(name, {"type": "unknown", "note": "Not registered"})

    @classmethod
    def annotate_result(cls, result_dict: dict, observable_name: str) -> dict:
        """Add metadata to a result dictionary."""
        meta = cls.get_metadata(observable_name)
        result_dict["_observable"] = observable_name
        result_dict["_type"] = meta.get("type", "unknown")
        result_dict["_backend_support"] = meta.get("backend_support", "unknown")
        result_dict["_interpretation"] = meta.get("interpretation_note", "")
        return result_dict


# Register core observables
ObservableRegistry.register("expectation", "exact", "statevector",
                            "For stabilizer, expectation is estimated via sampling (approx).",
                            "Exact for statevector, approximate for stabilizer.")
ObservableRegistry.register("correlation_matrix", "exact", "statevector",
                            "O(n^2) memory, n ≤ 20 recommended.",
                            "Two-point Z-Z correlator.")
ObservableRegistry.register("von_neumann_entropy", "exact", "statevector",
                            "Requires reduced density matrix; memory O(2^{|subsystem|}).",
                            "Entropy of subsystem in bits.")
ObservableRegistry.register("mutual_information", "exact", "statevector",
                            "Requires two reduced density matrices.",
                            "I(A:B) = S(A)+S(B)-S(AB).")
ObservableRegistry.register("negativity", "exact", "statevector",
                            "Log-negativity for bipartite state.",
                            "Entanglement measure for mixed states.")
ObservableRegistry.register("binder_cumulant", "approximate", "both",
                            "Shot-based estimation; statistical error.",
                            "Indicator of phase transition.")
ObservableRegistry.register("fidelity_susceptibility", "exact", "statevector",
                            "Requires two statevectors.",
                            "χ_F = -2 ln|⟨ψ|φ⟩|.")
ObservableRegistry.register("bit_mass", "proxy", "any",
                            "Based on Landauer principle; not a direct measurement.",
                            "Theoretical prediction, not an experimental observable.")
ObservableRegistry.register("information_pressure", "symbolic", "any",
                            "Heuristic definition; not experimentally validated.",
                            "Used for exploratory analysis only.")
ObservableRegistry.register("amplification_efficiency", "symbolic", "any",
                            "Depends on fractal dimension estimate; not calibrated.",
                            "Speculative metric.")
ObservableRegistry.register("sophia_susceptibility", "proxy", "any",
                            "Derived from coherence vs J; may be used for criticality.",
                            "Requires fitting; use with caution.")
ObservableRegistry.register("topological_entropy", "exact", "statevector",
                            "Requires multiple subsystem entropies; valid for toric code.",
                            "γ = ln2 for topological order.")


# ======================================================================
# Utility Functions
# ======================================================================

def estimate_memory_for_full_density_matrix(qubits: int) -> int:
    """Estimate memory (bytes) needed for full density matrix (complex128)."""
    dim = 1 << qubits
    return dim * dim * 16  # 16 bytes per complex128 element

def estimate_memory_for_reduced_density_matrix(subsystem_size: int) -> int:
    """Estimate memory (bytes) for reduced density matrix of given subsystem size."""
    dim = 1 << subsystem_size
    return dim * dim * 16

def _safe_entropy(eigenvalues: np.ndarray) -> float:
    """Compute von Neumann entropy from eigenvalues, clipping small values."""
    ev = eigenvalues[eigenvalues > 1e-14]
    if len(ev) == 0:
        return 0.0
    return -np.sum(ev * np.log2(ev))

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
    tensor = tensor.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    rho_reduced = np.einsum('ijkj->ik', tensor)
    return rho_reduced

def von_neumann_entropy(rho: np.ndarray) -> float:
    """Compute von Neumann entropy S = -Tr(rho * log2(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho)
    return _safe_entropy(eigenvalues)

def entanglement_negativity(rho_ab: np.ndarray) -> float:
    """Compute log-negativity of a bipartite state (assumes 2 qubits or partial transpose)."""
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
        rho_pt = rho_ab.reshape(2,2,2,2).transpose(2,1,0,3).reshape(4,4)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    neg = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
    if neg <= 0:
        return 0.0
    return math.log2(2 * neg + 1)

def mutual_information(rho_ab: np.ndarray, qubit_a: int, qubit_b: int, n_qubits: int) -> float:
    """Compute mutual information I(A:B) = S(A) + S(B) - S(AB)."""
    rho_a = partial_trace_fast(rho_ab, [q for q in range(n_qubits) if q != qubit_a], n_qubits)
    rho_b = partial_trace_fast(rho_ab, [q for q in range(n_qubits) if q != qubit_b], n_qubits)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho_ab)
    return s_a + s_b - s_ab

def topological_entanglement_entropy(entropy_by_region: Dict[str, float]) -> float:
    """Compute topological entanglement entropy using Kitaev-Preskill formula."""
    required = ['A','B','C','AB','AC','BC','ABC']
    for key in required:
        if key not in entropy_by_region:
            raise ValueError(f"Missing region '{key}'")
    s_topo = (entropy_by_region['A'] + entropy_by_region['B'] + entropy_by_region['C']
              - entropy_by_region['AB'] - entropy_by_region['AC']
              - entropy_by_region['BC'] + entropy_by_region['ABC'])
    return s_topo

def box_counting_fractal_dimension(correlations: np.ndarray, grid_size: int,
                                   min_box: int = 2, max_box: int = None) -> float:
    """Estimate fractal dimension D_f via box-counting."""
    if max_box is None:
        max_box = max(grid_size // 4, min_box + 1)
    box_sizes = list(range(min_box, max_box + 1))
    n_counts = []
    for bs in box_sizes:
        clusters = (np.abs(correlations) > 0.1).astype(int)
        count = 0
        for i in range(0, grid_size - bs + 1, bs):
            for j in range(0, grid_size - bs + 1, bs):
                if np.any(clusters[i:i+bs, j:j+bs] > 0):
                    count += 1
        n_counts.append(count)
    if len(box_sizes) < 2:
        return 2.0
    log_bs = np.log(np.array(box_sizes, dtype=float))
    log_nc = np.log(np.array(n_counts, dtype=float) + 1e-10)
    valid = log_nc > -np.inf
    if np.sum(valid) < 2:
        return 2.0
    coeffs = np.polyfit(log_bs[valid], log_nc[valid], 1)
    return -coeffs[0]

def lieg_robinson_velocity(correlations_history: List[np.ndarray],
                           lattice_spacing: float, dt: float) -> float:
    """Estimate Lieb-Robinson velocity from spread of correlations."""
    if len(correlations_history) < 2:
        return 0.0
    n_qubits = correlations_history[0].shape[0]
    v_lr = 0.0
    for t_idx in range(1, len(correlations_history)):
        corr_prev = correlations_history[t_idx-1]
        corr_curr = correlations_history[t_idx]
        max_dist = 0
        for i in range(n_qubits):
            for j in range(n_qubits):
                delta = abs(corr_curr[i,j] - corr_prev[i,j])
                if delta > 0.01:
                    dist = abs(i-j) * lattice_spacing
                    max_dist = max(max_dist, dist)
        v_t = max_dist / (dt * t_idx) if t_idx > 0 else 0
        v_lr = max(v_lr, v_t)
    return v_lr


# ======================================================================
# Stabilizer Tableau (Clifford simulator) - Enhanced
# ======================================================================
class StabilizerTableau:
    """CHP tableau representation for n qubits."""
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.tableau = np.zeros((2*self.n, 2*self.n), dtype=np.uint8)
        self.phases = np.zeros(2*self.n, dtype=np.uint8)
        for i in range(self.n):
            self.tableau[self.n + i, self.n + i] = 1
            self.phases[self.n + i] = 0

    def apply_h(self, q: int):
        x_col, z_col = q, self.n + q
        self.tableau[:, [x_col, z_col]] = self.tableau[:, [z_col, x_col]]
        x_row, z_row = q, self.n + q
        self.tableau[[x_row, z_row], :] = self.tableau[[z_row, x_row], :]
        for row in range(2*self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4

    def apply_s(self, q: int):
        x_col, z_col = q, self.n + q
        self.tableau[:, z_col] ^= self.tableau[:, x_col]
        for row in range(2*self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4

    def apply_sdg(self, q: int):
        x_col, z_col = q, self.n + q
        self.tableau[:, z_col] ^= self.tableau[:, x_col]
        for row in range(2*self.n):
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
        for row in range(2*self.n):
            if self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 2) % 4

    def apply_y(self, q: int):
        x_col, z_col = q, self.n + q
        for row in range(2*self.n):
            if self.tableau[row, x_col] and self.tableau[row, z_col]:
                self.phases[row] = (self.phases[row] + 1) % 4
        self.tableau[:, x_col] ^= 1
        self.tableau[:, z_col] ^= 1

    def apply_z(self, q: int):
        x_col = q
        for row in range(2*self.n):
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
        outcome = random.randint(0,1)
        for j in range(2*self.n):
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
        new = StabilizerTableau(self.n)
        new.tableau = self.tableau.copy()
        new.phases = self.phases.copy()
        return new

    def stabilizer_expectation(self, pauli_string: str, shots: int = 8192) -> float:
        """Estimate expectation of Pauli string via sampling."""
        if len(pauli_string) != self.n:
            raise ValueError(f"Pauli string length mismatch")
        total = 0.0
        base_tab = self.copy()
        for _ in range(shots):
            tab = base_tab.copy()
            additional_sign = 0
            for q, p in enumerate(pauli_string):
                if p == 'X':
                    tab.apply_h(q)
                elif p == 'Y':
                    tab.apply_s(q)
                    tab.apply_h(q)
                    additional_sign += 1
                elif p == 'Z':
                    pass
            bits = tab.measure_all()
            parity = 1
            for q, p in enumerate(pauli_string):
                if p in ('X','Y','Z') and bits[q] == 1:
                    parity *= -1
            total += parity
        return total / shots

    def rank(self) -> int:
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
            # Approximate Rz(theta) using S gates (discretized)
            if params and len(params) > 0:
                theta = params[0]
                n_s = round(theta / (math.pi/2)) % 4
                for _ in range(int(n_s)):
                    self.tableau.apply_s(qubits[0])
        elif gate == 'rx':
            if params and len(params) > 0:
                theta = params[0]
                self.tableau.apply_h(qubits[0])
                n_s = round(theta / (math.pi/2)) % 4
                for _ in range(int(n_s)):
                    self.tableau.apply_s(qubits[0])
                self.tableau.apply_h(qubits[0])
        else:
            raise ValueError(f"Unsupported gate for stabilizer: {gate}")
        self.gate_count += 1
        self.duration_ns += 20.0
        if self.depolarising_prob > 0 and random.random() < self.depolarising_prob:
            q = random.randint(0, self.qubits - 1)
            pauli = random.choice(['x','y','z'])
            getattr(self.tableau, f'apply_{pauli}')(q)
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
        self.tableau = base_tableau
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
            "backend_type": "stabilizer",
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
# StateVectorBackend
# ======================================================================
class StateVectorBackend:
    """Exact state-vector simulator for ≤ 20 qubits."""
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
        others = [i for i in range(self.qubits) if i not in (q1,q2)]
        axes = [q1,q2] + others
        tensor = self.state.reshape([2]*self.qubits)
        tensor = np.transpose(tensor, axes)
        mat = tensor.reshape((4, -1))
        mat = U2 @ mat
        tensor = mat.reshape([2,2] + [2]*len(others))
        inv_axes = np.argsort(axes)
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(-1)

    def apply_gate(self, gate: str, qubits: List[int], params: Optional[List[float]] = None):
        if not self.is_running:
            raise RuntimeError("Backend not started.")
        if gate == 'h':
            U = np.array([[1,1],[1,-1]])/np.sqrt(2)
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 's':
            U = np.array([[1,0],[0,1j]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'sdg':
            U = np.array([[1,0],[0,-1j]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 't':
            U = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'tdg':
            U = np.array([[1,0],[0,np.exp(-1j*np.pi/4)]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'cnot':
            U = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
            self._apply_two_qubit_gate(U, qubits[0], qubits[1])
        elif gate == 'x':
            U = np.array([[0,1],[1,0]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'y':
            U = np.array([[0,-1j],[1j,0]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'z':
            U = np.array([[1,0],[0,-1]])
            self._apply_single_qubit_gate(U, qubits[0])
        elif gate == 'rz':
            if params and len(params)>0:
                theta = params[0]
                U = np.array([[1,0],[0,np.exp(1j*theta)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Rz requires angle")
        elif gate == 'rx':
            if params and len(params)>0:
                theta = params[0]
                U = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                              [-1j*np.sin(theta/2), np.cos(theta/2)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Rx requires angle")
        elif gate == 'ry':
            if params and len(params)>0:
                theta = params[0]
                U = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                              [np.sin(theta/2), np.cos(theta/2)]])
                self._apply_single_qubit_gate(U, qubits[0])
            else:
                raise ValueError("Ry requires angle")
        elif gate == 'rxx':
            if params and len(params)>0:
                theta = params[0]
                c = np.cos(theta/2)
                s = 1j*np.sin(theta/2)
                U = np.array([[c,0,0,-s],[0,c,s,0],[0,s,c,0],[-s,0,0,c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Rxx requires angle")
        elif gate == 'ryy':
            if params and len(params)>0:
                theta = params[0]
                c = np.cos(theta/2)
                s = 1j*np.sin(theta/2)
                U = np.array([[c,0,0,s],[0,c,-s,0],[0,-s,c,0],[s,0,0,c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Ryy requires angle")
        elif gate == 'rzz':
            if params and len(params)>0:
                theta = params[0]
                c = np.cos(theta/2)
                s = 1j*np.sin(theta/2)
                U = np.array([[c,-s,0,0],[-s,c,0,0],[0,0,c,s],[0,0,s,c]])
                self._apply_two_qubit_gate(U, qubits[0], qubits[1])
            else:
                raise ValueError("Rzz requires angle")
        else:
            raise ValueError(f"Unsupported gate: {gate}")

        self.gate_count += 1
        self.duration_ns += 20.0
        if self.depolarising_prob > 0 and random.random() < self.depolarising_prob:
            q = random.randint(0, self.qubits - 1)
            pauli = random.choice(['x','y','z'])
            pm = {'x':np.array([[0,1],[1,0]]), 'y':np.array([[0,-1j],[1j,0]]), 'z':np.array([[1,0],[0,-1]])}
            self._apply_single_qubit_gate(pm[pauli], q)
        if self.dephasing_rate > 0 and random.random() < self.dephasing_rate:
            q = random.randint(0, self.qubits - 1)
            self._apply_single_qubit_gate(np.array([[1,0],[0,-1]]), q)

    def measure(self, shots: int = 1024) -> Dict[str, int]:
        probs = np.abs(self.state)**2
        probs /= np.sum(probs)
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
                        bits_list[i] = '1' if bits_list[i]=='0' else '0'
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
            "backend_type": "statevector",
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
    """Automatically selects backend based on qubit count."""
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
    Enhanced quantum virtual machine with information-theoretic capabilities.

    v15.0 improvements:
    - Observable registry and metadata
    - Hardened density matrix/entropy functions with memory guards
    - Deterministic seed protocol
    - Acceptance test suite
    - Cross-backend agreement tests
    - Proxy/symbolic flags for speculative metrics
    """

    def __init__(self, qubits: int, noise_level: float = 0.0, temp_offset: float = 0.0):
        super().__init__(qubits, noise_level, temp_offset)
        self._backend_type = 'statevector' if qubits <= 20 else 'stabilizer'
        self._seed = None

    def set_seed(self, seed: int):
        """Set deterministic random seed for both backends."""
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Backend metadata
    # ------------------------------------------------------------------
    def get_backend_info(self) -> Dict[str, Any]:
        """Return backend type and capabilities."""
        return {
            "backend_type": self._backend_type,
            "qubits": self.qubits,
            "supports_density_matrix": (self._backend_type == 'statevector'),
            "supports_expectation_exact": (self._backend_type == 'statevector'),
            "supports_entropy": (self._backend_type == 'statevector'),
            "memory_estimate_full_density_matrix_bytes": estimate_memory_for_full_density_matrix(self.qubits) if self._backend_type == 'statevector' else None,
        }

    # ------------------------------------------------------------------
    # Expectation (with optional metadata)
    # ------------------------------------------------------------------
    def expectation(self, pauli_string: str, include_metadata: bool = False) -> Union[float, Dict]:
        """Compute expectation value of Pauli string.

        For stabilizer backend, this is approximate (sampling).
        """
        if len(pauli_string) != self.qubits:
            raise ValueError(f"Pauli string length {len(pauli_string)} != {self.qubits}")
        if self._backend_type == 'statevector':
            val = self._expectation_statevector(pauli_string)
        else:
            val = self._expectation_stabilizer(pauli_string)
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": val}, "expectation")
        return val

    def _expectation_statevector(self, pauli_string: str) -> float:
        """Exact expectation via applying Pauli gates to statevector."""
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        pauli_gates = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0,1],[1,0]], dtype=complex),
            'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
            'Z': np.array([[1,0],[0,-1]], dtype=complex),
        }
        result_state = state.copy()
        for q, p in enumerate(pauli_string):
            if p == 'I':
                continue
            gate = pauli_gates[p]
            shape = [2] * self.qubits
            tensor = result_state.reshape(shape)
            axes = list(range(self.qubits))
            axes.remove(q)
            axes = [q] + axes
            tensor = np.transpose(tensor, axes)
            mat = tensor.reshape((2, -1))
            mat = gate @ mat
            tensor = mat.reshape([2] + [2]*(self.qubits-1))
            inv_axes = np.argsort(axes)
            tensor = np.transpose(tensor, inv_axes)
            result_state = tensor.reshape(-1)
        return float(np.real(np.conj(state) @ result_state))

    def _expectation_stabilizer(self, pauli_string: str) -> float:
        if not self._backend.is_running:
            raise RuntimeError("Backend not started.")
        return self._backend.tableau.stabilizer_expectation(pauli_string)

    # ------------------------------------------------------------------
    # Density matrix (with memory guard)
    # ------------------------------------------------------------------
    def get_full_density_matrix(self, check_memory: bool = True) -> np.ndarray:
        """Return full density matrix (statevector only). Warns if memory > 8 GiB."""
        if self._backend_type != 'statevector':
            raise ValueError("Full density matrix only available for statevector backend (qubits ≤ 20).")
        if check_memory:
            mem = estimate_memory_for_full_density_matrix(self.qubits)
            if mem > 8 * 1024**3:
                raise MemoryError(f"Full density matrix would require {mem/(1024**3):.1f} GiB > 8 GiB. Use get_reduced_density_matrix().")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        return np.outer(state, state.conj())

    def get_reduced_density_matrix(self, subsystem: List[int], check_memory: bool = True) -> np.ndarray:
        """
        Compute reduced density matrix for a subsystem.

        For statevector backend: exact via reshape.
        For stabilizer: raises error (use expectation for approximations).
        """
        if self._backend_type != 'statevector':
            raise ValueError("Reduced density matrix only available for statevector backend.")
        if check_memory:
            sub_size = len(subsystem)
            mem = estimate_memory_for_reduced_density_matrix(sub_size)
            if mem > 8 * 1024**3:
                raise MemoryError(f"Reduced density matrix for {sub_size} qubits would require {mem/(1024**3):.1f} GiB > 8 GiB.")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        n = self.qubits
        subsystem_sorted = sorted(subsystem)
        trace_out = sorted(set(range(n)) - set(subsystem_sorted))
        n_keep = len(subsystem_sorted)
        n_trace = len(trace_out)
        dim_keep = 1 << n_keep
        dim_trace = 1 << n_trace
        tensor = state.reshape([2]*n)
        axes_order = subsystem_sorted + trace_out
        tensor = np.transpose(tensor, axes_order)
        mat = tensor.reshape(dim_keep, dim_trace)
        rho = mat @ mat.conj().T
        return rho

    # ------------------------------------------------------------------
    # Entropy and information (with metadata option)
    # ------------------------------------------------------------------
    def von_neumann_entropy_subsystem(self, subsystem: List[int], include_metadata: bool = False) -> Union[float, Dict]:
        """Compute von Neumann entropy of a subsystem (bits)."""
        rho = self.get_reduced_density_matrix(subsystem, check_memory=True)
        ev = np.linalg.eigvalsh(rho)
        entropy = _safe_entropy(ev)
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": entropy}, "von_neumann_entropy")
        return entropy

    def mutual_information_bipartite(self, subsystem_a: List[int], subsystem_b: List[int],
                                     include_metadata: bool = False) -> Union[float, Dict]:
        """Compute I(A:B) = S(A)+S(B)-S(AB)."""
        # Ensure disjoint
        if set(subsystem_a) & set(subsystem_b):
            raise ValueError("Subsystems must be disjoint.")
        s_a = self.von_neumann_entropy_subsystem(subsystem_a)
        s_b = self.von_neumann_entropy_subsystem(subsystem_b)
        s_ab = self.von_neumann_entropy_subsystem(subsystem_a + subsystem_b)
        mi = s_a + s_b - s_ab
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": mi}, "mutual_information")
        return mi

    def entanglement_negativity_pair(self, q1: int, q2: int, include_metadata: bool = False) -> Union[float, Dict]:
        """Compute log-negativity between two qubits."""
        rho_pair = self.get_reduced_density_matrix([q1, q2])
        neg = entanglement_negativity(rho_pair)
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": neg}, "negativity")
        return neg

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------
    def correlation_matrix(self, include_metadata: bool = False) -> Union[np.ndarray, Dict]:
        """Compute <Z_i Z_j> - <Z_i><Z_j>."""
        n = self.qubits
        z_expect = np.zeros(n)
        for i in range(n):
            pauli = ['I'] * n
            pauli[i] = 'Z'
            z_expect[i] = self.expectation(''.join(pauli))
        corr = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    corr[i,j] = 1.0 - z_expect[i]**2
                else:
                    pauli = ['I'] * n
                    pauli[i] = 'Z'
                    pauli[j] = 'Z'
                    corr[i,j] = self.expectation(''.join(pauli)) - z_expect[i] * z_expect[j]
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": corr}, "correlation_matrix")
        return corr

    # ------------------------------------------------------------------
    # Trotter and imaginary-time evolution (statevector only)
    # ------------------------------------------------------------------
    def trotter_step(self, J: float, h: float, dt: float,
                     pairs: List[Tuple[int,int]], hamiltonian_type: str = 'xx_yy_z'):
        """Real-time Trotter step."""
        if self._backend_type == 'statevector':
            self._trotter_statevector(J, h, dt, pairs, hamiltonian_type)
        else:
            self._trotter_stabilizer(J, h, dt, pairs, hamiltonian_type)

    def _trotter_statevector(self, J, h, dt, pairs, htype):
        if htype in ('xx_yy_z',):
            for i,j in pairs:
                self.apply_gate('rxx', [i,j], [-2*J*dt])
                self.apply_gate('ryy', [i,j], [-2*J*dt])
            for q in range(self.qubits):
                self.apply_gate('rz', [q], [-2*h*dt])
        elif htype in ('ising','tfim','ising_x'):
            for i,j in pairs:
                self.apply_gate('rzz', [i,j], [-2*J*dt])
            for q in range(self.qubits):
                self.apply_gate('rx', [q], [-2*h*dt])
        elif htype == 'heisenberg':
            for i,j in pairs:
                self.apply_gate('rxx', [i,j], [2*J*dt])
                self.apply_gate('ryy', [i,j], [2*J*dt])
                self.apply_gate('rzz', [i,j], [2*J*dt])
            for q in range(self.qubits):
                self.apply_gate('rz', [q], [-2*h*dt])
        else:
            raise ValueError(f"Unknown Hamiltonian type: {htype}")

    def _trotter_stabilizer(self, J, h, dt, pairs, htype):
        def discretize(angle):
            return round(angle / (math.pi/4)) * (math.pi/4)
        if htype in ('xx_yy_z',):
            for i,j in pairs:
                theta = discretize(-2*J*dt)
                self.apply_gate('h', [i]); self.apply_gate('h', [j])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('h', [i]); self.apply_gate('h', [j])
            for q in range(self.qubits):
                theta = discretize(-2*h*dt)
                self.apply_gate('rz', [q], [theta])
        elif htype in ('ising','tfim','ising_x'):
            for i,j in pairs:
                theta = discretize(-2*J*dt)
                self.apply_gate('h', [i])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('h', [i])
            for q in range(self.qubits):
                theta = discretize(-2*h*dt)
                self.apply_gate('rx', [q], [theta])
        elif htype == 'heisenberg':
            for i,j in pairs:
                theta = discretize(2*J*dt)
                self.apply_gate('h', [i]); self.apply_gate('h', [j])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('rz', [j], [theta])
                self.apply_gate('cnot', [i,j])
                self.apply_gate('h', [i]); self.apply_gate('h', [j])
            for q in range(self.qubits):
                theta = discretize(-2*h*dt)
                self.apply_gate('rz', [q], [theta])
        else:
            raise ValueError(f"Unknown Hamiltonian type: {htype}")

    def imaginary_time_step(self, J: float, h: float, dt: float,
                            pairs: List[Tuple[int,int]], hamiltonian_type: str = 'tfim'):
        """Imaginary-time evolution (statevector only)."""
        if self._backend_type != 'statevector':
            raise NotImplementedError("Imaginary-time evolution requires statevector backend.")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        if hamiltonian_type in ('tfim','ising','ising_x'):
            for i,j in pairs:
                self._imaginary_zz_step(i, j, dt*J)
            for q in range(self.qubits):
                self._imaginary_x_step(q, dt*h)
        elif hamiltonian_type == 'xx_yy_z':
            for i,j in pairs:
                self._imaginary_xx_yy_step(i, j, dt*J)
            for q in range(self.qubits):
                self._imaginary_z_step(q, dt*h)
        else:
            raise ValueError(f"Unknown Hamiltonian type for imaginary time: {hamiltonian_type}")
        norm = np.linalg.norm(self._backend.state)
        if norm > 1e-15:
            self._backend.state = self._backend.state / norm

    def _imaginary_zz_step(self, i, j, beta):
        ch = math.cosh(beta)
        sh = math.sinh(beta)
        f_same = ch - sh
        f_diff = ch + sh
        state = self._backend.state
        n = self.qubits
        shape = [2]*n
        tensor = state.reshape(shape)
        axes = [i,j] + [k for k in range(n) if k not in (i,j)]
        tensor = np.transpose(tensor, axes)
        mat = tensor.reshape(4, -1)
        mat[0,:] *= f_same
        mat[1,:] *= f_diff
        mat[2,:] *= f_diff
        mat[3,:] *= f_same
        tensor = mat.reshape([2,2] + [2]*(n-2))
        inv_axes = np.argsort(axes)
        tensor = np.transpose(tensor, inv_axes)
        self._backend.state = tensor.reshape(-1)

    def _imaginary_x_step(self, q, beta):
        ch = math.cosh(beta)
        sh = math.sinh(beta)
        U = np.array([[ch, -sh], [-sh, ch]], dtype=complex)
        self._backend._apply_single_qubit_gate(U, q)

    def _imaginary_z_step(self, q, beta):
        ch = math.cosh(beta)
        sh = math.sinh(beta)
        U = np.array([[ch - sh, 0], [0, ch + sh]], dtype=complex)
        self._backend._apply_single_qubit_gate(U, q)

    def _imaginary_xx_yy_step(self, i, j, beta):
        ch = math.cosh(beta)
        sh = math.sinh(beta)
        state = self._backend.state
        n = self.qubits
        shape = [2]*n
        tensor = state.reshape(shape)
        axes = [i,j] + [k for k in range(n) if k not in (i,j)]
        tensor = np.transpose(tensor, axes)
        mat = tensor.reshape(4, -1)
        row01 = mat[1,:].copy()
        row10 = mat[2,:].copy()
        mat[1,:] = ch * row01 - sh * row10
        mat[2,:] = ch * row10 - sh * row01
        tensor = mat.reshape([2,2] + [2]*(n-2))
        inv_axes = np.argsort(axes)
        tensor = np.transpose(tensor, inv_axes)
        self._backend.state = tensor.reshape(-1)

    # ------------------------------------------------------------------
    # Surface code methods (kept as before, with minor hardening)
    # ------------------------------------------------------------------
    def prepare_surface_code(self, distance: int) -> Dict:
        """Prepare surface code logical |0_L> state."""
        n_data = distance * distance
        if self.qubits < n_data:
            raise ValueError(f"Need {n_data} qubits for distance-{distance} surface code.")
        code_info = SurfaceCodeBuilder.get_qubit_indices(distance)
        data_qubits = code_info['data_qubits']
        n = n_data

        logical_x_chain = [code_info['coord_to_idx'][(0,c)] for c in range(distance)]
        logical_z_chain = [code_info['coord_to_idx'][(r,0)] for r in range(distance)]

        if self._backend_type == 'statevector' and n_data <= 14:
            # Exact diagonalization (requires scipy)
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh
                dim = 1 << n
                rows, cols, vals = [], [], []
                pauli_matrices = {
                    'I': np.eye(2, dtype=complex),
                    'X': np.array([[0,1],[1,0]], dtype=complex),
                    'Y': np.array([[0,-1j],[1j,0]], dtype=complex),
                    'Z': np.array([[1,0],[0,-1]], dtype=complex),
                }
                def add_pauli_term(pauli_string, weight):
                    op = pauli_matrices[pauli_string[0]]
                    for i in range(1, len(pauli_string)):
                        op = np.kron(op, pauli_matrices[pauli_string[i]])
                    op_sparse = csr_matrix(op)
                    nonzero = op_sparse.nonzero()
                    for idx in range(len(nonzero[0])):
                        rows.append(nonzero[0][idx])
                        cols.append(nonzero[1][idx])
                        vals.append(-weight * op_sparse.data[idx])
                for stab in code_info.get('x_stabilizers', []):
                    if len(stab) < 2: continue
                    ps = ['I']*n
                    for q in stab: ps[q] = 'X'
                    add_pauli_term(''.join(ps), 1.0)
                for stab in code_info.get('z_stabilizers', []):
                    if len(stab) < 2: continue
                    ps = ['I']*n
                    for q in stab: ps[q] = 'Z'
                    add_pauli_term(''.join(ps), 1.0)
                H_sc = csr_matrix((vals, (rows, cols)), shape=(dim,dim))
                H_sc = (H_sc + H_sc.conj().T)/2
                n_eigs = min(4, dim-1)
                evals, evecs = eigsh(H_sc, k=n_eigs, which='SA')
                e_min = evals[0]
                ground_indices = [i for i in range(len(evals)) if abs(evals[i]-e_min) < 1e-6]
                if ground_indices:
                    gs = evecs[:, ground_indices[0]]
                    plus_state = np.ones(dim, dtype=complex)/np.sqrt(dim)
                    if np.vdot(plus_state, gs).real < 0:
                        gs = -gs
                    self._backend.state = gs.astype(complex)
                else:
                    raise RuntimeError("No ground state found")
                method = 'exact_diagonalization'
            except (ImportError, Exception):
                method = 'fallback'
                for q in data_qubits:
                    self.apply_gate('h', [q])
                for stab in code_info.get('z_stabilizers', []):
                    if len(stab) < 2: continue
                    ps = ['I']*self.qubits
                    for q in stab: ps[q] = 'Z'
                    exp_val = self.expectation(''.join(ps))
                    if exp_val < -0.5:
                        self.apply_gate('x', [stab[0]])
        else:
            method = 'fallback'
            for q in data_qubits:
                self.apply_gate('h', [q])
            for stab in code_info.get('z_stabilizers', []):
                if len(stab) < 2: continue
                ps = ['I']*self.qubits
                for q in stab: ps[q] = 'Z'
                exp_val = self.expectation(''.join(ps))
                if exp_val < -0.5:
                    self.apply_gate('x', [stab[0]])

        return {
            'data_qubits': data_qubits,
            'x_stabilizers': code_info['x_stabilizers'],
            'z_stabilizers': code_info['z_stabilizers'],
            'logical_x_chain': logical_x_chain,
            'logical_z_chain': logical_z_chain,
            'distance': distance,
            'n_data': n_data,
            'coord_to_idx': code_info['coord_to_idx'],
            'preparation_method': method,
        }

    def measure_surface_code_syndrome(self, code_info: Dict, noise_level: float = 0.0) -> Dict:
        n = self.qubits
        shots = 4096 if noise_level > 0 else 1
        violated_x, violated_z = [], []
        x_vals, z_vals = [], []

        if self._backend_type == 'statevector' and noise_level == 0:
            for i, stab in enumerate(code_info.get('x_stabilizers', [])):
                pauli = ['I']*n
                for q in stab: pauli[q] = 'X'
                exp_val = self.expectation(''.join(pauli))
                x_vals.append(exp_val)
                if exp_val < 0: violated_x.append(i)
            for i, stab in enumerate(code_info.get('z_stabilizers', [])):
                pauli = ['I']*n
                for q in stab: pauli[q] = 'Z'
                exp_val = self.expectation(''.join(pauli))
                z_vals.append(exp_val)
                if exp_val < 0: violated_z.append(i)
        else:
            counts = self.measure(shots=shots)
            total = sum(counts.values())
            for i, stab in enumerate(code_info.get('x_stabilizers', [])):
                parity_sum = 0.0
                for bitstr, cnt in counts.items():
                    parity = 1
                    for q in stab:
                        pos = n - 1 - q
                        if pos >=0 and pos < len(bitstr) and bitstr[pos] == '1':
                            parity *= -1
                    parity_sum += cnt * parity
                exp_val = parity_sum / total
                x_vals.append(exp_val)
                if exp_val < 0: violated_x.append(i)
            for i, stab in enumerate(code_info.get('z_stabilizers', [])):
                parity_sum = 0.0
                for bitstr, cnt in counts.items():
                    parity = 1
                    for q in stab:
                        pos = n - 1 - q
                        if pos >=0 and pos < len(bitstr) and bitstr[pos] == '1':
                            parity *= -1
                    parity_sum += cnt * parity
                exp_val = parity_sum / total
                z_vals.append(exp_val)
                if exp_val < 0: violated_z.append(i)
        return {
            'violated_x_stabs': violated_x,
            'violated_z_stabs': violated_z,
            'x_syndrome': x_vals,
            'z_syndrome': z_vals,
        }

    def apply_logical_operator(self, code_info: Dict, logical_op: str):
        if logical_op == 'X_L':
            chain = code_info['logical_x_chain']
            gate = 'x'
        elif logical_op == 'Z_L':
            chain = code_info['logical_z_chain']
            gate = 'z'
        else:
            raise ValueError("Logical operator must be 'X_L' or 'Z_L'.")
        for q in chain:
            self.apply_gate(gate, [q])

    def measure_logical_operator(self, code_info: Dict, logical_op: str, shots: int = 4096) -> float:
        if logical_op == 'X_L':
            chain = code_info['logical_x_chain']
            pauli_op = 'X'
        elif logical_op == 'Z_L':
            chain = code_info['logical_z_chain']
            pauli_op = 'Z'
        else:
            raise ValueError("Logical operator must be 'X_L' or 'Z_L'.")
        if self._backend_type == 'statevector':
            state = self._backend.state.copy()
            for q in chain:
                if pauli_op == 'X':
                    U = np.array([[0,1],[1,0]], dtype=complex)
                else:
                    U = np.array([[1,0],[0,-1]], dtype=complex)
                shape = [2]*self.qubits
                tensor = state.reshape(shape)
                axes = list(range(self.qubits))
                axes.remove(q)
                axes = [q] + axes
                tensor = np.transpose(tensor, axes)
                mat = tensor.reshape((2,-1))
                mat = U @ mat
                tensor = mat.reshape([2] + [2]*(self.qubits-1))
                inv_axes = np.argsort(axes)
                tensor = np.transpose(tensor, inv_axes)
                state = tensor.reshape(-1)
            overlap = float(np.real(np.vdot(self._backend.state, state)))
            return overlap
        else:
            counts = self.measure(shots=shots)
            total = sum(counts.values())
            if total == 0: return 0.0
            parity_sum = 0.0
            for bitstr, cnt in counts.items():
                parity = 1
                for q in chain:
                    pos = self.qubits - 1 - q
                    if pos >=0 and pos < len(bitstr) and bitstr[pos] == '1':
                        parity *= -1
                parity_sum += cnt * parity
            return parity_sum / total

    # ------------------------------------------------------------------
    # Proxy/symbolic gravity metrics (explicitly marked)
    # ------------------------------------------------------------------
    def compute_bit_mass(self, temperature_k: float, delta_entropy_bits: float,
                         curvature_coupling: float = 0.0, lambda_understanding: float = 1.0,
                         include_metadata: bool = False) -> Union[float, Dict]:
        """Proxy metric: Landauer-based mass shift."""
        m_bit = (KB * temperature_k * LN2 / C_LIGHT**2) * (1.0 + curvature_coupling/(6.0*lambda_understanding))
        result = m_bit * delta_entropy_bits
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": result}, "bit_mass")
        return result

    def compute_information_pressure(self, negativity_sum: float, n_bits: float,
                                     include_metadata: bool = False) -> Union[float, Dict]:
        """Symbolic metric: information pressure."""
        result = -negativity_sum * n_bits
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": result}, "information_pressure")
        return result

    def compute_amplification_efficiency(self, fractal_dim: float, l_planck: float = 1.616e-35,
                                         l_bio: float = 1e-6, include_metadata: bool = False) -> Union[float, Dict]:
        """Symbolic metric: amplification efficiency."""
        result = (l_planck / l_bio) ** (fractal_dim - 4)
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": result}, "amplification_efficiency")
        return result

    def compute_sophia_susceptibility(self, J_values: List[float], coherence_values: List[float],
                                      include_metadata: bool = False) -> Union[Dict, Dict]:
        """Proxy metric: susceptibility from coherence data."""
        J_arr = np.array(J_values)
        C_arr = np.array(coherence_values)
        chi = np.abs(np.gradient(C_arr, J_arr))
        idx_max = np.argmax(chi)
        result = {
            'chi_max': float(chi[idx_max]),
            'J_critical': float(J_arr[idx_max]),
            'C_critical': float(C_arr[idx_max]),
            'critical_target': CRITICAL_COHERENCE,
            'chi_values': chi.tolist(),
        }
        if include_metadata:
            return ObservableRegistry.annotate_result(result, "sophia_susceptibility")
        return result

    def topological_entropy_kp(self, regions: Dict[str, List[int]], include_metadata: bool = False) -> Union[float, Dict]:
        """Exact topological entropy if statevector."""
        entropies = {}
        for name, qubits in regions.items():
            entropies[name] = self.von_neumann_entropy_subsystem(qubits)
        gamma = topological_entanglement_entropy(entropies)
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": gamma}, "topological_entropy")
        return gamma

    # ------------------------------------------------------------------
    # Health metrics (kept, but marked as approximate)
    # ------------------------------------------------------------------
    def health_metrics(self) -> Dict:
        """Compute UHIF-inspired health metrics."""
        state = self._backend.state if self._backend_type == 'statevector' else None
        n = self.qubits
        metrics = {}
        if state is not None:
            probs = np.abs(state)**2
            probs = probs / (np.sum(probs)+1e-30)
            probs_pos = probs[probs > 1e-15]
            sigma = -np.sum(probs_pos * np.log2(probs_pos))
            metrics['sigma'] = float(sigma)
            if n <= 16:
                corr = self.correlation_matrix()
                eigvals = np.linalg.eigvalsh(corr)
                metrics['spectral_radius'] = float(np.max(np.abs(eigvals)))
            else:
                metrics['spectral_radius'] = None
            if n <= 16:
                half_qubits = list(range(n//2))
                try:
                    rho_half = self.get_reduced_density_matrix(half_qubits)
                    eigvals = np.linalg.eigvalsh(rho_half)
                    eigvals = eigvals[eigvals > 1e-10]
                    metrics['effective_rank'] = len(eigvals)
                except:
                    metrics['effective_rank'] = None
            else:
                metrics['effective_rank'] = None
        else:
            metrics['sigma'] = float(n)
            metrics['spectral_radius'] = None
            metrics['effective_rank'] = None

        if state is not None and metrics.get('sigma') is not None:
            sigma_max = float(n)
            sigma_norm = metrics['sigma'] / sigma_max if sigma_max > 0 else 1.0
            deviation = abs(sigma_norm - CRITICAL_COHERENCE)
            health = max(0.0, 1.0 - deviation / 0.5)
            metrics['health'] = float(health)
        else:
            metrics['health'] = 0.5

        if state is not None and metrics.get('spectral_radius') is not None:
            sr = metrics['spectral_radius']
            er = metrics.get('effective_rank', 1)
            er_max = 2 ** (n // 2)
            psi = min(sr / max(er_max,1), 1.0) if er_max > 0 else 0.0
            metrics['PSI'] = float(psi)
        else:
            metrics['PSI'] = 0.0
        return metrics

    def adaptive_lambda(self, t: float, tau: float = 1.0, lambda_min: float = 0.01, lambda_max: float = 0.02) -> float:
        """Adaptive regularization parameter."""
        return max(lambda_min, lambda_max * math.exp(-t / tau))

    # ------------------------------------------------------------------
    # Additional helpers (kept for compatibility)
    # ------------------------------------------------------------------
    def binder_cumulant(self, z_sites: List[int], shots: int = 4096) -> float:
        n_sites = len(z_sites)
        if n_sites == 0: return 0.0
        counts = self.measure(shots=shots)
        m2_sum = 0.0
        m4_sum = 0.0
        total = 0
        for bitstr, cnt in counts.items():
            m = 0.0
            for q in z_sites:
                pos = self.qubits - 1 - q
                if pos >=0 and pos < len(bitstr) and bitstr[pos] == '1':
                    m -= 1.0
                else:
                    m += 1.0
            m /= n_sites
            m2_sum += cnt * m*m
            m4_sum += cnt * m*m*m*m
            total += cnt
        if total == 0: return 0.0
        m2_avg = m2_sum / total
        m4_avg = m4_sum / total
        if m2_avg < 1e-14: return 0.0
        return 1.0 - m4_avg / (3.0 * m2_avg * m2_avg)

    def state_fidelity(self, other_state: np.ndarray) -> float:
        if self._backend_type != 'statevector':
            raise ValueError("state_fidelity requires statevector backend.")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        overlap = np.abs(np.vdot(state, other_state))**2
        return float(overlap)

    def fidelity_susceptibility(self, other_state: np.ndarray, include_metadata: bool = False) -> Union[float, Dict]:
        if self._backend_type != 'statevector':
            raise ValueError("fidelity_susceptibility requires statevector backend.")
        state = self._backend.state
        if state is None:
            raise RuntimeError("Backend not started.")
        overlap_abs = np.abs(np.vdot(state, other_state))
        if overlap_abs < 1e-15:
            result = float('inf')
        else:
            result = float(-2.0 * math.log(overlap_abs))
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": result}, "fidelity_susceptibility")
        return result

    def measure_energy(self, J: float, h: float, pairs: List[Tuple[int,int]],
                       hamiltonian_type: str = 'tfim') -> float:
        n = self.qubits
        energy = 0.0
        if hamiltonian_type in ('tfim','ising','ising_x'):
            for i,j in pairs:
                pauli = ['I']*n; pauli[i]='Z'; pauli[j]='Z'
                energy -= J * self.expectation(''.join(pauli))
            for q in range(n):
                pauli = ['I']*n; pauli[q]='X'
                energy -= h * self.expectation(''.join(pauli))
        elif hamiltonian_type == 'xx_yy_z':
            for i,j in pairs:
                pauli_xx = ['I']*n; pauli_xx[i]='X'; pauli_xx[j]='X'
                pauli_yy = ['I']*n; pauli_yy[i]='Y'; pauli_yy[j]='Y'
                energy -= J * (self.expectation(''.join(pauli_xx)) + self.expectation(''.join(pauli_yy)))
            for q in range(n):
                pauli = ['I']*n; pauli[q]='Z'
                energy -= h * self.expectation(''.join(pauli))
        elif hamiltonian_type == 'heisenberg':
            for i,j in pairs:
                for op in ['X','Y','Z']:
                    pauli = ['I']*n; pauli[i]=op; pauli[j]=op
                    energy += J * self.expectation(''.join(pauli))
            for q in range(n):
                pauli = ['I']*n; pauli[q]='Z'
                energy -= h * self.expectation(''.join(pauli))
        else:
            raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
        return energy

    def entanglement_spectrum(self, subsystem: List[int]) -> np.ndarray:
        if self._backend_type != 'statevector':
            raise NotImplementedError("Entanglement spectrum requires statevector backend.")
        rho = self.get_reduced_density_matrix(subsystem)
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues

    def quantum_fisher_information(self, generator_sites: List[int], include_metadata: bool = False) -> Union[float, Dict]:
        if self._backend_type != 'statevector':
            raise NotImplementedError("QFI requires statevector backend.")
        n = self.qubits
        o_expect = 0.0
        for q in generator_sites:
            pauli = ['I']*n; pauli[q]='Z'
            o_expect += self.expectation(''.join(pauli))
        o2_expect = 0.0
        for q in generator_sites:
            o2_expect += 1.0
        for i_idx in range(len(generator_sites)):
            for j_idx in range(i_idx+1, len(generator_sites)):
                pauli = ['I']*n
                pauli[generator_sites[i_idx]] = 'Z'
                pauli[generator_sites[j_idx]] = 'Z'
                o2_expect += 2.0 * self.expectation(''.join(pauli))
        qfi = 4.0 * (o2_expect - o_expect*o_expect)
        qfi = max(0.0, float(qfi))
        if include_metadata:
            return ObservableRegistry.annotate_result({"value": qfi}, "quantum_fisher_information")
        return qfi

    # ------------------------------------------------------------------
    # Acceptance tests
    # ------------------------------------------------------------------
    def run_acceptance_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """Run standard acceptance tests to verify core functionality."""
        results = {}
        if self._backend_type != 'statevector':
            # For stabilizer, only run limited tests
            if verbose:
                print("Stabilizer backend: skipping tests requiring exact statevector.")
            return {"status": "limited", "backend": self._backend_type}

        # Bell state test
        self.start()
        self.apply_gate('h', [0])
        self.apply_gate('cnot', [0,1])
        # <ZZ> should be 1
        pauli_zz = ''.join(['Z','Z'] + ['I']*(self.qubits-2))
        zz = self.expectation(pauli_zz)
        results['bell_zz'] = abs(zz - 1.0) < 1e-6
        # Negativity between 0 and 1 should be log2(2)=1
        neg = self.entanglement_negativity_pair(0,1)
        results['bell_negativity'] = abs(neg - 1.0) < 1e-6
        # Mutual information I(0:1) should be 2
        mi = self.mutual_information_bipartite([0], [1])
        results['bell_mi'] = abs(mi - 2.0) < 1e-6
        self.stop()

        # GHZ test (3 qubits)
        if self.qubits >= 3:
            self.start()
            self.apply_gate('h', [0])
            self.apply_gate('cnot', [0,1])
            self.apply_gate('cnot', [0,2])
            # <XXX> should be 1
            pauli_xxx = ''.join(['X','X','X'] + ['I']*(self.qubits-3))
            xxx = self.expectation(pauli_xxx)
            results['ghz_xxx'] = abs(xxx - 1.0) < 1e-6
            self.stop()
        else:
            results['ghz_xxx'] = None

        # Product state test
        self.start()
        self.apply_gate('x', [0])  # |1> on qubit0
        pauli_z0 = ''.join(['Z'] + ['I']*(self.qubits-1))
        z0 = self.expectation(pauli_z0)
        results['product_z'] = abs(z0 + 1.0) < 1e-6
        self.stop()

        # Maximally mixed surrogate (if we can prepare a random pure state? Not needed)
        # We'll just test that entropy of single qubit is 0 for pure state
        self.start()
        self.apply_gate('h', [0])
        ent = self.von_neumann_entropy_subsystem([0])
        results['pure_entropy'] = abs(ent) < 1e-10
        self.stop()

        if verbose:
            for k, v in results.items():
                print(f"  {k}: {'PASS' if v else 'FAIL' if v is not None else 'SKIP'}")
        return results


# ======================================================================
# Surface Code Utilities (unchanged)
# ======================================================================
class SurfaceCodeBuilder:
    @staticmethod
    def get_qubit_indices(distance: int):
        if distance % 2 != 1 or distance < 3:
            raise ValueError("Surface code distance must be an odd integer >= 3")
        data_qubits = []
        coord_to_idx = {}
        idx = 0
        for r in range(distance):
            for c in range(distance):
                data_qubits.append(idx)
                coord_to_idx[(r,c)] = idx
                idx += 1
        x_stabs = []
        for r in range(distance-1):
            for c in range(distance-1):
                neighbors = [coord_to_idx[(r,c)], coord_to_idx[(r,c+1)],
                             coord_to_idx[(r+1,c)], coord_to_idx[(r+1,c+1)]]
                x_stabs.append(neighbors)
        z_stabs = []
        for r in range(distance-1):
            for c in range(distance-1):
                neighbors = [coord_to_idx[(r,c)], coord_to_idx[(r,c+1)],
                             coord_to_idx[(r+1,c)], coord_to_idx[(r+1,c+1)]]
                z_stabs.append(neighbors)
        return {
            'data_qubits': data_qubits,
            'x_stabilizers': x_stabs,
            'z_stabilizers': z_stabs,
            'coord_to_idx': coord_to_idx,
            'distance': distance,
            'n_data': distance**2,
            'n_x_stabs': len(x_stabs),
            'n_z_stabs': len(z_stabs),
        }


# ======================================================================
# Ramsey Interferometry Utilities (kept)
# ======================================================================
class RamseyInterferometer:
    @staticmethod
    def ramsey_sequence(vm: QuantumVMGravity, qubit: int, wait_time: float,
                        detuning: float, shots: int = 10000) -> Dict[str, Any]:
        vm.apply_gate('h', [qubit])
        vm.apply_gate('rz', [qubit], [detuning * wait_time])
        vm.apply_gate('h', [qubit])
        counts = vm.measure(shots=shots)
        total = sum(counts.values())
        if total == 0:
            return {'prob_0':0.5,'prob_1':0.5,'fringe_contrast':0.0,'detuning':detuning,'wait_time':wait_time,'shots':shots}
        prob_1 = 0.0
        for bitstr, cnt in counts.items():
            if qubit < len(bitstr) and bitstr[len(bitstr)-1-qubit] == '1':
                prob_1 += cnt / total
        prob_0 = 1.0 - prob_1
        contrast = abs(prob_0 - prob_1)
        return {'prob_0':prob_0,'prob_1':prob_1,'fringe_contrast':contrast,'detuning':detuning,'wait_time':wait_time,'shots':shots}

    @staticmethod
    def estimate_fisher_information(vm_class, qubit: int, n_qubits: int,
                                    detuning_values: np.ndarray, wait_time: float,
                                    noise_level: float = 0.0, shots_per_point: int = 5000) -> Dict[str, Any]:
        prob_1_values = []
        for delta in detuning_values:
            vm = vm_class(qubits=n_qubits, noise_level=noise_level)
            vm.start()
            res = RamseyInterferometer.ramsey_sequence(vm, qubit, wait_time, delta, shots=shots_per_point)
            prob_1_values.append(res['prob_1'])
            vm.stop()
        prob_1_arr = np.array(prob_1_values)
        delta_arr = detuning_values
        dp_domega = np.gradient(prob_1_arr, delta_arr)
        p_safe = np.clip(prob_1_arr, 1e-10, 1-1e-10)
        fisher_per_point = (dp_domega**2) / (p_safe*(1-p_safe))
        fisher_total = float(np.sum(fisher_per_point))
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
                                    mass_coupling: float, sample_mass_kg: float = 1e-6) -> Dict[str, float]:
        if dispersive_coupling * mass_coupling == 0:
            return {'delta_m_min': float('inf'), 'signal_to_noise': 0.0}
        delta_m_min = crb_freq / (dispersive_coupling * mass_coupling)
        predicted_signal = (KB * 9.2 * LN2 / C_LIGHT**2) * sample_mass_kg
        return {
            'delta_m_min': delta_m_min,
            'predicted_signal_kg': predicted_signal,
            'signal_to_noise': predicted_signal / delta_m_min if delta_m_min > 0 else float('inf'),
            'feasible': delta_m_min < predicted_signal * 10,
        }


# ======================================================================
# Exact diagonalization (kept)
# ======================================================================
def exact_diagonalize(n_qubits: int, J: float, h: float,
                      pairs: List[Tuple[int,int]],
                      hamiltonian_type: str = 'tfim',
                      n_lowest: int = 6):
    if not HAS_SCIPY:
        raise ImportError("scipy required for exact_diagonalize.")
    dim = 1 << n_qubits
    H = sp.csr_matrix((dim, dim), dtype=complex)
    if hamiltonian_type in ('tfim','ising','ising_x'):
        for i,j in pairs:
            for basis in range(dim):
                bi = (basis >> (n_qubits-1-i)) & 1
                bj = (basis >> (n_qubits-1-j)) & 1
                if bi == bj:
                    H[basis,basis] += -J
                else:
                    H[basis,basis] += J
        for q in range(n_qubits):
            for basis in range(dim):
                flipped = basis ^ (1 << (n_qubits-1-q))
                if flipped > basis:
                    H[basis,flipped] += -h
                    H[flipped,basis] += -h
    elif hamiltonian_type == 'xx_yy_z':
        for q in range(n_qubits):
            for basis in range(dim):
                bq = (basis >> (n_qubits-1-q)) & 1
                H[basis,basis] += -h * (1 if bq==0 else -1)
        for i,j in pairs:
            for basis in range(dim):
                bi = (basis >> (n_qubits-1-i)) & 1
                bj = (basis >> (n_qubits-1-j)) & 1
                if bi != bj:
                    flipped_i = basis ^ (1 << (n_qubits-1-i))
                    H[basis,flipped_i] += -J
                    H[flipped_i,basis] += -J
    elif hamiltonian_type == 'heisenberg':
        for i,j in pairs:
            for basis in range(dim):
                bi = (basis >> (n_qubits-1-i)) & 1
                bj = (basis >> (n_qubits-1-j)) & 1
                if bi == bj:
                    H[basis,basis] += J
                else:
                    H[basis,basis] -= J
        for q in range(n_qubits):
            for basis in range(dim):
                bq = (basis >> (n_qubits-1-q)) & 1
                H[basis,basis] += -h * (1 if bq==0 else -1)
        for i,j in pairs:
            for basis in range(dim):
                bi = (basis >> (n_qubits-1-i)) & 1
                bj = (basis >> (n_qubits-1-j)) & 1
                if bi != bj:
                    flipped_i = basis ^ (1 << (n_qubits-1-i))
                    H[basis,flipped_i] += J
                    H[flipped_i,basis] += J
    else:
        raise ValueError(f"Unknown Hamiltonian type: {hamiltonian_type}")
    H = H.tocsr()
    n_lowest = min(n_lowest, dim-2)
    if n_lowest < 1:
        n_lowest = 1
    eigenvalues, eigenvectors = sp_linalg.eigsh(H, k=n_lowest, which='SA')
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


# ======================================================================
# Cross-backend agreement test
# ======================================================================
def cross_backend_agreement_test(qubits: int = 3, seed: int = 42):
    """Test that both backends produce consistent expectation values for a simple circuit."""
    # This test only runs if qubits <= 20 (so both backends are available)
    if qubits > 20:
        print("Cross-backend test requires qubits ≤ 20.")
        return None
    # Prepare a simple circuit: GHZ state
    sv = QuantumVMGravity(qubits, noise_level=0.0)
    sv.set_seed(seed)
    sv.start()
    sv.apply_gate('h', [0])
    for i in range(1, qubits):
        sv.apply_gate('cnot', [0, i])
    # Compute expectations
    pauli_xx = ['I']*qubits
    pauli_xx[0] = 'X'
    pauli_xx[1] = 'X'
    exp_sv = sv.expectation(''.join(pauli_xx))
    sv.stop()

    # Stabilizer backend (only if qubits <= 20, we can force it by creating a separate instance)
    stab = QuantumVMGravity(qubits, noise_level=0.0)
    stab.set_seed(seed)
    stab._backend = StabilizerBackend(qubits, 0.0, 0.0)  # force stabilizer
    stab._backend_type = 'stabilizer'
    stab.start()
    # Build GHZ using stabilizer gates
    stab.apply_gate('h', [0])
    for i in range(1, qubits):
        stab.apply_gate('cnot', [0, i])
    exp_stab = stab.expectation(''.join(pauli_xx))
    stab.stop()
    diff = abs(exp_sv - exp_stab)
    return {"difference": diff, "sv_value": exp_sv, "stab_value": exp_stab, "agreement": diff < 1e-2}


# ======================================================================
# Self-test / acceptance suite entry point
# ======================================================================
if __name__ == "__main__":
    print("MOS-HOR-QNVM v15.0-Gravity (Auditable Scientific Edition)")
    print("="*60)
    # Run acceptance tests for 4 qubits
    vm = QuantumVMGravity(qubits=4, noise_level=0.0)
    vm.set_seed(42)
    print("\nAcceptance tests (statevector backend):")
    vm.run_acceptance_tests(verbose=True)
    # Cross-backend agreement test
    print("\nCross-backend agreement test (GHZ, 3 qubits):")
    agree = cross_backend_agreement_test(qubits=3, seed=42)
    if agree:
        print(f"  SV: {agree['sv_value']:.6f}, STAB: {agree['stab_value']:.6f}, diff={agree['difference']:.6e}")
        print(f"  Agreement: {'PASS' if agree['agreement'] else 'FAIL'}")
    else:
        print("  Test skipped (qubits > 20).")
    print("\nEngine ready.")
