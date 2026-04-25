#!/usr/bin/env python3
"""
sim8_multi_qubit_heisenberg_metrology.py – Entanglement-Enhanced Quantum Metrology
==================================================================================

Simulates GHZ and spin-squeezed states for vacuum-weight detection via
Heisenberg-limited frequency estimation.  Uses the QuantumVMGravity engine
from qnvm_gravity.py (statevector for N<=20, stabilizer beyond).

Physical constants
------------------
  g_grav  = 2*pi*1e-6  Hz/kg   (gravitational coupling, from sim4)
  dm_true = 2.3e-23    kg       (predicted mass shift)

Output
------
  sim8_results/   JSON data + matplotlib figures

Author : MOS-HOR Quantum Physics Lab
Version: 1.0
"""

from __future__ import annotations
import sys
import os
import json
import time
import argparse
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

# Ensure qnvm_gravity is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qnvm_gravity import QuantumVMGravity

# ==========================================================================
# Physical / Default Constants
# ==========================================================================
G_GRAV_DEFAULT = 2.0 * np.pi * 1e-6   # Hz / kg
DM_TRUE_DEFAULT = 2.3e-23              # kg
HBAR = 1.054571817e-34                 # J*s


# ==========================================================================
# Module-level g_grav context (set once by main / CLI)
# ==========================================================================
_G_GRAV_CTX = G_GRAV_DEFAULT

def g_grav_from_ctx() -> float:
    return _G_GRAV_CTX

def set_g_grav_ctx(val: float):
    global _G_GRAV_CTX
    _G_GRAV_CTX = val


# ==========================================================================
# EntanglementMetrology
# ==========================================================================
class EntanglementMetrology:
    """Full simulation harness for GHZ and spin-squeezed quantum sensing."""

    def __init__(self, N: int, dephasing_rate: float = 0.0,
                 backaction_noise: float = 0.0):
        """
        Parameters
        ----------
        N : int
            Number of qubits (must be >= 1).
        dephasing_rate : float
            Independent dephasing rate gamma (Hz).  Each qubit gets a Z
            error with probability 1 - exp(-gamma * tau).
        backaction_noise : float
            Standard-deviation coefficient sigma_ba for common-mode
            radiation-pressure back-action.  A phase theta ~ N(0,
            sigma_ba * sqrt(tau)) is applied identically to every qubit.
        """
        self.N = N
        self.gamma = dephasing_rate
        self.sigma_ba = backaction_noise
        # noise_level=0 on the VM itself – we inject noise manually
        self.vm = QuantumVMGravity(qubits=N, noise_level=0.0)

    # ------------------------------------------------------------------
    # State preparation
    # ------------------------------------------------------------------
    def prepare_ghz(self):
        """Prepare the N-qubit GHZ state (|0...0> + |1...1>)/sqrt(2)."""
        self.vm.start()
        self.vm.apply_gate('h', [0])
        for i in range(1, self.N):
            self.vm.apply_gate('cnot', [0, i])

    def prepare_squeezed(self, mu_t: float):
        """
        Prepare a one-axis-twisted spin-squeezed state.

        Starting from |+>^N, applies  U = exp(-i mu_t J_z^2)  via
        pairwise RZZ(2*mu_t) rotations followed by single-qubit phase
        corrections  Rz(-(N-1)*mu_t)  that cancel the self-interaction
        terms from the J_z^2 = sum_{i<j} Z_i Z_j + N/2 decomposition.

        Parameters
        ----------
        mu_t : float
            Twisting strength (dimensionless, product mu * t_sq).
        """
        self.vm.start()
        # Coherent spin state: all |+>
        for i in range(self.N):
            self.vm.apply_gate('h', [i])
        # One-axis twisting: U = exp(-i mu_t J_z^2)
        # J_z^2 = (1/4)(N + 2 sum_{i<j} Z_i Z_j)
        # exp(-i mu_t J_z^2) = exp(-i mu_t N/4) * prod_{i<j} exp(-i mu_t/2 Z_i Z_j)
        # The global phase exp(-i mu_t N/4) is irrelevant.
        # Implement exp(-i mu_t/2 Z_i Z_j) via CNOT-Rz-CNOT:
        #   CNOT(i,j) · Rz(mu_t, j) · CNOT(i,j) ∝ exp(-i mu_t/2 Z_i Z_j)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.vm.apply_gate('cnot', [i, j])
                self.vm.apply_gate('rz', [j], [mu_t])
                self.vm.apply_gate('cnot', [i, j])

    # ------------------------------------------------------------------
    # Sensing evolution
    # ------------------------------------------------------------------
    def evolve_sensing(self, phase: float, tau: float):
        """
        Evolve under the gravitational sensing Hamiltonian for time tau.

        1. Collective Rz(phase) on every qubit  (signal from g_grav * dm * tau).
        2. Common-mode back-action:  theta ~ N(0, sigma_ba * sqrt(tau)).
        3. Independent dephasing: each qubit gets Z with prob p = 1 - exp(-gamma*tau).

        Parameters
        ----------
        phase : float
            Total accumulated phase (rad) from the signal.
        tau   : float
            Evolution time (s).
        """
        # (1) Signal phase
        for i in range(self.N):
            self.vm.apply_gate('rz', [i], [phase])

        # (2) Radiation-pressure back-action (common mode)
        if self.sigma_ba > 0.0:
            ba_phase = np.random.normal(0.0, self.sigma_ba * math.sqrt(tau))
            for i in range(self.N):
                self.vm.apply_gate('rz', [i], [ba_phase])

        # (3) Independent dephasing
        if self.gamma > 0.0:
            p_deph = 1.0 - math.exp(-self.gamma * tau)
            for i in range(self.N):
                if np.random.random() < p_deph:
                    self.vm.apply_gate('z', [i])

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------
    def measure_parity(self, shots: int = 10000) -> float:
        """
        Parity measurement for GHZ readout.

        Apply H on every qubit, measure in Z basis, compute
        P = <prod_i (-1)^{m_i}>.  For ideal GHZ, P = cos(N*phase).

        Returns
        -------
        float  in [-1, 1]
        """
        for i in range(self.N):
            self.vm.apply_gate('h', [i])
        counts = self.vm.measure(shots)
        parity = 0.0
        for bits, cnt in counts.items():
            p = 1
            for b in bits:
                if b == '1':
                    p *= -1
            parity += p * cnt / shots
        return parity

    def measure_Jz(self, shots: int = 10000) -> float:
        """
        Measure collective J_z = (1/2) * sum Z_i  via population counting.

        Returns
        -------
        float   (pop_1 - N/2) where pop_1 is the mean number of |1>s
        """
        counts = self.vm.measure(shots)
        jz = 0.0
        for bits, cnt in counts.items():
            pop = sum(1 for b in bits if b == '1')
            jz += (pop - self.N / 2.0) * cnt / shots
        return jz

    # ------------------------------------------------------------------
    # Collective spin expectation via Pauli decomposition
    # ------------------------------------------------------------------
    def _single_pauli_expectation(self, q: int, pauli: str) -> float:
        """Compute <P_q> for a single-qubit Pauli on qubit q."""
        pauli_str = ['I'] * self.N
        pauli_str[q] = pauli
        return self.vm.expectation(''.join(pauli_str))

    def _two_pauli_expectation(self, q1: int, q2: int,
                                p1: str, p2: str) -> float:
        """Compute <P_{q1} P_{q2}>."""
        pauli_str = ['I'] * self.N
        pauli_str[q1] = p1
        pauli_str[q2] = p2
        return self.vm.expectation(''.join(pauli_str))

    def _collective_J_moments(self, pauli: str) -> Tuple[float, float]:
        """
        Compute <J_P> and <J_P^2> for P in {X, Y, Z},
        where J_P = (1/2) sum_i P_i.

        Returns (<J_P>, <J_P^2>).
        """
        # <J_P> = (1/2) sum_i <P_i>
        jp = 0.0
        for i in range(self.N):
            jp += self._single_pauli_expectation(i, pauli)
        jp *= 0.5

        # <J_P^2> = (1/4) [ sum_i <P_i^2> + sum_{i!=j} <P_i P_j> ]
        # Since P_i^2 = I, <P_i^2> = 1 for all i
        # So <J_P^2> = (1/4) [ N + sum_{i!=j} <P_i P_j> ]
        jp2 = self.N  # the identity terms
        for i in range(self.N):
            for j in range(i + 1, self.N):
                jp2 += self._two_pauli_expectation(i, j, pauli, pauli)
        jp2 *= 0.25

        return jp, jp2

    # ------------------------------------------------------------------
    # Wineland squeezing parameter
    # ------------------------------------------------------------------
    def wineland_squeezing(self, shots: int = 8000) -> float:
        """
        Estimate the Wineland squeezing parameter xi_R^2.

        xi_R^2 = N * (Delta J_perp)^2 / <J_s>^2

        For a state prepared near the X direction (CSS |+>^N),
        J_s = J_x (mean spin direction) and J_perp = J_z (squeezed).

        Returns
        -------
        float  xi_R^2   (< 1 means squeezing, beating the SQL)
        """
        if self.vm._backend_type == 'statevector':
            jx, jx2 = self._collective_J_moments('X')
            jy, jy2 = self._collective_J_moments('Y')
            jz, jz2 = self._collective_J_moments('Z')

            mean_jx = jx
            mean_jx2 = jx ** 2
            var_jy = jy2 - jy ** 2
            var_jz = jz2 - jz ** 2
            if var_jy < 0.0:
                var_jy = 0.0
            if var_jz < 0.0:
                var_jz = 0.0
            # For OAT (J_z^2 Hamiltonian) starting from CSS along x,
            # squeezing occurs along J_y.  Use min variance in
            # the perpendicular plane (y-z) for a robust estimate.
            var_perp = min(var_jy, var_jz)
            if mean_jx2 < 1e-30:
                return float('inf')
            xi2 = self.N * var_perp / mean_jx2
            return max(xi2, 0.0)
        else:
            # Stabilizer: shot-based estimation via repeated sampling
            base_tab = self.vm._backend.tableau.copy()
            jz_samples = []
            jx_samples = []
            for _ in range(shots):
                # J_z: measure directly in computational basis
                tab_z = base_tab.copy()
                bits_z = tab_z.measure_all()
                jz_s = sum(1 - 2 * b for b in bits_z) / 2.0
                jz_samples.append(jz_s)
                # J_x: apply H on all qubits then measure
                tab_x = base_tab.copy()
                for q in range(self.N):
                    tab_x.apply_h(q)
                bits_x = tab_x.measure_all()
                jx_s = sum(1 - 2 * b for b in bits_x) / 2.0
                jx_samples.append(jx_s)
            var_jz = np.var(jz_samples)
            mean_jx = np.mean(jx_samples)
            mean_jx2 = mean_jx ** 2
            if mean_jx2 < 1e-30:
                return float('inf')
            # For stabilizer backend, J_z variance is used as proxy
            xi2 = self.N * var_jz / mean_jx2
            return max(xi2, 0.0)

    # ------------------------------------------------------------------
    # Fisher information
    # ------------------------------------------------------------------
    def fisher_information_ghz(self, phase_vals: np.ndarray,
                                tau: float,
                                shots: int = 5000) -> float:
        """
        Estimate classical Fisher information for phase estimation with GHZ.

        Sweeps the sensing phase, measures parity oscillation probability
        P(phi) = (1 + parity(phi))/2, then computes
        F = sum_k [ (dP/dphi)^2 / (P_k (1-P_k)) ].

        Returns
        -------
        float  Fisher information (rad^-2).
        """
        probs = []
        for phi in phase_vals:
            self.prepare_ghz()
            self.evolve_sensing(phi, tau)
            parity = self.measure_parity(shots)
            probs.append(0.5 * (1.0 + parity))
        dp = np.gradient(probs, phase_vals)
        p_arr = np.array(probs)
        p_arr = np.clip(p_arr, 1e-12, 1.0 - 1e-12)
        fisher = np.sum(dp ** 2 / (p_arr * (1.0 - p_arr)))
        return float(fisher)

    def quantum_fisher_ghz(self, tau: float) -> float:
        """
        Analytic quantum Fisher information for noisy GHZ.

        F_Q = N^2 * exp(-N * gamma * tau) * exp(-sigma_ba^2 * tau)

        Returns
        -------
        float
        """
        fq = float(self.N ** 2)
        if self.gamma > 0.0:
            fq *= math.exp(-self.N * self.gamma * tau)
        if self.sigma_ba > 0.0:
            fq *= math.exp(-(self.sigma_ba ** 2) * tau)
        return fq

    # ------------------------------------------------------------------
    # Optimal sensitivity
    # ------------------------------------------------------------------
    def optimal_sensitivity(self, g_grav: float,
                             tau_vals: np.ndarray) -> Tuple[float, float]:
        """
        Find the minimum detectable mass delta_m_min and the optimal
        sensing time tau_opt via the quantum Cramér-Rao bound.

        delta_m_min = 1 / (g_grav * sqrt(F_Q))

        Returns
        -------
        (dm_min, tau_opt)
        """
        best_dm = float('inf')
        best_tau = tau_vals[0]
        for tau in tau_vals:
            fq = self.quantum_fisher_ghz(tau)
            if fq <= 0.0:
                continue
            dm = 1.0 / (g_grav * math.sqrt(fq))
            if dm < best_dm:
                best_dm = dm
                best_tau = tau
        return best_dm, float(best_tau)

    def sensitivity_vs_N(self, N_vals: List[int], tau: float,
                          shots: int = 5000) -> Dict[str, list]:
        """
        Sweep over qubit numbers and compute GHZ sensitivity.

        Returns dict with keys 'N', 'dm_min_ghz', 'dm_sql', 'fi_sim',
        'fq_analytic', 'scaling'.
        """
        g = g_grav_from_ctx()
        out = {k: [] for k in ['N', 'dm_min_ghz', 'dm_sql',
                                 'fi_sim', 'fq_analytic', 'scaling']}
        for N in N_vals:
            em = EntanglementMetrology(N, self.gamma, self.sigma_ba)
            fq = em.quantum_fisher_ghz(tau)
            if fq > 0:
                dm_ghz = 1.0 / (g * math.sqrt(fq))
            else:
                dm_ghz = float('inf')
            dm_sql = 1.0 / (g * math.sqrt(N))
            # Simulated FI (small phase sweep)
            phase_span = max(1e-6, 2.0 * np.pi * N * 1e-3)
            phase_vals = np.linspace(-phase_span, phase_span, 12)
            try:
                fi = em.fisher_information_ghz(phase_vals, tau, shots=shots)
            except Exception:
                fi = 0.0
            out['N'].append(N)
            out['dm_min_ghz'].append(dm_ghz)
            out['dm_sql'].append(dm_sql)
            out['fi_sim'].append(fi)
            out['fq_analytic'].append(fq)
            out['scaling'].append(dm_ghz / dm_sql if dm_sql > 0 else float('inf'))
        return out

    def sensitivity_vs_time(self, N: int, tau_vals: np.ndarray,
                             shots: int = 5000) -> Dict[str, list]:
        """
        Sweep sensing time for a fixed N.

        Returns dict with 'tau', 'dm_min', 'fq', 'sql_dm'.
        """
        g = g_grav_from_ctx()
        out = {k: [] for k in ['tau', 'dm_min', 'fq', 'sql_dm']}
        for tau in tau_vals:
            fq = self.quantum_fisher_ghz(tau)
            if fq > 0:
                dm = 1.0 / (g * math.sqrt(fq))
            else:
                dm = float('inf')
            sql_dm = 1.0 / (g * math.sqrt(N))
            out['tau'].append(tau)
            out['dm_min'].append(dm)
            out['fq'].append(fq)
            out['sql_dm'].append(sql_dm)
        return out


# ==========================================================================
# Plotting helpers
# ==========================================================================
def _ensure_output_dir(d: str):
    os.makedirs(d, exist_ok=True)

def _save_fig(fig, name: str, out_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [plot] {path}")
    return path

def plot_xi_vs_mut(mu_t_vals, xi2_vals, N_sq, out_dir):
    """xi_R^2 vs twisting strength mu_t."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(mu_t_vals, xi2_vals, 'o-', color='#2196F3', lw=1.5,
                markersize=5)
    ax.axhline(1.0, ls='--', color='grey', lw=1.2, label='SQL ($\\xi_R^2=1$)')
    ax.set_xlabel('Twisting strength $\\mu t$')
    ax.set_ylabel('$\\xi_R^2$')
    ax.set_title(f'Wineland Squeezing Parameter  ($N={N_sq}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, 'xiR2_vs_mut.png', out_dir)
    plt.close(fig)

def plot_fi_vs_N(data: dict, out_dir):
    """Fisher information vs N (Heisenberg scaling)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    Ns = np.array(data['N'], dtype=float)
    fq = np.array(data['FQ_analytic'])
    fi = np.array(data['FI_simulated'])
    ax.loglog(Ns, fq, 's-', color='#E91E63',
              label='QFI (analytic)', lw=1.5, markersize=6)
    valid = fi > 0
    if np.any(valid):
        ax.loglog(Ns[valid], fi[valid], '^--', color='#4CAF50',
                  label='FI (simulated)', lw=1.2, markersize=5)
    ax.loglog(Ns, Ns, ':', color='#9E9E9E', lw=1.2, label='SQL ($\\propto N$)')
    ax.loglog(Ns, Ns**2, ':', color='#9E9E9E', alpha=0.5,
              lw=1.2, label='HL ($\\propto N^2$)')
    ax.set_xlabel('Number of qubits $N$')
    ax.set_ylabel('Fisher information')
    ax.set_title('Fisher Information Scaling')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    _save_fig(fig, 'FI_vs_N.png', out_dir)
    plt.close(fig)

def plot_dm_vs_Tint(tau_vals, dm_vals, sql_vals, N, out_dir):
    """Minimum detectable mass vs sensing time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(tau_vals, dm_vals, '-', color='#E91E63', lw=2,
              label=f'GHZ ($N={N}$)')
    ax.loglog(tau_vals, sql_vals, '--', color='#9E9E9E', lw=1.5,
              label='SQL ($N=1$)')
    ax.set_xlabel('Sensing time $\\tau$ (s)')
    ax.set_ylabel('$\\delta m_{\\min}$ (kg)')
    ax.set_title('Mass Sensitivity vs Sensing Time')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    _save_fig(fig, 'dm_min_vs_tau.png', out_dir)
    plt.close(fig)

def plot_dm_vs_N(Ns, dm_ghz, dm_sql, dm_target, out_dir):
    """dm_min vs N."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(Ns, dm_ghz, 's-', color='#E91E63', lw=1.5,
              label='GHZ (analytic)', markersize=6)
    ax.loglog(Ns, dm_sql, '^--', color='#9E9E9E', lw=1.5,
              label='SQL', markersize=5)
    ax.axhline(dm_target, ls=':', color='#FF9800', lw=1.5,
               label=f'$\\Delta m_{{\\rm true}}$ = {dm_target:.1e} kg')
    ax.set_xlabel('Number of qubits $N$')
    ax.set_ylabel('$\\delta m_{\\min}$ (kg)')
    ax.set_title('GHZ Mass Sensitivity vs Qubit Count')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    _save_fig(fig, 'dm_min_vs_N.png', out_dir)
    plt.close(fig)

def plot_squeezed_phase_distribution(phase_samples, out_dir):
    """Histogram of measured phase for squeezed-state sensing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(phase_samples, bins=40, density=True, color='#4CAF50', alpha=0.7,
            edgecolor='#2E7D32')
    ax.set_xlabel('Estimated phase $\\hat{\\phi}$ (rad)')
    ax.set_ylabel('Density')
    ax.set_title('Squeezed-State Phase Estimation Distribution')
    ax.grid(True, alpha=0.3)
    _save_fig(fig, 'squeezed_phase_dist.png', out_dir)
    plt.close(fig)

def plot_feasibility(Ns, dm_ghz, dm_target, gamma, sigma_ba, g_grav, out_dir):
    """Feasibility assessment: can we reach the target?"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    Ns_f = np.array(Ns, dtype=float)
    dm_f = np.array(dm_ghz, dtype=float)
    ratio = np.where(np.isfinite(dm_f) & (dm_f > 0),
                     dm_f / dm_target, 1e10)
    ax.semilogy(Ns_f, ratio, 'o-', color='#2196F3', lw=1.5, markersize=6)
    ax.axhline(1.0, ls='--', color='#FF9800', lw=1.5, label='Target reached')
    ax.set_xlabel('Qubit count $N$')
    ax.set_ylabel('$\\delta m_{\\min} / \\Delta m_{\\rm true}$')
    ax.set_title(f'Feasibility  ($\\gamma$={gamma:.0e} Hz, '
                 f'$\\sigma_{{BA}}$={sigma_ba:.1e})')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(1e-2, 1e6)
    _save_fig(fig, 'feasibility.png', out_dir)
    plt.close(fig)


# ==========================================================================
# CLI & Main Simulation Driver
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='sim8 – Entanglement-Enhanced Quantum Metrology')
    p.add_argument('--N-max', type=int, default=10,
                   help='Maximum number of qubits (default 10)')
    p.add_argument('--dephasing', type=float, default=1e2,
                   help='Dephasing rate gamma in Hz (default 100)')
    p.add_argument('--backaction', type=float, default=0.1,
                   help='Back-action noise coeff sigma_ba (default 0.1)')
    p.add_argument('--g-grav', type=float, default=G_GRAV_DEFAULT,
                   help=f'Gravitational coupling Hz/kg (default {G_GRAV_DEFAULT})')
    p.add_argument('--signal-dm', type=float, default=DM_TRUE_DEFAULT,
                   help=f'True mass shift kg (default {DM_TRUE_DEFAULT})')
    p.add_argument('--tau', type=float, default=1e-4,
                   help='Nominal sensing time in s (default 1e-4)')
    p.add_argument('--shots', type=int, default=4000,
                   help='Shots per measurement (default 4000)')
    p.add_argument('--output-dir', type=str, default='sim8_results',
                   help='Output directory (default sim8_results)')
    p.add_argument('--sweep-N', action='store_true', default=True,
                   help='Run N-sweep (default True)')
    p.add_argument('--sweep-squeezing', action='store_true', default=True,
                   help='Run squeezing sweep (default True)')
    return p.parse_args()


def run_simulation(args):
    """Execute the full simulation and produce outputs."""
    t0 = time.time()
    _ensure_output_dir(args.output_dir)
    set_g_grav_ctx(args.g_grav)
    g_grav = args.g_grav
    dm_true = args.signal_dm
    gamma = args.dephasing
    sigma_ba = args.backaction
    tau_nom = args.tau
    shots = args.shots
    N_max = args.N_max

    # Choose N values that fit in statevector (<= 20)
    N_vals = [n for n in [1, 2, 3, 4, 6, 8, 10, 12, 16]
              if n <= min(N_max, 20)]
    if not N_vals:
        N_vals = [N_max] if N_max <= 20 else [1, 2, 4, 8]
    tau_vals = np.logspace(-6, -2, 50)

    json_data: Dict = {
        'parameters': {
            'N_max': N_max, 'N_vals': N_vals,
            'dephasing_rate_Hz': gamma,
            'backaction_noise': sigma_ba,
            'g_grav_Hz_per_kg': g_grav,
            'signal_dm_kg': dm_true,
            'tau_nominal_s': tau_nom,
            'shots': shots,
        },
        'squeezing': {'mu_t': [], 'xi_R2': []},
        'fisher_vs_N': {},
        'sensitivity_vs_time': {},
        'sensitivity_vs_N': {},
        'feasibility': {'N': [], 'dm_min': [], 'ratio': []},
    }

    print('=' * 72)
    print('  SIMULATION 8: Entanglement-Enhanced Quantum Metrology')
    print('  GHZ & Spin-Squeezed States for Vacuum-Weight Detection')
    print('=' * 72)
    print(f'  g_grav       = {g_grav:.4e} Hz/kg')
    print(f'  Delta_m_true = {dm_true:.4e} kg')
    print(f'  gamma        = {gamma:.2e} Hz')
    print(f'  sigma_ba     = {sigma_ba:.2e}')
    print(f'  tau_nom      = {tau_nom:.2e} s')
    print(f'  shots        = {shots}')
    print(f'  N_vals       = {N_vals}')
    print(f'  Backend      : statevector (N <= 20)')
    print('-' * 72)

    # =================================================================
    # 1. Spin-squeezing sweep: xi_R^2 vs mu_t
    # =================================================================
    if args.sweep_squeezing:
        print('\n[1] Spin-Squeezing Parameter Sweep')
        print('    Measuring xi_R^2 vs twisting strength mu_t ...')
        N_sq = min(N_max, 8)  # keep small for speed
        mu_t_vals = np.linspace(0, 0.6, 16)
        xi2_vals = []
        for mu in mu_t_vals:
            em = EntanglementMetrology(N_sq, dephasing_rate=0.0,
                                       backaction_noise=0.0)
            em.prepare_squeezed(mu)
            xi2 = em.wineland_squeezing(shots=min(shots, 3000))
            xi2_vals.append(xi2)
            tag = '  ** SQUEEZED' if xi2 < 1.0 else ''
            print(f'      mu_t={mu:.4f}  ->  xi_R^2={xi2:.6f}{tag}')
        json_data['squeezing']['N'] = N_sq
        json_data['squeezing']['mu_t'] = mu_t_vals.tolist()
        json_data['squeezing']['xi_R2'] = [float(x) for x in xi2_vals]
        try:
            plot_xi_vs_mut(mu_t_vals, np.array(xi2_vals), N_sq, args.output_dir)
        except Exception as exc:
            print(f'    [warn] xi plot failed: {exc}')

        # Squeezed-state phase estimation distribution
        print('    Squeezed-state phase estimation...')
        xi2_arr = np.array(xi2_vals)
        valid_mask = xi2_arr > 0
        if np.any(valid_mask & (xi2_arr < 10)):
            best_idx = np.argmin(xi2_arr)
            mu_opt = mu_t_vals[best_idx]
        else:
            mu_opt = mu_t_vals[1]
        phase_true = 0.05
        n_reps = 100
        phase_ests = []
        em = EntanglementMetrology(N_sq, dephasing_rate=gamma * 0.01,
                                   backaction_noise=0.0)
        for _ in range(n_reps):
            em.prepare_squeezed(mu_opt)
            em.evolve_sensing(phase_true, tau_nom * 0.1)
            jz = em.measure_Jz(shots=min(shots, 2000))
            # Linear estimate: phi ~ 2 * Jz / (N * <Jx>)
            phase_ests.append(jz * 2.0 / N_sq)
        try:
            plot_squeezed_phase_distribution(phase_ests, args.output_dir)
        except Exception as exc:
            print(f'    [warn] phase dist plot failed: {exc}')
        json_data['squeezed_phase'] = [float(x) for x in phase_ests]
        std_est = np.std(phase_ests) if phase_ests else float('inf')
        print(f'    Phase estimate std = {std_est:.6f} rad '
              f'(sql = {1/math.sqrt(N_sq):.6f} rad)')

    # =================================================================
    # 2. GHZ Fisher Information vs N
    # =================================================================
    if args.sweep_N:
        print('\n[2] GHZ Fisher Information vs N')
        print('    Sweeping qubit count ...')
        em0 = EntanglementMetrology(N_vals[0] if N_vals else 4,
                                    gamma, sigma_ba)
        sweep = em0.sensitivity_vs_N(N_vals, tau_nom, shots=shots)
        json_data['fisher_vs_N'] = {
            'N': sweep['N'],
            'FQ_analytic': [float(x) for x in sweep['fq_analytic']],
            'FI_simulated': [float(x) for x in sweep['fi_sim']],
            'dm_ghz': [float(x) for x in sweep['dm_min_ghz']],
            'dm_sql': [float(x) for x in sweep['dm_sql']],
            'scaling_ratio': [float(x) for x in sweep['scaling']],
        }
        print(f'    {"N":>4s}  {"FQ_analytic":>12s}  {"FI_sim":>12s}'
              f'  {"dm_GHZ":>12s}  {"dm_SQL":>12s}  {"ratio":>8s}')
        for i, N in enumerate(sweep['N']):
            fq = sweep['fq_analytic'][i]
            fi = sweep['fi_sim'][i]
            dm_g = sweep['dm_min_ghz'][i]
            dm_s = sweep['dm_sql'][i]
            r = sweep['scaling'][i]
            fq_s = f'{fq:12.3e}' if np.isfinite(fq) else '         inf'
            fi_s = f'{fi:12.3e}' if np.isfinite(fi) and fi > 0 else '          0'
            dm_g_s = f'{dm_g:12.3e}' if np.isfinite(dm_g) else '         inf'
            dm_s_s = f'{dm_s:12.3e}' if np.isfinite(dm_s) else '         inf'
            r_s = f'{r:8.3f}' if np.isfinite(r) else '      inf'
            print(f'    {N:4d}  {fq_s}  {fi_s}  {dm_g_s}  {dm_s_s}  {r_s}')
        try:
            plot_fi_vs_N(json_data['fisher_vs_N'], args.output_dir)
        except Exception as exc:
            print(f'    [warn] FI plot failed: {exc}')

    # =================================================================
    # 3. Sensitivity vs time (for each N)
    # =================================================================
    print('\n[3] Sensitivity vs Sensing Time')
    time_results = {}
    for N in N_vals:
        em = EntanglementMetrology(N, gamma, sigma_ba)
        tv = em.sensitivity_vs_time(N, tau_vals, shots=shots)
        time_results[str(N)] = {
            'tau': [float(x) for x in tv['tau']],
            'dm_min': [float(x) for x in tv['dm_min']],
            'FQ': [float(x) for x in tv['fq']],
            'sql_dm': [float(x) for x in tv['sql_dm']],
        }
        dm_min, tau_opt = em.optimal_sensitivity(g_grav, tau_vals)
        dm_s = f'{dm_min:.3e}' if np.isfinite(dm_min) else '     inf'
        print(f'    N={N:3d}:  dm_min={dm_s} kg  @ tau={tau_opt:.3e} s')
    json_data['sensitivity_vs_time'] = time_results
    # Plot for the largest N
    largest = max(N_vals)
    if str(largest) in time_results:
        tr = time_results[str(largest)]
        dm_arr = np.array(tr['dm_min'])
        sql_arr = np.array(tr['sql_dm'])
        valid = np.isfinite(dm_arr) & (dm_arr > 0)
        if np.any(valid):
            try:
                plot_dm_vs_Tint(tau_vals[valid], dm_arr[valid],
                                sql_arr[valid], largest, args.output_dir)
            except Exception as exc:
                print(f'    [warn] time plot failed: {exc}')

    # =================================================================
    # 4. dm_min vs N (at optimal tau)
    # =================================================================
    print('\n[4] Minimum Detectable Mass vs N (at optimal tau)')
    dm_ghz_list = []
    dm_sql_list = []
    tau_opt_list = []
    for N in N_vals:
        em = EntanglementMetrology(N, gamma, sigma_ba)
        dm_min, tau_opt = em.optimal_sensitivity(g_grav, tau_vals)
        dm_ghz_list.append(dm_min)
        dm_sql_list.append(1.0 / (g_grav * math.sqrt(N)))
        tau_opt_list.append(tau_opt)
        json_data['feasibility']['N'].append(N)
        json_data['feasibility']['dm_min'].append(dm_min)
        json_data['feasibility']['ratio'].append(
            dm_min / dm_true if dm_true > 0 else float('inf'))
        dm_s = f'{dm_min:.3e}' if np.isfinite(dm_min) else '     inf'
        print(f'    N={N:3d}:  dm_min={dm_s} kg  tau_opt={tau_opt:.3e} s  '
              f'ratio_to_target={dm_min/dm_true:.2e}')
    json_data['sensitivity_vs_N'] = {
        'N': N_vals,
        'dm_ghz': [float(x) for x in dm_ghz_list],
        'dm_sql': [float(x) for x in dm_sql_list],
    }
    try:
        plot_dm_vs_N(N_vals, dm_ghz_list, dm_sql_list, dm_true, args.output_dir)
        plot_feasibility(N_vals, dm_ghz_list, dm_true, gamma, sigma_ba,
                         g_grav, args.output_dir)
    except Exception as exc:
        print(f'    [warn] N-sweep plots failed: {exc}')

    # =================================================================
    # 5. Heisenberg scaling exponent
    # =================================================================
    print('\n[5] Heisenberg Scaling Analysis')
    valid = [(n, dm) for n, dm in zip(N_vals, dm_ghz_list)
             if np.isfinite(dm) and dm > 0]
    alpha = None
    if len(valid) >= 2:
        ns = np.log10(np.array([v[0] for v in valid], dtype=float))
        dms = np.log10(np.array([v[1] for v in valid]))
        coeffs = np.polyfit(ns, dms, 1)
        alpha = -coeffs[0]
        print(f'    Scaling exponent  delta_m ~ N^(-{alpha:.3f})')
        print(f'    (ideal Heisenberg = -1.0, SQL = -0.5)')
        json_data['scaling_exponent'] = float(alpha)
    else:
        print('    Insufficient data for scaling fit.')

    # =================================================================
    # 6. Comparison table: GHZ vs SQL vs target
    # =================================================================
    print('\n[6] Comparison Summary')
    print(f'    {"N":>4s}  {"dm_GHZ":>12s}  {"dm_SQL":>12s}'
          f'  {"dm_true":>12s}  {"GHZ/SQL":>8s}  {"SQL/dm":>10s}')
    for i, N in enumerate(N_vals):
        dm_g = dm_ghz_list[i]
        dm_s = dm_sql_list[i]
        r_gs = dm_g / dm_s if dm_s > 0 and np.isfinite(dm_g) else float('inf')
        r_st = dm_s / dm_true if dm_true > 0 else float('inf')
        dm_g_s = f'{dm_g:.3e}' if np.isfinite(dm_g) else '     inf'
        r_gs_s = f'{r_gs:8.3f}' if np.isfinite(r_gs) else '      inf'
        print(f'    {N:4d}  {dm_g_s}  {dm_s:12.3e}'
              f'  {dm_true:12.3e}  {r_gs_s}  {r_st:10.2e}')

    # =================================================================
    # 7. Integration-time analysis
    # =================================================================
    print('\n[7] Integration-Time Analysis')
    T_int_vals = np.logspace(0, 6, 30)  # 1 s to 10^6 s
    print(f'    {"T_int (s)":>12s}', end='')
    for N in [n for n in N_vals if n in [1, 4, max(N_vals)]]:
        print(f'  {"dm(N="+str(N)+")":>14s}', end='')
    print()
    for T in T_int_vals:
        print(f'    {T:12.1e}', end='')
        for N in [n for n in N_vals if n in [1, 4, max(N_vals)]]:
            em = EntanglementMetrology(N, gamma, sigma_ba)
            dm_min, tau_opt = em.optimal_sensitivity(g_grav, tau_vals)
            if not np.isfinite(dm_min) or dm_min <= 0 or tau_opt <= 0:
                print(f'  {"inf":>14s}', end='')
                continue
            # Average over nu = T / tau_opt repetitions
            nu = T / tau_opt
            dm_T = dm_min / math.sqrt(nu)
            print(f'  {dm_T:14.3e}', end='')
        print()

    # =================================================================
    # 8. Feasibility verdict
    # =================================================================
    print('\n' + '=' * 72)
    print('  FEASIBILITY VERDICT')
    print('=' * 72)
    reachable = [n for n, dm in zip(N_vals, dm_ghz_list)
                 if np.isfinite(dm) and dm < dm_true]
    if reachable:
        N_best = min(reachable)
        dm_best = dm_ghz_list[N_vals.index(N_best)]
        print(f'  [OK] Target delta_m = {dm_true:.2e} kg IS REACHABLE '
              f'with N >= {N_best}')
        print(f'       At N={N_best}: dm_min = {dm_best:.3e} kg  '
              f'({dm_best/dm_true:.2f}x the target)')
    else:
        # Extrapolate: find N where dm_min = dm_true
        if len(valid) >= 2 and alpha is not None:
            ns_v = np.array([v[0] for v in valid], dtype=float)
            dm_v = np.array([v[1] for v in valid])
            log_n = np.log10(ns_v)
            log_dm = np.log10(dm_v)
            c = np.polyfit(log_n, log_dm, 1)
            log_n_target = (math.log10(dm_true) - c[1]) / c[0]
            N_needed = 10 ** log_n_target
            print(f'  [--] Target NOT reachable with N <= {max(N_vals)}.')
            print(f'       Extrapolated N needed: ~{N_needed:.0f} qubits')
            print(f'       (scaling exponent alpha = {alpha:.3f})')

            # Also estimate with integration time
            N_test = max(N_vals)
            em = EntanglementMetrology(N_test, gamma, sigma_ba)
            dm_min, tau_opt = em.optimal_sensitivity(g_grav, tau_vals)
            if np.isfinite(dm_min) and dm_min > 0 and tau_opt > 0:
                T_needed = (dm_min / dm_true) ** 2 * tau_opt
                print(f'       At N={N_test}: need T_int ~ {T_needed:.2e} s '
                      f'({T_needed/3600:.1e} hrs) to reach target')
        else:
            print('  [--] Insufficient data for feasibility assessment.')

    print(f'\n  Noise parameters: gamma={gamma:.1e} Hz, sigma_ba={sigma_ba:.1e}')
    print(f'  Optimal tau range: [{min(tau_opt_list):.3e}, '
          f'{max(tau_opt_list):.3e}] s')

    elapsed = time.time() - t0
    print(f'\n  Total wall time: {elapsed:.1f} s')
    print('=' * 72)

    # =================================================================
    # Save JSON
    # =================================================================
    json_path = os.path.join(args.output_dir, 'sim8_results.json')
    json_data['elapsed_seconds'] = elapsed
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=float)
    print(f'\n  JSON saved: {json_path}')

    return json_data


# ==========================================================================
def main():
    args = parse_args()
    run_simulation(args)


if __name__ == '__main__':
    main()
