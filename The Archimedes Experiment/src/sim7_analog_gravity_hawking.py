#!/usr/bin/env python3
"""
sim7_analog_gravity_hawking.py
===============================
Sonic Black Hole Analog — Hawking Radiation in a 1D Fermionic Qubit Chain

Simulates a quantum quench from a uniform tight-binding chain to one with
position-dependent hopping J_i = (J0/2)(1 − tanh((i−i_h)/σ)) that creates
a sonic horizon.  An on-site potential μ_i = μ0 tanh((i−i_h)/σ) breaks the
particle-hole symmetry so that particle density redistributes across the
horizon — a direct analog of Hawking radiation.

Tracks entanglement-entropy growth, correlation spreading, and thermal
features using an efficient single-particle correlation-matrix formalism
for Gaussian (free-fermion) states.

Ground state via Jordan-Wigner Slater determinant (full 2^L statevector).
Time evolution: exp(−i h dt) on the L×L single-particle Hamiltonian.
Observables: ⟨n_i⟩, S_EE(cut), ⟨c_i†c_j⟩, bond current, Bogoliubov |β_k|².
"""

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import argparse
import os
import time as clock

# ═══════════════════════════════════════════════════════════════
#  Vectorised popcount for numpy int64 arrays
# ═══════════════════════════════════════════════════════════════

def _popcount(x):
    """Hamming weight of every element in a numpy int64 array."""
    x = np.asarray(x, dtype=np.int64)
    x -= (x >> 1) & np.int64(0x5555555555555555)
    x = (x & np.int64(0x3333333333333333)) + ((x >> 2) & np.int64(0x3333333333333333))
    x = (x + (x >> 4)) & np.int64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.int64(0x0101010101010101)) >> 56).astype(int)


# ═══════════════════════════════════════════════════════════════
#  Jordan-Wigner fermionic operators on 2^L statevectors
# ═══════════════════════════════════════════════════════════════

def _jw_parity(idx, site, L):
    """(-1)^{Σ_{l<site} n_l}  for every index in *idx*."""
    if site == 0:
        return np.ones(len(idx), dtype=float)
    shift = L - site
    mask = ((np.int64(1) << site) - 1) << shift
    bits = (idx & mask) >> shift
    return np.where(_popcount(bits) % 2, -1.0, 1.0)


def _jw_create(state, site, L):
    """Apply c†_site  via  c†_j = (Π_{l<j} Z_l) ⊗ σ^+_j.
    Bit convention: site 0 ↔ bit (L−1) (MSB-first)."""
    dim = 1 << L
    bp = L - 1 - site
    idx = np.arange(dim, dtype=np.int64)
    vacant = ((idx >> bp) & 1) == 0
    src = idx[vacant]
    par = _jw_parity(src, site, L)
    amp = par * state[vacant]
    dst = (src | (np.int64(1) << bp)).astype(int)
    out = np.zeros(dim, dtype=complex)
    np.add.at(out, dst, amp)
    return out


def _jw_annihilate(state, site, L):
    """Apply c_site  via  c_j = (Π_{l<j} Z_l) ⊗ σ^-_j."""
    dim = 1 << L
    bp = L - 1 - site
    idx = np.arange(dim, dtype=np.int64)
    occ = ((idx >> bp) & 1) == 1
    src = idx[occ]
    par = _jw_parity(src, site, L)
    amp = par * state[occ]
    dst = (src ^ (np.int64(1) << bp)).astype(int)
    out = np.zeros(dim, dtype=complex)
    np.add.at(out, dst, amp)
    return out


# ═══════════════════════════════════════════════════════════════
#  Slater determinant → full 2^L statevector
# ═══════════════════════════════════════════════════════════════

def slater_to_statevector(eigvecs, n_occ, L):
    """Build the normalised 2^L statevector for a Slater determinant.
    eigvecs[:, k] = k-th single-particle orbital;  occupy 0 … n_occ−1."""
    dim = 1 << L
    psi = np.zeros(dim, dtype=complex)
    psi[0] = 1.0
    for k in range(n_occ):
        c = eigvecs[:, k]
        tmp = np.zeros(dim, dtype=complex)
        for j in range(L):
            if abs(c[j]) > 1e-14:
                tmp += c[j] * _jw_create(psi, j, L)
        nrm = np.linalg.norm(tmp)
        psi = tmp / nrm if nrm > 1e-12 else tmp
    return psi


# ═══════════════════════════════════════════════════════════════
#  Single-particle Hamiltonian builders  (L × L, real symmetric)
# ═══════════════════════════════════════════════════════════════

def ham_uniform(L, J0):
    """Open-boundary uniform tight-binding chain."""
    H = np.zeros((L, L))
    for i in range(L - 1):
        H[i, i + 1] = H[i + 1, i] = -J0
    return H


def ham_inhomogeneous(L, J0, sigma, mu0, ih):
    """Position-dependent hopping  J(i+½) = (J0/2)[1−tanh((i+½−ih)/σ)]
    and on-site potential  μ(i) = μ0 tanh((i−ih)/σ)."""
    H = np.zeros((L, L))
    for i in range(L - 1):
        x = i + 0.5
        J = (J0 / 2.0) * (1.0 - np.tanh((x - ih) / sigma))
        H[i, i + 1] = H[i + 1, i] = -J
    for i in range(L):
        H[i, i] = mu0 * np.tanh((i - ih) / sigma)
    return H


def hopping_array(L, J0, sigma, ih):
    """1-D bond hopping amplitudes  (length L−1)."""
    xs = np.arange(L - 1) + 0.5
    return (J0 / 2.0) * (1.0 - np.tanh((xs - ih) / sigma))


def potential_array(L, mu0, sigma, ih):
    """1-D on-site potentials  (length L)."""
    return mu0 * np.tanh((np.arange(L, dtype=float) - ih) / sigma)


# ═══════════════════════════════════════════════════════════════
#  Correlation-matrix helpers  (Gaussian / free-fermion states)
# ═══════════════════════════════════════════════════════════════

def corr_from_slater(eigvecs, n_occ):
    r"""$C_{ij} = \sum_{k<n_\mathrm{occ}} U_{ik} U^*_{jk}$."""
    U = eigvecs[:, :n_occ]
    return U @ U.conj().T


def density_from_corr(C):
    r"""$\langle n_i \rangle = C_{ii}$."""
    return np.real(np.diag(C))


def ee_from_corr(C, cut):
    """Von Neumann entropy of subsystem A = sites [0, cut).
    S_A = −Σ_k [ν_k ln ν_k + (1−ν_k) ln(1−ν_k)]  with ν = eig(C_A)."""
    nu = np.linalg.eigvalsh(C[:cut, :cut])
    nu = np.clip(nu, 1e-15, 1.0 - 1e-15)
    return -float(np.sum(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)))


def current_from_corr(C):
    """Bond current  I_i = −2 Im C_{i,i+1}  (length L−1)."""
    return np.array([-2.0 * np.imag(C[i, i + 1]) for i in range(C.shape[0] - 1)])


def energy_density_from_corr(C, h):
    """Local energy density  ε_i = Σ_j h_{ij} C_{ji}  (length L)."""
    return np.real(np.diag(h @ C))


def ee_from_statevector(psi, L, cut):
    """S_EE via explicit partial trace (verification only)."""
    dA, dB = 1 << cut, 1 << (L - cut)
    M = psi.reshape(dA, dB)
    rho = M @ M.conj().T
    ev = np.linalg.eigvalsh(rho)
    ev = ev[ev > 1e-14]
    return -float(np.sum(ev * np.log(ev)))


def correlation_from_statevector(psi, L):
    """C_{ij} = <ψ|c†_i c_j|ψ> via Jordan-Wigner (for small L only)."""
    C = np.zeros((L, L), dtype=complex)
    for j in range(L):
        cj = _jw_annihilate(psi, j, L)
        for i in range(L):
            C[i, j] = np.vdot(psi, _jw_create(cj, i, L))
    return C


# ═══════════════════════════════════════════════════════════════
#  Bogoliubov / scattering analysis  (step-function horizon)
# ═══════════════════════════════════════════════════════════════

def bogoliubov_step_coefficients(L, J0, ih, J_ratio=0.05):
    """Numerical + analytic |β_k|² for a step-function hopping change.
    Left (i<ih): J_L = J0.  Right (i≥ih): J_R = J0·J_ratio.
    Returns (k_modes, beta_numeric, beta_analytic)."""
    J_L, J_R = J0, J0 * J_ratio

    # step-function Hamiltonian
    H = np.zeros((L, L))
    for i in range(L - 1):
        Ji = J_L if i < ih - 1 else (0.5 * (J_L + J_R) if i == ih - 1 else J_R)
        H[i, i + 1] = H[i + 1, i] = -Ji
    evals, evecs = np.linalg.eigh(H)

    N_k = 80
    k_modes = np.linspace(0.05, np.pi - 0.05, N_k)
    beta_num = np.zeros(N_k)
    beta_ana = np.zeros(N_k)

    for ik, k in enumerate(k_modes):
        E = -2.0 * J_L * np.cos(k)
        # analytic reflection
        cos_q = (J_L / J_R) * np.cos(k)
        if abs(cos_q) <= 1.0:
            q = np.arccos(cos_q)
            num = -J_L * np.exp(1j * k) + J_R * np.exp(1j * q)
            den = J_L * np.exp(-1j * k) - J_R * np.exp(1j * q)
            beta_ana[ik] = float(np.abs(num / den) ** 2)
        else:
            beta_ana[ik] = 1.0
        # numerical decomposition
        idx_e = np.argmin(np.abs(evals - E))
        psi = evecs[:, idx_e]
        ns = np.arange(ih, dtype=float)
        M = np.column_stack([np.exp(1j * k * ns), np.exp(-1j * k * ns)])
        coeffs, *_ = np.linalg.lstsq(M, psi[:ih], rcond=None)
        A, B = coeffs
        if abs(A) > 1e-10:
            beta_num[ik] = float(np.abs(B / A) ** 2)
    return k_modes, beta_num, beta_ana


# ═══════════════════════════════════════════════════════════════
#  Hawking-temperature and central-charge extraction
# ═══════════════════════════════════════════════════════════════

def extract_hawking_temp_entropy(times, entropies):
    """T_H = (6/π) × dS/dt  fitted to the RISING portion of S_EE(t).
    Finds the entropy peak and fits linearly from t=0 to t_peak."""
    if len(times) < 6:
        return 0.0, 0.0
    ent = np.asarray(entropies)
    tms = np.asarray(times)
    peak_idx = np.argmax(ent)
    if peak_idx < 3:
        return 0.0, 0.0
    # fit to [1, peak_idx] to capture the rising phase
    seg_t = tms[1:peak_idx + 1]
    seg_s = ent[1:peak_idx + 1]
    c = np.polyfit(seg_t, seg_s, 1)
    slope = max(c[0], 0.0)
    T_H = (6.0 / np.pi) * slope
    return T_H, slope


def extract_central_charge(slope, T_H):
    """c = (6/(π T_H)) × dS/dt."""
    if T_H < 1e-12:
        return 0.0
    return (6.0 / (np.pi * T_H)) * max(slope, 0.0)


def extract_hawking_temp_density(excess, positions, ih):
    """Fit |δn(x)| on subsonic side to exp(−x/ξ);  T_H ∝ 1/ξ.
    Returns 0 if the fit slope is positive (no exponential decay)."""
    mask = positions >= ih
    if np.sum(mask) < 3:
        return 0.0
    x = positions[mask].astype(float)
    y = np.abs(excess[mask])
    good = y > 1e-13
    if np.sum(good) < 3:
        return 0.0
    x, y = x[good], y[good]
    logy = np.log(np.clip(y, 1e-15, None))
    c = np.polyfit(x, logy, 1)
    # Positive slope means density grows → no thermal decay
    if c[0] < -1e-12:
        xi = -1.0 / c[0]
        return min(1.0 / max(xi, 1e-10), 100.0)
    return 0.0


# ═══════════════════════════════════════════════════════════════
#  Dispersion relation (left vs right of horizon)
# ═══════════════════════════════════════════════════════════════

def dispersion_sides(L, J0, sigma, mu0, ih):
    """Diagonalise local Hamiltonians on each side; return eigenvalue arrays."""
    nL, nR = max(ih, 2), max(L - ih, 2)
    HL, HR = np.zeros((nL, nL)), np.zeros((nR, nR))
    for i in range(nL - 1):
        x = i + 0.5
        J = (J0 / 2.0) * (1.0 - np.tanh((x - ih) / sigma))
        HL[i, i + 1] = HL[i + 1, i] = -J
    for i in range(nR - 1):
        x = ih + i + 0.5
        J = (J0 / 2.0) * (1.0 - np.tanh((x - ih) / sigma))
        HR[i, i + 1] = HR[i + 1, i] = -J
    return np.sort(np.linalg.eigvalsh(HL)), np.sort(np.linalg.eigvalsh(HR))


# ═══════════════════════════════════════════════════════════════
#  Main simulation class
# ═══════════════════════════════════════════════════════════════

class SonicBlackHole:
    """Sonic black-hole analog in a 1D fermionic qubit chain."""

    def __init__(self, L, sigma, J0=1.0, mu0=0.5, filling=0.5):
        self.L = L
        self.sigma = sigma
        self.J0 = J0
        self.mu0 = mu0          # non-zero to break particle-hole symmetry
        self.filling = filling
        self.ih = L // 2
        self.n_occ = int(filling * L)
        self.positions = np.arange(L, dtype=float)

    # ── Hamiltonians ─────────────────────────────────────────
    def _h_uniform(self):
        return ham_uniform(self.L, self.J0)

    def _h_inhom(self):
        return ham_inhomogeneous(self.L, self.J0, self.sigma, self.mu0, self.ih)

    def hopping_profile(self):
        return hopping_array(self.L, self.J0, self.sigma, self.ih)

    def potential_profile(self):
        return potential_array(self.L, self.mu0, self.sigma, self.ih)

    # ── Ground state ─────────────────────────────────────────
    def ground_state_corr(self):
        """(C0, evals, evecs) of the uniform-chain ground state."""
        H0 = self._h_uniform()
        ev, U = np.linalg.eigh(H0)
        return corr_from_slater(U, self.n_occ), ev, U

    def ground_state_psi(self):
        """Full 2^L statevector (for verification only)."""
        H0 = self._h_uniform()
        ev, U = np.linalg.eigh(H0)
        return slater_to_statevector(U, self.n_occ, self.L), ev, U

    # ── Time evolution ───────────────────────────────────────
    def evolve(self, C0, t_max, dt, measure_every=1):
        """C(t) = e^{-iht} C(0) e^{+iht}  via single-particle propagator."""
        h = self._h_inhom()
        Udt = expm(-1j * h * dt)
        Udt_dag = Udt.conj().T
        n_steps = int(round(t_max / dt))

        ts = [0.0]
        Cs = [C0.copy()]
        dens = [density_from_corr(C0)]
        ees = [ee_from_corr(C0, self.ih)]
        curs = [current_from_corr(C0)]
        edens = [energy_density_from_corr(C0, self._h_uniform())]

        C = C0.copy()
        h_inhom = h
        for step in range(1, n_steps + 1):
            C = Udt @ C @ Udt_dag
            if step % measure_every == 0 or step == n_steps:
                ts.append(step * dt)
                Cs.append(C.copy())
                dens.append(density_from_corr(C))
                ees.append(ee_from_corr(C, self.ih))
                curs.append(current_from_corr(C))
                edens.append(energy_density_from_corr(C, h_inhom))
        return dict(times=np.array(ts), C=Cs,
                    density=np.array(dens), entropy=np.array(ees),
                    current=np.array(curs), energy_density=np.array(edens))

    # ── Full run ─────────────────────────────────────────────
    def run(self, t_max=20.0, dt=0.05, measure_every=4):
        t0 = clock.time()
        C0, evals, evecs = self.ground_state_corr()
        E_gs = float(np.trace(self._h_uniform() @ C0).real)
        print(f"  Ground state: {self.n_occ} fermions in L={self.L}")
        print(f"  Band: [{evals[0]:.4f}, {evals[-1]:.4f}],  "
              f"E_F={evals[self.n_occ - 1]:.4f},  <H>={E_gs:.4f}")

        res = self.evolve(C0, t_max, dt, measure_every)
        times = res["times"];  dens = res["density"]
        ent   = res["entropy"]; curs = res["current"]
        edens = res["energy_density"]
        print(f"  Evolved {int(t_max/dt)} steps → {len(times)} snapshots "
              f"in {clock.time()-t0:.1f}s")

        excess = dens[-1] - dens[0]
        T_H_ent, slope = extract_hawking_temp_entropy(times, ent)
        T_H_dens = extract_hawking_temp_density(excess, self.positions, self.ih)
        # Prefer entropy-based T_H (more robust for free fermions);
        # the density-based estimate includes classical drift from μ₀.
        T_H = T_H_ent if T_H_ent > 1e-12 else T_H_dens
        c_ext = extract_central_charge(slope, T_H) if T_H > 1e-12 else 0.0
        max_exc = float(np.max(np.abs(excess)))

        print(f"  T_H (entropy rise):  {T_H_ent:.6f}  ← primary")
        print(f"  T_H (density fit):   {T_H_dens:.6f}  (includes classical drift)")
        print(f"  T_H (adopted):       {T_H:.6f}")
        print(f"  Central charge c:    {c_ext:.4f}  (expect ≈ 1.0)")
        print(f"  dS_EE/dt (rise):     {slope:.6f}")
        print(f"  Max |δn_i|:         {max_exc:.6f}")
        print(f"  Max |δε_i|:         {np.max(np.abs(edens[-1]-edens[0])):.6f}")

        return dict(
            L=self.L, sigma=self.sigma, J0=self.J0, mu0=self.mu0,
            filling=self.filling, ih=self.ih, n_occ=self.n_occ,
            t_max=t_max, dt=dt, E_gs=E_gs,
            times=times.tolist(), densities=dens.tolist(),
            entropies=ent.tolist(), currents=curs.tolist(),
            energy_densities=edens.tolist(),
            T_H_density=float(T_H_dens), T_H_entropy=float(T_H_ent),
            T_H=float(T_H), central_charge=float(c_ext),
            dSdt=float(slope), max_density_excess=float(max_exc),
            hopping_profile=self.hopping_profile().tolist(),
            potential_profile=self.potential_profile().tolist(),
            excess_density=excess.tolist(),
            corr_final_real=np.real(res["C"][-1]).tolist(),
            corr_final_imag=np.imag(res["C"][-1]).tolist(),
            corr_delta_norm=float(np.max(np.abs(
                np.array(res["C"][-1]) - np.array(res["C"][0])))),
        )


# ═══════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════

def _save(fig, d, name):
    p = os.path.join(d, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_hopping(sim, d):
    J = sim.hopping_profile(); mu = sim.potential_profile()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    bonds = np.arange(len(J)) + 0.5
    a1.plot(bonds, J, "b-o", ms=3)
    a1.axvline(sim.ih, color="r", ls="--", lw=1, label=f"Horizon i_h={sim.ih}")
    a1.set_ylabel(r"Hopping $J_i$")
    a1.set_title(f"Profile  ($L={sim.L}$, $\\sigma={sim.sigma}$, "
                 f"$J_0={sim.J0}$, $\\mu_0={sim.mu0}$)")
    a1.legend()
    a2.plot(np.arange(sim.L), mu, "g-s", ms=3)
    a2.axvline(sim.ih, color="r", ls="--", lw=1)
    a2.set_xlabel("Site / bond index");  a2.set_ylabel(r"Potential $\mu_i$")
    plt.tight_layout()
    return _save(fig, d, f"hopping_L{sim.L}_sigma{sim.sigma}.png")


def plot_density(sim, r, d, n_snaps=6):
    ts = np.array(r["times"]); dn = np.array(r["densities"]); ih = sim.ih
    idxs = np.linspace(0, len(ts) - 1, n_snaps, dtype=int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for idx in idxs:
        a1.plot(dn[idx], alpha=0.8, label=f"t={ts[idx]:.1f}")
    a1.axvline(ih, color="k", ls="--", lw=1)
    a1.set_xlabel("Site $i$");  a1.set_ylabel(r"$\langle n_i \rangle$")
    a1.set_title("Particle Density Evolution");  a1.legend(fontsize=7, ncol=2)
    exc = dn[-1] - dn[0]
    cols = ["steelblue" if i < ih else "coral" for i in range(sim.L)]
    a2.bar(range(sim.L), exc, color=cols)
    a2.axvline(ih, color="k", ls="--", lw=1)
    a2.set_xlabel("Site $i$");  a2.set_ylabel(r"$\delta n_i$")
    a2.set_title(f"Excess Density at $t={ts[-1]:.1f}$")
    plt.tight_layout()
    return _save(fig, d, f"density_L{sim.L}_sigma{sim.sigma}.png")


def plot_entropy(r, d, label=""):
    ts = np.array(r["times"]); ent = np.array(r["entropies"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, ent, "b-", lw=2, label=r"$S_{\mathrm{EE}}(t)$")
    # mark peak and rising slope
    pk = np.argmax(ent)
    ax.axvline(ts[pk], color="g", ls=":", lw=1, label=f"peak t={ts[pk]:.1f}")
    if pk > 3:
        c = np.polyfit(ts[1:pk+1], ent[1:pk+1], 1)
        ax.plot(ts[1:pk+1], c[0]*ts[1:pk+1]+c[1], "r--",
                label=f"rise slope = {c[0]:.4f}")
    ax.set_xlabel("Time $t$");  ax.set_ylabel(r"$S_{\mathrm{EE}}$")
    ax.set_title(f"Entanglement Entropy {label}");  ax.legend();  ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, f"entropy_L{r['L']}_sigma{r['sigma']}.png")


def plot_correlation(r, d, L):
    C = np.array(r["corr_final_real"]) + 1j * np.array(r["corr_final_imag"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    im = a1.imshow(np.abs(C), cmap="viridis", origin="lower")
    plt.colorbar(im, ax=a1, label=r"$|C_{ij}|$")
    a1.axhline(L//2, color="r", ls="--", lw=.8); a1.axvline(L//2, color="r", ls="--", lw=.8)
    a1.set_xlabel("$j$"); a1.set_ylabel("$i$")
    a1.set_title(r"$|\langle c_i^\dagger c_j\rangle|$")
    # show correlation change (off-diagonal)
    C0_diag = np.eye(L) * 0.5   # initial: half-filling
    dC = np.abs(C - C0_diag)
    np.fill_diagonal(dC, 0)
    im2 = a2.imshow(dC, cmap="magma", origin="lower")
    plt.colorbar(im2, ax=a2, label=r"$|\Delta C_{ij}|$")
    a2.axhline(L//2, color="c", ls="--", lw=.8); a2.axvline(L//2, color="c", ls="--", lw=.8)
    a2.set_xlabel("$j$"); a2.set_ylabel("$i$")
    a2.set_title("Correlation change (off-diagonal)")
    plt.tight_layout()
    return _save(fig, d, f"corr_L{L}_sigma{r['sigma']}.png")


def plot_current(r, d, L):
    ts = np.array(r["times"]); cur = np.array(r["currents"])
    ns = min(6, len(ts)); idxs = np.linspace(0, len(ts)-1, ns, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in idxs:
        ax.plot(cur[idx], alpha=.8, label=f"t={ts[idx]:.1f}")
    ax.axvline(L//2, color="k", ls="--", lw=1)
    ax.set_xlabel("Bond $i$"); ax.set_ylabel("Current $I_i$")
    ax.set_title("Bond Current"); ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, f"current_L{L}_sigma{r['sigma']}.png")


def plot_energy_density(r, d, L, ih):
    ts = np.array(r["times"]); ed = np.array(r["energy_densities"])
    ns = min(6, len(ts)); idxs = np.linspace(0, len(ts)-1, ns, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in idxs:
        ax.plot(ed[idx], alpha=.8, label=f"t={ts[idx]:.1f}")
    ax.axvline(ih, color="k", ls="--", lw=1)
    ax.set_xlabel("Site $i$"); ax.set_ylabel(r"$\varepsilon_i$")
    ax.set_title("Local Energy Density"); ax.legend(fontsize=7); ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, f"energy_density_L{L}_sigma{r['sigma']}.png")


def plot_bogoliubov(L, J0, d):
    ih = L // 2
    k, bn, ba = bogoliubov_step_coefficients(L, J0, ih)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(k/np.pi, bn, "b-", lw=2, label="Numerical")
    a1.plot(k/np.pi, ba, "r--", lw=1.5, label="Analytic")
    a1.set_xlabel(r"$k/\pi$");  a1.set_ylabel(r"$|\beta_k|^2$")
    a1.set_title("Bogoliubov Coefficients (Step Horizon)")
    a1.legend();  a1.grid(True, alpha=.3)
    omega = 2*J0*np.abs(np.cos(k)); si = np.argsort(omega)
    a2.plot(omega[si], bn[si], "b-o", ms=2)
    a2.set_xlabel(r"$\omega_k$");  a2.set_ylabel(r"$|\beta_k|^2$")
    a2.set_title("Spectrum vs Frequency");  a2.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, f"bogoliubov_L{L}.png")


def plot_dispersion(L, J0, sigma, mu0, ih, d):
    eL, eR = dispersion_sides(L, J0, sigma, mu0, ih)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.linspace(0, 1, len(eL)), eL, "b-o", ms=3, label="Left (supersonic)")
    ax.plot(np.linspace(0, 1, len(eR)), eR, "r-s", ms=3, label="Right (subsonic)")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel(r"$k/\pi$");  ax.set_ylabel(r"$\omega(k)$")
    ax.set_title(f"Dispersion ($\\sigma={sigma}$)");  ax.legend();  ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, f"dispersion_sigma{sigma}.png")


def plot_temp_vs_sigma(all_r, d):
    sig = np.array([r["sigma"] for r in all_r])
    tmp = np.array([r["T_H"] for r in all_r])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sig, tmp, "bo-", ms=8, label="Numerical $T_H$")
    v = tmp > 1e-10
    if np.sum(v) > 1:
        c = np.polyfit(1.0/sig[v], tmp[v], 1)
        sf = np.linspace(sig[v].min(), sig[v].max(), 100)
        ax.plot(sf, c[0]/sf+c[1], "r--", label=r"Fit: $T_H \propto 1/\sigma$")
        ax.legend()
    ax.set_xlabel(r"$\sigma$");  ax.set_ylabel(r"$T_H$")
    ax.set_title(r"$T_H$ vs $\sigma$ Scaling");  ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, "T_H_vs_sigma.png")


def plot_central_charge(all_r, d):
    ds = np.array([r["dSdt"] for r in all_r])
    th = np.array([r["T_H"] for r in all_r])
    sg = np.array([r["sigma"] for r in all_r])
    fig, ax = plt.subplots(figsize=(8, 5))
    v = th > 1e-10
    if np.sum(v) > 0:
        ax.plot(th[v], ds[v], "bo-", ms=8)
        for i in range(len(sg)):
            if th[i] > 1e-10:
                ax.annotate(f"σ={sg[i]:.1f}", (th[i], ds[i]),
                            textcoords="offset points", xytext=(5, 5), fontsize=8)
        if np.sum(v) > 1:
            cf = np.polyfit(th[v], ds[v], 1)
            cv = 6*cf[0]/np.pi
            xf = np.linspace(0, th[v].max()*1.15, 100)
            ax.plot(xf, cf[0]*xf+cf[1], "r--", label=f"Fit: c = {cv:.3f}")
            ax.legend()
    ax.set_xlabel(r"$T_H$");  ax.set_ylabel(r"$dS_{\mathrm{EE}}/dt$")
    ax.set_title(r"Central Charge: $dS/dt = \pi\, c\, T_H / 6$")
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, "central_charge.png")


def plot_entropy_comparison(all_r, d):
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in all_r:
        ax.plot(r["times"], r["entropies"], lw=2,
                label=rf"$\sigma={r['sigma']:.1f}$")
    ax.set_xlabel("Time $t$");  ax.set_ylabel(r"$S_{\mathrm{EE}}$")
    ax.set_title("Entropy Growth Comparison");  ax.legend();  ax.grid(True, alpha=.3)
    plt.tight_layout()
    return _save(fig, d, "entropy_comparison.png")


# ═══════════════════════════════════════════════════════════════
#  Verification (small L)
# ═══════════════════════════════════════════════════════════════

def verify_statevector(L_test=8, J0=1.0):
    n = L_test // 2
    H0 = ham_uniform(L_test, J0)
    ev, U = np.linalg.eigh(H0)
    C_ref = corr_from_slater(U, n)
    psi = slater_to_statevector(U, n, L_test)
    C_sv = correlation_from_statevector(psi, L_test)
    err = np.max(np.abs(C_ref - C_sv))
    nm = np.linalg.norm(psi)
    S_c = ee_from_corr(C_ref, n)
    S_s = ee_from_statevector(psi, L_test, n)
    print(f"  L={L_test}: |ψ|={nm:.12f}, max|ΔC|={err:.2e}")
    print(f"    <N>={np.trace(C_ref).real:.6f} (exp {n}), "
          f"<H>={np.trace(H0@C_ref).real:.6f}")
    print(f"    S_EE(corr)={S_c:.6f}, S_EE(ψ)={S_s:.6f}, ΔS={abs(S_c-S_s):.2e}")
    return err < 1e-8


def verify_metric(sim):
    J = sim.hopping_profile()
    kF = np.pi * sim.filling
    vl = 2*J[0]*np.sin(kF) if len(J) > 0 else 0
    vr = 2*J[-1]*np.sin(kF) if len(J) > 0 else 0
    kappa = abs(vl - vr) / sim.sigma if sim.sigma > 0 else 0
    print(f"  v_left={vl:.4f}, v_right={vr:.4f}")
    print(f"  κ ≈ {kappa:.4f},  T_H(κ/2π) ≈ {kappa/(2*np.pi):.4f}")
    return kappa


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def _json_clean(obj):
    """Recursively convert numpy / complex types for JSON serialisation."""
    if isinstance(obj, (complex, np.complexfloating)):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, (np.floating, np.integer, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(
        description="Sim7 — Sonic Black Hole: Hawking Radiation "
                    "in a 1D Fermionic Qubit Chain")
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--J0", type=float, default=1.0)
    ap.add_argument("--mu0", type=float, default=0.5,
                    help="On-site potential (breaks particle-hole symmetry)")
    ap.add_argument("--filling", type=float, default=0.5)
    ap.add_argument("--t-max", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--output-dir", type=str, default="sim7_results")
    ap.add_argument("--sigma-sweep", action="store_true")
    ap.add_argument("--measure-every", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    t_start = clock.time()

    print("=" * 72)
    print("  SIM 7:  Analog Gravity — Sonic Black Hole Hawking Radiation")
    print("=" * 72)
    print(f"  L={a.L},  σ={a.sigma},  J₀={a.J0},  μ₀={a.mu0}")
    print(f"  filling={a.filling},  t_max={a.t_max},  dt={a.dt}")
    print(f"  output → {a.output_dir}/\n")

    # Phase 0 — statevector verification
    print("─" * 52)
    print("  Phase 0 — Jordan-Wigner Statevector Verification")
    print("─" * 52)
    Lv = min(a.L, 10)
    if Lv >= 4:
        ok = verify_statevector(Lv, a.J0)
        print(f"  ► {'PASS ✓' if ok else 'FAIL ✗'}\n")

    # Phase 1 — Bogoliubov + dispersion
    print("─" * 52)
    print("  Phase 1 — Bogoliubov & Dispersion Analysis")
    print("─" * 52)
    plot_bogoliubov(a.L, a.J0, a.output_dir)
    sim0 = SonicBlackHole(a.L, a.sigma, a.J0, a.mu0, a.filling)
    plot_dispersion(a.L, a.J0, a.sigma, a.mu0, sim0.ih, a.output_dir)
    print("  ► Bogoliubov & dispersion plots saved.\n")

    # Phase 2 — time evolution
    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0] if a.sigma_sweep else [a.sigma]
    all_res = []
    for sigma in sigmas:
        print("─" * 52)
        print(f"  Phase 2 — Time Evolution  (σ = {sigma})")
        print("─" * 52)
        sim = SonicBlackHole(a.L, sigma, a.J0, a.mu0, a.filling)
        kappa = verify_metric(sim); print()
        r = sim.run(a.t_max, a.dt, a.measure_every)
        r["kappa"] = float(kappa)
        all_res.append(r)

        plot_hopping(sim, a.output_dir)
        plot_density(sim, r, a.output_dir)
        plot_entropy(r, a.output_dir, label=f"(σ={sigma})")
        plot_correlation(r, a.output_dir, a.L)
        plot_current(r, a.output_dir, a.L)
        plot_energy_density(r, a.output_dir, a.L, sim.ih)
        print(f"  ► Plots saved.\n")

    # Phase 3 — scaling
    if len(sigmas) > 1:
        print("─" * 52)
        print("  Phase 3 — Scaling Analysis")
        print("─" * 52)
        plot_temp_vs_sigma(all_res, a.output_dir)
        plot_central_charge(all_res, a.output_dir)
        plot_entropy_comparison(all_res, a.output_dir)
        print("  ► Scaling plots saved.\n")
        hdr = f"  {'σ':>6s} {'T_H':>10s} {'c':>8s} {'dS/dt':>10s} {'max|δn|':>10s}"
        print(hdr);  print("  " + "─"*48)
        for r in all_res:
            print(f"  {r['sigma']:6.2f} {r['T_H']:10.6f} {r['central_charge']:8.4f} "
                  f"{r['dSdt']:10.6f} {r['max_density_excess']:10.6f}")
    else:
        r = all_res[0]
        print("─" * 52);  print("  Final Results");  print("─" * 52)
        print(f"  T_H        = {r['T_H']:.6f}")
        print(f"  c          = {r['central_charge']:.4f}")
        print(f"  dS/dt      = {r['dSdt']:.6f}")
        print(f"  max |δn|   = {r['max_density_excess']:.6f}")
        print(f"  κ          = {r.get('kappa',0):.4f}")

    # Save JSON
    out = dict(parameters=dict(L=a.L, J0=a.J0, mu0=a.mu0,
                               filling=a.filling, t_max=a.t_max, dt=a.dt,
                               sigma_sweep=a.sigma_sweep),
               results=_json_clean(all_res))
    jp = os.path.join(a.output_dir, "sim7_results.json")
    with open(jp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ► JSON saved: {jp}")

    print(f"\n{'=' * 72}")
    print(f"  Done.  Wall time: {clock.time()-t_start:.1f}s")
    print(f"{'=' * 72}")
    return all_res


if __name__ == "__main__":
    main()
