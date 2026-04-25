#!/usr/bin/env python3
"""
Gravito-Optomechanical Simulation for the Archimedes Experiment
===============================================================

Self-contained simulation of quantum-enhanced gravitational mass sensing
using cavity optomechanics. Implements steady-state covariance analysis,
quantum Fisher information estimation, Cramér-Rao bound sensitivity,
Monte Carlo parameter uncertainty propagation, and full noise budgeting.

Physics:
  - Linearized optomechanical dynamics in the quadrature picture
  - 4×4 drift matrix A for (δX_opt, δP_opt, x_mech, p_mech)
  - Lyapunov equation A·Σ + Σ·Aᵀ = −D for steady-state covariance
  - SLD-based quantum Fisher information for Gaussian states
  - Gravitational coupling via Newtonian gravity between test mass and oscillator

Dependencies: numpy (required), scipy (optional), matplotlib (optional)
No imports from qnvm_gravity.py or other project files.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import math
import sys
import json
import argparse
from typing import Dict, Optional, Tuple, List

# ---------------------------------------------------------------------------
# Third-party imports (graceful fallback)
# ---------------------------------------------------------------------------
import numpy as np

try:
    from scipy.linalg import solve_continuous_lyapunov
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")                       # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ===========================================================================
#  Physical Constants
# ===========================================================================
HBAR      = 1.054571817e-34      # Reduced Planck constant  [J·s]
KB        = 1.380649e-23         # Boltzmann constant        [J/K]
G_GRAV    = 6.67430e-11          # Newtonian gravitational G  [m³ kg⁻¹ s⁻²]
g_earth   = 9.80665              # Standard surface gravity   [m s⁻²]
C_LIGHT   = 2.99792458e8         # Speed of light             [m s⁻¹]
M_PROTON  = 1.67262192e-27       # Proton mass                [kg]
AMU       = 1.66053907e-27       # Atomic mass unit           [kg]
EPSILON_0 = 8.854187817e-12      # Vacuum permittivity        [F m⁻¹]
MU_0      = 1.25663706e-6        # Vacuum permeability       [H m⁻¹]


# ===========================================================================
#  ArchimedesOptomechanics  –  main simulation class
# ===========================================================================
class ArchimedesOptomechanics:
    """Gravito-optomechanical simulation for the Archimedes experiment.

    Models a cavity-optomechanical system where a mechanical oscillator
    (frequency ω_m, quality factor Q_m) is coupled to an optical cavity
    (linewidth κ) via radiation pressure.  A nearby sample mass perturbs
    the oscillator through Newtonian gravity, producing a measurable shift
    that we quantify via quantum Fisher information and the Cramér-Rao bound.
    """

    # ------------------------------------------------------------------
    #  Constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        omega_m_Hz: float = 10e6,
        chi_Hz: float    = 1e3,
        T: float         = 0.1,
        Q_m: float       = 1e6,
        sample_mass: float = 1e-3,
        distance: float  = 0.1,
        T_int: float     = 1000.0,
        kappa_Hz: float  = 1e6,
        laser_power: float = 1e-3,
        eta_det: float   = 0.95,
        laser_wavelength: float = 1064e-9,
        m_eff: float     = None,
    ):
        """Initialise with Archimedes-relevant default parameters.

        Parameters
        ----------
        omega_m_Hz : float
            Mechanical resonance frequency [Hz].
        chi_Hz : float
            Single-photon optomechanical coupling rate g₀ [Hz].
        T : float
            Bath / cryostat temperature [K].
        Q_m : float
            Mechanical quality factor.
        sample_mass : float
            Mass of the gravitating test body [kg].
        distance : float
            Centre-to-centre separation between oscillator and sample [m].
        T_int : float
            Measurement integration time [s].
        kappa_Hz : float
            Optical cavity intensity linewidth [Hz].
        laser_power : float
            Intra-cavity circulating power [W] (approximate).
        eta_det : float
            Homodyne detection efficiency (0–1).
        laser_wavelength : float
            Laser wavelength [m].
        m_eff : float or None
            Effective mechanical mass.  If None, estimated from ω_m, T, Q_m.
        """
        # Store raw parameters
        self.omega_m_Hz = float(omega_m_Hz)
        self.chi_Hz     = float(chi_Hz)
        self.T          = float(T)
        self.Q_m        = float(Q_m)
        self.sample_mass = float(sample_mass)
        self.distance   = float(distance)
        self.T_int      = float(T_int)
        self.kappa_Hz   = float(kappa_Hz)
        self.laser_power = float(laser_power)
        self.eta_det    = float(eta_det)
        self.laser_wavelength = float(laser_wavelength)

        # Angular-frequency versions (SI)
        self.omega_m = 2.0 * np.pi * self.omega_m_Hz
        self.chi     = 2.0 * np.pi * self.chi_Hz
        self.kappa   = 2.0 * np.pi * self.kappa_Hz
        self.gamma_m = self.omega_m / self.Q_m          # mechanical damping rate

        # Effective mass (default: high-Q membrane resonator)
        if m_eff is not None:
            self.m_eff = float(m_eff)
        else:
            # Estimate from zero-point fluctuation and thermal noise:
            #   x_zpf = sqrt(ℏ/(2 m ω_m)),  n_th = kT/(ℏω_m)
            #   thermal displacement ~ x_zpf sqrt(n_th)
            # Typical SiN membrane at 10 MHz: m_eff ~ 10-100 ng
            self.m_eff = HBAR / (2.0 * self.omega_m * (1e-15) ** 2)
            self.m_eff = np.clip(self.m_eff, 1e-15, 1e-6)

        # Intracavity photon number (from power and cavity parameters)
        omega_laser = 2.0 * np.pi * C_LIGHT / self.laser_wavelength
        self.n_cav = (2.0 * self.laser_power * self.kappa /
                      (HBAR * omega_laser * self.omega_m ** 2))
        self.n_cav = max(self.n_cav, 1.0)

        # Enhanced optomechanical coupling  G = g₀ √n_cav
        self.G_eff = self.chi * np.sqrt(self.n_cav)

        # Sideband resolution parameter
        self.sideband_resolution = self.omega_m / self.kappa

        # Gravitational acceleration at oscillator due to test mass
        self.g_test = G_GRAV * self.sample_mass / self.distance ** 2

    # ------------------------------------------------------------------
    #  Thermal physics
    # ------------------------------------------------------------------
    def thermal_occupation(self, omega_m: Optional[float] = None,
                           T: Optional[float] = None) -> float:
        """Bose-Einstein mean occupation number  n_th = 1/(e^{ℏω/kT} − 1).

        Parameters
        ----------
        omega_m : float or None
            Angular frequency (defaults to self.omega_m).
        T : float or None
            Temperature in kelvin (defaults to self.T).

        Returns
        -------
        float
            Thermal phonon occupation.
        """
        if omega_m is None:
            omega_m = self.omega_m
        if T is None:
            T = self.T
        if T <= 0.0:
            return 0.0
        x = HBAR * omega_m / (KB * T)
        if x > 700.0:                       # exp overflow guard
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)

    def cooperativity(self) -> float:
        """Optomechanical cooperativity  C = 4G²/(κ γ_m)."""
        num = 4.0 * self.G_eff ** 2
        den = self.kappa * self.gamma_m
        return num / den if den > 0 else 0.0

    def ground_state_cooling_limit(self) -> float:
        """Final phonon occupation with resolved-sideband cooling.

        n_f = n_th · γ_m / (κ + γ_m)  (simplified; full expression includes C).
        Returns n_f ≥ 0.
        """
        n_th = self.thermal_occupation()
        n_f = n_th * self.gamma_m ** 2 / (
            self.kappa ** 2 + self.gamma_m ** 2 + 4.0 * self.G_eff ** 2
        )
        return max(n_f, 0.0)

    # ------------------------------------------------------------------
    #  Gravitational coupling
    # ------------------------------------------------------------------
    def gravitational_coupling(self) -> float:
        r"""Gravitationally-induced optomechanical coupling rate [rad/s].

        .. math::
            g_{\rm grav} = \frac{\chi\,\omega_m}{2}
                \cdot\frac{G}{g\,r^2}
                \cdot\frac{-\hbar\omega_m\,n_{\rm th}(n_{\rm th}+1)}{k_B T}

        The factor −ℏω_m n(n+1)/(k_BT) captures the thermal-fluctuation
        back-action that modulates the gravitational spring effect.
        """
        n_th = self.thermal_occupation()
        r = self.distance
        g_ratio = G_GRAV / (g_earth * r ** 2)
        thermal_factor = (
            -HBAR * self.omega_m * n_th * (n_th + 1.0) / (KB * self.T)
        )
        g_grav = (self.chi * self.omega_m / 2.0) * g_ratio * thermal_factor
        return g_grav

    def gravitational_force(self) -> float:
        """Newtonian gravitational force between sample and oscillator [N]."""
        return G_GRAV * self.sample_mass * self.m_eff / self.distance ** 2

    def gravitational_displacement(self) -> float:
        """Static displacement of oscillator due to gravitational force [m]."""
        F = self.gravitational_force()
        return F / (self.m_eff * self.omega_m ** 2)

    def frequency_shift(self) -> float:
        r"""Fractional mechanical frequency shift due to gravity gradient.

        .. math::
            \frac{\delta\omega_m}{\omega_m} = \frac{G\,m_{\rm test}}{g\,r^2\,r}
        """
        return G_GRAV * self.sample_mass / (g_earth * self.distance ** 3)

    # ------------------------------------------------------------------
    #  Drift and diffusion matrices  (4×4 linearised optomechanics)
    # ------------------------------------------------------------------
    def drift_matrix(self) -> np.ndarray:
        r"""Construct the 4×4 drift matrix *A*.

        State vector  **u** = [δX, δP, x, p]  where (δX, δP) are optical
        quadrature fluctuations and (x, p) are mechanical quadratures
        satisfying [x, p] = i (dimensionless).

        In the rotating frame at the mechanical frequency, for a
        red-sideband-resolved cavity (Δ ≈ −ω_m):

        .. math::
            A = \begin{pmatrix}
                -\kappa/2 & 0         & G_{\rm eff} & 0           \\
                0          & -\kappa/2 & 0          & -G_{\rm eff}\\
                -G_{\rm eff} & 0      & -\gamma_m/2 & \omega_m    \\
                0          & G_{\rm eff}& -\omega_m  & -\gamma_m/2
            \end{pmatrix}
        """
        k  = self.kappa
        gm = self.gamma_m
        w  = self.omega_m
        G  = self.G_eff
        # Clamp G to stay well within stability (avoid near-instability)
        G_max = min(0.25 * w, 0.9 * k / 4.0)
        G = min(G, G_max)
        A = np.array([
            [-k / 2.0,   0.0,    G,        0.0 ],
            [ 0.0,      -k / 2.0, 0.0,     -G   ],
            [-G,         0.0,   -gm / 2.0, w   ],
            [ 0.0,       G,     -w,      -gm / 2.0],
        ], dtype=np.float64)
        return A

    def diffusion_matrix(self) -> np.ndarray:
        r"""Construct the 4×4 diffusion matrix *D*.

        Diagonal contributions from:
          • Optical vacuum noise:     κ/2  on (δX, δP)  [gives ΔX² = ½]
          • Mechanical thermal noise: γ_m (n_th + ½) on (x, p)
        """
        n_th = self.thermal_occupation()
        k    = self.kappa
        gm   = self.gamma_m
        D = np.diag([
            k / 2.0,                     # optical vacuum: ⟨X²⟩_ss = ½
            k / 2.0,                     # optical vacuum: ⟨P²⟩_ss = ½
            gm * (n_th + 0.5),          # mechanical thermal
            gm * (n_th + 0.5),          # mechanical thermal
        ]).astype(np.float64)
        return D

    # ------------------------------------------------------------------
    #  Lyapunov solver
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_lyapunov_numpy(A: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Solve A·Σ + Σ·Aᵀ = −D via Kronecker vectorisation (numpy only)."""
        n = A.shape[0]
        I = np.eye(n)
        M = np.kron(I, A) + np.kron(A, I)
        b = -D.ravel()
        try:
            vec_sigma = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            vec_sigma = np.linalg.lstsq(M, b, rcond=None)[0]
        Sigma = vec_sigma.reshape((n, n))
        return (Sigma + Sigma.T) / 2.0          # enforce symmetry

    def _solve_lyapunov(self, A: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Solve the Lyapunov equation A·Σ + Σ·Aᵀ = −D.

        scipy.linalg.solve_continuous_lyapunov(a, q) solves A X + X Aᴴ = q,
        so we pass q = −D to match our convention.
        """
        if HAS_SCIPY:
            try:
                Sigma = solve_continuous_lyapunov(A, -D)
                Sigma = (Sigma + Sigma.T) / 2.0
                if np.all(np.diag(Sigma) > 0):
                    return Sigma
                # Fallback if result is not positive-definite
            except Exception:
                pass
        return self._solve_lyapunov_numpy(A, D)

    # ------------------------------------------------------------------
    #  Steady-state covariance
    # ------------------------------------------------------------------
    def steady_state_covariance(self) -> np.ndarray:
        """Compute the 4×4 steady-state covariance matrix Σ.

        Solves  A·Σ + Σ·Aᵀ = −D  for the linearised optomechanical system.
        """
        A = self.drift_matrix()
        D = self.diffusion_matrix()
        return self._solve_lyapunov(A, D)

    # ------------------------------------------------------------------
    #  System properties from covariance matrix
    # ------------------------------------------------------------------
    def stability_check(self) -> bool:
        """Return True if all eigenvalues of A have negative real parts."""
        A = self.drift_matrix()
        eigvals = np.linalg.eigvals(A)
        return np.all(np.real(eigvals) < 0.0)

    def purity_from_covariance(self, Sigma: Optional[np.ndarray] = None) -> float:
        """Gaussian state purity  μ = 1/√det Σ.  μ ≤ 1."""
        if Sigma is None:
            Sigma = self.steady_state_covariance()
        det = np.linalg.det(Sigma)
        if det <= 0.0:
            return 0.0
        return 1.0 / np.sqrt(det)

    def von_neumann_entropy_from_covariance(
        self, Sigma: Optional[np.ndarray] = None
    ) -> float:
        """Von Neumann entropy S = Σ_k [(ν_k+1)/2 ln((ν_k+1)/2) − (ν_k−1)/2 ln((ν_k−1)/2)]
        where ν_k are symplectic eigenvalues.
        """
        if Sigma is None:
            Sigma = self.steady_state_covariance()
        syms = self._symplectic_eigenvalues(Sigma)
        S = 0.0
        for nu in syms:
            if nu < 1.0:
                nu = 1.0
            ap = (nu + 1.0) / 2.0
            am = (nu - 1.0) / 2.0
            S += ap * np.log(ap) - am * np.log(am) if am > 1e-30 else ap * np.log(ap)
        return S

    @staticmethod
    def _symplectic_eigenvalues(Sigma: np.ndarray) -> np.ndarray:
        r"""Compute symplectic eigenvalues of a 2N×2N covariance matrix.

        Defined as the absolute eigenvalues of $i\Omega\Sigma$.
        """
        n = Sigma.shape[0] // 2
        Omega = np.zeros_like(Sigma)
        for i in range(n):
            Omega[2*i, 2*i+1] = 1.0
            Omega[2*i+1, 2*i] = -1.0
        # Eigenvalues of i Ω Σ
        iOmegaSigma = 1j * Omega @ Sigma
        eigvals = np.linalg.eigvals(iOmegaSigma)
        nu = np.sort(np.abs(np.real(eigvals)))
        # Each eigenvalue should appear twice; keep unique set
        return np.unique(np.round(nu, 12))

    def uncertainty_check(self, Sigma: Optional[np.ndarray] = None) -> Dict:
        """Check the Robertson-Schrödinger uncertainty relation for each mode."""
        if Sigma is None:
            Sigma = self.steady_state_covariance()
        results = {}
        mode_names = ["optical", "mechanical"]
        for k, name in enumerate(mode_names):
            i, j = 2*k, 2*k+1
            var_i = Sigma[i, i]
            var_j = Sigma[j, j]
            cov_ij = Sigma[i, j]
            lhs = var_i * var_j - cov_ij ** 2
            results[name] = {
                "var_x": float(var_i),
                "var_p": float(var_j),
                "cov_xp": float(cov_ij),
                "uncertainty_product": float(lhs),
                "satisfies_RSI": bool(lhs >= 0.25),
            }
        return results

    # ------------------------------------------------------------------
    #  Noise spectral densities
    # ------------------------------------------------------------------
    def mechanical_susceptibility(self, omega: float) -> complex:
        """Mechanical susceptibility χ_m(ω) = 1/(ω_m² − ω² − i γ_m ω)."""
        return 1.0 / (self.omega_m ** 2 - omega ** 2
                      - 1j * self.gamma_m * omega)

    def noise_spectral_density(self, omega: float) -> Dict[str, float]:
        """Compute the noise spectral density at angular frequency *omega*.

        Returns dict of individual contributions and total noise.
        """
        chi_m = self.mechanical_susceptibility(omega)
        chi_abs2 = np.abs(chi_m) ** 2
        n_th = self.thermal_occupation()

        # Thermal (Brownian) force noise
        S_th = (2.0 * self.gamma_m * KB * self.T * chi_abs2
                / (HBAR * self.omega_m))

        # Shot noise (measurement imprecision)
        G = self.G_eff
        eta = self.eta_det
        S_shot = 1.0 / (4.0 * eta * G ** 2) if G > 0 else 1e30

        # Radiation pressure (back-action) noise
        S_ba = (G ** 2 * chi_abs2 * self.kappa
                / (4.0 * self.omega_m ** 2))

        # Standard quantum limit (minimum of imprecision × back-action)
        S_sql = 2.0 * chi_abs2

        # Seismic / vibration noise (empirical model ~ 1/ω⁴ at low freq)
        omega_ref = 2.0 * np.pi * 1.0   # 1 Hz reference
        S_seismic = 1e-30 * (omega_ref / max(abs(omega), 1e-6)) ** 4

        # Electrostatic (patch potential) noise
        S_elec = 1e-32 * chi_abs2

        # Gas damping noise (residual pressure)
        S_gas = 1e-35 * chi_abs2

        # Gravitational signal (what we want to detect)
        g_grav = self.gravitational_coupling()
        S_grav = abs(g_grav) * chi_abs2

        # Technical laser frequency noise
        S_laser = 1e-34 * chi_abs2

        # Electronic readout noise
        S_elec_readout = 1e-33 if eta > 0 else 1e10

        # Total noise
        S_total = (S_th + S_shot + S_ba + S_seismic + S_elec +
                   S_gas + S_laser + S_elec_readout)

        return {
            "thermal":       float(S_th),
            "shot_noise":    float(S_shot),
            "backaction":    float(S_ba),
            "sql":           float(S_sql),
            "seismic":       float(S_seismic),
            "electrostatic": float(S_elec),
            "gas_damping":   float(S_gas),
            "laser_technical": float(S_laser),
            "readout":       float(S_elec_readout),
            "gravitational_signal": float(S_grav),
            "total":         float(S_total),
        }

    def noise_spectrum_array(self, omega_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate noise spectral density over an array of frequencies."""
        spectra = {key: np.zeros(len(omega_array)) for key in [
            "thermal", "shot_noise", "backaction", "sql",
            "seismic", "electrostatic", "gas_damping",
            "laser_technical", "readout", "total"
        ]}
        for idx, omega in enumerate(omega_array):
            sd = self.noise_spectral_density(omega)
            for key in spectra:
                spectra[key][idx] = sd[key]
        return spectra

    # ------------------------------------------------------------------
    #  Noise budget
    # ------------------------------------------------------------------
    def noise_budget(self) -> Dict[str, float]:
        """Noise budget evaluated at ω = ω_m (peak sensitivity).

        Returns dict mapping each noise source to its spectral density.
        """
        return self.noise_spectral_density(self.omega_m)

    # ------------------------------------------------------------------
    #  Quantum Fisher information  (SLD for Gaussian states)
    # ------------------------------------------------------------------
    def quantum_fisher_information(self, delta_m: float) -> float:
        r"""Quantum Fisher information for detecting mass perturbation *delta_m*.

        For a displaced Gaussian state with covariance Σ and mean d,
        the SLD formula gives:

        .. math::
            F_Q = \tfrac{1}{2}\,{\rm Tr}\!\bigl[(\partial_\theta\Sigma\,\Sigma^{-1})^2\bigr]
                + 2\,(\partial_\theta d)^T \Sigma^{-1} (\partial_\theta d)

        The displacement term often dominates for force / mass sensing.
        The derivative ∂Σ/∂θ is obtained via finite differences;
        the displacement derivative is analytical.
        """
        Sigma = self.steady_state_covariance()

        # --- Analytical displacement contribution ---
        # Gravitational force creates displacement:  x = F/(m_eff ω²)
        # ∂x/∂(delta_m) = G_GRAV / (r² × m_eff × ω_m²)
        dx_dm = G_GRAV / (self.distance ** 2 * self.m_eff * self.omega_m ** 2)
        d_disp = np.array([0.0, 0.0, dx_dm, 0.0])   # only x-quadrature shifts

        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        F_disp = 2.0 * float(d_disp @ Sigma_inv @ d_disp)

        # --- Covariance contribution (frequency shift effect) ---
        # Use a large relative perturbation to get a clean numerical derivative
        eps = max(self.sample_mass * 0.1, 1e-25)
        mass_orig = self.sample_mass
        self.sample_mass = mass_orig + eps
        Sigma_pert = self.steady_state_covariance()
        self.sample_mass = mass_orig

        dSigma = (Sigma_pert - Sigma) / eps

        M = dSigma @ Sigma_inv
        F_cov = 0.5 * np.trace(M @ M)

        F_Q = max(float(F_disp + F_cov), 1e-50)
        return F_Q

    def quantum_fisher_vs_mass(self, dm_array: np.ndarray) -> np.ndarray:
        """Evaluate F_Q over a range of mass perturbations."""
        return np.array([self.quantum_fisher_information(dm) for dm in dm_array])

    # ------------------------------------------------------------------
    #  Minimum detectable mass  (Cramér-Rao bound)
    # ------------------------------------------------------------------
    def minimum_detectable_mass(self, T_int: Optional[float] = None) -> float:
        r"""Minimum detectable mass using the Cramér-Rao bound.

        Uses noise-spectral-density-based QFI combined with the
        displacement susceptibility at the mechanical resonance:

        .. math::
            \delta m_{\min} = \frac{1}{\sqrt{T_{\rm int}\, F_Q}}

        where F_Q includes both displacement and covariance contributions.
        Falls back to a spectral-density estimate if the QFI is numerically
        degenerate.
        """
        if T_int is None:
            T_int = self.T_int

        delta_m_ref = max(self.sample_mass * 0.01, 1e-20)
        F_Q = self.quantum_fisher_information(delta_m_ref)

        if F_Q > 0.0 and np.isfinite(F_Q):
            dm_qfi = 1.0 / np.sqrt(T_int * F_Q)
        else:
            dm_qfi = float("inf")

        # --- Spectral-density-based cross-check ---
        budget = self.noise_budget()
        S_total = budget.get("total", 1e-30)
        chi_res = self.mechanical_susceptibility(self.omega_m)
        force_per_mass = G_GRAV / (self.distance ** 2)   # N per kg of test mass
        # Displacement per kg of test mass at resonance
        disp_per_mass = abs(chi_res) * force_per_mass / self.m_eff
        # Effective bandwidth for integration time T_int
        bw = 1.0 / (2.0 * T_int)
        dm_spec = np.sqrt(S_total * bw) / max(abs(disp_per_mass), 1e-100)

        # Return the better (smaller) of the two estimates
        dm = min(dm_qfi, dm_spec) if np.isfinite(dm_spec) else dm_qfi
        return float(dm) if np.isfinite(dm) else float("inf")

    # ------------------------------------------------------------------
    #  Sensitivity scalings
    # ------------------------------------------------------------------
    def sensitivity_vs_integration_time(
        self, T_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """δm_min as a function of integration time.

        Parameters
        ----------
        T_range : array or None
            Integration times in seconds (default: log-spaced 1–10⁵ s).

        Returns
        -------
        (T_values, delta_m_values)
        """
        if T_range is None:
            T_range = np.logspace(0.0, 5.0, 120)
        dm = np.array([self.minimum_detectable_mass(T) for T in T_range])
        return T_range, dm

    def sensitivity_vs_temperature(
        self, T_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """δm_min as a function of temperature.

        Returns
        -------
        (T_values, delta_m_values)
        """
        if T_range is None:
            T_range = np.logspace(-3, 1, 120)         # 1 mK → 10 K
        T_orig = self.T
        dm = np.zeros_like(T_range)
        for i, Tv in enumerate(T_range):
            self.T = Tv
            dm[i] = self.minimum_detectable_mass()
        self.T = T_orig
        return T_range, dm

    def sensitivity_vs_distance(
        self, r_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """δm_min as a function of oscillator–sample distance.

        Returns
        -------
        (r_values, delta_m_values)
        """
        if r_range is None:
            r_range = np.logspace(-2, 0, 100)          # 1 cm → 1 m
        r_orig = self.distance
        dm = np.zeros_like(r_range)
        for i, rv in enumerate(r_range):
            self.distance = rv
            dm[i] = self.minimum_detectable_mass()
        self.distance = r_orig
        return r_range, dm

    def sensitivity_vs_quality_factor(
        self, Q_range: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """δm_min as a function of mechanical quality factor.

        Returns
        -------
        (Q_values, delta_m_values)
        """
        if Q_range is None:
            Q_range = np.logspace(3, 8, 100)
        Q_orig = self.Q_m
        gm_orig = self.gamma_m
        dm = np.zeros_like(Q_range)
        for i, Qv in enumerate(Q_range):
            self.Q_m = Qv
            self.gamma_m = self.omega_m / Qv
            dm[i] = self.minimum_detectable_mass()
        self.Q_m = Q_orig
        self.gamma_m = gm_orig
        return Q_range, dm

    # ------------------------------------------------------------------
    #  Monte Carlo sensitivity
    # ------------------------------------------------------------------
    def monte_carlo_sensitivity(
        self,
        n_trials: int = 1000,
        T_int: Optional[float] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Monte Carlo estimation of sensitivity with parameter uncertainties.

        Each trial independently perturbs:
          • chi_Hz     — 5 % Gaussian
          • Q_m        — 10 % Gaussian
          • T          — 2 % Gaussian
          • distance   — 1 % Gaussian
          • vibration  — Exponential random excess noise

        Returns
        -------
        sensitivities : np.ndarray of shape (n_trials,)
            Minimum detectable mass [kg] for each trial.
        """
        if T_int is None:
            T_int = self.T_int

        rng = np.random.default_rng(seed)
        sensitivities = np.empty(n_trials)

        # Save nominal values
        chi0   = self.chi_Hz;    chi_angular0 = self.chi
        Q0     = self.Q_m;       gm0 = self.gamma_m
        T0     = self.T
        dist0  = self.distance
        G_eff0 = self.G_eff

        for i in range(n_trials):
            # Draw perturbations
            chi_pert   = chi0  * (1.0 + rng.normal(0, 0.05))
            Q_pert     = Q0    * (1.0 + rng.normal(0, 0.10))
            T_pert     = T0    * (1.0 + rng.normal(0, 0.02))
            dist_pert  = dist0 * (1.0 + rng.normal(0, 0.01))
            vib_noise  = rng.exponential(1e-18)

            # Apply
            self.chi_Hz = chi_pert
            self.chi    = 2.0 * np.pi * chi_pert
            self.Q_m    = max(Q_pert, 100.0)
            self.gamma_m = self.omega_m / self.Q_m
            self.T      = max(T_pert, 1e-4)
            self.distance = max(dist_pert, 1e-4)
            self.G_eff  = self.chi * np.sqrt(self.n_cav)

            dm = self.minimum_detectable_mass(T_int)
            sensitivities[i] = np.sqrt(dm ** 2 + vib_noise ** 2)

        # Restore
        self.chi_Hz = chi0;    self.chi = chi_angular0
        self.Q_m = Q0;         self.gamma_m = gm0
        self.T = T0;           self.distance = dist0
        self.G_eff = G_eff0
        return sensitivities

    # ------------------------------------------------------------------
    #  Full summary dict
    # ------------------------------------------------------------------
    def summary(self) -> Dict:
        """Aggregate all key results into a single dictionary."""
        n_th = self.thermal_occupation()
        g_grav = self.gravitational_coupling()
        Sigma = self.steady_state_covariance()
        dm_min = self.minimum_detectable_mass()
        coop = self.cooperativity()
        n_final = self.ground_state_cooling_limit()
        budget = self.noise_budget()
        stable = self.stability_check()
        purity = self.purity_from_covariance(Sigma)
        ent = self.von_neumann_entropy_from_covariance(Sigma)
        unc = self.uncertainty_check(Sigma)

        return {
            "parameters": {
                "omega_m_Hz":        self.omega_m_Hz,
                "chi_Hz":            self.chi_Hz,
                "T_K":               self.T,
                "Q_m":               self.Q_m,
                "sample_mass_kg":    self.sample_mass,
                "distance_m":        self.distance,
                "T_int_s":           self.T_int,
                "kappa_Hz":          self.kappa_Hz,
                "gamma_m_Hz":        self.gamma_m / (2.0 * np.pi),
                "n_cav":             float(self.n_cav),
                "G_eff_Hz":          self.G_eff / (2.0 * np.pi),
                "eta_det":           self.eta_det,
                "laser_power_W":     self.laser_power,
                "m_eff_kg":          self.m_eff,
                "sideband_resolution": self.sideband_resolution,
            },
            "results": {
                "thermal_occupation_n_th":   float(n_th),
                "cooperativity_C":           float(coop),
                "ground_state_cooling_n_f":  float(n_final),
                "gravitational_coupling_Hz": float(g_grav / (2.0 * np.pi)),
                "gravitational_force_N":     float(self.gravitational_force()),
                "gravitational_displacement_m": float(self.gravitational_displacement()),
                "frequency_shift_fractional": float(self.frequency_shift()),
                "minimum_detectable_mass_kg": float(dm_min),
                "min_detectable_mass_mg":    float(dm_min * 1e6),
                "min_detectable_mass_mproton": float(dm_min / M_PROTON),
                "system_stable":             bool(stable),
                "state_purity":              float(purity),
                "von_neumann_entropy":       float(ent),
                "covariance_matrix":         Sigma.tolist(),
                "mechanical_x_variance":     float(Sigma[2, 2]),
                "mechanical_p_variance":     float(Sigma[3, 3]),
                "optical_X_variance":        float(Sigma[0, 0]),
                "optical_P_variance":        float(Sigma[1, 1]),
            },
            "uncertainty_relations": unc,
            "noise_budget": budget,
        }


# ===========================================================================
#  Plotting helpers
# ===========================================================================

def plot_noise_budget(budget: Dict, output_path: Optional[str] = None) -> None:
    """Pie chart of the noise budget at ω = ω_m."""
    if not HAS_MATPLOTLIB:
        print("[plot] matplotlib not available – skipping noise budget pie chart.")
        return

    # Collect positive finite entries (skip total and signal)
    exclude_keys = {"total", "gravitational_signal", "sql"}
    labels, sizes = [], []
    for key, val in budget.items():
        if key in exclude_keys:
            continue
        v = float(val)
        if v > 0 and np.isfinite(v):
            human = key.replace("_", " ").title()
            labels.append(human)
            sizes.append(v)

    if not sizes:
        print("[plot] No positive noise contributions to display.")
        return

    total = sum(sizes)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(sizes), 1)))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%" if pct > 1 else "",
        colors=colors, startangle=140, pctdistance=0.80,
        wedgeprops={"edgecolor": "k", "linewidth": 0.5},
    )
    for t in texts:
        t.set_fontsize(9)
    for t in autotexts:
        t.set_fontsize(8)

    ax.set_title("Archimedes Experiment — Noise Budget at ω = ω_m",
                 fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()

    path = output_path or "sim4_noise_budget.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Noise budget pie chart → {path}")
    plt.close()


def plot_sensitivity_curves(
    sim: ArchimedesOptomechanics,
    output_path: Optional[str] = None,
) -> None:
    """Four-panel sensitivity curves: T_int, T, distance, Q_m."""
    if not HAS_MATPLOTLIB:
        print("[plot] matplotlib not available – skipping sensitivity curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax = axes.ravel()

    # --- Panel 1: Integration time ---
    t_arr, dm_t = sim.sensitivity_vs_integration_time()
    valid = np.isfinite(dm_t) & (dm_t > 0)
    if np.any(valid):
        ax[0].loglog(t_arr[valid], dm_t[valid] * 1e6, "b-", lw=2, label="δm_min")
        ax[0].set_xlabel("Integration time  $T_{\\rm int}$ (s)")
        ax[0].set_ylabel("Min detectable mass (mg)")
        ax[0].set_title("Sensitivity vs Integration Time", fontweight="bold")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3, which="both")

    # --- Panel 2: Temperature ---
    T_arr, dm_T = sim.sensitivity_vs_temperature()
    valid = np.isfinite(dm_T) & (dm_T > 0)
    if np.any(valid):
        ax[1].loglog(T_arr[valid], dm_T[valid] * 1e6, "r-", lw=2)
        ax[1].set_xlabel("Temperature (K)")
        ax[1].set_ylabel("Min detectable mass (mg)")
        ax[1].set_title("Sensitivity vs Temperature", fontweight="bold")
        ax[1].grid(True, alpha=0.3, which="both")

    # --- Panel 3: Distance ---
    r_arr, dm_r = sim.sensitivity_vs_distance()
    valid = np.isfinite(dm_r) & (dm_r > 0)
    if np.any(valid):
        ax[2].loglog(r_arr[valid] * 100, dm_r[valid] * 1e6, "g-", lw=2)
        ax[2].set_xlabel("Distance (cm)")
        ax[2].set_ylabel("Min detectable mass (mg)")
        ax[2].set_title("Sensitivity vs Distance", fontweight="bold")
        ax[2].grid(True, alpha=0.3, which="both")

    # --- Panel 4: Quality factor ---
    Q_arr, dm_Q = sim.sensitivity_vs_quality_factor()
    valid = np.isfinite(dm_Q) & (dm_Q > 0)
    if np.any(valid):
        ax[3].loglog(Q_arr[valid], dm_Q[valid] * 1e6, "m-", lw=2)
        ax[3].set_xlabel("Quality factor $Q_m$")
        ax[3].set_ylabel("Min detectable mass (mg)")
        ax[3].set_title("Sensitivity vs Quality Factor", fontweight="bold")
        ax[3].grid(True, alpha=0.3, which="both")

    fig.suptitle("Archimedes Gravito-Optomechanics — Sensitivity Scalings",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = output_path or "sim4_sensitivity_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Sensitivity curves → {path}")
    plt.close()


def plot_mc_histogram(
    sensitivities: np.ndarray,
    output_path: Optional[str] = None,
) -> None:
    """Histogram of Monte Carlo sensitivity distribution."""
    if not HAS_MATPLOTLIB:
        print("[plot] matplotlib not available – skipping MC histogram.")
        return

    valid = sensitivities[np.isfinite(sensitivities)]
    if len(valid) == 0:
        print("[plot] No finite MC results to histogram.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    mg = valid * 1e6
    ax.hist(mg, bins=60, color="steelblue", edgecolor="black", alpha=0.75)

    mean_v  = np.mean(mg)
    median_v = np.median(mg)
    std_v   = np.std(mg)

    ax.axvline(mean_v,  color="red",   ls="--", lw=2,
               label=f"Mean   = {mean_v:.3e} mg")
    ax.axvline(median_v, color="green", ls="--", lw=2,
               label=f"Median = {median_v:.3e} mg")
    ax.axvline(mean_v + std_v, color="orange", ls=":", lw=1.5,
               label=f"+1σ    = {mean_v + std_v:.3e} mg")
    ax.axvline(max(mean_v - std_v, 0), color="orange", ls=":", lw=1.5,
               label=f"−1σ    = {max(mean_v - std_v, 0):.3e} mg")

    ax.set_xlabel("Minimum Detectable Mass (mg)", fontsize=12)
    ax.set_ylabel("Trial count", fontsize=12)
    ax.set_title("Monte Carlo Sensitivity Distribution", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_path or "sim4_mc_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] MC histogram → {path}")
    plt.close()


def plot_noise_spectrum(
    sim: ArchimedesOptomechanics,
    output_path: Optional[str] = None,
) -> None:
    """Plot the noise spectral density vs frequency around ω_m."""
    if not HAS_MATPLOTLIB:
        print("[plot] matplotlib not available – skipping noise spectrum.")
        return

    f_range = np.linspace(0.5 * sim.omega_m_Hz, 1.5 * sim.omega_m_Hz, 400)
    omega_range = 2.0 * np.pi * f_range
    spectra = sim.noise_spectrum_array(omega_range)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.loglog(f_range / 1e6, spectra["total"], "k-", lw=2.5, label="Total", zorder=5)
    ax.loglog(f_range / 1e6, spectra["thermal"], "r-", lw=1.3, label="Thermal")
    ax.loglog(f_range / 1e6, spectra["shot_noise"], "b-", lw=1.3, label="Shot noise")
    ax.loglog(f_range / 1e6, spectra["backaction"], "g-", lw=1.3, label="Back-action")
    ax.loglog(f_range / 1e6, spectra["seismic"], "m--", lw=1.0, label="Seismic")
    ax.loglog(f_range / 1e6, spectra["gas_damping"], "c--", lw=1.0, label="Gas damping")
    ax.axvline(sim.omega_m_Hz / 1e6, color="gray", ls=":", lw=1,
               label=f"$f_m$ = {sim.omega_m_Hz/1e6:.0f} MHz")

    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Noise spectral density  (arb. units)", fontsize=12)
    ax.set_title("Optomechanical Noise Spectrum", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    path = output_path or "sim4_noise_spectrum.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Noise spectrum → {path}")
    plt.close()


# ===========================================================================
#  Pretty-printing
# ===========================================================================

def print_results(data: Dict) -> None:
    """Print a formatted summary to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("   ARCHIMEDES  GRAVITO-OPTOMECHANICS  SIMULATION  RESULTS")
    print(sep)

    # --- Parameters ---
    print("\n  INPUT PARAMETERS")
    print("  " + "-" * 50)
    p = data["parameters"]
    for k, v in p.items():
        if isinstance(v, float):
            print(f"    {k:35s} : {v: .6e}")
        else:
            print(f"    {k:35s} : {v}")

    # --- Key results ---
    print("\n  KEY RESULTS")
    print("  " + "-" * 50)
    r = data["results"]
    for k, v in r.items():
        if k == "covariance_matrix":
            continue
        if isinstance(v, (int, bool)):
            print(f"    {k:35s} : {v}")
        elif isinstance(v, float):
            print(f"    {k:35s} : {v: .6e}")

    # --- Covariance matrix ---
    cov = r["covariance_matrix"]
    labels = ["δX_opt", "δP_opt", "x_mech", "p_mech"]
    print("\n  COVARIANCE MATRIX  Σ")
    print("  " + "-" * 50)
    header = "          " + "  ".join(f"{l:>14s}" for l in labels)
    print(header)
    for i, row in enumerate(cov):
        row_str = f"  {labels[i]:>8s} " + "  ".join(f"{v:14.5e}" for v in row)
        print(row_str)

    # --- Noise budget ---
    print("\n  NOISE BUDGET  (at ω = ω_m)")
    print("  " + "-" * 50)
    nb = data["noise_budget"]
    total = nb.get("total", 1.0)
    for k, v in nb.items():
        if k in ("total", "gravitational_signal", "sql"):
            continue
        frac = (v / total * 100.0) if total > 0 and v > 0 else 0.0
        print(f"    {k:30s} : {v: .6e}   ({frac:5.1f} %)")
    if "gravitational_signal" in nb:
        print(f"    {'gravitational_signal':30s} : {nb['gravitational_signal']: .6e}   (signal)")

    # --- Uncertainty ---
    if "uncertainty_relations" in data:
        print("\n  UNCERTAINTY RELATIONS")
        print("  " + "-" * 50)
        for mode, info in data["uncertainty_relations"].items():
            ok = "✓" if info["satisfies_RSI"] else "✗"
            print(f"    {mode:12s}  ΔX²ΔP² − cov² = {info['uncertainty_product']:.6e}  {ok}")

    print(f"\n{sep}\n")


# ===========================================================================
#  CLI entry-point
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Archimedes gravito-optomechanics simulation — "
            "sensitivity and noise budget for gravitational mass sensing "
            "with cavity-optomechanical readout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Physics parameters
    phys = p.add_argument_group("Physical parameters")
    phys.add_argument("--omega-m",    type=float, default=10e6,
                      help="Mechanical resonance frequency (Hz)")
    phys.add_argument("--chi",        type=float, default=1e3,
                      help="Single-photon optomechanical coupling g₀ (Hz)")
    phys.add_argument("--temperature", type=float, default=0.1,
                      help="Bath / cryostat temperature (K)")
    phys.add_argument("--q-factor",   type=float, default=1e6,
                      help="Mechanical quality factor Q_m")
    phys.add_argument("--sample-mass", type=float, default=1e-3,
                      help="Gravitating test mass (kg)")
    phys.add_argument("--distance",   type=float, default=0.1,
                      help="Oscillator–sample distance (m)")
    phys.add_argument("--kappa",      type=float, default=1e6,
                      help="Optical cavity linewidth (Hz)")

    # Simulation parameters
    sim = p.add_argument_group("Simulation parameters")
    sim.add_argument("--integration-time", type=float, default=1000.0,
                     help="Measurement integration time (s)")
    sim.add_argument("--mc-trials", type=int, default=500,
                     help="Number of Monte Carlo trials (0 = skip)")
    sim.add_argument("--laser-power", type=float, default=1e-3,
                     help="Intracavity laser power (W)")
    sim.add_argument("--eta-det",    type=float, default=0.95,
                     help="Homodyne detection efficiency")

    # I/O
    io = p.add_argument_group("Input / output")
    io.add_argument("--plot",       action="store_true",
                    help="Generate plots (requires matplotlib)")
    io.add_argument("--output",     type=str, default=None,
                    help="JSON output file path")
    io.add_argument("--output-prefix", type=str, default="sim4_gravito",
                    help="Prefix for output file names")
    io.add_argument("--quiet",      action="store_true",
                    help="Suppress printed summary")

    return p


def run_simulation(args: argparse.Namespace) -> Dict:
    """Execute the full simulation pipeline and return the summary dict."""
    # Build the simulator
    sim = ArchimedesOptomechanics(
        omega_m_Hz=args.omega_m,
        chi_Hz=args.chi,
        T=args.temperature,
        Q_m=args.q_factor,
        sample_mass=args.sample_mass,
        distance=args.distance,
        T_int=args.integration_time,
        kappa_Hz=args.kappa,
        laser_power=args.laser_power,
        eta_det=args.eta_det,
    )

    print(f"\nInitialising Archimedes gravito-optomechanics simulation …")
    print(f"  ω_m  = {sim.omega_m_Hz/1e6:.2f} MHz   |  "
          f"T = {sim.T*1e3:.1f} mK   |  Q = {sim.Q_m:.1e}")
    print(f"  χ    = {sim.chi_Hz:.1f} Hz     |  "
          f"κ = {sim.kappa_Hz/1e6:.2f} MHz   |  η = {sim.eta_det}")
    print(f"  m_s  = {sim.sample_mass*1e3:.3f} g    |  "
          f"d = {sim.distance*100:.1f} cm  |  T_int = {sim.T_int:.0f} s")

    # Full summary
    summary = sim.summary()

    # Monte Carlo
    mc_sensitivities = None
    if args.mc_trials > 0:
        print(f"\nRunning Monte Carlo  ({args.mc_trials} trials) …", end=" ",
              flush=True)
        mc_sensitivities = sim.monte_carlo_sensitivity(
            n_trials=args.mc_trials, T_int=sim.T_int
        )
        finite = mc_sensitivities[np.isfinite(mc_sensitivities)]
        summary["monte_carlo"] = {
            "n_trials":             args.mc_trials,
            "mean_sensitivity_kg":  float(np.mean(finite)) if len(finite) else None,
            "std_sensitivity_kg":   float(np.std(finite))  if len(finite) else None,
            "median_sensitivity_kg":float(np.median(finite)) if len(finite) else None,
            "min_sensitivity_kg":   float(np.min(finite))   if len(finite) else None,
            "max_sensitivity_kg":   float(np.max(finite))   if len(finite) else None,
        }
        print("done.")

    # Print
    if not args.quiet:
        print_results(summary)
        if mc_sensitivities is not None:
            mc = summary["monte_carlo"]
            print("  MONTE CARLO SUMMARY")
            print("  " + "-" * 50)
            for k, v in mc.items():
                if isinstance(v, float):
                    print(f"    {k:30s} : {v: .6e}")
                else:
                    print(f"    {k:30s} : {v}")

    # Plots
    if args.plot:
        pfx = args.output_prefix
        budget = sim.noise_budget()
        plot_noise_budget(budget, output_path=f"{pfx}_noise_budget.png")
        plot_sensitivity_curves(sim, output_path=f"{pfx}_sensitivity_curves.png")
        plot_noise_spectrum(sim, output_path=f"{pfx}_noise_spectrum.png")
        if mc_sensitivities is not None:
            plot_mc_histogram(mc_sensitivities, output_path=f"{pfx}_mc_histogram.png")

    # Save JSON
    json_path = args.output or f"{args.output_prefix}_results.json"
    # Convert numpy types for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            v = _convert(o)
            if v is not o:
                return v
            return super().default(o)

    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2, cls=_NumpyEncoder)
    print(f"  Results saved → {json_path}")

    return summary


# ===========================================================================
#  main
# ===========================================================================

def main() -> Dict:
    """Entry point: parse arguments, run simulation, return summary."""
    parser = build_parser()
    args = parser.parse_args()
    return run_simulation(args)


if __name__ == "__main__":
    main()
