#!/usr/bin/env python3
"""
sim9_end_to_end_archimedes_twin.py – Full Digital Twin of the Archimedes Experiment

Capstone simulation integrating the complete Archimedes measurement chain:
  1. Superconducting phase transition  → bit-mass Δm from entropy change ΔS
  2. Optomechanical transducer         → gravitational frequency shift Δω_m
  3. Entanglement-enhanced readout     → GHZ / squeezed quantum sensor array
  4. Full noise budget                 → thermal, back-action, seismic, amplifier
  5. SNR, matched filter, Monte Carlo  → feasibility verdict

COMPLETELY SELF-CONTAINED – no imports from sim1/sim4/sim7/sim8.

Key physics:
  Δm  ≈ ΔS · kB · Tc / c²             (Landauer bit-mass)
  Δωm = (ωm/2) · G·Δm / (g·r²)       (gravitational frequency shift on oscillator)
  SNR = Δωm / √(S_ω_total(0) / T_int)  (white-noise idealised SNR)

Noise model – all PSDs in units of [rad²/s / Hz] (angular-frequency noise PSD):
  Thermal     : χ² · 2 γ_m n_th(n_th+1) / D(f)        (phonon-number fluctuations)
  Back-action : ℏ ω_m χ² N_eff / (4 Q_m)               (quantum radiation pressure)
  Seismic     : (ω_m/2g)² · S_a(f) · |H_susp(f)|²      (ground vibration × isolation)
  Amplifier   : ℏ² χ² n_add / (2 γ_m τ_meas)           (SQUID / HEMT readout)
"""

import argparse
import os
import sys
import json
import time as _time

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
HBAR    = 1.054571817e-34    # Reduced Planck constant   [J·s]
KB      = 1.380649e-23       # Boltzmann constant        [J/K]
C_LIGHT = 2.99792458e8       # Speed of light            [m/s]
G_GRAV  = 6.6743e-11         # Gravitational constant    [m³ kg⁻¹ s⁻²]
g_earth = 9.81               # Standard gravity          [m/s²]


# ===================================================================
#  ArchimedesDigitalTwin
# ===================================================================
class ArchimedesDigitalTwin:
    """End-to-end digital twin of the Archimedes vacuum-weight experiment."""

    def __init__(self, config: dict):
        self.cfg = config
        # Derived angular frequencies / rates
        self._omega_m     = 2.0 * np.pi * config["omega_m_Hz"]
        self._f_m         = config["omega_m_Hz"]
        self._chi         = 2.0 * np.pi * config["chi_Hz"]
        self._gamma_deph  = 2.0 * np.pi * config["gamma_deph_Hz"]
        self._gamma_m     = self._omega_m / config["Q_m"]
        self._Q_m         = config["Q_m"]

    # ------------------------------------------------------------------
    #  Core physics helpers
    # ------------------------------------------------------------------
    def bit_mass_prediction(self, T_c: float, delta_S_kB: float,
                            volume: float) -> float:
        """Δm ≈ ΔS · kB · Tc / c²   (bit-mass / Landauer relation)."""
        return delta_S_kB * KB * T_c / C_LIGHT**2

    def thermal_occupation(self, omega_m: float, T: float) -> float:
        """Bose-Einstein mean phonon number  n_th = 1/(e^{ℏω/kBT} − 1)."""
        x = HBAR * omega_m / (KB * T)
        if x > 500:
            return 0.0
        return 1.0 / (np.exp(x) - 1.0)

    def gravitational_coupling(self) -> float:
        """g_grav [rad/s / kg] = (ω_m/2) · G / (g · r²)."""
        r = self.cfg["distance"]
        return 0.5 * self._omega_m * G_GRAV / (g_earth * r**2)

    def frequency_shift(self, delta_m: float) -> float:
        """Δω_m = g_grav · Δm."""
        return self.gravitational_coupling() * delta_m

    # ------------------------------------------------------------------
    #  Noise PSD methods  (all return [rad²/s / Hz])
    # ------------------------------------------------------------------
    @staticmethod
    def _lorentzian_D(freqs: np.ndarray, f_m: float, Q_m: float) -> np.ndarray:
        """Mechanical susceptibility denominator D(f) = (1−u²)² + (u/Q)²."""
        u = freqs / f_m
        return (1.0 - u**2)**2 + (u / Q_m)**2

    def thermal_noise_psd(self, freqs: np.ndarray, omega_m: float, T: float,
                          Q_m: float, gamma_m: float) -> np.ndarray:
        """
        Phonon-number fluctuation noise transduced through dispersive coupling.
        S_th(f) = χ² · 2 γ_m n_th(n_th+1) / D(f)
        At DC (f≪f_m):  S_th ≈ χ² · 2 γ_m n_th(n_th+1)
        """
        n_th = self.thermal_occupation(omega_m, T)
        D = self._lorentzian_D(freqs, omega_m / (2*np.pi), Q_m)
        D = np.maximum(D, 1e-60)
        return self._chi**2 * 2.0 * gamma_m * n_th * (n_th + 1.0) / D

    def backaction_noise_psd(self, freqs: np.ndarray, chi: float,
                             N_eff: float) -> np.ndarray:
        """
        Quantum back-action from continuous dispersive measurement.
        Back-action adds phonon noise: S_n_ba = κ_meas N_eff / (4 ω_m)
        Referred to frequency: S_ω_ba = χ² S_n_ba.
        """
        kappa_meas = 2.0 * np.pi * 1e6
        S_n_ba = kappa_meas * N_eff / (4.0 * self._omega_m)
        return chi**2 * S_n_ba * np.ones_like(freqs)

    def seismic_noise_psd(self, freqs: np.ndarray, S0: float = 1e-12,
                          f0: float = 1.0,
                          Q_susp: float = 100.0) -> np.ndarray:
        """
        Ground vibration noise filtered by pendulum / platform isolation.
        Ground acceleration PSD:  S_a(f) = S0 · (f0/f)^2   for f > f0
        Isolation transfer:       H(f) = f² / √[(f²−f_s²)² + (f·f_s/Q_s)²]
        Frequency noise:           S_seis = (ω_m/(2g))² · S_a · H²
        """
        f_s = f0  # suspension corner frequency
        safe_f = np.maximum(freqs, 1e-8)
        # Ground acceleration PSD  [m²/s³/Hz]
        S_a = S0 * (f0 / safe_f)**2
        # Pendulum transfer function magnitude squared
        num = safe_f**4
        den = (safe_f**2 - f_s**2)**2 + (safe_f * f_s / Q_susp)**2
        den = np.maximum(den, 1e-60)
        H2 = num / den
        # Convert to oscillator frequency noise via  Δω = (ω_m/2g)·a
        S_seis = (self._omega_m / (2.0 * g_earth))**2 * S_a * H2
        return S_seis

    def amplifier_noise_psd(self, freqs: np.ndarray, n_add: float,
                            chi: float) -> np.ndarray:
        """
        Readout imprecision noise referred to oscillator frequency.
        Continuous dispersive readout with bandwidth κ_meas:
            S_n_imp = n_add / κ_meas   [phonons/Hz]
            S_ω_amp = χ² · S_n_imp
        Using κ_meas = 2π × 1 MHz (typical transmon readout).
        """
        kappa_meas = 2.0 * np.pi * 1e6     # readout bandwidth [rad/s]
        S_amp = chi**2 * n_add / kappa_meas
        return S_amp * np.ones_like(freqs)

    def total_noise_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Sum of all noise contributions."""
        N_eff = self._effective_N()
        return (self.thermal_noise_psd(freqs, self._omega_m,
                                       self.cfg["temperature_K"],
                                       self._Q_m, self._gamma_m)
                + self.backaction_noise_psd(freqs, self._chi, N_eff)
                + self.seismic_noise_psd(freqs,
                                         S0=self.cfg["seismic_S0"],
                                         f0=self.cfg["suspension_freq_Hz"],
                                         Q_susp=self.cfg["suspension_Q"])
                + self.amplifier_noise_psd(freqs, self.cfg["n_add"],
                                           self._chi))

    # ------------------------------------------------------------------
    #  Noise budget at DC
    # ------------------------------------------------------------------
    def noise_budget(self) -> dict:
        """Evaluate each noise source near DC (f = 0.01 Hz)."""
        f_dc = np.array([0.01])
        N_eff = self._effective_N()
        S_th   = self.thermal_noise_psd(
            f_dc, self._omega_m, self.cfg["temperature_K"],
            self._Q_m, self._gamma_m)[0]
        S_ba   = self.backaction_noise_psd(f_dc, self._chi, N_eff)[0]
        S_seis = self.seismic_noise_psd(
            f_dc, self.cfg["seismic_S0"],
            self.cfg["suspension_freq_Hz"],
            self.cfg["suspension_Q"])[0]
        S_amp  = self.amplifier_noise_psd(
            f_dc, self.cfg["n_add"], self._chi)[0]
        S_tot  = S_th + S_ba + S_seis + S_amp
        return dict(thermal=S_th, backaction=S_ba, seismic=S_seis,
                    amplifier=S_amp, total=S_tot)

    # ------------------------------------------------------------------
    #  Effective coherent qubit number
    # ------------------------------------------------------------------
    def _effective_N(self) -> float:
        """N_eff = N · exp(−γ_deph · T₂)."""
        T2 = self.cfg["T2_us"] * 1e-6
        return self.cfg["N_qubits"] * np.exp(-self._gamma_deph * T2)

    # ------------------------------------------------------------------
    #  Sensitivity bounds
    # ------------------------------------------------------------------
    def ghz_sensitivity(self, N: int, gamma: float, g_grav: float,
                        T_int: float) -> float:
        """
        Minimum detectable Δm with GHZ entangled state.
        δm_GHZ = ℏ / (g_grav · N² · exp(−N γ T₂) · √(T_int))
        """
        T2 = self.cfg["T2_us"] * 1e-6
        coh = np.exp(-N * gamma * T2)
        if coh < 1e-300:
            return np.inf
        return HBAR / (g_grav * N**2 * coh * np.sqrt(T_int))

    def sql_sensitivity(self, N: int, g_grav: float,
                        T_int: float) -> float:
        """Standard Quantum Limit:  δm_SQL = ℏ / (g_grav · √(N T_int))."""
        return HBAR / (g_grav * np.sqrt(N * T_int))

    def squeezed_sensitivity(self, N: int, gamma: float, g_grav: float,
                             squeezing_db: float, T_int: float) -> float:
        """Spin-squeezed:  δm_sqz = δm_SQL / (ξ² · √coherence)."""
        xi2 = 10.0 ** (-squeezing_db / 10.0)
        dm_sql = self.sql_sensitivity(N, g_grav, T_int)
        T2 = self.cfg["T2_us"] * 1e-6
        coh = np.exp(-N * gamma * T2 * 0.5)
        return dm_sql / max(xi2 * coh, 1e-300)

    # ------------------------------------------------------------------
    #  SNR
    # ------------------------------------------------------------------
    def compute_snr(self, delta_m: float, T_int: float) -> float:
        """SNR = Δω_m / √(S_total(DC) / T_int)."""
        S_tot = self.noise_budget()["total"]
        dw = self.frequency_shift(delta_m)
        rms = np.sqrt(S_tot / T_int)
        return dw / max(rms, 1e-300)

    def matched_filter_snr(self, delta_m_array: np.ndarray,
                           T_int: float, dt: float) -> float:
        """
        Matched-filter SNR for a known template s(t) in white noise with PSD S_n.
        Optimal:  SNR²_mf = (1/S_n) ∫|s(t)|² dt
        """
        s = self.gravitational_coupling() * np.asarray(delta_m_array, dtype=float)
        S_n = self.noise_budget()["total"]
        if S_n < 1e-300:
            return 0.0
        energy = np.sum(s**2) * dt          # ∫|s(t)|² dt
        return float(np.sqrt(energy / S_n))

    # ------------------------------------------------------------------
    #  Monte Carlo
    # ------------------------------------------------------------------
    def monte_carlo(self, n_trials: int = 100) -> dict:
        """Propagate parameter uncertainties; return SNR statistics."""
        rng = np.random.default_rng(42)
        snr_arr = np.zeros(n_trials)
        dm_arr  = np.zeros(n_trials)
        # snapshot originals
        base = {k: self.cfg[k]
                for k in ("T_c", "delta_S_kB", "temperature_K",
                          "gamma_deph_Hz", "Q_m", "distance", "N_qubits")}

        for i in range(n_trials):
            self.cfg["T_c"]            = rng.normal(base["T_c"], 0.1)
            self.cfg["delta_S_kB"]     = base["delta_S_kB"] * rng.lognormal(0, 0.3)
            self.cfg["temperature_K"]  = max(base["temperature_K"] * rng.lognormal(0, 0.5), 0.01)
            self.cfg["gamma_deph_Hz"]  = base["gamma_deph_Hz"] * rng.lognormal(0, 0.3)
            self.cfg["Q_m"]            = base["Q_m"] * rng.lognormal(0, 0.2)
            self.cfg["distance"]       = max(base["distance"] * rng.lognormal(0, 0.2), 0.01)
            self.cfg["N_qubits"]       = max(int(round(base["N_qubits"] * rng.normal(1, 0.1))), 1)
            self._gamma_m    = self._omega_m / self.cfg["Q_m"]
            self._gamma_deph = 2.0 * np.pi * self.cfg["gamma_deph_Hz"]

            dm  = self.bit_mass_prediction(self.cfg["T_c"], self.cfg["delta_S_kB"],
                                           self.cfg["volume"])
            snr = self.compute_snr(dm, self.cfg["T_int"])
            snr_arr[i] = snr
            dm_arr[i]  = dm

        # restore
        for k, v in base.items():
            self.cfg[k] = v
        self._gamma_m    = self._omega_m / base["Q_m"]
        self._gamma_deph = 2.0 * np.pi * base["gamma_deph_Hz"]

        return dict(
            snr_mean=float(np.mean(snr_arr)),
            snr_std=float(np.std(snr_arr)),
            snr_median=float(np.median(snr_arr)),
            snr_min=float(np.min(snr_arr)),
            snr_max=float(np.max(snr_arr)),
            p_detect_5sigma=float(np.mean(snr_arr > 5.0)),
            dm_mean_kg=float(np.mean(dm_arr)),
            dm_std_kg=float(np.std(dm_arr)),
            snr_samples=snr_arr,
            dm_samples=dm_arr,
        )

    # ------------------------------------------------------------------
    #  Full pipeline
    # ------------------------------------------------------------------
    def run_full(self) -> dict:
        t0 = _time.time()

        # 1 ── signal
        delta_m    = self.bit_mass_prediction(
                        self.cfg["T_c"], self.cfg["delta_S_kB"], self.cfg["volume"])
        g_grav     = self.gravitational_coupling()
        delta_omega = self.frequency_shift(delta_m)

        # 2 ── noise budget at DC
        budget = self.noise_budget()

        # 3 ── frequency-domain noise curves
        freqs = np.logspace(-2, np.log10(self._f_m), 500)
        N_eff = self._effective_N()
        S_th   = self.thermal_noise_psd(freqs, self._omega_m,
                                        self.cfg["temperature_K"],
                                        self._Q_m, self._gamma_m)
        S_ba   = self.backaction_noise_psd(freqs, self._chi, N_eff)
        S_seis = self.seismic_noise_psd(freqs, self.cfg["seismic_S0"],
                                        self.cfg["suspension_freq_Hz"],
                                        self.cfg["suspension_Q"])
        S_amp  = self.amplifier_noise_psd(freqs, self.cfg["n_add"], self._chi)
        S_tot  = S_th + S_ba + S_seis + S_amp

        # 4 ── sensitivity bounds
        gamma  = self._gamma_deph
        T_int  = self.cfg["T_int"]
        N      = self.cfg["N_qubits"]
        dm_ghz = self.ghz_sensitivity(N, gamma, g_grav, T_int)
        dm_sql = self.sql_sensitivity(N, g_grav, T_int)
        dm_sqz = self.squeezed_sensitivity(N, gamma, g_grav, 10.0, T_int)

        # 5 ── SNR
        snr = self.compute_snr(delta_m, T_int)

        # 6 ── time-domain signal & matched filter
        n_t = 1000
        t_sig = np.linspace(0, T_int, n_t)
        dt    = t_sig[1] - t_sig[0]
        t_mid = T_int / 2.0
        width = T_int / 10.0
        mass_curve = delta_m / (1.0 + np.exp(-(t_sig - t_mid) / width))
        snr_mf = self.matched_filter_snr(mass_curve, T_int, dt)

        # 7 ── diagnostics
        n_th   = self.thermal_occupation(self._omega_m, self.cfg["temperature_K"])
        eff_N  = N_eff
        names  = ["thermal", "backaction", "seismic", "amplifier"]
        vals   = [budget[n] for n in names]
        dom_i  = int(np.argmax(vals))
        dom_name = names[dom_i]
        dom_pct  = vals[dom_i] / budget["total"] * 100.0

        detectable = snr >= 5.0
        verdict    = "DETECTABLE" if detectable else "NOT DETECTABLE"
        improvements = self._improvements(snr, budget, dom_name, vals)

        return dict(
            bit_mass_kg=delta_m, bit_mass_ag=delta_m * 1e18,
            frequency_shift_Hz=delta_omega, g_grav_Hz_per_kg=g_grav,
            n_thermal=n_th, effective_qubits=eff_N,
            snr=snr, snr_matched_filter=snr_mf,
            dm_ghz_kg=dm_ghz, dm_sql_kg=dm_sql, dm_squeezed_kg=dm_sqz,
            noise_budget=budget,
            dominant_noise=dom_name, dominant_fraction_pct=dom_pct,
            verdict=verdict, detectable=detectable,
            required_improvements=improvements,
            freqs=freqs, S_total_freq=S_tot,
            S_thermal_freq=S_th, S_backaction_freq=S_ba,
            S_seismic_freq=S_seis, S_amplifier_freq=S_amp,
            t_signal=t_sig, mass_curve=mass_curve,
            elapsed_s=_time.time() - t0,
        )

    # ------------------------------------------------------------------
    #  Improvement analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _improvements(snr, budget, dom, vals) -> list:
        target = 5.0
        if snr >= target:
            return ["All requirements met – experiment is feasible."]
        if snr < 1e-30:
            return ["Signal negligible – increase ΔS by many orders of magnitude."]
        fac = target / snr
        out = []
        total = budget["total"]
        label_map = {
            "thermal":   "thermal noise (lower T or increase Q_m)",
            "backaction":"back-action (lower χ or fewer qubits)",
            "seismic":   "seismic noise (better vibration isolation)",
            "amplifier": "amplifier noise (lower n_add / JPA readout)",
        }
        for name, val in zip(["thermal","backaction","seismic","amplifier"], vals):
            if val / total > 0.05:
                red = fac**2
                out.append(f"Reduce {label_map[name]} by {red:.1e}×")
        out.append(f"Increase ΔS by {fac:.1e}×  or  decrease distance by {fac**0.5:.1e}×")
        out.append(f"Increase T_int by {fac**2:.1e}×")
        return out


    # ==================================================================
    #  Plotting
    # ==================================================================
    def generate_plots(self, results: dict,
                       output_dir: str = "sim9_results"):
        os.makedirs(output_dir, exist_ok=True)
        self._plot_noise_pie(results, output_dir)
        self._plot_sens_vs_tint(results, output_dir)
        self._plot_sens_vs_N(results, output_dir)
        self._plot_mc_histogram(results, output_dir)
        self._plot_signal_chain(results, output_dir)
        self._plot_noise_psd(results, output_dir)
        print(f"  [plots] 6 figures saved to {output_dir}/")

    def _import_mpl(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            return None

    # P1 ── noise pie ──────────────────────────────────────────────────
    def _plot_noise_pie(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        labels = ["Thermal", "Back-action", "Seismic", "Amplifier"]
        vals   = [r["noise_budget"][k] for k in
                  ["thermal","backaction","seismic","amplifier"]]
        if sum(vals) < 1e-300:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(vals, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=["#e74c3c","#3498db","#2ecc71","#f39c12"])
        ax.set_title("Noise Budget – Dominant Sources", fontsize=13)
        plt.tight_layout()
        fig.savefig(os.path.join(out, "noise_budget_pie.png"), dpi=150)
        plt.close(fig)

    # P2 ── sensitivity vs T_int ───────────────────────────────────────
    def _plot_sens_vs_tint(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        Ts = np.logspace(1, 5, 80)
        gg = r["g_grav_Hz_per_kg"]; N = self.cfg["N_qubits"]
        g  = self._gamma_deph
        d_ghz = np.array([self.ghz_sensitivity(N, g, gg, T) for T in Ts])
        d_sql = np.array([self.sql_sensitivity(N, gg, T) for T in Ts])
        d_sqz = np.array([self.squeezed_sensitivity(N, g, gg, 10, T) for T in Ts])
        sig = r["bit_mass_kg"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.loglog(Ts, np.clip(d_ghz, 1e-100, None), "r-",  lw=2, label="GHZ")
        ax.loglog(Ts, d_sql,                           "b--", lw=2, label="SQL")
        ax.loglog(Ts, np.clip(d_sqz, 1e-100, None),    "g-.", lw=2, label="Squeezed 10 dB")
        ax.axhline(sig, color="k", ls=":", lw=1.5,
                   label=f"Signal Δm = {sig:.2e} kg")
        ax.set_xlabel("Integration time $T_{int}$ [s]")
        ax.set_ylabel("Min. detectable Δm [kg]")
        ax.set_title("Sensitivity vs Integration Time")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out, "sensitivity_vs_Tint.png"), dpi=150)
        plt.close(fig)

    # P3 ── sensitivity vs N ───────────────────────────────────────────
    def _plot_sens_vs_N(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        Ns = np.arange(1, 33)
        gg = r["g_grav_Hz_per_kg"]; g = self._gamma_deph
        Ti = self.cfg["T_int"]
        d_ghz = [self.ghz_sensitivity(int(n), g, gg, Ti) for n in Ns]
        d_sql = [self.sql_sensitivity(int(n), gg, Ti) for n in Ns]
        d_sqz = [self.squeezed_sensitivity(int(n), g, gg, 10, Ti) for n in Ns]
        sig = r["bit_mass_kg"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(Ns, np.clip(d_ghz, 1e-100, None), "r-o", ms=3, lw=1.5, label="GHZ")
        ax.semilogy(Ns, d_sql,                           "b--s", ms=3, lw=1.5, label="SQL")
        ax.semilogy(Ns, np.clip(d_sqz, 1e-100, None),    "g-.^", ms=3, lw=1.5, label="Squeezed 10 dB")
        ax.axhline(sig, color="k", ls=":", lw=1.5,
                   label=f"Signal Δm = {sig:.2e} kg")
        ax.set_xlabel("Number of qubits  N")
        ax.set_ylabel("Min. detectable Δm [kg]")
        ax.set_title("Sensitivity vs Qubit Count")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out, "sensitivity_vs_N.png"), dpi=150)
        plt.close(fig)

    # P4 ── MC histogram ──────────────────────────────────────────────
    def _plot_mc_histogram(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        mc = self.monte_carlo(n_trials=200)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mc["snr_samples"], bins=30, color="#3498db",
                edgecolor="w", alpha=0.85)
        ax.axvline(5.0, color="r", ls="--", lw=2, label="5σ threshold")
        ax.axvline(mc["snr_mean"], color="k", ls="-", lw=2,
                   label=f"Mean = {mc['snr_mean']:.4f}")
        ax.set_xlabel("SNR"); ax.set_ylabel("Trials")
        ax.set_title(f"Monte Carlo SNR  "
                     f"(P>5σ = {mc['p_detect_5sigma']:.1%})")
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(out, "mc_snr_histogram.png"), dpi=150)
        plt.close(fig)

    # P5 ── signal chain schematic ────────────────────────────────────
    def _plot_signal_chain(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
        ax.set_title("Archimedes Signal Chain", fontsize=14, weight="bold")
        boxes = [
            (0.5, 3.5, "SC Phase\nTransition\nΔS → Δm"),
            (3.0, 3.5, "Gravitational\nCoupling\nΔm → Δω_m"),
            (5.5, 3.5, "Mech. Osc.\nω_m=2π·10 MHz"),
            (8.0, 3.5, "Qubit\nEnsemble\n(N GHZ)"),
            (10.5, 3.5, "Readout\n& SNR"),
        ]
        cols = ["#e74c3c","#f39c12","#2ecc71","#3498db","#9b59b6"]
        for (x, y, txt), c in zip(boxes, cols):
            ax.add_patch(plt.Rectangle((x, y), 2.0, 1.8,
                         fc=c, ec="k", alpha=.75, lw=1.5, zorder=2))
            ax.text(x+1, y+.9, txt, ha="center", va="center",
                    fontsize=8, weight="bold", color="w", zorder=3)
        for i in range(len(boxes)-1):
            ax.annotate("", xy=(boxes[i+1][0], 4.4),
                        xytext=(boxes[i][0]+2.0, 4.4),
                        arrowprops=dict(arrowstyle="->", lw=2, color="k"),
                        zorder=4)
        nb = r["noise_budget"]
        noise_txt = (f"Noise Sources:\n"
                     f"• Thermal:   {nb['thermal']:.2e}\n"
                     f"• Back-act:  {nb['backaction']:.2e}\n"
                     f"• Seismic:   {nb['seismic']:.2e}\n"
                     f"• Amplifier: {nb['amplifier']:.2e}")
        ax.text(1.0, 1.5, noise_txt, fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=.9))
        res_txt = (f"Δm  = {r['bit_mass_kg']:.3e} kg\n"
                   f"SNR = {r['snr']:.4f}\n"
                   f"{r['verdict']}")
        ax.text(8.5, 1.5, res_txt, fontsize=10, family="monospace",
                weight="bold",
                bbox=dict(boxstyle="round",
                          fc="lightgreen" if r["detectable"] else "#ffcccc",
                          alpha=.9))
        plt.tight_layout()
        fig.savefig(os.path.join(out, "signal_chain_schematic.png"), dpi=150)
        plt.close(fig)

    # P6 ── noise PSD spectral ────────────────────────────────────────
    def _plot_noise_psd(self, r, out):
        plt = self._import_mpl()
        if plt is None:
            return
        f = r["freqs"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.loglog(f, r["S_thermal_freq"],    "r-",  lw=1.5, label="Thermal")
        ax.loglog(f, r["S_backaction_freq"], "b--", lw=1.5, label="Back-action")
        ax.loglog(f, r["S_seismic_freq"],    "g-.", lw=1.5, label="Seismic")
        ax.loglog(f, r["S_amplifier_freq"],  "m:",  lw=1.5, label="Amplifier")
        ax.loglog(f, r["S_total_freq"],      "k-",  lw=2.5, label="Total")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"$S_\omega(f)$  [rad² s⁻¹ Hz⁻¹]")
        ax.set_title("Noise Power Spectral Density")
        ax.legend(fontsize=9); ax.grid(True, alpha=.3, which="both")
        plt.tight_layout()
        fig.savefig(os.path.join(out, "noise_psd_spectral.png"), dpi=150)
        plt.close(fig)


# ======================================================================
#  Summary printer
# ======================================================================
def print_summary(results: dict, mc: dict = None):
    H = "=" * 72
    print(H)
    print("   ARCHIMEDES DIGITAL TWIN – FULL RESULTS SUMMARY")
    print(H)

    print(f"\n  SIGNAL")
    print(f"    Bit-mass  Δm          = {results['bit_mass_kg']:.4e} kg  "
          f"({results['bit_mass_ag']:.2f} ag)")
    print(f"    Freq. shift  Δω_m     = {results['frequency_shift_Hz']:.4e} rad/s")
    print(f"    Grav. coupling g_grav = {results['g_grav_Hz_per_kg']:.4e} rad·s⁻¹/kg")

    print(f"\n  SYSTEM")
    print(f"    Thermal occupation n_th = {results['n_thermal']:.4e}")
    print(f"    Effective qubits N_eff  = {results['effective_qubits']:.2f}")

    nb = results["noise_budget"]
    tot = nb["total"]
    print(f"\n  NOISE BUDGET  (near DC,  [rad² s⁻¹ Hz⁻¹])")
    for k in ("thermal","backaction","seismic","amplifier"):
        pct = nb[k]/tot*100 if tot > 0 else 0
        print(f"    {k:<16s} {nb[k]:.4e}  ({pct:5.1f}%)")
    print(f"    {'TOTAL':<16s} {tot:.4e}")
    print(f"    Dominant: {results['dominant_noise']} "
          f"({results['dominant_fraction_pct']:.1f}%)")

    print(f"\n  SENSITIVITY  (min. detectable Δm)")
    print(f"    GHZ           : {results['dm_ghz_kg']:.4e} kg")
    print(f"    SQL           : {results['dm_sql_kg']:.4e} kg")
    print(f"    Squeezed 10dB : {results['dm_squeezed_kg']:.4e} kg")

    print(f"\n  SIGNAL-TO-NOISE RATIO")
    print(f"    Simple SNR          : {results['snr']:.4f}")
    print(f"    Matched-filter SNR  : {results['snr_matched_filter']:.4f}")

    if mc is not None:
        print(f"\n  MONTE CARLO  (parameter uncertainties)")
        print(f"    SNR mean ± std   : {mc['snr_mean']:.4f} ± {mc['snr_std']:.4f}")
        print(f"    SNR median       : {mc['snr_median']:.4f}")
        print(f"    SNR range        : [{mc['snr_min']:.4f}, {mc['snr_max']:.4f}]")
        print(f"    P(SNR > 5σ)      : {mc['p_detect_5sigma']:.1%}")
        print(f"    Δm mean ± std    : {mc['dm_mean_kg']:.3e} ± "
              f"{mc['dm_std_kg']:.3e} kg")

    print("\n" + "-" * 72)
    clr = "\033[92m" if results["detectable"] else "\033[91m"
    print(f"  VERDICT:  {clr}{results['verdict']}\033[0m")
    print("\n  REQUIRED IMPROVEMENTS:")
    for imp in results.get("required_improvements", []):
        print(f"    • {imp}")
    print(H)


# ======================================================================
#  CLI
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Archimedes Digital Twin – End-to-End Simulation")
    p.add_argument("--T-c",       type=float, default=9.2,
                   help="Critical temperature [K]")
    p.add_argument("--delta-S",   type=float, default=1e23,
                   help="Entropy change ΔS/kB")
    p.add_argument("--volume",    type=float, default=1e-6,
                   help="Sample volume [m³]")
    p.add_argument("--distance",  type=float, default=0.1,
                   help="Sample–oscillator distance [m]")
    p.add_argument("--N-qubits",  type=int,   default=8,
                   help="Number of entangled qubits")
    p.add_argument("--T-int",     type=float, default=1000.0,
                   help="Integration time [s]")
    p.add_argument("--mc-trials", type=int,   default=200,
                   help="Monte Carlo trials")
    p.add_argument("--output-dir",type=str,   default="sim9_results",
                   help="Output directory for plots")
    return p


def main():
    args = build_parser().parse_args()

    config = dict(
        T_c=9.2, delta_S_kB=1e23, volume=1e-6, sample_mass=1e-3,
        distance=0.1, omega_m_Hz=10e6, Q_m=1e6, chi_Hz=1e3,
        temperature_K=0.1, N_qubits=8, T2_us=50, gamma_deph_Hz=2e3,
        T_int=1000.0, n_add=15, suspension_freq_Hz=1.0, suspension_Q=100,
        seismic_S0=1e-12,
    )
    # CLI overrides
    config["T_c"]          = args.T_c
    config["delta_S_kB"]   = args.delta_S
    config["volume"]       = args.volume
    config["distance"]     = args.distance
    config["N_qubits"]     = args.N_qubits
    config["T_int"]        = args.T_int

    print("=" * 72)
    print("  Archimedes Digital Twin – End-to-End Simulation")
    print("=" * 72)
    print("\n  Configuration:")
    for k, v in config.items():
        print(f"    {k:.<30s} {v}")

    twin = ArchimedesDigitalTwin(config)

    print("\n  Running full pipeline …")
    results = twin.run_full()
    print(f"  Pipeline completed in {results['elapsed_s']:.2f} s")

    print(f"\n  Monte Carlo ({args.mc_trials} trials) …")
    mc = twin.monte_carlo(n_trials=args.mc_trials)

    print("\n  Generating plots …")
    twin.generate_plots(results, output_dir=args.output_dir)

    print()
    print_summary(results, mc=mc)

    # JSON
    jpath = os.path.join(args.output_dir, "sim9_results.json")
    blob = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    blob["mc"] = {k: v for k, v in mc.items() if not isinstance(v, np.ndarray)}
    with open(jpath, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    print(f"\n  JSON → {jpath}")

    sys.exit(0 if results["detectable"] else 1)


if __name__ == "__main__":
    main()
