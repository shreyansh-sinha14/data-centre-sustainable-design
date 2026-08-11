"""
Joint Monte Carlo Sensitivity Analysis
=======================================
Reproduces Section 3.9.2 of:
  Sinha, S., Shukla, S., Gupta, P., Taneja, S. (2026).
  "An Integrated Analytical Framework for Sustainable Data Centre Design:
   CFD-Based Cooling, Cryogenic Energy Storage, and Quantum-Secure AI
   Optimisation." Discover Energy.

This script draws N=100,000 joint samples of nine uncertain input
parameters (each triangular-distributed, mode = the paper's central
estimate, bounds = +/-10% except CAPEX at +/-25%), propagates them
through the facility-level analytical model of Sections 3.8-3.9.1, and
reports P10/P50/P90 percentiles for the four headline outputs: PUE,
renewable fraction, CUE, and simple payback period.

Usage:
    python monte_carlo_sensitivity.py

Outputs:
    ../results/monte_carlo_draws.csv       (all N draws, inputs + outputs)
    ../results/monte_carlo_summary.csv     (P10/P50/P90 summary table)
"""

import numpy as np
import pandas as pd
import os

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
N = 100_000
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------
# Fixed reference-scenario parameters (Jaipur, Section 3.8)
# ---------------------------------------------------------------------
IT_LOAD_MW = 5.0
UTILISATION = 0.70                       # Stage 6, confirmed by diurnal analysis
E_IT = IT_LOAD_MW * UTILISATION * 8760   # MWh/yr = 30,660

SOLAR_MW = 4.0
WIND_MW = 1.5

LIGHTING_OVERHEAD = 0.01                 # fixed, LED lighting

# Traditional baseline (fixed reference, not varied — Section 3.8, Stage 1)
BASELINE_COOLING_OVERHEAD = 0.51
BASELINE_PDIST_LOSS = 0.06
PUE_BASELINE = 1 + BASELINE_COOLING_OVERHEAD + BASELINE_PDIST_LOSS + LIGHTING_OVERHEAD  # 1.58
E_TOTAL_BASELINE = E_IT * PUE_BASELINE   # MWh/yr = 48,443 (grid energy, no renewables)

# Immersion / air-cooling mix
IMMERSION_FRACTION = 0.60
AIR_FRACTION = 0.40

CAPEX_CENTRAL = 11.0   # $M

# ---------------------------------------------------------------------
# Triangular-distribution parameters: (low, mode, high)
# mode = paper's central estimate; bounds = +/-10% (CAPEX: +/-25%)
# ---------------------------------------------------------------------
params = {
    "immersion_overhead":  (0.05  * 0.90, 0.05,  0.05  * 1.10),
    "air_overhead":        (0.20  * 0.90, 0.20,  0.20  * 1.10),
    "power_dist_loss":     (0.06  * 0.90, 0.06,  0.06  * 1.10),
    "laes_rte":            (0.60  * 0.90, 0.60,  0.60  * 1.10),
    "grid_intensity":      (0.82  * 0.90, 0.82,  0.82  * 1.10),   # kg CO2/kWh
    "solar_cf":            (0.22  * 0.90, 0.22,  0.22  * 1.10),
    "wind_cf":             (0.25  * 0.90, 0.25,  0.25  * 1.10),
    "capex_musd":          (11.0  * 0.75, 11.0,  11.0  * 1.25),
    "tariff_usd_kwh":      (0.10  * 0.90, 0.10,  0.10  * 1.10),
}

draws = {name: rng.triangular(lo, mode, hi, size=N)
         for name, (lo, mode, hi) in params.items()}

df = pd.DataFrame(draws)

# ---------------------------------------------------------------------
# Facility-level analytical model (Sections 3.8-3.9.1)
# ---------------------------------------------------------------------

# --- PUE (mixed 60/40 configuration), Eq. 13-14 ---
cooling_overhead = (IMMERSION_FRACTION * df["immersion_overhead"]
                     + AIR_FRACTION * df["air_overhead"])
df["PUE"] = 1 + cooling_overhead + df["power_dist_loss"] + LIGHTING_OVERHEAD

# --- Renewable generation, Eq. 16-18 ---
E_solar = SOLAR_MW * df["solar_cf"] * 8760
E_wind = WIND_MW * df["wind_cf"] * 8760
E_renew = E_solar + E_wind

# --- Total facility energy, Eq. 20 ---
E_total = E_IT * df["PUE"]

# --- Direct renewable fraction, LAES-shifted energy, total fraction, Eq. 21-22 ---
f_direct = E_renew / E_total
E_shifted = f_direct * E_renew * df["laes_rte"]
df["renewable_fraction_pct"] = 100 * (E_renew + E_shifted) / E_total

# --- Grid energy draw, CO2 emissions, CUE, Eq. 23-25 ---
E_grid_proposed = E_total - E_renew - E_shifted
CO2_proposed_kg = E_grid_proposed * 1000 * df["grid_intensity"]
df["CUE_kgCO2_per_kWh"] = CO2_proposed_kg / (E_IT * 1000)

# --- Simple payback, Eq. 26-27 ---
annual_savings_usd = (E_TOTAL_BASELINE - E_grid_proposed) * 1000 * df["tariff_usd_kwh"]
df["payback_years"] = (df["capex_musd"] * 1e6) / annual_savings_usd

# ---------------------------------------------------------------------
# Save full draw set (rounded for readability / smaller file size)
# ---------------------------------------------------------------------
round_map = {
    "immersion_overhead": 5, "air_overhead": 5, "power_dist_loss": 5,
    "laes_rte": 5, "grid_intensity": 5, "solar_cf": 5, "wind_cf": 5,
    "capex_musd": 4, "tariff_usd_kwh": 5,
    "PUE": 5, "renewable_fraction_pct": 4,
    "CUE_kgCO2_per_kWh": 5, "payback_years": 4,
}
for col, dp in round_map.items():
    df[col] = df[col].round(dp)

out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(out_dir, exist_ok=True)

draws_path = os.path.join(out_dir, "monte_carlo_draws.csv")
df.to_csv(draws_path, index=False)

# ---------------------------------------------------------------------
# Summary table: P10 / P50 / P90 for the four headline outputs
# ---------------------------------------------------------------------
outputs = {
    "PUE (mixed 60/40)": "PUE",
    "Renewable fraction (%)": "renewable_fraction_pct",
    "CUE (kg CO2/kWh)": "CUE_kgCO2_per_kWh",
    "Simple payback (yr)": "payback_years",
}

summary_rows = []
for label, col in outputs.items():
    p10, p50, p90 = np.percentile(df[col], [10, 50, 90])
    summary_rows.append({"Output": label, "P10": round(p10, 3),
                          "P50 (median)": round(p50, 3), "P90": round(p90, 3)})

summary = pd.DataFrame(summary_rows)
summary_path = os.path.join(out_dir, "monte_carlo_summary.csv")
summary.to_csv(summary_path, index=False)

print(f"N = {N:,} draws (seed={SEED})")
print(summary.to_string(index=False))
print(f"\nFull draws written to:  {os.path.abspath(draws_path)}")
print(f"Summary written to:     {os.path.abspath(summary_path)}")
