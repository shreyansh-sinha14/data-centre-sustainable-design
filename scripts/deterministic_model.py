"""
Deterministic Point-Estimate Model
====================================
Reproduces the exact facility-level equations of Section 3.8 (Jaipur
reference scenario) and Section 3.8 Stages 8-9 (Delhi, Chennai) from:

  Sinha, S., Shukla, S., Gupta, P., Taneja, S. (2026). "An Integrated
  Analytical Framework for Sustainable Data Centre Design." Discover Energy.

Running this script reproduces every point estimate quoted in the paper's
Tables 4 and 6 (PUE, renewable fraction, CUE, payback) from first
principles, with no random sampling. Use this to sanity-check the paper's
arithmetic independently of the Monte Carlo script.

Usage:
    python deterministic_model.py
"""

IT_LOAD_MW = 5.0
UTILISATION = 0.70
E_IT = IT_LOAD_MW * UTILISATION * 8760          # MWh/yr

SOLAR_MW = 4.0
WIND_MW = 1.5
LAES_RTE = 0.60
LIGHTING_OVERHEAD = 0.01
POWER_DIST_LOSS = 0.06
CAPEX_MUSD = 11.0
TARIFF = 0.10

IMMERSION_FRACTION = 0.60
AIR_FRACTION = 0.40
IMMERSION_OVERHEAD = 0.05
AIR_OVERHEAD = 0.20

CITIES = {
    "Jaipur":  dict(solar_cf=0.22, wind_cf=0.25, baseline_cooling_overhead=0.51),
    "Delhi":   dict(solar_cf=0.19, wind_cf=0.22, baseline_cooling_overhead=0.47),
    "Chennai": dict(solar_cf=0.20, wind_cf=0.28, baseline_cooling_overhead=0.55),
}

GRID_INTENSITY = 0.82  # kg CO2/kWh, India grid average (Section 3.8)


def run_city(name, solar_cf, wind_cf, baseline_cooling_overhead):
    # --- Baseline (traditional, air-cooled) ---
    pue_baseline = 1 + baseline_cooling_overhead + POWER_DIST_LOSS + LIGHTING_OVERHEAD
    e_total_baseline = E_IT * pue_baseline
    cue_baseline = (e_total_baseline * 1000 * GRID_INTENSITY) / (E_IT * 1000)

    # --- Proposed (mixed 60/40) ---
    cooling_overhead = IMMERSION_FRACTION * IMMERSION_OVERHEAD + AIR_FRACTION * AIR_OVERHEAD
    pue_mixed = 1 + cooling_overhead + POWER_DIST_LOSS + LIGHTING_OVERHEAD
    e_total = E_IT * pue_mixed

    e_solar = SOLAR_MW * solar_cf * 8760
    e_wind = WIND_MW * wind_cf * 8760
    e_renew = e_solar + e_wind

    f_direct = e_renew / e_total
    e_shifted = f_direct * e_renew * LAES_RTE
    f_total = (e_renew + e_shifted) / e_total

    e_grid = e_total - e_renew - e_shifted
    co2_kg = e_grid * 1000 * GRID_INTENSITY
    cue = co2_kg / (E_IT * 1000)

    cue_reduction_pct = 100 * (1 - cue / cue_baseline)

    annual_savings = (e_total_baseline - e_grid) * 1000 * TARIFF
    payback_years = (CAPEX_MUSD * 1e6) / annual_savings

    return {
        "City": name,
        "PUE_baseline": round(pue_baseline, 3),
        "PUE_mixed": round(pue_mixed, 3),
        "Renewable_generation_MWh": round(e_renew, 0),
        "Renewable_fraction_pct": round(100 * f_total, 1),
        "CUE_baseline_kgCO2_kWh": round(cue_baseline, 3),
        "CUE_proposed_kgCO2_kWh": round(cue, 3),
        "CUE_reduction_pct": round(cue_reduction_pct, 1),
        "Payback_years": round(payback_years, 1),
    }


if __name__ == "__main__":
    import pandas as pd
    import os

    rows = [run_city(name, **params) for name, params in CITIES.items()]
    df = pd.DataFrame(rows)

    print(df.to_string(index=False))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deterministic_climate_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWritten to: {os.path.abspath(out_path)}")
