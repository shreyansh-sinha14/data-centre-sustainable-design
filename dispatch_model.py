"""
dispatch_model.py
=================
8,760-hour dispatch model for the 5 MW reference data centre described in
"An Integrated Analytical Framework for Sustainable Data Centre Design".

Reconstructed from the manuscript (Section 3.8, Stages 1-8, Eqs. 11-28 and
the sizing / TEE / coupling / LCA sections). Every parameter that the paper
states is a named field in `Params`; nothing is tuned to hit a target.

Inputs (CSV, 2005-2023 hourly):
    nasa_power_<city>_csv.csv   WS50M, WS10M, T2M   (local solar time)
    pvgis_<city>_csv.csv        P (W) for 4 kWp      (UTC, stamped hh:30)

Conventions
-----------
* 29 Feb removed -> 19 years x 8,760 h.
* PVGIS is shifted UTC -> local solar time by round(lon/15) h (5 h for all
  three cities) within each year (circular, so edge effects touch 5 h/yr).
* IT load profile (0.90 for 08:00-20:00, 0.50 otherwise) is applied in local
  solar time, which is also the time base of the NASA POWER files.
* Energy in MWh == mean MW per hour.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

N_YEARS, N_HOURS = 19, 8760
YEARS = list(range(2005, 2024))

# --------------------------------------------------------------------------
# City definitions (baseline cooling overheads from Stage 2 / Stage 8)
# --------------------------------------------------------------------------
CITIES = {
    "jaipur":  dict(lon=75.819, base_cool=0.51),
    "delhi":   dict(lon=77.209, base_cool=0.47),
    "chennai": dict(lon=80.270, base_cool=0.55),
}


# --------------------------------------------------------------------------
# Parameters (every value is taken from the manuscript)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # facility / load
    it_mw: float = 5.0
    load_peak: float = 0.90           # 08:00-20:00 local solar
    load_off: float = 0.50            # 20:00-08:00
    peak_start: int = 8
    peak_end: int = 20
    util_scale: float = 1.0           # +/-10 % utilisation sensitivity
    # cooling mix and overheads
    x_imm: float = 0.60               # immersion fraction
    o_air_jaipur: float = 0.20        # CFD-informed air-cooled overhead (Jaipur)
    o_aux: float = 0.045              # immersion pumps/auxiliaries
    f_des: float = 0.03               # dry-cooler fan share at design approach
    dT_des: float = 10.0              # design approach, K
    t_cond: float = 50.0              # condenser temperature, C
    cop_ch: float = 4.0               # trim-chiller COP
    dist: float = 0.06                # power-distribution loss
    dist_full: float = 0.05           # ... when fully immersion cooled (x = 1)
    light: float = 0.01
    # renewables
    pv_mw: float = 4.0
    wind_mw: float = 0.0
    wind_cf_override: float | None = None   # e.g. 0.25 for a better site
    hub_h: float = 80.0
    cut_in: float = 3.0
    rated: float = 12.0
    cut_out: float = 25.0
    # LAES
    laes_mwh: float = 10.0            # output-referenced capacity
    laes_mw: float = 5.0
    rte: float = 0.60
    boiloff_day: float = 0.001
    # TEE confidential computing
    f_tee: float = 0.25
    eta_tee: float = 0.0
    # carbon / money
    ci: float = 0.71                  # kg CO2/kWh (CEA v21.0)
    tariff: float = 0.10              # $/kWh
    capex_cooling: float = 3.0        # $M
    capex_laes: float = 5.0
    capex_pv_per_mw: float = 321.96e5 / 95.9927 / 1e6 / 1.0  # Rs/MW -> $M/MW
    capex_wind: float = 5.04          # $M (NREL 2024: 3,362 $/kW x 1.5 MW)
    opex_wind: float = 0.0615         # $M/yr
    include_laes_capex: bool = True


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _find(data_dir, prefix, city):
    f = os.path.join(data_dir, f"{prefix}_{city}_csv.csv")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    return f


def _drop_feb29(arr_flat, month, day):
    keep = ~((month == 2) & (day == 29))
    return arr_flat[keep]


@lru_cache(maxsize=None)
def load_city(city: str, data_dir: str = "/mnt/user-data/uploads") -> dict:
    """Return dict of (19, 8760) arrays: T (C), ws10, ws50, pv_cf (fraction of
    installed kWp), plus elevation (m) and the data-derived wind exponent."""
    cache = os.path.join(data_dir if os.access(data_dir, os.W_OK) else "/tmp",
                         f".cache_{city}.npz")
    nasa_f = _find(data_dir, "nasa_power", city)
    with open(nasa_f, "r", errors="ignore") as fh:
        head = [next(fh) for _ in range(12)]
    elev = float(re.search(r"=\s*([\d.]+)\s*meters", " ".join(head)).group(1))

    if os.path.exists(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
        out["elev"] = elev
        return out

    # NASA POWER (local solar time)
    n = pd.read_csv(nasa_f, skiprows=11)
    assert len(n) == 166536, f"{city}: unexpected NASA length {len(n)}"
    assert (n[["WS50M", "WS10M", "T2M"]] > -990).all().all(), "missing data"
    mo, dy = n.MO.values, n.DY.values
    sh = lambda c: _drop_feb29(n[c].values, mo, dy).reshape(N_YEARS, N_HOURS)
    T, ws10, ws50 = sh("T2M"), sh("WS10M"), sh("WS50M")

    # PVGIS (UTC, hh:30) -> local solar time
    p = pd.read_csv(_find(data_dir, "pvgis", city), skiprows=10, nrows=166536)
    assert len(p) == 166536 and p.time.iloc[0].startswith("20050101"), "PVGIS parse"
    pm = p.time.str[4:6].astype(int).values
    pd_ = p.time.str[6:8].astype(int).values
    pv = _drop_feb29(p["P"].values.astype(float), pm, pd_).reshape(N_YEARS, N_HOURS)
    shift = int(round(CITIES[city]["lon"] / 15.0))
    pv = np.roll(pv, shift, axis=1) / 4000.0          # 4 kWp -> capacity factor

    np.savez(cache, T=T, ws10=ws10, ws50=ws50, pv_cf=pv)
    return dict(T=T, ws10=ws10, ws50=ws50, pv_cf=pv, elev=elev)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def it_profile(p: Params) -> np.ndarray:
    """Hourly IT load (MW), length 8760, local solar time."""
    hod = np.arange(N_HOURS) % 24
    lam = np.where((hod >= p.peak_start) & (hod < p.peak_end), p.load_peak, p.load_off)
    return p.it_mw * lam * p.util_scale


def o_imm_rest(T, p: Params):
    """Fan + trim-chiller terms of Eq. (13) (everything except o_aux)."""
    dT = p.t_cond - T
    fan = p.f_des * (p.dT_des / np.maximum(dT, p.dT_des)) ** 3
    chiller = np.maximum(0.0, 1.0 - dT / p.dT_des) / p.cop_ch
    return fan + chiller


def o_imm(T, p: Params):
    return p.o_aux + o_imm_rest(T, p)


def o_air_city(city, p: Params):
    return p.o_air_jaipur * CITIES[city]["base_cool"] / CITIES["jaipur"]["base_cool"]


def dist_loss(p: Params):
    return p.dist_full if p.x_imm >= 1.0 else p.dist


def facility_load(T, city, p: Params):
    """Hourly facility load (MW) for one weather year T (8760,). Includes the
    TEE energy penalty as extra IT load (Eq. TEE-1)."""
    it = it_profile(p) * (1.0 + p.f_tee * p.eta_tee)
    cool = p.x_imm * o_imm(T, p) + (1.0 - p.x_imm) * o_air_city(city, p)
    return it * (1.0 + cool + dist_loss(p) + p.light), it


def alpha_shear(w) -> float:
    """Power-law exponent from the 10 m and 50 m series (ratio of means)."""
    return float(np.log(w["ws50"].mean() / w["ws10"].mean()) / np.log(50.0 / 10.0))


def wind_cf(w, y, p: Params):
    """Hourly wind capacity factor for year y: shear to hub height, air-density
    correction, generic cubic-ramp curve (3 / 12 / 25 m/s)."""
    a = alpha_shear(w)
    v = w["ws50"][y] * (p.hub_h / 50.0) ** a
    Tk = w["T"][y] + 273.15
    pres = 101325.0 * (1 - 2.25577e-5 * w["elev"]) ** 5.25588
    rho = pres / (287.05 * Tk)
    v = v * (rho / 1.225) ** (1 / 3)
    cf = np.where(v < p.cut_in, 0.0,
         np.where(v < p.rated, (v**3 - p.cut_in**3) / (p.rated**3 - p.cut_in**3),
         np.where(v <= p.cut_out, 1.0, 0.0)))
    return cf


def dispatch(r, l, rte, cap_mwh, pw_mw, boiloff_day, use_laes=True):
    """Hour-by-hour dispatch. Returns (direct, laes_out, curtailed) in MWh.

    r >= l : direct = l; surplus charges LAES (power- and capacity-limited) or
             is curtailed.
    r <  l : direct = r; LAES covers the deficit from stored energy.
    Stored energy is output-referenced (charged energy x RTE) so discharge can
    never exceed what was stored; boil-off decays the store each hour.
    """
    surplus = np.maximum(r - l, 0.0)
    direct = np.minimum(r, l).sum()
    if not use_laes or surplus.sum() == 0.0:
        return direct, 0.0, surplus.sum()
    deficit = np.maximum(l - r, 0.0)
    keep = 1.0 - boiloff_day / 24.0
    stored = out = curt = 0.0
    for s, d in zip(surplus, deficit):
        stored *= keep
        if s > 0.0:
            ch = min(s, pw_mw, (cap_mwh - stored) / rte)
            ch = max(ch, 0.0)
            stored += ch * rte
            curt += s - ch
        elif d > 0.0 and stored > 0.0:
            dis = min(d, stored, pw_mw)
            stored -= dis
            out += dis
    return direct, out, curt


# --------------------------------------------------------------------------
# One city, one parameter set -> annual results for each of the 19 years
# --------------------------------------------------------------------------
def simulate(city: str, p: Params = Params(), use_laes: bool = True,
             data_dir: str = "/mnt/user-data/uploads") -> pd.DataFrame:
    w = load_city(city, data_dir)
    base_pue = 1.0 + CITIES[city]["base_cool"] + 0.06 + 0.01
    rows = []
    for y in range(N_YEARS):
        l, it = facility_load(w["T"][y], city, p)
        r = p.pv_mw * w["pv_cf"][y]
        if p.wind_mw > 0:
            wc = (np.full(N_HOURS, p.wind_cf_override) if p.wind_cf_override
                  else wind_cf(w, y, p))
            r = r + p.wind_mw * wc
        direct, lout, curt = dispatch(r, l, p.rte, p.laes_mwh, p.laes_mw,
                                      p.boiloff_day, use_laes)
        e_it, e_tot = it.sum(), l.sum()
        e_it_useful = it_profile(p).sum()            # CUE denominator (paper)
        grid = e_tot - direct - lout
        base_e = e_it_useful * base_pue
        rows.append(dict(
            year=YEARS[y], E_IT=e_it_useful, E_total=e_tot, PUE=e_tot / e_it,
            gen=r.sum(), pv_gen=(p.pv_mw * w["pv_cf"][y]).sum(),
            direct=direct, laes_out=lout, curtailed=curt, grid=grid,
            f_ren=(direct + lout) / e_tot,
            co2_t=grid * p.ci, CUE=grid * p.ci / e_it_useful,
            base_E=base_e, base_co2_t=base_e * p.ci,
            base_CUE=base_e * p.ci / e_it_useful, base_PUE=base_pue))
    return pd.DataFrame(rows)


def capex(p: Params) -> float:
    c = p.capex_cooling + p.capex_pv_per_mw * p.pv_mw + p.capex_wind * (p.wind_mw / 1.5)
    if p.include_laes_capex:
        c += p.capex_laes
    return c


def economics(m: dict, p: Params, capex_musd: float | None = None) -> dict:
    """m = mean-over-years results. Savings use the traditional-baseline energy."""
    capx = capex(p) if capex_musd is None else capex_musd
    savings = (m["base_E"] - m["grid"]) * p.tariff * 1e3 - p.opex_wind * (p.wind_mw > 0) * 1e6
    return dict(CAPEX=capx, savings_usd=savings, payback=capx * 1e6 / savings)


def summarise(city, p=Params(), use_laes=True, data_dir="/mnt/user-data/uploads"):
    df = simulate(city, p, use_laes, data_dir)
    m = df.mean(numeric_only=True).to_dict()
    m.update(economics(m, p))
    m["cue_reduction"] = 1 - m["CUE"] / m["base_CUE"]
    m["co2_reduction"] = 1 - m["co2_t"] / m["base_co2_t"]
    m["f_ren_min"], m["f_ren_max"] = df.f_ren.min(), df.f_ren.max()
    return m, df


# --------------------------------------------------------------------------
# Auxiliary analyses from the manuscript
# --------------------------------------------------------------------------
def coupling_bound(city, t_cond=50.0, d_hx=5.0, rte0=0.60, data_dir="/mnt/user-data/uploads"):
    """Eq. (drte): ideal-gas upper bound on the RTE gain from condenser-heat
    preheating, averaged over the hourly series. Returns (annual pp, monthly pp)."""
    T = load_city(city, data_dir)["T"]
    gain = rte0 * np.maximum(0.0, (t_cond - d_hx) - T) / (T + 273.15)
    month = np.repeat(np.arange(12), np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 24)
    monthly = np.array([gain[:, month == m].mean() for m in range(12)]) * 100
    return gain.mean() * 100, monthly


def hours_above(city, thr=40.0, data_dir="/mnt/user-data/uploads"):
    return float((load_city(city, data_dir)["T"] > thr).sum(axis=1).mean())


def mean_immersion_overhead(city, p=Params(), data_dir="/mnt/user-data/uploads"):
    return float(o_imm(load_city(city, data_dir)["T"], p).mean())


def pareto_sweep(city="jaipur", p=Params(), step=0.05, data_dir="/mnt/user-data/uploads"):
    """Immersion-fraction sweep (Fig. 5): PUE, CUE, CAPEX. CAPEX is cooling
    (piecewise-linear through $1.0M @0, $3.0M @0.6, $5.5M @1.0) + LAES + PV."""
    rows = []
    for x in np.arange(0, 1 + 1e-9, step):
        q = replace(p, x_imm=float(x))
        m, _ = summarise(city, q, True, data_dir)
        cool_capex = np.interp(x, [0, 0.6, 1.0], [1.0, 3.0, 5.5])
        rows.append(dict(x_imm=x, PUE=m["PUE"], CUE=m["CUE"], f_ren=m["f_ren"],
                         CAPEX_musd=cool_capex + p.capex_laes + p.capex_pv_per_mw * p.pv_mw))
    return pd.DataFrame(rows)


def lca(m, p=Params()):
    """Table 'lca': simplified annualised embodied carbon (t/yr)."""
    pv = m["pv_gen"] * 1e3 * 40e-6                 # 40 g/kWh -> t
    wind = (m["gen"] - m["pv_gen"]) * 1e3 * 6.5e-6
    laes = p.laes_mwh * 60 * 1.85 / 25
    fluid = 200 * (0.6 * 625) * 12 / 1e3 / 15
    emb = pv + wind + laes + fluid
    avoided = m["base_co2_t"] - m["co2_t"]
    return dict(pv=pv, wind=wind, laes=laes, fluid=fluid, embodied=emb,
                avoided=avoided, net=avoided - emb,
                net_reduction=(avoided - emb) / m["base_co2_t"])


if __name__ == "__main__":
    for c in CITIES:
        m, _ = summarise(c)
        print(f"{c:8s} PUE={m['PUE']:.3f} gen={m['pv_gen']:.0f} f_ren={m['f_ren']*100:.1f}% "
              f"CUE={m['CUE']:.3f} red={m['cue_reduction']*100:.1f}% payback={m['payback']:.2f}")
