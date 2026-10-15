"""
monte_carlo.py
==============
Joint Monte Carlo sensitivity analysis (manuscript Section 3.9.2, Table 5).

* N = 100,000 draws; each draw runs the full 8,760-h dispatch on one of the 19
  real weather years chosen uniformly at random (inter-annual variability is
  therefore part of the spread).
* Eight inputs, independent triangular distributions with mode at the central
  estimate and bounds +/-10 % (+/-25 % for CAPEX):
      immersion auxiliary overhead, air-cooled overhead, distribution loss,
      LAES round-trip efficiency, grid carbon intensity, solar capacity factor,
      total CAPEX, electricity tariff.
* The solar profile of the drawn year is rescaled by CF_sampled / CF_central
  (clipped at installed capacity), so hourly shape is that of the real year.
* The traditional-baseline energy used in the payback denominator is fixed at
  its point estimate (30,660 MWh x 1.58), as stated in the paper.

Draws are grouped by weather year and evaluated as (batch x 8760) matrices;
any draw that actually has hours with generation > load (not the case at the
reference sizing) falls back to the exact sequential LAES dispatch.

Usage:  python monte_carlo.py [--n 100000] [--seed 20260415] [--city jaipur]
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import dispatch_model as dm

CAPEX_CENTRAL = None   # computed from Params


def tri(rng, centre, rel, n):
    return rng.triangular(centre * (1 - rel), centre, centre * (1 + rel), n)


def run(n=100_000, seed=20260415, city="jaipur", data_dir="/mnt/user-data/uploads",
        batch=1000, noLAES_mode="flat"):
    p0 = dm.Params()
    w = dm.load_city(city, data_dir)
    rng = np.random.default_rng(seed)

    capex0 = dm.capex(p0)
    # Point-estimate quantities
    pt, _ = dm.summarise(city, p0, True, data_dir)
    cf_central = w["pv_cf"].mean()                       # 19-yr mean solar CF
    base_E = pt["base_E"]                                # fixed: 48,443 MWh
    it = dm.it_profile(p0)
    e_it_useful = it.sum()
    o_air0 = dm.o_air_city(city, p0)

    # --- draw the eight inputs and the weather year ------------------------
    d = dict(
        o_aux=tri(rng, p0.o_aux, 0.10, n),
        o_air=tri(rng, o_air0, 0.10, n),
        dist=tri(rng, p0.dist, 0.10, n),
        rte=tri(rng, p0.rte, 0.10, n),
        ci=tri(rng, p0.ci, 0.10, n),
        cf=tri(rng, cf_central, 0.10, n),
        capex=tri(rng, capex0, 0.25, n),
        tariff=tri(rng, p0.tariff, 0.10, n),
    )
    year = rng.integers(0, dm.N_YEARS, n)

    E_tot = np.empty(n); direct = np.empty(n); lout = np.empty(n)
    x, light = p0.x_imm, p0.light
    n_fallback = 0

    for y in range(dm.N_YEARS):
        idx = np.where(year == y)[0]
        rest = dm.o_imm_rest(w["T"][y], p0)                  # (8760,)
        pv_y = w["pv_cf"][y]
        sum_it = it.sum()
        sum_it_rest = (it * rest).sum()
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            # closed-form facility energy: sum IT(t)(1 + x*(o_aux+rest) + (1-x)*o_air + dist + light)
            c = x * d["o_aux"][b] + (1 - x) * d["o_air"][b] + d["dist"][b] + light
            E_tot[b] = sum_it * (1 + c) + x * sum_it_rest
            # generation (clipped at installed capacity) and load matrices
            r = np.minimum(p0.pv_mw * pv_y[None, :] * (d["cf"][b] / cf_central)[:, None], p0.pv_mw)
            l = it[None, :] * (1 + c[:, None] + x * rest[None, :])
            direct[b] = np.minimum(r, l).sum(axis=1)
            lout[b] = 0.0
            surplus_rows = np.where((r - l).max(axis=1) > 0)[0]
            for j in surplus_rows:                            # exact LAES dispatch
                n_fallback += 1
                dr, lo, _ = dm.dispatch(r[j], l[j], d["rte"][b[j]], p0.laes_mwh,
                                        p0.laes_mw, p0.boiloff_day, True)
                direct[b[j]], lout[b[j]] = dr, lo

    grid = E_tot - direct - lout
    out = pd.DataFrame(dict(
        PUE=E_tot / e_it_useful,
        f_ren=(direct + lout) / E_tot,
        CUE=grid * d["ci"] / e_it_useful,
        payback=d["capex"] * 1e6 / ((base_E - grid) * d["tariff"] * 1e3),
    ))
    # variant without LAES: the paper subtracts the flat $5.0M LAES CAPEX (verified against its 1.6-3.0 yr range)
    if noLAES_mode == "proportional":
        capex_no = d["capex"] * (capex0 - p0.capex_laes) / capex0
    else:                                                     # subtract flat $5.0M
        capex_no = d["capex"] - p0.capex_laes
    out["payback_noLAES"] = capex_no * 1e6 / ((base_E - grid) * d["tariff"] * 1e3)
    out.attrs["n_fallback"] = n_fallback
    out.attrs["point"] = pt
    return out


def pctl(s):
    return np.percentile(s, [10, 50, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20260415)
    ap.add_argument("--city", default="jaipur")
    ap.add_argument("--out", default="/mnt/user-data/outputs/monte_carlo_draws.csv.gz")
    a = ap.parse_args()

    df = run(a.n, a.seed, a.city)
    pt = df.attrs["point"]
    print(f"N={a.n:,}  city={a.city}  seed={a.seed}  exact-LAES fallbacks: {df.attrs['n_fallback']}")
    print(f"{'Output':22s} {'P10':>8s} {'P50':>8s} {'P90':>8s} {'point':>8s}")
    pts = dict(PUE=pt["PUE"], f_ren=pt["f_ren"], CUE=pt["CUE"], payback=pt["payback"])
    for k in ["PUE", "f_ren", "CUE", "payback"]:
        p10, p50, p90 = pctl(df[k])
        sc = 100 if k == "f_ren" else 1
        print(f"{k:22s} {p10*sc:8.3f} {p50*sc:8.3f} {p90*sc:8.3f} {pts[k]*sc:8.3f}")
    p10, p50, p90 = pctl(df["payback_noLAES"])
    print(f"{'payback, no LAES':22s} {p10:8.2f} {p50:8.2f} {p90:8.2f}")
    df.round(6).to_csv(a.out, index=False, compression="gzip")
    print("draws saved ->", a.out)


if __name__ == "__main__":
    main()
