"""
run_validation.py
=================
Reproduces the manuscript's quantitative claims from the raw NASA POWER and
PVGIS files and reports PASS/FAIL against the published value.

    python run_validation.py            # fast checks (seconds)
    python run_validation.py --mc       # also the 100,000-draw Monte Carlo

Each check is (label, paper value, model value, tolerance). Tolerances are set
to the rounding precision the paper prints (e.g. +/-0.05 pp for "18.0 %"), so a
FAIL means the paper's printed number does not follow from its own stated model.
"""
import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

import dispatch_model as dm
from dispatch_model import Params

rows = []


def chk(section, label, paper, model, tol):
    ok = abs(paper - model) <= tol
    rows.append(dict(section=section, check=label, paper=paper,
                     model=round(float(model), 4), tol=tol, status="PASS" if ok else "FAIL"))


def main(run_mc=False):
    P = Params()
    S = {c: dm.summarise(c, P) for c in dm.CITIES}          # (mean dict, per-year df)
    j, jdf = S["jaipur"]

    # ---- Table 4 / Stages 2-4, 8 -----------------------------------------
    t4 = {
        "jaipur":  dict(cf=.186, h40=323, oimm=5.1, pue=1.182, gen=6505, fren=18.0, cue=.688, red=38.6, pb=5.0),
        "delhi":   dict(cf=.183, h40=453, oimm=5.3, pue=1.177, gen=6404, fren=17.7, cue=.687, red=37.1, pb=5.3),
        "chennai": dict(cf=.168, h40=8,   oimm=4.9, pue=1.186, gen=5875, fren=16.2, cue=.706, red=38.6, pb=4.9),
    }
    for c, e in t4.items():
        m = S[c][0]
        w = dm.load_city(c)
        chk("T4", f"{c} solar CF", e["cf"], w["pv_cf"].mean(), .0005)
        chk("T4", f"{c} h>40C", e["h40"], dm.hours_above(c), .5)
        chk("T4", f"{c} mean immersion overhead %", e["oimm"], 100 * dm.mean_immersion_overhead(c), .05)
        chk("T4", f"{c} proposed PUE", e["pue"], m["PUE"], .0005)
        chk("T4", f"{c} solar MWh/yr", e["gen"], m["pv_gen"], 0.5)
        chk("T4", f"{c} renewable fraction %", e["fren"], 100 * m["f_ren"], .05)
        chk("T4", f"{c} CUE", e["cue"], m["CUE"], .0005)
        chk("T4", f"{c} CUE reduction %", e["red"], 100 * m["cue_reduction"], .05)
        chk("T4", f"{c} payback yr", e["pb"], m["payback"], .05)
    chk("T4", "baseline PUE delhi", 1.54, S["delhi"][0]["base_PUE"], 1e-9)
    chk("T4", "baseline CUE delhi", 1.093, S["delhi"][0]["base_CUE"], .0005)
    chk("T4", "baseline CUE chennai", 1.150, S["chennai"][0]["base_CUE"], .0005)

    # ---- Jaipur reference (Stages 1-5, Table 6 mixed column) --------------
    chk("Ref", "E_IT MWh", 30660, j["E_IT"], .5)
    chk("Ref", "E_total MWh", 36228, j["E_total"], 1)
    chk("Ref", "baseline energy MWh", 48443, j["base_E"], 1)
    chk("Ref", "baseline CO2 t", 34394, j["base_co2_t"], 1)
    chk("Ref", "baseline CUE", 1.122, j["base_CUE"], .0005)
    chk("Ref", "grid MWh", 29723, j["grid"], 1)
    chk("Ref", "CO2 t", 21103, j["co2_t"], 2)
    chk("Ref", "f_ren min across years %", 17.3, 100 * j["f_ren_min"], .05)
    chk("Ref", "f_ren max across years %", 18.5, 100 * j["f_ren_max"], .05)
    chk("Ref", "solar CF range min", .179, jdf.pv_gen.min() / (4 * 8760), .0005)
    chk("Ref", "solar CF range max", .192, jdf.pv_gen.max() / (4 * 8760), .0005)
    chk("Ref", "LAES discharge MWh (all yrs max)", 0, jdf.laes_out.max(), 1e-9)
    chk("Ref", "curtailment MWh (all yrs max)", 0, jdf.curtailed.max(), 1e-9)
    chk("Ref", "CAPEX $M", 9.34, j["CAPEX"], .005)
    chk("Ref", "annual savings $M", 1.872, j["savings_usd"] / 1e6, .001)
    chk("Ref", "payback no LAES", 2.3, dm.economics(j, replace(P, include_laes_capex=False))["payback"], .05)
    o = dm.o_imm(np.array([40.0, 45.0]), P)
    chk("Stage2", "o_imm at 40C %", 7.5, 100 * o[0], .05)
    chk("Stage2", "o_imm at 45C %", 20.0, 100 * o[1], .05)

    # ---- Table 6 full immersion -------------------------------------------
    full = replace(P, x_imm=1.0)
    f, _ = dm.summarise("jaipur", full)
    chk("T6", "full-imm E MWh", 34115, f["E_total"], 2)
    chk("T6", "full-imm PUE", 1.11, f["PUE"], .005)
    chk("T6", "full-imm f_ren %", 19.1, 100 * f["f_ren"], .05)
    chk("T6", "full-imm grid MWh", 27610, f["grid"], 2)
    chk("T6", "full-imm CO2 t", 19603, f["co2_t"], 2)
    chk("T6", "full-imm CUE", 0.639, f["CUE"], .0005)
    chk("T6", "full-imm reduction %", 43.0, 100 * f["cue_reduction"], .05)
    chk("T6", "full-imm payback (CAPEX 11.84)", 5.7, dm.economics(f, full, 11.84)["payback"], .05)

    # ---- Table 3 tariff ----------------------------------------------------
    for tar, gc, sv, pb, pb0 in [(.07, 2.08, 1.31, 7.1, 3.3), (.10, 2.97, 1.87, 5.0, 2.3), (.13, 3.86, 2.43, 3.8, 1.8)]:
        q = replace(P, tariff=tar)
        m, _ = dm.summarise("jaipur", q)
        chk("T3", f"${tar} grid cost $M", gc, m["grid"] * tar * 1e3 / 1e6, .005)
        chk("T3", f"${tar} savings $M", sv, m["savings_usd"] / 1e6, .005)
        chk("T3", f"${tar} payback", pb, m["payback"], .05)
        chk("T3", f"${tar} payback no LAES", pb0, dm.economics(m, replace(q, include_laes_capex=False))["payback"], .05)

    # ---- Table 'sizing' (PV oversizing) -----------------------------------
    sizing = {8: (13011, 35.7, 35.3, 136, 227, 0), 10: (16264, 43.1, 40.4, 983, 1638, 0),
              12: (19516, 49.6, 43.6, 2201, 3731, 61)}
    for mw, (g, fl, fn, lo, cn, cl) in sizing.items():
        a, _ = dm.summarise("jaipur", replace(P, pv_mw=mw), True)
        b, _ = dm.summarise("jaipur", replace(P, pv_mw=mw), False)
        chk("Sizing", f"{mw} MW generation", g, a["pv_gen"], 1)
        chk("Sizing", f"{mw} MW f_ren with LAES %", fl, 100 * a["f_ren"], .05)
        chk("Sizing", f"{mw} MW f_ren no LAES %", fn, 100 * b["f_ren"], .05)
        chk("Sizing", f"{mw} MW LAES out MWh", lo, a["laes_out"], 2)
        chk("Sizing", f"{mw} MW curtailed no LAES MWh", cn, b["curtailed"], 2)
        chk("Sizing", f"{mw} MW curtailed with LAES MWh", cl, a["curtailed"], 2)

    # ---- TEE table ---------------------------------------------------------
    for eta, fr, cue in [(0, 18.0, .688), (.05, 17.7, .699), (.15, 17.3, .720), (.40, 16.3, .772)]:
        m, _ = dm.summarise("jaipur", replace(P, eta_tee=eta))
        chk("TEE", f"eta={eta} f_ren %", fr, 100 * m["f_ren"], .05)
        chk("TEE", f"eta={eta} CUE", cue, m["CUE"], .0005)
    chk("TEE", "dE_TEE MWh/yr", 1150, .25 * 30660 * .15, 1)

    # ---- Thermal coupling bound -------------------------------------------
    for c, e in [("jaipur", 4.0), ("delhi", 4.0), ("chennai", 3.4)]:
        chk("Coupling", f"{c} mean dRTE pp", e, dm.coupling_bound(c)[0], .05)
    _, mon = dm.coupling_bound("jaipur")
    chk("Coupling", "Jaipur Jan pp", 6.4, mon[0], .05)
    chk("Coupling", "Jaipur Jun pp", 2.1, mon[5], .05)
    chk("Coupling", "Tcond 45C pp", 3.0, dm.coupling_bound("jaipur", t_cond=45)[0], .05)
    chk("Coupling", "Tcond 55C pp", 5.0, dm.coupling_bound("jaipur", t_cond=55)[0], .05)
    a, _ = dm.summarise("jaipur", replace(P, pv_mw=12, rte=.60), True)
    b, _ = dm.summarise("jaipur", replace(P, pv_mw=12, rte=.64), True)
    chk("Coupling", "12 MW f_ren at RTE 60% (text: 49.7)", 49.7, 100 * a["f_ren"], .05)
    chk("Coupling", "12 MW f_ren at RTE 64% (text: 50.0)", 50.0, 100 * b["f_ren"], .05)
    chk("Coupling", "4 MW change from coupling < 0.1 pp", 0.0,
        100 * (dm.summarise("jaipur", replace(P, rte=.64))[0]["f_ren"] - j["f_ren"]), .1)

    # ---- Wind sensitivity ---------------------------------------------------
    for c, a_ in [("jaipur", .24), ("delhi", .26), ("chennai", .18)]:
        chk("Wind", f"{c} shear exponent", a_, dm.alpha_shear(dm.load_city(c)), .005)
    wp = replace(P, wind_mw=1.5)
    w, wdf = dm.summarise("jaipur", wp)
    wcf = (wdf.gen - wdf.pv_gen) / (1.5 * 8760)
    chk("Wind", "Jaipur wind CF mean", .116, wcf.mean(), .0005)
    chk("Wind", "Jaipur wind CF min", .090, wcf.min(), .0005)
    chk("Wind", "Jaipur wind CF max", .134, wcf.max(), .0005)
    chk("Wind", "wind MWh/yr", 1524, (wdf.gen - wdf.pv_gen).mean(), 5)
    chk("Wind", "f_ren %", 22.2, 100 * w["f_ren"], .05)
    chk("Wind", "CUE", .653, w["CUE"], .0005)
    chk("Wind", "reduction %", 41.8, 100 * w["cue_reduction"], .05)
    chk("Wind", "payback yr", 7.3, w["payback"], .05)
    w2, _ = dm.summarise("jaipur", replace(wp, wind_cf_override=.25))
    chk("Wind", "CF 0.25 f_ren %", 26.5, 100 * w2["f_ren"], .05)
    chk("Wind", "CF 0.25 CUE", .617, w2["CUE"], .0005)
    chk("Wind", "CF 0.25 payback", 6.8, w2["payback"], .05)

    # ---- Deterministic +/-10 % / tornado ----------------------------------
    pu = [dm.summarise("jaipur", replace(P, util_scale=u))[0]["payback"] for u in (1.1, 0.9)]
    chk("Sens", "utilisation +10% payback", 4.7, pu[0], .05)
    chk("Sens", "utilisation -10% payback", 5.3, pu[1], .05)
    fr = [dm.summarise("jaipur", replace(P, pv_mw=4 * s))[0]["f_ren"] * 100 for s in (0.9, 1.1)]
    chk("Sens", "solar CF -10% f_ren %", 16.2, fr[0], .05)
    chk("Sens", "solar CF +10% f_ren %", 19.8, fr[1], .05)
    pue = [dm.summarise("jaipur", replace(P, o_aux=.045 * s))[0]["PUE"] for s in (0.9, 1.1)]
    chk("Sens", "o_aux -10% PUE", 1.179, pue[0], .0005)
    chk("Sens", "o_aux +10% PUE", 1.184, pue[1], .0005)
    cue = [dm.summarise("jaipur", replace(P, ci=.71 * s))[0]["CUE"] for s in (0.9, 1.1)]
    chk("Sens", "CI -10% CUE", .619, cue[0], .0005)
    chk("Sens", "CI +10% CUE", .757, cue[1], .0005)
    chk("Sens", "CAPEX -25% payback", 3.7, 0.75 * 9.34e6 / j["savings_usd"], .05)
    chk("Sens", "CAPEX +25% payback", 6.2, 1.25 * 9.34e6 / j["savings_usd"], .05)

    # ---- LCA ----------------------------------------------------------------
    L = dm.lca(j, P)
    chk("LCA", "PV embodied t/yr", 260.2, L["pv"], .1)
    chk("LCA", "LAES embodied t/yr", 44.4, L["laes"], .05)
    chk("LCA", "fluid embodied t/yr", 60.0, L["fluid"], .05)
    chk("LCA", "total embodied t/yr", 364.6, L["embodied"], .2)
    chk("LCA", "operational avoided t/yr", 13291, L["avoided"], 2)
    chk("LCA", "net avoided t/yr", 12926, L["net"], 2)
    chk("LCA", "net reduction %", 37.6, 100 * L["net_reduction"], .05)

    # ---- Pareto boundaries (Fig. 5) ---------------------------------------
    pf = dm.pareto_sweep("jaipur", P, 0.2)
    chk("Pareto", "x=0.6 PUE", 1.18, pf.loc[np.isclose(pf.x_imm, .6), "PUE"].iloc[0], .005)
    chk("Pareto", "x=1.0 PUE", 1.11, pf.loc[np.isclose(pf.x_imm, 1), "PUE"].iloc[0], .005)
    chk("Pareto", "x=1.0 CUE", 0.639, pf.loc[np.isclose(pf.x_imm, 1), "CUE"].iloc[0], .0005)
    chk("Pareto", "x=0.6 CAPEX", 9.34, pf.loc[np.isclose(pf.x_imm, .6), "CAPEX_musd"].iloc[0], .005)
    chk("Pareto", "x=1.0 CAPEX", 11.84, pf.loc[np.isclose(pf.x_imm, 1), "CAPEX_musd"].iloc[0], .005)

    # ---- Monte Carlo (Table 5) --------------------------------------------
    if run_mc:
        import monte_carlo as mc
        df = mc.run(100_000, 20260415)
        tgt = {"PUE": (1.176, 1.182, 1.187, 1), "f_ren": (16.9, 18.0, 19.0, 100),
               "CUE": (.649, .688, .727, 1), "payback": (4.27, 4.99, 5.74, 1)}
        for k, (a, b, c, s) in tgt.items():
            q = np.percentile(df[k], [10, 50, 90]) * s
            tol = .0015 if k == "PUE" else (.06 if k == "f_ren" else (.0015 if k == "CUE" else .02))
            for nm, pv, mv in zip(("P10", "P50", "P90"), (a, b, c), q):
                chk("MC", f"{k} {nm}", pv, mv, tol)
        q = np.percentile(df["payback_noLAES"], [10, 50, 90])
        chk("MC", "no-LAES payback P10", 1.6, q[0], .05)
        chk("MC", "no-LAES payback P50", 2.3, q[1], .05)
        chk("MC", "no-LAES payback P90", 3.0, q[2], .05)

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 500, "display.max_colwidth", 60)
    print(R.to_string(index=False))
    print("\nSummary by section:")
    print(R.groupby("section").status.value_counts().unstack(fill_value=0).to_string())
    print(f"\nTOTAL: {(R.status=='PASS').sum()} PASS, {(R.status=='FAIL').sum()} FAIL of {len(R)}")
    R.to_csv("/mnt/user-data/outputs/validation_report.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", action="store_true")
    main(ap.parse_args().mc)
