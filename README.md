# Data and Code Availability

Supporting code for:

> Sinha, S., Shukla, S., Gupta, P., Taneja, S.: *An Integrated Analytical Framework for Sustainable Data Centre Design: CFD-Informed Immersion Cooling, Cryogenic Energy Storage, and Quantum-Secure AI Optimisation.* (under review)

The study is an analytical design framework. No experimental data were generated and no physical prototype was built.

## What this repository contains

The repository has two independent parts:

| Folder | Part | Manuscript item |
|---|---|---|
| `monte_carlo_facility_level/` | 8,760-hour facility dispatch model, 100,000-draw joint Monte Carlo analysis, and a validation script that checks the model against the manuscript's printed numbers | Tables 3 to 6, Section 3.9.2 (Table 5), sizing, TEE, wind, LCA and Pareto results |
| `scripts/`, `results/` | Preliminary Q-learning controller study on a simplified simulator | Fig. 7 |

```
monte_carlo_facility_level/
  dispatch_model.py            8,760-h dispatch model of the 5 MW reference
                               facility (Stages 1-8, Eqs. 11-28). Every
                               parameter is a named field of `Params`.
  monte_carlo.py               Joint Monte Carlo sensitivity analysis
                               (N = 100,000 draws, 8 uncertain inputs).
  run_validation.py            Re-derives the manuscript's quantitative
                               claims and reports PASS/FAIL for each.
  monte_carlo_draws_csv.gz     All 100,000 draws (outputs), gzipped CSV.

scripts/
  train_qlearning.py           Simulator, Q-learning training, and evaluation
                               of the three policies (one run per PV size)
  make_fig7.py                 Regenerates Fig. 7 from the files in results/

results/
  results_pv4.json             Evaluation results, PV 4 MW (mean and std over 5 years)
  results_pv10.json            Evaluation results, PV 10 MW
  learning_curve_pv4.npy       Return per training episode, PV 4 MW
  learning_curve_pv10.npy      Return per training episode, PV 10 MW
  Q_pv4.npy, Q_pv10.npy        Trained Q-tables
  controller_training.png      Fig. 7

requirements.txt
```

---

# Part 1: Facility-level model and Monte Carlo analysis (`monte_carlo_facility_level/`)

## Model

`dispatch_model.py` simulates the 5 MW IT reference facility hour by hour for each of 19 weather years (2005 to 2023, 29 February removed, 19 x 8,760 h). For each hour it computes the facility load (IT load plus cooling, distribution and lighting overheads), the solar (and optional wind) generation, and dispatches the 10 MWh / 5 MW liquid-air energy store (LAES). From the annual totals it derives PUE, renewable fraction, CUE, avoided CO2, annual savings and simple payback.

Conventions:

- IT load is 90 % of nominal from 08:00 to 20:00 and 50 % otherwise, in local solar time.
- NASA POWER data are already in local solar time. PVGIS data are in UTC and are shifted to local solar time by `round(lon/15)` hours (5 h for all three cities).
- Energy in MWh equals mean MW per hour.
- The traditional-baseline energy is PUE_baseline x useful IT energy (48,443 MWh/yr for Jaipur) and is held fixed in the payback calculation.
- No parameter is tuned to hit a target output. Central estimates come from the manuscript.

## Monte Carlo design (`monte_carlo.py`, manuscript Section 3.9.2)

| Item | Setting |
|---|---|
| Draws | N = 100,000 (default seed `20260415`, city `jaipur`) |
| Weather | Each draw runs the full 8,760-h dispatch on one of the 19 real weather years, chosen uniformly at random, so inter-annual variability is part of the spread |
| Inputs | 8 independent triangular distributions, mode at the central estimate |
| Bounds | +/-10 % for all inputs except total CAPEX (+/-25 %) |
| Solar | The drawn year's profile is rescaled by CF_sampled / CF_central and clipped at installed capacity, so the hourly shape stays that of the real year |
| Baseline | Fixed at its point estimate, as stated in the manuscript |

The eight uncertain inputs are: immersion auxiliary overhead, air-cooled overhead, distribution loss, LAES round-trip efficiency, grid carbon intensity, solar capacity factor, total CAPEX, and electricity tariff.

For speed, draws are grouped by weather year and evaluated as (batch x 8,760) matrices, using a closed-form expression for facility energy. Any draw with hours where generation exceeds load would fall back to the exact sequential LAES dispatch; at the reference sizing this does not occur, so LAES discharge is zero in every draw.

Outputs per draw: `PUE`, `f_ren` (renewable fraction), `CUE`, `payback`, and `payback_noLAES` (payback with the LAES CAPEX removed).

## Results (Jaipur, 100,000 draws)

| Output | P10 | P50 | P90 | Point estimate |
|---|---|---|---|---|
| PUE | 1.176 | 1.182 | 1.187 | 1.182 |
| Renewable fraction (%) | 16.9 | 18.0 | 19.0 | 18.0 |
| CUE (kg CO2/kWh IT) | 0.649 | 0.688 | 0.727 | 0.688 |
| Simple payback (yr) | 4.27 | 4.99 | 5.74 | 5.0 |

These percentiles were recomputed from `monte_carlo_draws_csv.gz` and match the manuscript's Table 5 to the printed precision.

**Payback without LAES.** The manuscript reports a P10 to P90 range of 1.6 to 3.0 years (median 2.3). `monte_carlo.py` has two ways of removing the LAES CAPEX, selected by `noLAES_mode`:

- `"flat"` (script default): subtracts the flat $5.0M LAES CAPEX from each drawn total CAPEX. This is the variant `run_validation.py` checks against the 1.6 to 3.0 yr range.
- `"proportional"`: scales each drawn CAPEX by (CAPEX_0 - 5.0) / CAPEX_0, which gives a narrower band.

The `payback_noLAES` column in the shipped draws file corresponds to the `"proportional"` variant (P10/P50/P90 = 1.98 / 2.32 / 2.67 yr). The other four columns do not depend on this choice. To obtain the 1.6 to 3.0 yr band, regenerate the draws with the default settings (see below).

## Reproducing the results

Python 3.9 or later, with `numpy` and `pandas`.

**1. Input data (not included in this repository).** The scripts need hourly 2005 to 2023 files for each city (`jaipur`, `delhi`, `chennai`) in one directory:

```
nasa_power_<city>_csv.csv    NASA POWER hourly: WS50M, WS10M, T2M (local solar time).
                             166,536 data rows after an 11-line header that includes
                             the site elevation ("... = <x> meters").
pvgis_<city>_csv.csv         PVGIS hourly output P (W) for a 4 kWp system (UTC, stamped hh:30).
                             10-line header, 166,536 data rows.
```

**2. Set the paths.** The scripts currently default to `/mnt/user-data/uploads` (input data) and `/mnt/user-data/outputs` (outputs). Change `data_dir` in `dispatch_model.py` and `monte_carlo.py`, the `--out` default in `monte_carlo.py`, and the report path in `run_validation.py` to match your machine.

**3. Run** from inside `monte_carlo_facility_level/`:

```
python dispatch_model.py                 # per-city point estimates (PUE, f_ren, CUE, payback)
python run_validation.py                 # fast checks against the manuscript (seconds)
python run_validation.py --mc            # also runs the 100,000-draw Monte Carlo
python monte_carlo.py --n 100000 --seed 20260415 --city jaipur
```

`run_validation.py` writes `validation_report.csv`. Each check has a paper value, the model value, a tolerance set to the rounding precision the manuscript prints, and a PASS/FAIL status. A FAIL means the printed number does not follow from the stated model.

---

# Part 2: Preliminary controller study (`scripts/`, `results/`)

A tabular Q-learning agent compared with a static policy and a rule-based policy on a simplified hourly simulator (Fig. 7 of the manuscript).

## Important notes on scope

- The simulator uses **synthetic** weather, grid carbon intensity and tariff series, generated inside `make_year()`. It does **not** use the NASA POWER and PVGIS data of the facility-level model in Part 1. Absolute values (for example annual emissions) therefore differ from Tables 4 to 6 of the manuscript and are meant only for comparing the policies with each other.
- The reward follows Eq. (10) of the manuscript with weights alpha = 0.4 and beta = 0.4. The thermal weight is raised from 0.2 to 2.0 so that the thermal SLA acts as a near-hard constraint, a small backlog penalty (0.02 per unit) is added, and the security term is not included.
- Training uses one seed (0). Evaluation uses five synthetic years (seeds 100 to 104); the error bars in Fig. 7 are the standard deviation over these years.
- The facility-level tables (PUE, renewable fraction, CUE, payback, Monte Carlo) are reproduced by Part 1, not by these scripts.

## Reproducing the results

Python 3.9 or later.

```
pip install -r requirements.txt
python scripts/train_qlearning.py 4
python scripts/train_qlearning.py 10
python scripts/make_fig7.py
```

Run the commands from the repository root. The training script writes its outputs to the current directory, so either run it from inside `results/` (`cd results && python ../scripts/train_qlearning.py 4`) or move the generated files into `results/` before running `make_fig7.py`. The runs are deterministic for the fixed seeds (re-running reproduced the reported numbers exactly). Each training run executes 80,000 two-day episodes and takes a few minutes on a laptop CPU.

### Hyperparameters

| Item | Value |
|---|---|
| Episodes | 80,000 (two days each, random start day) |
| Actions | 18 (LAES idle/charge/discharge x workload normal/defer/catch-up x cooling boost off/on) |
| State grid | 12 x 4 x 4 x 4 x 3 (hour block, temperature, net renewable surplus, LAES state of charge, backlog) |
| Discount factor | 0.8 |
| Exploration | epsilon falls from 1.0 to 0.03 at episode 40,000, then stays at 0.03 |
| Learning rate | max(0.02, 1 / (1 + 0.05 N(s,a))) |
| Reference plant | 5 MW IT load, 10 MWh LAES, 5 MW LAES power, 60 % round-trip efficiency |

### Results (mean over 5 evaluation years)

| PV size | Policy | Emissions (t CO2/yr) | Return | Thermal violations (h/yr) | Curtailed (MWh/yr) |
|---|---|---|---|---|---|
| 4 MW | Static | 21,775 | -5,952 | 0.0 | 0 |
| 4 MW | Rule-based | 21,206 | -6,205 | 0.0 | 0 |
| 4 MW | Q-learning | 21,455 | -5,934 | 0.8 | 0 |
| 10 MW | Static | 16,320 | -5,515 | 0.0 | 1,029 |
| 10 MW | Rule-based | 15,762 | -5,770 | 0.0 | 290 |
| 10 MW | Q-learning | 16,038 | -5,478 | 2.8 | 1,100 |

Q-learning reaches the highest return at both sizes, but the rule-based policy gives the lowest emissions and no thermal violations. The study shows that the reward is learnable on the simplified simulator; it does not show that Q-learning outperforms a well-tuned rule-based controller.

## Citation

If you use this code, please cite the manuscript above. A DOI will be added here when one is assigned.

## Contact

Pooja Gupta, Department of Data Science and Engineering, Manipal University Jaipur.
Email: pooja.gupta@jaipur.manipal.edu
