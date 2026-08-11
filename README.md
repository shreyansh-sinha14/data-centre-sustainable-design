# Data and Code Availability

Supporting analytical scripts and derived datasets for:

> Sinha, S., Shukla, S., Gupta, P., Taneja, S. (2026). *An Integrated
> Analytical Framework for Sustainable Data Centre Design: CFD-Based
> Cooling, Cryogenic Energy Storage, and Quantum-Secure AI Optimisation.*
> Discover Energy.

This repository contains no new experimental data (the study is a
theoretical/analytical design framework, as stated in the manuscript's
Availability of Data and Materials declaration). It contains the
analytical model and Monte Carlo simulation code used to derive the
paper's quantitative results, so that every figure reported in the
manuscript can be independently reproduced.

## Contents

```
scripts/
  deterministic_model.py         Reproduces the point estimates in
                                  Table 4 (Jaipur / Delhi / Chennai
                                  climate comparison) and Table 6
                                  (PUE, renewable fraction, CUE,
                                  payback) via the exact equations
                                  in Section 3.8 of the manuscript.

  monte_carlo_sensitivity.py     Reproduces the joint Monte Carlo
                                  sensitivity analysis of Section
                                  3.9.2 (N = 100,000 triangular-
                                  distributed draws across nine
                                  uncertain input parameters),
                                  reported in Table 5 / Fig. 6.

results/
  deterministic_climate_comparison.csv   Output of deterministic_model.py
  monte_carlo_draws.csv                  All 100,000 Monte Carlo draws
                                          (inputs + outputs)
  monte_carlo_summary.csv                P10 / P50 / P90 summary table
                                          (reproduces Table 5)

requirements.txt                 Python dependencies (numpy, pandas)
```

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/deterministic_model.py
python scripts/monte_carlo_sensitivity.py
```

## Correspondence with the manuscript

| Manuscript item | Reproduced by | Paper value | Script output |
|---|---|---|---|
| Table 4 (Jaipur/Delhi/Chennai) | `deterministic_model.py` | see Table 4 | `deterministic_climate_comparison.csv` |
| Table 6 (facility PUE, CUE, payback) | `deterministic_model.py` | PUE 1.18, CUE 0.620, payback ≈4.4 yr | same, Jaipur row |
| Table 5 / Fig. 6 (Monte Carlo P10/P50/P90) | `monte_carlo_sensitivity.py` | PUE 1.174/1.180/1.186; Renewable fraction 34.2/35.9/37.7%; CUE 0.582/0.620/0.658; Payback 3.53/4.35/5.40 yr | `monte_carlo_summary.csv` |

The deterministic script reproduces the manuscript's Table 4 and Table 6
figures exactly (to within rounding). The Monte Carlo script reproduces
the PUE, renewable-fraction, and CUE percentile bands from Table 5
essentially exactly; the simple-payback percentile band reproduces
closely but not to the last decimal (P10/P90 differ by a few tenths of
a year from the manuscript's reported 3.53–5.40 yr range), most likely
because the manuscript does not fully specify whether the traditional
baseline energy consumption was held fixed or itself resampled across
draws. This script holds the baseline fixed at its point-estimate value
(48,443 MWh/yr), which is the more conservative and transparent choice.
Users needing an exact bit-for-bit reproduction of the original run
should contact the corresponding author.

## Model assumptions

All parameter central estimates, triangular-distribution bounds
(±10% for all parameters except CAPEX at ±25%), and the underlying
equations are drawn directly from Sections 3.8–3.9 of the manuscript.
No parameter is calibrated to a target output; see Section 3.10
(Model Validation Strategy) of the manuscript for the full validation
discussion.

## Citation

If you use this code, please cite the manuscript above. A DOI will be
added here once assigned by Discover Energy.

## Contact

Pooja Gupta, Associate Professor, Department of Data Science and
Engineering, Manipal University Jaipur.
Email: pooja.gupta@jaipur.manipal.edu
