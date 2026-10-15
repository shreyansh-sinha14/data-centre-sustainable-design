# Data and Code Availability

Supporting code for:

> Sinha, S., Shukla, S., Gupta, P., Taneja, S.: *An Integrated Analytical Framework for Sustainable Data Centre Design: CFD-Informed Immersion Cooling, Cryogenic Energy Storage, and Quantum-Secure AI Optimisation.* (under review)

The study is an analytical design framework. No experimental data were generated and no physical prototype was built.

## What this repository contains

The code and results of the **preliminary controller study** (Fig. 7 of the manuscript): a tabular Q-learning agent compared with a static policy and a rule-based policy on a simplified hourly simulator.

```
scripts/
  train_qlearning.py     Simulator, Q-learning training, and evaluation of the
                         three policies (one run per PV size)
  make_fig7.py           Regenerates Fig. 7 from the files in results/

results/
  results_pv4.json       Evaluation results, PV 4 MW (mean and std over 5 years)
  results_pv10.json      Evaluation results, PV 10 MW
  learning_curve_pv4.npy   Return per training episode, PV 4 MW
  learning_curve_pv10.npy  Return per training episode, PV 10 MW
  Q_pv4.npy, Q_pv10.npy    Trained Q-tables
  controller_training.png  Fig. 7

archive_old_model/       Earlier scripts and outputs, superseded (see below)
requirements.txt
```

## Important notes on scope

- The simulator uses **synthetic** weather, grid carbon intensity and tariff series, generated inside `make_year()`. It does **not** use the NASA POWER and PVGIS data of the manuscript's 8,760-hour dispatch model. Absolute values (for example annual emissions) therefore differ from Tables 4 to 6 of the manuscript and are meant only for comparing the policies with each other.
- The reward follows Eq. (10) of the manuscript with weights alpha = 0.4 and beta = 0.4. The thermal weight is raised from 0.2 to 2.0 so that the thermal SLA acts as a near-hard constraint, a small backlog penalty (0.02 per unit) is added, and the security term is not included.
- Training uses one seed (0). Evaluation uses five synthetic years (seeds 100 to 104); the error bars in Fig. 7 are the standard deviation over these years.
- This repository does **not** reproduce the facility-level tables of the manuscript (PUE, renewable fraction, CUE, payback, Monte Carlo). Those results are described by the equations and parameters given in the manuscript. The code for them is available from the corresponding author on reasonable request.

## Reproducing the results

Python 3.9 or later.

```
pip install -r requirements.txt
python scripts/train_qlearning.py 4
python scripts/train_qlearning.py 10
python scripts/make_fig7.py
```

Run the commands from the repository root. The scripts write to `results/` and are deterministic for the fixed seeds. Each training run executes 80,000 two-day episodes and takes a few minutes on a laptop CPU.

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

## Results (mean over 5 evaluation years)

| PV size | Policy | Emissions (t CO2/yr) | Return | Thermal violations (h/yr) | Curtailed (MWh/yr) |
|---|---|---|---|---|---|
| 4 MW | Static | 21,775 | -5,952 | 0.0 | 0 |
| 4 MW | Rule-based | 21,206 | -6,205 | 0.0 | 0 |
| 4 MW | Q-learning | 21,455 | -5,934 | 0.8 | 0 |
| 10 MW | Static | 16,320 | -5,515 | 0.0 | 1,029 |
| 10 MW | Rule-based | 15,762 | -5,770 | 0.0 | 290 |
| 10 MW | Q-learning | 16,038 | -5,478 | 2.8 | 1,100 |

Q-learning reaches the highest return at both sizes, but the rule-based policy gives the lowest emissions and no thermal violations. The study shows that the reward is learnable on the simplified simulator; it does not show that Q-learning outperforms a well-tuned rule-based controller.

## Archived files

`archive_old_model/` holds scripts and outputs from an earlier version of the analysis. They use a different renewable-shifting rule and report values that no longer match the current manuscript, so they should not be used to reproduce its results.

## Citation

If you use this code, please cite the manuscript above. A DOI will be added here when one is assigned.

## Contact

Pooja Gupta, Department of Data Science and Engineering, Manipal University Jaipur.
Email: pooja.gupta@jaipur.manipal.edu
