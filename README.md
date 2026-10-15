# When Validation Misleads

**Group Dependence, Temporal Change and the Reliability of Performance Estimates in Supply Chain Machine Learning**

Author: Tewogbade Shakir Adeyemi

This repository holds the full analysis code, registered hypotheses, and results for
Paper A of the research programme described in `Updated-Proposal.pdf`. It uses two
public datasets (DataCo Smart Supply Chain, Olist Brazilian E-Commerce) to study how
group dependence and temporal change bias performance estimates for supply-chain
late-delivery models, whether common validation-design contrasts identify the
mechanism or just describe a difference, and when the choice of validation design
changes which model a practitioner would pick.

## Data

Not committed to this repository (large, separately licensed). Place the raw files at:

```
DataCo-Dataset/DataCoSupplyChainDataset.csv      (Constante, Silva & Pereira, 2019 — Mendeley Data)
Olist-Dataset/olist_*.csv                         (Olist, 2018 — Kaggle)
```

## Setup

```
pip install -r requirements.txt
```

## Repository layout

```
src/                         Core library
  config.py                  Paths, seeds, registered constants (delta = 0.01 AUC)
  dataco_prep.py / olist_prep.py   O1: decision time, population, predictors, audit
  metrics.py                 AUC, AP, Brier, calibration-in-the-large, slope, ECE
  bootstrap.py                Cluster/block bootstrap, equivalence test, Holm's method
  stats_tests.py              Kendall's tau, slope test, PSI, standardised mean shift
  splitters.py                 The five validation designs + rolling-origin splits
  models.py                    Model families (logreg/RF/GBM) + baselines, time-aware tuning
  o2_pipeline.py               O2: validation-design comparison
  mechanism_experiments.py     O3: fixed-test-set A/B/C/D mechanism experiments
  simulation.py                 O3: factorial simulation with known mechanisms
  rolling_origin.py             O4: rolling-origin reliability analysis
  gate_integration.py           BDP Model Gate integration (Section 7.7)

scripts/                      One runnable script per objective (run_o2_*, run_o3_*, run_o4_*,
                               run_hypothesis_tests.py)

results/                      All numeric outputs (CSV/JSON) and the gate's own reports
  REPORT.md                   Consolidated findings and the O5 reporting protocol
```

## Reproducing the analysis

```
python scripts/run_o2_dataco.py           # O2: validation-design comparison, DataCo
python scripts/run_o2_olist.py            # O2: validation-design comparison, Olist
python scripts/run_hypothesis_tests.py    # H1, H2, H6 (RQ1, RQ4)
python scripts/run_o3_mechanism.py        # H3 (RQ2): fixed-test-set dose-response
python scripts/run_o3_simulation.py       # H4 (RQ2): factorial simulation, recovery error
python scripts/run_o4_rolling_origin.py   # H5 (RQ3): rolling-origin reliability
```

See `results/REPORT.md` for the findings and the O5 reporting protocol.

## BDP Model Gate

Section 7.7 of the proposal names the [BDP Model Gate](https://pypi.org/project/bdp-model-gate/)
(Oden, 2026) as the tool producing standardised validation, calibration and drift reports
for every design. `src/gate_integration.py` runs it for every (validation design, model)
combination fitted in O2 and saves its PASS / NEEDS_REVIEW / BLOCKED report to
`results/o2_design_comparison/gate_reports/<dataset>/`. The proposal cites v0.6.0; this repository is pinned to exactly that version
(`pip show bdp-model-gate` reports `Version: 0.6.0`), installed from the wheel
supplied alongside the proposal rather than from PyPI, whose current release at
analysis time had moved to v0.7.0. The gate's
performance/security thresholds were relaxed from its deployment-grade defaults
(`min_score` 0.8 -> 0.6, adversarial flip-rate 0.05 -> 0.25) because this study evaluates
validation *methodology*, not production readiness; the validation-category thresholds
(leakage, split overlap, drift) were left at their defaults, since those are exactly what
Section 7.7 asks the gate to report on. Its output is reported descriptively, alongside —
not in place of — the paper's own bootstrap intervals and hypothesis tests.
