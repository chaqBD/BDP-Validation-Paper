"""Shared paths and constants for Paper A: When Validation Misleads.

Author: Tewogbade Shakir Adeyemi
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATACO_CSV = ROOT / "DataCo-Dataset" / "DataCoSupplyChainDataset.csv"
OLIST_DIR = ROOT / "Olist-Dataset"

RESULTS_DIR = ROOT / "results"
AUDIT_DIR = RESULTS_DIR / "data_audit"
O2_DIR = RESULTS_DIR / "o2_design_comparison"
O3_MECH_DIR = RESULTS_DIR / "o3_mechanism"
O3_SIM_DIR = RESULTS_DIR / "o3_simulation"
O4_DIR = RESULTS_DIR / "o4_rolling_origin"
FIG_DIR = RESULTS_DIR / "figures"

for d in [AUDIT_DIR, O2_DIR, O3_MECH_DIR, O3_SIM_DIR, O4_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20261015

# Equivalence margin registered in the analysis plan (Section 4 of the proposal).
DELTA_AUC = 0.01

# DataCo primary population window (Section 5.2).
DATACO_PRIMARY_START = "2015-01-01"
DATACO_PRIMARY_END = "2017-09-30"
DATACO_SENSITIVITY_START = "2017-10-01"
DATACO_SENSITIVITY_END = "2018-01-31"

# Olist primary population window (Section 5.2).
OLIST_PRIMARY_START = "2017-01-01"
OLIST_PRIMARY_END = "2018-08-31"
