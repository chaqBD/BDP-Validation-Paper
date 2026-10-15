"""Produce one human-readable BDP Model Gate HTML report per dataset (the primary
estimand, gradient-boosted model), illustrating "the report a reviewer reads"
(Section 7.7). The JSON reports for every (design, model) combination are already
saved by run_o2_dataco.py / run_o2_olist.py; this just renders one of them as HTML.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bdp_model_gate import ModelGate, StructuredGateContext
from bdp_model_gate.structured import default_structured_checks

from src import dataco_prep as dc
from src import olist_prep as ol
from src.config import O2_DIR, RANDOM_SEED
from src.gate_integration import GATE_CONFIG, MODEL_CARD_TEMPLATE
from src.models import time_aware_tune, build_preprocessor
from src.o2_pipeline import build_designs


def demo_for_dataco():
    df = dc.load_raw()
    X, y, meta, num, cat = dc.build_population(df)
    designs, _ = build_designs(meta, test_size=0.2, seed=RANDOM_SEED)
    train_idx, test_idx = designs["chronological_primary"]
    preprocessor = build_preprocessor(num, cat)
    pipe, params, score = time_aware_tune(
        "gbm", X.iloc[train_idx], y.iloc[train_idx],
        meta["decision_time"].iloc[train_idx], preprocessor, seed=RANDOM_SEED)
    p_test = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    model_card = dict(MODEL_CARD_TEMPLATE, training_data_source="dataco",
                       validation_strategy="out_of_time")
    ctx = StructuredGateContext(model=pipe, X=X.iloc[test_idx], y_true=y.iloc[test_idx].values,
                                 y_pred=p_test, X_train=X.iloc[train_idx], model_card=model_card)
    gate = ModelGate(checks=default_structured_checks(GATE_CONFIG))
    report = gate.run(ctx)
    out = O2_DIR / "gate_reports" / "dataco_chronological_primary_gbm_demo.html"
    report.to_html(str(out), title="DataCo late-delivery model — chronological, primary estimand")
    print("DataCo demo gate status:", report.gate_status, "->", out)


def demo_for_olist():
    orders, customers, items, payments, products = ol.load_raw()
    X, y, meta, num, cat = ol.build_population(orders, customers, items, payments, products)
    designs, _ = build_designs(meta, test_size=0.2, seed=RANDOM_SEED)
    train_idx, test_idx = designs["chronological_primary"]
    preprocessor = build_preprocessor(num, cat)
    pipe, params, score = time_aware_tune(
        "gbm", X.iloc[train_idx], y.iloc[train_idx],
        meta["decision_time"].iloc[train_idx], preprocessor, seed=RANDOM_SEED)
    p_test = pipe.predict_proba(X.iloc[test_idx])[:, 1]
    model_card = dict(MODEL_CARD_TEMPLATE, training_data_source="olist",
                       validation_strategy="out_of_time")
    ctx = StructuredGateContext(model=pipe, X=X.iloc[test_idx], y_true=y.iloc[test_idx].values,
                                 y_pred=p_test, X_train=X.iloc[train_idx], model_card=model_card)
    gate = ModelGate(checks=default_structured_checks(GATE_CONFIG))
    report = gate.run(ctx)
    out = O2_DIR / "gate_reports" / "olist_chronological_primary_gbm_demo.html"
    report.to_html(str(out), title="Olist late-delivery model — chronological, primary estimand")
    print("Olist demo gate status:", report.gate_status, "->", out)


if __name__ == "__main__":
    demo_for_dataco()
    demo_for_olist()
