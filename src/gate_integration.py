"""Integration with the BDP Model Gate (Section 7.7): produces standardised
validation, calibration and drift reports for every (design, model) combination
in O2. The gate's PASS/NEEDS_REVIEW/BLOCKED status and its flags are reported
descriptively alongside our own metrics and bootstrap intervals; the gate does
not replace the hypothesis tests in Section 4.
"""
import json
import warnings
from pathlib import Path

import numpy as np

from bdp_model_gate import GateConfig, StructuredGateContext, ModelGate
from bdp_model_gate.structured import default_structured_checks

# This paper evaluates validation *methodology*, not production readiness, so the
# performance/security bars are relaxed from the gate's deployment-grade defaults
# (min_score 0.8, flip-rate 0.05) to thresholds appropriate for a research-stage
# model with a narrow, shared tuning budget (Section 7.1). The validation-category
# thresholds (leakage, split overlap, drift) are left at their defaults: those are
# exactly what Section 7.7 asks the gate to report on for every design.
GATE_CONFIG = GateConfig()
GATE_CONFIG.performance.min_score = 0.6
GATE_CONFIG.security.adversarial_flip_rate_threshold = 0.25

MODEL_CARD_TEMPLATE = {
    "legal_basis": "legitimate interest - internal supply-chain operations research",
    "data_minimization_justification": ("order/customer identifiers excluded from the "
                                         "feature set; see Section 5.2 of the proposal"),
    "training_data_source": None,  # filled per call
    "use_case": "supply_chain_late_delivery_prediction",
    "validation_strategy": None,  # filled per call
}


MAX_GATE_ROWS = 4000  # cap per side so per-feature checks (leakage, drift, robustness)
# stay fast; the gate's report here is descriptive (Section 7.7), not the source of
# this paper's performance numbers -- those come from the full-data evaluation already
# computed in o2_pipeline.run_o2 before the gate is ever called.


def _cap(df_like, n, seed=0):
    if len(df_like) <= n:
        return df_like
    return df_like.sample(n=n, random_state=seed)


def run_gate_report(pipe, X_train, X_test, y_test, p_test, out_dir: Path, tag: str,
                     dataset_name=None, validation_strategy=None):
    """Run the gate for one fitted pipeline and persist its JSON (+ HTML where
    possible) report. Never raises: a gate failure is recorded, not propagated,
    consistent with the library's own 'it degrades; it never fails' contract.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"tag": tag}
    model_card = dict(MODEL_CARD_TEMPLATE, training_data_source=dataset_name,
                       validation_strategy=validation_strategy)
    rng = np.random.default_rng(0)
    X_train_g = _cap(X_train, MAX_GATE_ROWS)
    if len(X_test) > MAX_GATE_ROWS:
        test_pos = np.sort(rng.choice(len(X_test), size=MAX_GATE_ROWS, replace=False))
        X_test_g, y_test_g, p_test_g = X_test.iloc[test_pos], y_test[test_pos], p_test[test_pos]
    else:
        X_test_g, y_test_g, p_test_g = X_test, y_test, p_test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ctx = StructuredGateContext(
                model=pipe, X=X_test_g, y_true=y_test_g, y_pred=p_test_g, X_train=X_train_g,
                model_card=model_card)
            gate = ModelGate(checks=default_structured_checks(GATE_CONFIG))
            report = gate.run(ctx)
        result["gate_status"] = report.gate_status
        result["summary"] = report.summary()
        try:
            flags = {}
            for cat in ["validation", "performance", "compliance", "security", "fairness"]:
                cat_results = [r for r in report.results if getattr(r, "category", None) == cat]
                flags[cat] = {
                    "n_checks": len(cat_results),
                    "n_flagged": sum(1 for r in cat_results if getattr(r, "status", "OK") != "OK"),
                    "checks": [{"name": getattr(r, "name", None),
                                "status": getattr(r, "status", None),
                                "detail": getattr(r, "detail", None)}
                               for r in cat_results],
                }
            result["flags_by_category"] = flags
        except Exception as e:
            result["flags_by_category_error"] = str(e)

        json_path = out_dir / f"{tag}.json"
        try:
            report.to_json(str(json_path))
        except Exception as e:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            result["to_json_fallback_error"] = str(e)
    except Exception as e:
        result["gate_status"] = "ERROR"
        result["error"] = str(e)
        with open(out_dir / f"{tag}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result
