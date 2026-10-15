"""Regenerates the capped-vs-full sampling-sensitivity table for the two
row-comparison-dependent defects (exact duplicate, group/customer overlap), one
representative model family (LightGBM), both datasets, using the same (now
row-reuse-corrected) case construction as the main battery. Extracts every
`split_overlap` sub-check correctly (the gate reports two: duplicates-within-
validation and overlap-with-training) rather than taking the first match found.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import RESULTS_DIR
from src.gate_integration import GATE_CONFIG
from src.release_cases import (_reference, case_exact_duplicate, case_group_overlap_dataco,
                                case_group_overlap_olist)
from src.release_review import fit_predict
from bdp_model_gate import StructuredGateContext, ModelGate
from bdp_model_gate.structured import default_structured_checks

OUT_DIR = RESULTS_DIR / "release_review"


def extract_split_overlap(report):
    out = {}
    for r in report.results:
        if r.check_name == "split_overlap":
            out[r.metadata.get("check", "unknown")] = {
                "flag": r.flag, "detail": r.detail, "metadata": r.metadata}
    return out


def run_uncapped(dataset, case, family="gbm"):
    ref = _reference(dataset)
    X, y = ref["X"], ref["y"]
    X_train, y_train = X.iloc[case["train_idx"]], y.iloc[case["train_idx"]]
    X_test, y_test = X.iloc[case["test_idx"]], y.iloc[case["test_idx"]]
    pipe, p_test = fit_predict(family, X_train, y_train, X_test, ref["num"], ref["cat"])
    model_card = {"training_data_source": dataset, "validation_strategy": "out_of_time",
                  "use_case": "supply_chain_late_delivery_prediction",
                  "legal_basis": "legitimate interest - internal supply-chain operations research",
                  "data_minimization_justification": "order/customer identifiers excluded"}
    ctx = StructuredGateContext(model=pipe, X=X_test, y_true=y_test.values, y_pred=p_test,
                                X_train=X_train, model_card=model_card)
    gate = ModelGate(checks=default_structured_checks(GATE_CONFIG))
    report = gate.run(ctx)
    overlap = extract_split_overlap(report)
    return {
        "n_train": len(case["train_idx"]), "n_test": len(case["test_idx"]),
        "gate_status": report.gate_status, "model_score": report.model_score,
        "duplicates_within_validation_flag": overlap.get("duplicates_within_validation", {}).get("flag"),
        "duplicates_within_validation_fraction":
            overlap.get("duplicates_within_validation", {}).get("metadata", {}).get("duplicate_fraction"),
        "overlap_with_training_flag": overlap.get("overlap_with_training", {}).get("flag"),
        "overlap_with_training_fraction":
            overlap.get("overlap_with_training", {}).get("metadata", {}).get("overlap_fraction"),
        "overlap_with_training_n":
            overlap.get("overlap_with_training", {}).get("metadata", {}).get("n_overlapping"),
    }


def capped_row_from_battery(dataset, case_name, family="gbm"):
    battery = pd.read_csv(OUT_DIR / "release_review_battery.csv")
    row = battery[(battery.dataset == dataset) & (battery.case == case_name)
                  & (battery.model == family)]
    return row.iloc[0].to_dict() if len(row) else None


if __name__ == "__main__":
    rows = []
    for dataset, case_name, case_builder in [
        ("dataco", "defect_exact_duplicate", lambda ref=None: case_exact_duplicate(_reference("dataco"))),
        ("olist", "defect_exact_duplicate", lambda ref=None: case_exact_duplicate(_reference("olist"))),
        ("dataco", "defect_group_overlap", lambda ref=None: case_group_overlap_dataco(_reference("dataco"))),
        ("olist", "defect_group_overlap", lambda ref=None: case_group_overlap_olist(_reference("olist"))),
    ]:
        case = case_builder()
        full = run_uncapped(dataset, case)
        full["dataset"] = dataset
        full["case"] = case_name
        full["sample"] = "full"
        rows.append(full)
        print(dataset, case_name, "FULL:", full["overlap_with_training_flag"],
              full["overlap_with_training_fraction"], full["gate_status"])

        capped = capped_row_from_battery(dataset, case_name)
        if capped is not None:
            rows.append({
                "dataset": dataset, "case": case_name, "sample": "capped_4000",
                "n_train": 4000, "n_test": 4000,
                "gate_status": capped["verdict_stock_gate"],
                "model_score": capped["auc"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sampling_sensitivity.csv", index=False)
    print("\nSaved to", OUT_DIR / "sampling_sensitivity.csv")
    print(out.to_string())
