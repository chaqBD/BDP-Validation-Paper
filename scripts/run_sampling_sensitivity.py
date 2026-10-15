"""Regenerates the capped-vs-full sampling-sensitivity table for the two
row-comparison-dependent defects (exact duplicate, group/customer overlap), one
representative model family (LightGBM), both datasets, using the same (now
row-reuse-corrected) case construction as the main battery. Extracts every
`split_overlap` sub-check correctly (the gate reports two: duplicates-within-
validation and overlap-with-training) rather than taking the first match found --
for BOTH the uncapped, live-report run and the capped run already saved by the
main battery (`run_release_review.py`), so the capped row is never left with
blank overlap fields.

Also distinguishes genuinely injected row duplication from coincidental
feature-vector collisions: `case_exact_duplicate` injects duplicate rows by
exact source-row index (ground truth, known exactly), while the gate's
overlap-with-training check matches on hashed feature vectors, which can also
count rows that merely happen to collide on a narrower or lower-cardinality
feature set without being an injected duplicate at all.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
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


def extract_split_overlap_json(gjson):
    out = {}
    for cat, items in gjson.get("results_by_category", {}).items():
        if cat != "validation":
            continue
        for it in items:
            if it.get("check_name") == "split_overlap":
                meta = it.get("metadata", {})
                out[meta.get("check", "unknown")] = {"flag": it.get("flag"), "metadata": meta}
    return out


def capped_row_from_battery(dataset, case_name, family="gbm"):
    battery = pd.read_csv(OUT_DIR / "release_review_battery.csv")
    row = battery[(battery.dataset == dataset) & (battery.case == case_name)
                  & (battery.model == family)]
    if not len(row):
        return None
    row = row.iloc[0].to_dict()

    gjson_path = OUT_DIR / "gate_json" / dataset / f"{dataset}_{case_name}_{family}.json"
    if gjson_path.exists():
        with open(gjson_path, encoding="utf-8") as f:
            gjson = json.load(f)
        overlap = extract_split_overlap_json(gjson)
        row["duplicates_within_validation_flag"] = overlap.get("duplicates_within_validation", {}).get("flag")
        row["duplicates_within_validation_fraction"] = \
            overlap.get("duplicates_within_validation", {}).get("metadata", {}).get("duplicate_fraction")
        row["overlap_with_training_flag"] = overlap.get("overlap_with_training", {}).get("flag")
        row["overlap_with_training_fraction"] = \
            overlap.get("overlap_with_training", {}).get("metadata", {}).get("overlap_fraction")
        row["overlap_with_training_n"] = \
            overlap.get("overlap_with_training", {}).get("metadata", {}).get("n_overlapping")
    return row


def coincidental_match_count(dataset, case):
    """Distinguishes test rows the gate's hash-based overlap-with-training check
    would flag because they were genuinely injected (their exact row position is
    recorded by `case_exact_duplicate` as `injected_test_rows`, the ground truth)
    from test rows whose feature vector merely happens to coincide with some
    training row's hash without having been injected at all.
    """
    ref = _reference(dataset)
    X = ref["X"]
    train_idx, test_idx = case["train_idx"], case["test_idx"]
    injected_test_rows = set(case.get("injected_test_rows", np.array([], dtype=int)).tolist())

    train_hashes = pd.util.hash_pandas_object(X.iloc[train_idx], index=False).values
    test_hashes = pd.util.hash_pandas_object(X.iloc[test_idx], index=False).values
    train_hash_set = set(train_hashes.tolist())
    matched_mask = np.isin(test_hashes, list(train_hash_set))
    matched_test_rows = set(test_idx[matched_mask].tolist())

    n_injected_recovered = len(injected_test_rows & matched_test_rows)
    n_coincidental = len(matched_test_rows - injected_test_rows)
    n_injected_missed = len(injected_test_rows - matched_test_rows)
    return {
        "n_injected": len(injected_test_rows),
        "n_hash_matches": len(matched_test_rows),
        "n_injected_recovered": n_injected_recovered,
        "n_coincidental": n_coincidental,
        "n_injected_missed": n_injected_missed,
    }


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
            capped["dataset"] = dataset
            capped["case"] = case_name
            capped["sample"] = "capped_4000"
            rows.append(capped)
            print(dataset, case_name, "CAPPED:", capped.get("overlap_with_training_flag"),
                  capped.get("overlap_with_training_fraction"), capped.get("verdict_stock_gate"))

        if case_name == "defect_exact_duplicate":
            cm = coincidental_match_count(dataset, case)
            print(dataset, case_name, "coincidental-match check:", cm)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "sampling_sensitivity.csv", index=False)
    print("\nSaved to", OUT_DIR / "sampling_sensitivity.csv")

    # Dataset-specific capped/full understatement ratio, computed directly from
    # this run's own numbers rather than stated generically for both datasets.
    print("\n" + "=" * 100)
    print("Capped-vs-full understatement ratio (overlap_with_training_fraction), exact-duplicate defect")
    print("=" * 100)
    for dataset in ["dataco", "olist"]:
        d = out[(out.case == "defect_exact_duplicate") & (out.dataset == dataset)]
        full_frac = d[d["sample"] == "full"]["overlap_with_training_fraction"].iloc[0]
        capped_frac = d[d["sample"] == "capped_4000"]["overlap_with_training_fraction"].iloc[0]
        if capped_frac and capped_frac > 0:
            print(f"  {dataset}: full={full_frac:.4f} capped={capped_frac:.4f} "
                  f"ratio={full_frac/capped_frac:.1f}x understated")
        else:
            print(f"  {dataset}: full={full_frac:.4f} capped={capped_frac} (capped fraction is zero/undefined)")
    print(out.to_string())
