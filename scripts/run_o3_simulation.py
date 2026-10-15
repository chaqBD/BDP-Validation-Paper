"""Run the O3 factorial simulation with known mechanisms (Section 7.3, H4)."""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import O3_SIM_DIR, RANDOM_SEED
from src.simulation import run_factorial

if __name__ == "__main__":
    t0 = time.time()
    df = run_factorial(n_orders_rep=3200, n_orders_truth=40000, n_reps=8, seed=RANDOM_SEED)
    df.to_csv(O3_SIM_DIR / "simulation_results.csv", index=False)
    print(f"Done in {time.time()-t0:.1f}s, {len(df)} rows")
    print("\nMean recovery error (random-vs-grouped contrast) by W (within-order correlation):")
    print(df.groupby("W")["recovery_error_random_vs_grouped"].agg(["mean", "std"]))
    print("\nMean recovery error (grouped-vs-chronological contrast) by S,R,P (temporal factors):")
    print(df.groupby(["S", "R", "P"])["recovery_error_grouped_vs_chronological"].agg(["mean", "std"]))
    print("\nEstimated contrast magnitude by condition (sanity check of direction):")
    print(df.groupby("W")["est_contrast_random_vs_grouped"].mean())
    print(df.groupby(["S", "R", "P"])["est_contrast_grouped_vs_chronological"].mean())
