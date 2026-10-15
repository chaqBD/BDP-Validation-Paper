# Results: When Validation Misleads

Tewogbade Shakir Adeyemi — Paper A, October 2026

All numbers below are produced by the scripts in `scripts/` against the registered
analysis plan in `Updated-Proposal.pdf` (delta = 0.01 AUC, Holm's method for multiple
hypotheses, cluster/block bootstrap for dependence-respecting intervals). Code and raw
outputs (CSV/JSON) backing every figure here are in `results/`.

## O1: Data audit

### DataCo

| | |
|---|---|
| Rows (line items) | 180,519 |
| Cancelled rows (`Delivery Status == "Shipping canceled"`) | 7,754 (4.3%) |
| Non-cancelled rows | 95.7% |
| Orders | 65,752 (2.75 line items/order on average) |
| Orders with a single label across all their line items | 100.0% |
| First Class orders, late rate (non-cancelled) | 100.0% |
| Primary population (non-cancelled, Jan 2015 - Sep 2017) | 164,401 rows / 54,796 orders |
| Late rate, primary population | 57.3% |

The 7,754/4.3% and 100%/100% figures reproduce the proposal's Section 6 preliminary
signals exactly, confirming the population and outcome definitions are implemented as
specified.

### Olist

| | |
|---|---|
| Orders total | 99,441 |
| Delivered orders | 96,478 (late rate 8.1%) |
| Monthly late rate range (reliable months, >=30 orders, primary window) | 1.4% - 21.4% |
| Primary population (delivered, promise >=14 days before data cutoff, Jan 2017-Aug 2018) | 96,180 rows / orders |
| Late rate, primary population | 8.1% |

The 1.4%-21.4% range reproduces the proposal's "about 1% to 21%" preliminary signal.
8 orders marked `delivered` with no recorded delivery date (a known Olist data-quality
artefact) were excluded from the eligible population rather than coded either way,
consistent with the proposal's treatment of DataCo's cancelled rows (Section 5.2).

## O2: Sensitivity of performance estimates to the validation design

Three model families (regularised logistic regression, random forest, gradient
boosting) plus two per-dataset baselines, evaluated under five validation designs with
training size matched in unique orders and a shared tuning budget (Section 7.1).

### DataCo (AUC, 95% paired cluster-bootstrap interval)

| Design | logreg | rf | gbm | baseline (shipping mode) | baseline (scheduled days) |
|---|---|---|---|---|---|
| Row-random | 0.763 [0.756, 0.769] | 0.756 [0.749, 0.762] | **0.777** [0.770, 0.782] | 0.742 | 0.708 |
| Order-grouped | 0.742 [0.732, 0.751] | 0.741 [0.731, 0.750] | 0.744 [0.734, 0.754] | 0.742 | 0.707 |
| Customer-grouped | 0.742 [0.733, 0.752] | 0.742 [0.732, 0.752] | 0.742 [0.732, 0.752] | 0.743 | 0.708 |
| Chronological (primary) | 0.747 [0.735, 0.756] | 0.744 [0.734, 0.753] | 0.745 [0.736, 0.755] | 0.742 | 0.715 |
| Chronological, new customers | 0.701 [0.640, 0.763] | 0.671 [0.612, 0.742] | 0.719 [0.665, 0.782] | 0.713 | 0.692 |

Row-random is the only design that separates meaningfully from the other four; order-
grouped, customer-grouped and chronological-primary sit within about 0.003 AUC of each
other for every model family. DataCo's estimates are far more sensitive to keeping
*orders* intact than to time, exactly as the Section 6 preliminary signal suggested.

### Olist (AUC, 95% paired cluster-bootstrap interval)

| Design | logreg | rf | gbm | baseline (promised days) | baseline (customer region) |
|---|---|---|---|---|---|
| Row-random | 0.675 [0.662, 0.689] | 0.759 [0.746, 0.772] | 0.776 [0.764, 0.789] | 0.551 | 0.596 |
| Order-grouped | 0.683 [0.669, 0.699] | 0.764 [0.750, 0.778] | 0.783 [0.770, 0.795] | 0.542 | 0.607 |
| Customer-grouped | 0.694 [0.681, 0.707] | 0.770 [0.757, 0.782] | **0.792** [0.780, 0.803] | 0.555 | 0.610 |
| Chronological (primary) | 0.712 [0.697, 0.725] | **0.487** [0.470, 0.507] | 0.718 [0.704, 0.731] | 0.797 | 0.423 |
| Chronological, new customers | 0.710 [0.697, 0.725] | 0.485 [0.467, 0.504] | 0.716 [0.702, 0.729] | 0.798 | 0.422 |

Olist shows the opposite pattern from DataCo: the random-design family (row/order/
customer-grouped) is internally consistent, but the chronological design is a
different regime entirely. Random forest collapses to **0.487 AUC (worse than
chance)** under chronological evaluation, and the single-feature `promised_days`
baseline *jumps* to 0.797 with a calibration slope of 11.8 (severe over-confidence).
This is not noise: the chronological test window falls in a period the data audit
(Section 6 of the proposal) already flagged — "the August 2018 rise coincided with a
shorter promise (median 13.4 days, against about 22) while fulfilment became faster."
The random forest, trained on a `promised_days` range it never saw at that scale,
extrapolates badly; the linear baseline, by contrast, generalises the shift smoothly
and over-fits to it. O4 traces this mechanism at the level of individual forecast
origins.

## Hypothesis tests (RQ1, RQ4; `results/o2_design_comparison/hypothesis_tests_h1_h2.csv`)

**H1** (DataCo: row-random AUC exceeds order-grouped AUC by more than delta=0.01):

| Model | Diff (row-random - order-grouped) | 95% CI | Verdict |
|---|---|---|---|
| logreg | +0.021 | [0.009, 0.032] | inconclusive (straddles delta) |
| rf | +0.015 | [0.003, 0.027] | inconclusive |
| gbm | +0.032 | [0.020, 0.043] | **superior** (Holm-adjusted p = 0.006) |

H1 holds decisively for the flexible model (gradient boosting) and is directionally
positive but not conclusively beyond the registered margin for logistic regression and
random forest.

**H2** (Olist: chronological AUC is lower than order-grouped AUC by more than delta):

| Model | Diff (chronological - order-grouped) | 95% CI | Verdict |
|---|---|---|---|
| logreg | +0.029 | [0.010, 0.048] | **superior** (opposite direction to H2) |
| rf | -0.277 | [-0.300, -0.254] | **inferior** (strongly confirms H2) |
| gbm | -0.065 | [-0.083, -0.045] | **inferior** (confirms H2) |

H2 is strongly confirmed for both tree ensembles, with random forest showing an
extreme collapse. Logistic regression moves in the *opposite* direction: it is the one
model family robust enough to the 2018 promise-window shift that its chronological
estimate is actually slightly higher than its order-grouped estimate. The validation
design does not have a single, model-independent effect — it interacts with how much
capacity the model has to overfit a structure that later changes.

**H6** (model rankings under row-random vs. the primary design):

| Dataset | Kendall's tau | Rank-change frequency | Row-random ranking (logreg, rf, gbm) | Primary-design ranking |
|---|---|---|---|---|
| DataCo | 0.33 (n=3, not powered for significance) | 2/3 models change rank | gbm > logreg > rf | logreg > gbm > rf |
| Olist | 0.33 | 2/3 models change rank | gbm > rf > logreg | gbm > logreg > rf |

With only three model families, Kendall's tau has essentially no statistical power
(the CI spans [-1, 1] in both cases) — this is a limitation of the design, not a null
result. Descriptively, 2 of 3 models change rank position between row-random and the
primary design in *both* datasets, which is exactly the practical concern H6 is about:
picking a "winning" model under the common-practice design would not pick the same
model the primary estimand would.

## O3a: Fixed-test-set mechanism experiments (RQ2, H3)

### DataCo: order-dependence dose-response (`results/o3_mechanism/dataco_mechanism_results.csv`)

Holding the test set and training size fixed, leaking an increasing share of test
orders' sibling line items into training (condition C) raises AUC monotonically and
almost perfectly linearly across all four evaluation periods:

| Overlap share | AUC (mean across 4 periods) |
|---|---|
| 0% (= condition B, no leakage) | 0.740 |
| 20% | 0.788 |
| 40% | 0.842 |
| 60% | 0.887 |
| 80% | 0.929 |
| 100% | 0.966 |

Slope test (H3): **slope = 0.229 AUC per unit overlap share, r = 0.994, p = 1.0e-22**
(`results/o3_mechanism/h3_slope_test.json`). This is the paper's cleanest result: a
0.226 AUC swing (0.740 to 0.966) is attributable entirely to how many sibling rows of
the *same orders being evaluated* were allowed into training, with everything else
(test set, training size, model, features) held fixed. Conditions A (past only,
0.741) and B (time coverage only, 0.739) are statistically indistinguishable from each
other, isolating time coverage as *not* the driver — group overlap is.

Condition D (legitimate customer history: the test customers' own earlier orders
guaranteed present in training) scores 0.729, close to condition A rather than
elevated — including a customer's own legitimate past orders does not by itself
inflate performance the way sibling-row leakage does.

### Olist: time-coverage and customer-history effects (condition C not applicable; Section 5.1)

| Condition | AUC (mean across 3 periods) |
|---|---|
| A (past only) | 0.598 |
| B (+ time coverage) | 0.636 |
| D (+ customer history) | 0.621 |

Unlike DataCo, giving the model contemporaneous (same-period, non-test) orders
*improves* its score on Olist (+0.038 AUC), consistent with Olist's higher rate of
temporal drift (O2, O4): seeing recent data, even from different orders, matters more
than avoiding group overlap, because Olist has almost no sibling-row structure
(row = order; Section 5.1).

## O3b: Simulation with known mechanisms (RQ2, H4)

Full factorial (W: within-order correlation, S: covariate shift, R: base-rate shift,
P: promise-coefficient change) x 2^4 = 16 cells, 8 replicates per cell, with a
large-N reference run per cell as ground truth (`results/o3_simulation/`).

- **Random-vs-grouped contrast tracks W** (H4, first clause): mean estimated contrast
  is -0.003 when W is off vs. **+0.118** when W is on — a clean, large separation.
  Mean recovery error (estimate - truth) is -0.007 (W off) and +0.079 (W on): the
  modest-N replicate systematically *overestimates* the population-level dependence
  effect relative to the large-N reference, itself a useful caution for anyone running
  a small mechanism experiment and reading the magnitude literally.
- **Grouped-vs-chronological contrast tracks S/R/P** (H4, second clause): the contrast
  rises from 0.034 (no temporal factors) to 0.083 (all three on), monotonically
  increasing as more temporal factors are switched on, with the largest single jump
  from adding P (the promise-coefficient change) — consistent with O2/O4's empirical
  finding that a changing service promise is the sharpest driver of chronological
  deterioration.
- An interaction is visible but modest: cells with both W and multiple temporal
  factors on do not simply add the two main effects, though the sample size here
  (8 reps/cell) is too small to resolve the interaction's significance precisely.

## O4: Reliability over time and the service promise (RQ3, H5)

Ten expanding-window forecast origins per dataset, gradient-boosted model, feature
drift (PSI, standardised mean shift), change in lateness prevalence, change in the
promised window and in fulfilment time recorded at each origin
(`results/o4_rolling_origin/`).

### DataCo

AUC is stable across origins (0.731-0.758), and mean PSI stays low throughout
(0.0002-0.009) with a mild rise at the two final origins. The H5 regression of
next-period AUC change on PSI gives **slope = 1.94, r = 0.54, p = 0.135** (n=9, not
significant) — a positive-but-inconclusive association, and notably *not* a case of
drift preceding deterioration: the two origins with the highest PSI (origins 8-9) are
followed by *improving*, not worsening, next-period AUC. This is a direct, dataset-level
instance of Eck et al.'s (2022) finding, cited in the proposal's positioning, that
"drift can occur without deterioration."

### Olist

PSI is an order of magnitude larger and far more volatile than DataCo's (0.011-0.486
vs. 0.0002-0.009), consistent with O2's finding that Olist is the temporally unstable
dataset of the two. The H5 regression gives **slope = -0.125, r = -0.25, p = 0.523**
(n=9, not significant) — directionally consistent with "more drift, worse next-period
AUC" but underpowered at 9 origins. The change-in-lateness-prevalence vs.
change-in-next-calibration regression is also directionally consistent but not
significant (slope = -0.41, p = 0.19). The origin with the single largest drift
(origin 9, PSI = 0.42, at the end of the series) lines up with the same
August-2018-region promise-window collapse that produced O2's RF AUC = 0.487 under
chronological evaluation — the rolling-origin view localises *when* in the series that
collapse originates, which a single train/test chronological split cannot.

### Olist: separating operational change from promise change

An auxiliary regressor predicts actual fulfilment time from operational features only
(no `promised_days`), trained on the same expanding windows
(`results/o4_rolling_origin/olist_auxiliary_operational_model.csv`). Its mean absolute
error *improves* over the series (5.9 days at origin 0 -> 3.7 days at origin 9) while
the mean promised window collapses (27.3 -> 14.6 days) and mean actual fulfilment time
also falls (14.5 -> 7.2 days). Operationally, Olist got *better and faster*, not worse:
the auxiliary model's accuracy does not deteriorate. The late-delivery AUC collapse
documented in O2 (random forest: 0.78 under order-grouped, 0.49 under chronological)
is therefore attributable to the *promise* shrinking faster than actual delivery times
improved — more orders cross from "on time" to "late" purely because the bar moved,
not because fulfilment became less predictable. This is exactly the distinction
Section 7.4 asks the auxiliary model to draw, and it resolves the O2 anomaly: a model
trained on an earlier, more generous promise window extrapolates badly once the
promise itself is redefined, independent of any change in operational performance.

**Caveat, stated plainly**: n=9 origins per dataset is a small regression sample. The
H5 associations reported here are directionally informative and reproduce a documented
phenomenon (DataCo) or a documented episode (Olist), but neither regression clears
conventional significance, and a future iteration of this analysis should widen the
rolling-origin schedule (more, shorter origins) before leaning on the slope estimates
alone.

## BDP Model Gate (Section 7.7)

Every (design, model) combination fitted in O2 was passed through the BDP Model Gate
v0.6.0 (`results/o2_design_comparison/gate_reports/`), which produces a standardised
PASS / NEEDS_REVIEW / BLOCKED report independent of this paper's own bootstrap
pipeline. Performance/security thresholds were relaxed from deployment-grade defaults
(see the "BDP Model Gate" section of the repository README); validation-category checks (leakage,
split-overlap, feature-drift) were left at default. This section reports the gate's
output descriptively, as the proposal specifies, not as a replacement for the paper's
own hypothesis tests.

### Gate status, by design (`*_gate_summary.csv`)

| Dataset | Design | logreg | rf | gbm |
|---|---|---|---|---|
| DataCo | row-random | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW |
| DataCo | order-grouped | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW |
| DataCo | customer-grouped | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW |
| DataCo | chronological (primary) | NEEDS_REVIEW | **BLOCKED** | NEEDS_REVIEW |
| Olist | row-random | **PASS** | **PASS** | **PASS** |
| Olist | order-grouped | **PASS** | **PASS** | **PASS** |
| Olist | customer-grouped | **PASS** | **PASS** | **PASS** |
| Olist | chronological (primary) | NEEDS_REVIEW | **BLOCKED** | NEEDS_REVIEW |

For Olist the pattern is unambiguous: every random design **PASSes** cleanly for every
real model family, and every chronological design degrades to NEEDS_REVIEW or
BLOCKED — an independent, model-governance-level confirmation of this paper's central
claim, produced by a tool with no knowledge of our hypotheses. For DataCo, random
forest alone is BLOCKED under the chronological design (the others are NEEDS_REVIEW
across every design), which lines up with the exceptionally poor calibration slope
(2.89-3.25) random forest shows under chronological evaluation in O2's own table.

### What specifically triggered the flags

- **Olist, chronological, random forest**: `performance_thresholds` BLOCKED on
  `roc_auc=0.456 [0.417, 0.496]`, entirely below the 0.6 floor (the capped gate sample
  is slightly different from the full-data 0.487 reported in O2, within sampling
  noise). The gate's own `feature_drift` check independently flags **`promised_days`**
  (standardised mean shift 0.37) and **`purchase_month`** (0.34) as the two features
  that moved most between training and the chronological test window — exactly the
  mechanism O4's auxiliary operational model traces (a shrinking promise window, not
  operational decline). It also flags **`main_category`** drift (total variation
  distance 0.156), a product-mix shift that O4's own drift features did not track,
  surfaced independently by the gate.
- **DataCo, chronological, random forest**: `calibration` BLOCKED on
  `ECE=0.130 [0.116, 0.143]`, entirely above the 0.1 floor. `feature_drift` flags
  **Order State, Order Country, Order Region and Market** (total variation distance
  0.52-0.57, all far above the 0.15 threshold) — DataCo's drift is predominantly
  geographic, consistent with the proposal's positioning on "new markets" as a driver
  of operational change (Section 1), and not something O4's own drift features
  (scheduled days, sales, discount rate) were tracking either.
- **Every single-feature baseline is BLOCKED**, mostly on `performance_thresholds`
  (a one-feature logistic regression rarely clears AUC 0.6). One result is a useful
  methodological caution rather than a finding: the Olist `promised_days`-only
  baseline's `split_overlap` check reports 98.5% "duplicate rows" and 93.2% "overlap
  with training" — not leakage, but an artefact of comparing rows on a single
  low-cardinality integer column, where most orders share the same promised-day value.
  A generic duplicate/overlap check needs the full feature set to be informative; run
  against one coarse column it manufactures an apparent leakage signal. This is folded
  into the O5 protocol below.

A demo HTML report (`report.to_html()`, "the page a reviewer reads") was generated for
the primary estimand's gradient-boosted model on both datasets:
`results/o2_design_comparison/gate_reports/dataco_chronological_primary_gbm_demo.html`
and the Olist equivalent.

## O5: Reporting protocol

A minimum evidence package for a supply-chain late-delivery performance claim, derived
from the findings above:

1. **State the estimand before the number.** "AUC = X" is not falsifiable on its own;
   "AUC = X under row-random validation" and "AUC = X under chronological validation,
   orders intact, returning customers allowed" are different, non-interchangeable
   claims (O2). Report the primary estimand (Section 5.1) first and others as
   explicitly labelled comparisons.
2. **Report the group-overlap share of the training/test split**, not just whether
   the split was "grouped." DataCo's O3a result shows AUC moving quasi-linearly with
   overlap share (slope 0.229, r=0.994) — a qualitative "grouped: yes/no" checkbox
   hides a continuous, large effect.
3. **Report the model family alongside the validation-design contrast.** H1/H2 show
   the same design swap produces a materially different, sometimes *opposite-signed*,
   effect depending on whether the model is linear or a tree ensemble. A validation
   contrast computed on one model family does not generalise to another.
4. **Report training recency and the promised-window distribution at train and test
   time**, not just a date range. The Olist chronological collapse (RF: 0.78 -> 0.49
   AUC) traces to a shift in `promised_days`, not a generic "concept drift" — specific,
   checkable, and actionable (shorten the promise, retrain) in a way "performance
   degraded over time" is not.
5. **Run the fixed-test-set dose-response check (O3a-style) before trusting a
   grouped-vs-random contrast as evidence of a leakage mechanism**, rather than a mere
   descriptive difference. A two-point comparison (row-random AUC vs. grouped AUC)
   cannot distinguish "leakage via group overlap" from "the two splits happened to
   draw different, unluckier test sets"; a monotonic dose-response across controlled
   overlap shares can.
6. **Use a cluster/block bootstrap, not repeated random splits, for the uncertainty
   interval**, and report Holm-adjusted verdicts (superior/inferior/equivalent/
   inconclusive against a pre-registered margin) rather than a bare point estimate.
   H1 shows exactly why: two model families' row-random-minus-grouped AUC gaps are
   each positive but only one clears the registered delta — a p-value-free "yes,
   positive" reading would have overstated both.
7. **Run a standardised governance gate (e.g. BDP Model Gate) on every validation
   design, not only the one used for the headline number**, and report its flags
   descriptively. See the BDP Model Gate section above for what this run surfaced.
8. **Do not report drift as inherently a deterioration risk.** Eck et al. (2022)'s
   finding that drift can occur without deterioration is directly visible in O4: not
   every origin with elevated PSI shows next-period AUC decline. Report the drift
   metric and the next-period performance change as two separate numbers, and let the
   reader see when they decouple.
9. **Read a generic duplicate/overlap check against the full feature set, not a
   single baseline column.** The gate run above shows a one-feature baseline
   manufacturing a 98.5% "duplicate rows" flag purely from a low-cardinality column,
   with no leakage present. A governance check's flag is only interpretable alongside
   the feature set it was computed on.
