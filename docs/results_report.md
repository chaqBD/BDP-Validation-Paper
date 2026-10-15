---
title: "Full Results Report — When Validation Misleads"
subtitle: "Group Dependence, Temporal Change and the Reliability of Performance Estimates in Supply Chain Machine Learning (Paper A)"
author: "Tewogbade Shakir Adeyemi"
date: "October 2026"
---

# Purpose and scope

This document reports every numerical result produced by the Paper A analysis codebase (`github.com/chaqBD/BDP-Validation-Paper`) against the registered research framework. It is organised by objective (O1–O5) exactly as the proposal specifies, and every table here is reproducible from the CSV/JSON files in `results/` by running the corresponding script in `scripts/`. Where a finding is discussed rather than merely tabulated, the discussion is confined to what the numbers show; interpretation at publication depth is reserved for the companion paper.

Two datasets are used throughout: **DataCo** (a simulated/semi-synthetic smart-supply-chain dataset, order-line-item granularity) and **Olist** (a real Brazilian e-commerce marketplace dataset, order granularity). All code ran on Python 3.13 with scikit-learn 1.7, LightGBM 4.6, and BDP Model Gate 0.6.0.

---

# O1 — Data audit and task definition

## DataCo

| Quantity | Value |
|---|---|
| Total rows (order line items) | 180,519 |
| Cancelled rows (`Delivery Status == "Shipping canceled"`) | 7,754 (4.30%) |
| Non-cancelled rows | 172,765 (95.70%) |
| Unique orders | 65,752 |
| Mean line items per order | 2.75 |
| Unique customers (full dataset) | 20,652 |
| Date range | 2015-01-01 to 2018-01-31 |
| Overall late-delivery rate (`Late_delivery_risk`) | 54.83% |
| First Class orders, late rate (non-cancelled) | 100.00% |
| Orders where every line item shares one label | 100.00% |

**Primary population** (non-cancelled, 2015-01-01 to 2017-09-30):

| Quantity | Value |
|---|---|
| Rows | 164,401 |
| Orders | 54,796 |
| Customers | 12,296 |
| Late rate | 57.28% |
| Features | 23 (12 numeric, 11 categorical) |

Numeric features: `Days for shipment (scheduled)`, `Order Item Discount`, `Order Item Discount Rate`, `Order Item Product Price`, `Order Item Quantity`, `Sales`, `Order Item Total`, `Product Price`, `Latitude`, `Longitude`, `order_month`, `order_dow`.
Categorical features: `Type`, `Category Name`, `Customer Segment`, `Customer State`, `Customer Country`, `Market`, `Order Region`, `Order State`, `Order Country`, `Shipping Mode`, `Department Name`.
Excluded as leaky (known only after the decision): `Days for shipping (real)`, `Delivery Status`, `Order Status`, `shipping date (DateOrders)`. Excluded as identifiers/PII/free text/constant: `Customer Id`, `Customer Email`, `Customer Fname`, `Customer Lname`, `Customer Password`, `Customer Street`, `Customer Zipcode`, `Order Customer Id`, `Order Id`, `Order Item Id`, `Order Item Cardprod Id`, `Order Zipcode`, `Product Card Id`, `Product Category Id`, `Category Id`, `Department Id`, `Product Description`, `Product Image`, `Product Name`, `Customer City`, `Order City`, `Product Status` (constant).

## Olist

| Quantity | Value |
|---|---|
| Orders, all statuses | 99,441 |
| Delivered orders | 96,478 |
| Late rate among delivered orders | 8.11% |
| Purchase date range | 2016-09-04 to 2018-10-17 |
| Mean items per order | 1.13 |
| Monthly late rate, reliable months (>=30 orders), primary window | 1.36% – 21.36% |

**Primary population** (delivered, promise >=14 days before the data cut-off, purchases 2017-01-01 to 2018-08-31; 8 orders marked "delivered" with no recorded delivery date excluded as a data-quality artefact rather than coded either way):

| Quantity | Value |
|---|---|
| Rows (= orders) | 96,180 |
| Customers (`customer_unique_id`) | 93,078 |
| Late rate | 8.13% |
| Features | 17 (14 numeric, 3 categorical) |

Numeric features: `promised_days`, `n_items`, `n_sellers`, `n_categories`, `total_price`, `total_freight`, `avg_item_price`, `total_payment_value`, `payment_installments`, `avg_product_weight_g`, `avg_product_volume_cm3`, `purchase_month`, `purchase_dow`, `n_prior_orders`.
Categorical features: `customer_state`, `payment_type_mode`, `main_category`.
Excluded as leaky: approval/carrier/delivery timestamps, review data.

---

# O2 — Sensitivity of performance estimates to the validation design

Five validation designs x three tuned model families (regularised logistic regression, random forest, gradient boosting [LightGBM]) x two per-dataset baselines, with training size matched in unique orders across the four non-secondary designs (DataCo: 43,804 orders; Olist: 76,904 orders), a shared two-configuration tuning grid per family, time-aware inner validation (`TimeSeriesSplit`, 2 folds), and 400-resample paired cluster bootstrap (clustered on order id) for every AUC interval.

## DataCo — full metrics by design and model

| Design | Model | AUC | 95% CI | AP | Brier | Cal-in-large | Cal slope | ECE |
|---|---|---|---|---|---|---|---|---|
| Row-random | logreg | 0.7627 | 0.7565–0.7686 | 0.8437 | 0.1862 | -0.0047 | 1.015 | 0.0085 |
| Row-random | rf | 0.7557 | 0.7494–0.7618 | 0.8407 | 0.2070 | -0.0049 | 2.668 | 0.1107 |
| Row-random | gbm | 0.7765 | 0.7699–0.7823 | 0.8496 | 0.1834 | -0.0046 | 1.107 | 0.0169 |
| Row-random | baseline: shipping mode | 0.7418 | 0.7361–0.7472 | 0.7833 | 0.1878 | -0.0046 | 1.007 | 0.0047 |
| Row-random | baseline: scheduled days | 0.7083 | 0.7014–0.7142 | 0.7122 | 0.2039 | -0.0054 | 0.977 | 0.0747 |
| Order-grouped | logreg | 0.7418 | 0.7318–0.7507 | 0.8409 | 0.1889 | 0.0144 | 0.962 | 0.0344 |
| Order-grouped | rf | 0.7409 | 0.7311–0.7501 | 0.8407 | 0.1888 | 0.0149 | 1.060 | 0.0362 |
| Order-grouped | gbm | 0.7444 | 0.7338–0.7540 | 0.8411 | 0.1885 | 0.0145 | 0.964 | 0.0238 |
| Order-grouped | baseline: shipping mode | 0.7415 | 0.7331–0.7489 | 0.7928 | 0.1867 | 0.0149 | 1.058 | 0.0181 |
| Order-grouped | baseline: scheduled days | 0.7065 | 0.6971–0.7161 | 0.7230 | 0.2030 | 0.0153 | 0.971 | 0.0663 |
| Customer-grouped | logreg | 0.7420 | 0.7332–0.7524 | 0.8314 | 0.1899 | -0.0104 | 0.914 | 0.0351 |
| Customer-grouped | rf | 0.7416 | 0.7322–0.7521 | 0.8319 | 0.1897 | -0.0099 | 1.035 | 0.0336 |
| Customer-grouped | gbm | 0.7421 | 0.7323–0.7524 | 0.8308 | 0.1888 | -0.0103 | 0.947 | 0.0176 |
| Customer-grouped | baseline: shipping mode | 0.7433 | 0.7355–0.7526 | 0.7829 | 0.1875 | -0.0105 | 0.993 | 0.0106 |
| Customer-grouped | baseline: scheduled days | 0.7084 | 0.6987–0.7177 | 0.7100 | 0.2043 | -0.0107 | 0.962 | 0.0768 |
| Chronological (primary) | logreg | 0.7467 | 0.7353–0.7561 | 0.8348 | 0.1902 | -0.0008 | 0.930 | 0.0159 |
| Chronological (primary) | rf | 0.7437 | 0.7341–0.7534 | 0.8339 | 0.2080 | -0.0032 | 2.893 | 0.1228 |
| Chronological (primary) | gbm | 0.7450 | 0.7357–0.7551 | 0.8325 | 0.1896 | -0.0014 | 0.949 | 0.0092 |
| Chronological (primary) | baseline: shipping mode | 0.7421 | 0.7337–0.7503 | 0.7816 | 0.1891 | -0.0020 | 0.981 | 0.0092 |
| Chronological (primary) | baseline: scheduled days | 0.7154 | 0.7064–0.7248 | 0.7210 | 0.2015 | -0.0024 | 1.058 | 0.0707 |
| Chronological, new customers | logreg | 0.7005 | 0.6403–0.7633 | 0.7648 | 0.2072 | -0.0224 | 0.746 | 0.0252 |
| Chronological, new customers | rf | 0.6711 | 0.6120–0.7418 | 0.7616 | 0.2197 | -0.0381 | 2.477 | 0.1210 |
| Chronological, new customers | gbm | 0.7186 | 0.6647–0.7825 | 0.7824 | 0.2034 | -0.0225 | 0.822 | 0.0250 |
| Chronological, new customers | baseline: shipping mode | 0.7131 | 0.6578–0.7657 | 0.7266 | 0.2022 | -0.0244 | 0.865 | 0.0294 |
| Chronological, new customers | baseline: scheduled days | 0.6918 | 0.6285–0.7494 | 0.6665 | 0.2134 | -0.0306 | 0.945 | 0.0665 |

(n test rows: row-random/order-grouped/customer-grouped/chronological-primary ~32,800–33,100; chronological new-customers n=896.)

## Olist — full metrics by design and model

| Design | Model | AUC | 95% CI | AP | Brier | Cal-in-large | Cal slope | ECE |
|---|---|---|---|---|---|---|---|---|
| Row-random | logreg | 0.6832 | 0.6689–0.6969 | 0.1591 | 0.0716 | -0.0003 | 0.959 | 0.0021 |
| Row-random | rf | 0.7610 | 0.7478–0.7735 | 0.2605 | 0.0677 | -0.0007 | 1.356 | 0.0141 |
| Row-random | gbm | 0.7782 | 0.7665–0.7893 | 0.2652 | 0.0665 | 0.0002 | 1.067 | 0.0062 |
| Row-random | baseline: promised days | 0.5511 | 0.5351–0.5639 | 0.1036 | 0.0739 | -0.0008 | 1.053 | 0.0038 |
| Row-random | baseline: customer region | 0.6104 | 0.5966–0.6243 | 0.1144 | 0.0731 | -0.0009 | 0.917 | 0.0017 |
| Order-grouped | logreg | 0.6847 | 0.6711–0.6974 | 0.1677 | 0.0702 | -0.0019 | 0.991 | 0.0033 |
| Order-grouped | rf | 0.7637 | 0.7503–0.7753 | 0.2710 | 0.0662 | -0.0030 | 1.380 | 0.0148 |
| Order-grouped | gbm | 0.7795 | 0.7671–0.7908 | 0.2758 | 0.0650 | -0.0015 | 1.038 | 0.0067 |
| Order-grouped | baseline: promised days | 0.5548 | 0.5384–0.5693 | 0.1080 | 0.0728 | -0.0024 | 1.126 | 0.0024 |
| Order-grouped | baseline: customer region | 0.6072 | 0.5930–0.6216 | 0.1182 | 0.0720 | -0.0025 | 0.927 | 0.0031 |
| Customer-grouped | logreg | 0.6826 | 0.6685–0.6954 | 0.1655 | 0.0731 | 0.0011 | 0.948 | 0.0023 |
| Customer-grouped | rf | 0.7639 | 0.7503–0.7761 | 0.2677 | 0.0689 | 0.0004 | 1.368 | 0.0175 |
| Customer-grouped | gbm | **0.7817** | 0.7695–0.7928 | 0.2751 | 0.0676 | 0.0011 | 1.073 | 0.0071 |
| Customer-grouped | baseline: promised days | 0.5546 | 0.5332–0.5598 | 0.1114 | 0.0755 | 0.0017 | 1.010 | 0.0041 |
| Customer-grouped | baseline: customer region | 0.6073 | 0.5919–0.6212 | 0.1213 | 0.0747 | 0.0011 | 0.921 | 0.0038 |
| Chronological (primary) | logreg | 0.7117 | 0.6969–0.7244 | 0.1122 | 0.0525 | -0.0473 | 0.937 | 0.0473 |
| Chronological (primary) | rf | **0.4870** | 0.4435–0.4803 | 0.0561 | 0.0515 | -0.0246 | -0.322 | 0.0246 |
| Chronological (primary) | gbm | 0.7203 | 0.7054–0.7332 | 0.1080 | 0.0490 | -0.0041 | 1.039 | 0.0042 |
| Chronological (primary) | baseline: promised days | 0.7974 | 0.7819–0.8110 | 0.2375 | 0.0507 | -0.0386 | **11.887** | 0.0585 |
| Chronological (primary) | baseline: customer region | 0.4229 | 0.4061–0.4393 | 0.0520 | 0.0535 | -0.0316 | -0.345 | 0.0316 |
| Chronological, new customers | logreg | 0.7090 | 0.6956–0.7234 | 0.1133 | 0.0530 | -0.0470 | 0.933 | 0.0470 |
| Chronological, new customers | rf | 0.4604 | 0.4418–0.4796 | 0.0566 | 0.0520 | -0.0240 | -0.338 | 0.0240 |
| Chronological, new customers | gbm | 0.7188 | 0.7041–0.7308 | 0.1085 | 0.0496 | -0.0036 | 1.032 | 0.0039 |
| Chronological, new customers | baseline: promised days | 0.7984 | 0.7834–0.8113 | 0.2401 | 0.0512 | -0.0380 | 11.943 | 0.0587 |
| Chronological, new customers | baseline: customer region | 0.4220 | 0.4058–0.4384 | 0.0526 | 0.0541 | -0.0310 | -0.351 | 0.0310 |

(n test rows: row-random/order-grouped ~19,236; customer-grouped n=19,276; chronological-primary n=19,236; chronological new-customers n=18,786.)

---

# Registered hypothesis tests (RQ1, RQ4)

Delta (equivalence margin) = 0.01 AUC, registered in advance. Verdicts follow Lakens (2017): *superior* (CI entirely above delta), *inferior* (CI entirely below -delta), *equivalent* (CI entirely within ±delta), *inconclusive* (CI straddles delta). Paired cluster bootstrap, 1,000 resamples, clustered on order id. Holm-adjusted across the per-model family of tests within each hypothesis.

## H1 — DataCo: row-random AUC exceeds order-grouped AUC by more than delta

| Model | Row-random minus order-grouped AUC | 95% CI | Verdict |
|---|---|---|---|
| logreg | +0.0209 | 0.0090–0.0323 | Inconclusive |
| rf | +0.0148 | 0.0030–0.0267 | Inconclusive |
| gbm | +0.0321 | 0.0203–0.0434 | **Superior** |

## H2 — Olist: chronological AUC is lower than order-grouped AUC by more than delta

| Model | Chronological minus order-grouped AUC | 95% CI | Verdict |
|---|---|---|---|
| logreg | +0.0287 | 0.0104–0.0480 | **Superior** (opposite direction to H2) |
| rf | -0.2769 | [-0.2998, -0.2535] | **Inferior** (confirms H2) |
| gbm | -0.0651 | [-0.0834, -0.0454] | **Inferior** (confirms H2) |

## H6 — model rankings under row-random vs. the primary (chronological) design

| Dataset | Row-random AUC (logreg, rf, gbm) | Row-random rank | Primary-design AUC | Primary-design rank | Kendall's tau | Rank-change frequency |
|---|---|---|---|---|---|---|
| DataCo | 0.7627, 0.7557, 0.7765 | logreg=2, rf=3, gbm=1 | 0.7467, 0.7437, 0.7450 | logreg=1, rf=3, gbm=2 | 0.333 (CI [-1, 1]) | 2/3 |
| Olist | 0.6832, 0.7610, 0.7782 | logreg=3, rf=2, gbm=1 | 0.7117, 0.4870, 0.7203 | logreg=2, rf=3, gbm=1 | 0.333 (CI [-1, 1]) | 2/3 |

With only three model families, Kendall's tau's bootstrap interval spans the full [-1, 1] range in both datasets — the test is not powered to detect a significant association at this n. The rank-change frequency is reported descriptively.

---

# O3a — Fixed-test-set mechanism experiments (RQ2, H3)

Test set held fixed per evaluation period; training set varies across four conditions (A: past only; B: past + contemporaneous, test orders excluded; C: B + a controlled share of test orders' sibling rows [DataCo only]; D: past only + the test customers' own earlier orders guaranteed present), training size held constant (DataCo: n=6,000; Olist: n=4,000) within each period. Model: LightGBM, fixed hyperparameters.

## DataCo (four periods: 2017-03, 2017-05, 2017-07, 2017-09)

| Period | A (past only) | B (+ time coverage) | C @ 0% | C @ 20% | C @ 40% | C @ 60% | C @ 80% | C @ 100% | D (+ customer history) |
|---|---|---|---|---|---|---|---|---|---|
| 2017-03 | 0.7337 | 0.7275 | 0.7275 | 0.7883 | 0.8311 | 0.8799 | 0.9183 | 0.9604 | 0.7267 |
| 2017-05 | 0.7445 | 0.7410 | 0.7410 | 0.7797 | 0.8349 | 0.8783 | 0.9298 | 0.9657 | 0.7512 |
| 2017-07 | 0.7521 | 0.7410 | 0.7410 | 0.7968 | 0.8539 | 0.8953 | 0.9357 | 0.9697 | 0.7214 |
| 2017-09 | 0.7343 | 0.7459 | 0.7459 | 0.7893 | 0.8472 | 0.8938 | 0.9336 | 0.9677 | 0.7169 |
| **Mean** | **0.7412** | **0.7389** | **0.7389** | **0.7885** | **0.8418** | **0.8868** | **0.9294** | **0.9659** | **0.7290** |

**Dose-response slope test (H3)**: AUC on overlap share, pooled across all four periods: slope = 0.2289 AUC per unit overlap share, intercept = 0.7441, r = 0.9941, p = 1.00 x 10⁻²². Number of leaked sibling orders at 100% overlap ranged from 612–688 across periods (out of 791–855 test orders with >=2 line items in the test window).

## Olist (three periods: 2017-11, 2018-02, 2018-05; condition C not run — row = order, Section 5.1)

| Period | A (past only) | B (+ time coverage) | D (+ customer history) |
|---|---|---|---|
| 2017-11 | 0.5542 | 0.6096 | 0.5633 |
| 2018-02 | 0.6376 | 0.6295 | 0.6576 |
| 2018-05 | 0.6019 | 0.6683 | 0.6430 |
| **Mean** | **0.5979** | **0.6358** | **0.6213** |

---

# O3b — Simulation with known mechanisms (RQ2, H4)

Full factorial design: W (within-order correlation) x S (covariate shift) x R (base-rate shift) x P (promise-coefficient change), each on/off, 2⁴ = 16 cells, 8 replicates per cell at n=3,200 orders, with a large-N (n=40,000 orders) reference run per cell as ground truth. Fitting model: random forest (capacity needed for the leakage channel to manifest; see the companion paper's methodology section for why a well-specified linear model cannot show it). Contrasts: *random-vs-grouped* (row-random AUC minus order-grouped AUC, isolates W) and *grouped-vs-chronological* (order-grouped AUC minus chronological AUC, isolates S/R/P).

## Main effect of W on the random-vs-grouped contrast

| W | Estimated contrast (mean, 8 reps x 8 cells) | Truth contrast (large-N reference) | Recovery error (estimate − truth) |
|---|---|---|---|
| Off | -0.0026 | +0.0046 | -0.0072 |
| On | +0.1184 | +0.0397 | +0.0789 |

## Main pattern of S, R, P on the grouped-vs-chronological contrast

| S | R | P | Estimated contrast | Truth contrast | Recovery error |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.0339 | 0.0338 | -0.0033 |
| 0 | 0 | 1 | 0.0811 | 0.0974 | -0.0164 |
| 0 | 1 | 0 | 0.0305 | 0.0409 | -0.0104 |
| 0 | 1 | 1 | 0.0992 | 0.1113 | -0.0121 |
| 1 | 0 | 0 | 0.0538 | 0.0500 | +0.0038 |
| 1 | 0 | 1 | 0.1005 | 0.1191 | -0.0187 |
| 1 | 1 | 0 | 0.0393 | 0.0653 | -0.0259 |
| 1 | 1 | 1 | 0.1192 | 0.1403 | -0.0211 |

## Full 16-cell factorial (W x S x R x P)

| W | S | R | P | Est. random-vs-grouped | Est. grouped-vs-chronological | Recovery error (r-v-g) | Recovery error (g-v-c) |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | -0.0103 | 0.0353 | -0.0154 | +0.0014 |
| 0 | 0 | 0 | 1 | 0.0020 | 0.0811 | -0.0039 | -0.0164 |
| 0 | 0 | 1 | 0 | -0.0014 | 0.0305 | -0.0043 | -0.0104 |
| 0 | 0 | 1 | 1 | -0.0047 | 0.0992 | -0.0101 | -0.0121 |
| 0 | 1 | 0 | 0 | -0.0042 | 0.0538 | -0.0078 | +0.0038 |
| 0 | 1 | 0 | 1 | -0.0038 | 0.1005 | -0.0085 | -0.0187 |
| 0 | 1 | 1 | 0 | 0.0061 | 0.0393 | 0.0026 | -0.0259 |
| 0 | 1 | 1 | 1 | -0.0046 | 0.1192 | -0.0099 | -0.0211 |
| 1 | 0 | 0 | 0 | 0.1237 | 0.0326 | 0.0920 | -0.0080 |
| 1 | 0 | 0 | 1 | 0.1168 | 0.0373 | 0.0741 | -0.0253 |
| 1 | 0 | 1 | 0 | 0.1151 | 0.0197 | 0.0755 | -0.0252 |
| 1 | 0 | 1 | 1 | 0.1200 | 0.0515 | 0.0762 | -0.0140 |
| 1 | 1 | 0 | 0 | 0.1230 | 0.0244 | 0.0829 | -0.0184 |
| 1 | 1 | 0 | 1 | 0.1184 | 0.0363 | 0.0770 | -0.0300 |
| 1 | 1 | 1 | 0 | 0.1138 | 0.0233 | 0.0774 | -0.0315 |
| 1 | 1 | 1 | 1 | 0.1164 | 0.0477 | 0.0762 | -0.0336 |

Recovery error is consistently positive (overestimation) for the W=1 random-vs-grouped contrast (mean +0.079) and consistently negative (underestimation) for the grouped-vs-chronological contrast once any temporal factor is on (mean roughly -0.02). Both biases are systematic, not centred on zero, at this replicate count (8 reps/cell, n=3,200 orders).

---

# O4 — Rolling-origin reliability analysis (RQ3, H5)

Ten expanding-window forecast origins per dataset (minimum training fraction 40%, ~6–8% test window per origin), LightGBM, drift measured as PSI and standardised mean shift (SMD) on a small set of operationally meaningful features per dataset.

## DataCo — origin-by-origin

| Origin | Train n | Test n | AUC | Mean PSI | Change in lateness prevalence | Change in next-period AUC |
|---|---|---|---|---|---|---|
| 0 | 65,760 | 9,864 | 0.7345 | 0.00069 | -0.0061 | +0.0097 |
| 1 | 75,624 | 9,864 | 0.7442 | 0.00080 | +0.0125 | -0.0015 |
| 2 | 85,488 | 9,864 | 0.7427 | 0.00034 | -0.0058 | -0.0119 |
| 3 | 95,352 | 9,864 | 0.7308 | 0.00084 | -0.0091 | +0.0055 |
| 4 | 105,216 | 9,864 | 0.7363 | 0.00020 | +0.0009 | +0.0078 |
| 5 | 115,080 | 9,864 | 0.7441 | 0.00085 | -0.0054 | +0.0032 |
| 6 | 124,944 | 9,864 | 0.7473 | 0.00055 | +0.0010 | -0.0018 |
| 7 | 134,808 | 9,864 | 0.7455 | 0.00324 | -0.0097 | -0.0054 |
| 8 | 144,672 | 9,864 | 0.7401 | 0.00792 | -0.0039 | +0.0184 |
| 9 | 154,536 | 9,864 | 0.7585 | 0.00895 | +0.0054 | (no next origin) |

**H5 regressions**: PSI vs. change in next-period AUC: slope = 1.938, r = 0.538, p = 0.135 (n=9). Change in lateness prevalence vs. change in next-period |calibration-in-the-large|: slope = -0.185, r = -0.194, p = 0.618 (n=9). Neither clears conventional significance; direction is positive for drift-vs-next-AUC (i.e., *not* preceding deterioration).

## Olist — origin-by-origin

| Origin | Train n | Test n | AUC | Mean PSI | Change in lateness prevalence | Change in next-period AUC |
|---|---|---|---|---|---|---|
| 0 | 38,472 | 5,770 | 0.7247 | 0.4863 | +0.0096 | -0.0923 |
| 1 | 44,242 | 5,770 | 0.6324 | 0.0328 | +0.0021 | +0.0374 |
| 2 | 50,012 | 5,770 | 0.6698 | 0.0115 | +0.0615 | -0.0138 |
| 3 | 55,782 | 5,770 | 0.6560 | 0.0521 | +0.1914 | +0.0767 |
| 4 | 61,552 | 5,770 | 0.7327 | 0.0458 | +0.0017 | -0.0267 |
| 5 | 67,322 | 5,770 | 0.7060 | 0.0143 | -0.0442 | +0.0109 |
| 6 | 73,092 | 5,770 | 0.7169 | 0.1028 | -0.0089 | +0.0367 |
| 7 | 78,862 | 5,770 | 0.7536 | 0.0274 | -0.0711 | -0.0928 |
| 8 | 84,632 | 5,770 | 0.6607 | 0.1730 | -0.0224 | +0.1462 |
| 9 | 90,402 | 5,770 | 0.8069 | 0.4205 | +0.0208 | (no next origin) |

**H5 regressions**: PSI vs. change in next-period AUC: slope = -0.125, r = -0.246, p = 0.523 (n=9). Change in lateness prevalence vs. change in next-period |calibration-in-the-large|: slope = -0.406, r = -0.482, p = 0.189 (n=9). Direction is consistent with "more drift, worse next-period AUC" but underpowered at 9 origins.

## Olist auxiliary operational model (separating operational change from promise change)

A random-forest regressor predicts actual fulfilment time from operational features only (`promised_days` excluded), trained on the same expanding windows.

| Origin | MAE (days) | Mean promised days (test) | Mean actual fulfilment days (test) |
|---|---|---|---|
| 0 | 5.95 | 27.27 | 14.54 |
| 1 | 5.78 | 25.89 | 13.66 |
| 2 | 6.40 | 24.63 | 15.82 |
| 3 | 7.87 | 22.05 | 17.54 |
| 4 | 5.69 | 21.94 | 12.27 |
| 5 | 4.66 | 23.08 | 10.50 |
| 6 | 4.88 | 25.94 | 11.14 |
| 7 | 4.55 | 24.65 | 8.26 |
| 8 | 4.40 | 18.41 | 8.70 |
| 9 | 3.66 | 14.65 | 7.16 |

The auxiliary model's mean absolute error *falls* from 5.95 to 3.66 days across the series (i.e., operational fulfilment-time prediction does not deteriorate), while the mean promised window collapses from 27.3 to 14.6 days and mean actual fulfilment time also falls from 14.5 to 7.2 days. This indicates that the O2 chronological AUC collapse for Olist is attributable to the promise shrinking, not to operational unpredictability.

---

# BDP Model Gate (Section 7.7) — full results

Every (design, model) combination fitted in O2 was passed through BDP Model Gate v0.6.0 with `performance.min_score=0.6`, `security.adversarial_flip_rate_threshold=0.25` (relaxed from deployment-grade defaults of 0.8/0.05; validation-category thresholds left at default), a declared `model_card` (validation strategy, use case, legal basis), and rows capped at 4,000/side for tractability.

## Gate status summary

| Dataset | Design | logreg | rf | gbm | baseline 1 | baseline 2 |
|---|---|---|---|---|---|---|
| DataCo | Row-random | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | BLOCKED | BLOCKED |
| DataCo | Order-grouped | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | BLOCKED | BLOCKED |
| DataCo | Customer-grouped | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | BLOCKED | BLOCKED |
| DataCo | Chronological (primary) | NEEDS_REVIEW | **BLOCKED** | NEEDS_REVIEW | BLOCKED | BLOCKED |
| DataCo | Chronological, new customers | NEEDS_REVIEW | NEEDS_REVIEW | NEEDS_REVIEW | BLOCKED | BLOCKED |
| Olist | Row-random | **PASS** | **PASS** | **PASS** | BLOCKED | BLOCKED |
| Olist | Order-grouped | **PASS** | **PASS** | **PASS** | BLOCKED | BLOCKED |
| Olist | Customer-grouped | **PASS** | **PASS** | **PASS** | BLOCKED | BLOCKED |
| Olist | Chronological (primary) | NEEDS_REVIEW | **BLOCKED** | NEEDS_REVIEW | BLOCKED | BLOCKED |
| Olist | Chronological, new customers | NEEDS_REVIEW | **BLOCKED** | NEEDS_REVIEW | BLOCKED | BLOCKED |

(DataCo baselines: 1 = shipping mode, 2 = scheduled days. Olist baselines: 1 = promised days, 2 = customer region.)

## Flagged checks, selected examples

**Olist, chronological-primary, random forest** — BLOCKED.
- `performance_thresholds`: roc_auc = 0.456 [0.417, 0.496] on the capped gate sample, entirely below the 0.6 floor.
- `feature_drift`: `promised_days` SMD = 0.370 (threshold 0.25); `purchase_month` SMD = 0.337; `main_category` total variation distance = 0.156 (threshold 0.15).

**DataCo, chronological-primary, random forest** — BLOCKED.
- `calibration`: ECE = 0.130 [0.116, 0.143], entirely above the 0.1 floor.
- `feature_drift`: `Order State` TVD = 0.574; `Order Country` TVD = 0.555; `Order Region` TVD = 0.548; `Market` TVD = 0.521 (threshold 0.15 for all).

**Olist, chronological-primary, baseline (promised days)** — BLOCKED.
- `split_overlap`: 3,938/4,000 (98.5%) rows flagged as duplicates within the validation set; 3,729/4,000 (93.2%) rows flagged as overlapping with training, both computed over the single `promised_days` column. This reflects the column's low cardinality (most orders share the same promised-day value), not genuine leakage — see the O5 protocol note below.

A full demonstration HTML report (`report.to_html()`) was generated for the chronological-primary gradient-boosting model on each dataset: `results/o2_design_comparison/gate_reports/dataco_chronological_primary_gbm_demo.html` and the Olist equivalent.

---

# O5 — Reporting protocol (minimum evidence package)

1. State the estimand before the number — label every AUC/Brier/calibration figure with its validation design.
2. Report the group-overlap share of the training/test split, not just "grouped: yes/no."
3. Report the model family alongside the validation-design contrast; the same design swap can be superior for one family and inferior for another (H1/H2).
4. Report training recency and the promised-window distribution at train and test time, not just a date range.
5. Run a fixed-test-set dose-response check before treating a grouped-vs-random contrast as evidence of a leakage mechanism rather than a descriptive difference.
6. Use a cluster/block bootstrap and Holm-adjusted equivalence verdicts, not a bare point estimate or repeated-random-split "confidence interval."
7. Run a standardised governance gate on every validation design used, not only the one behind the headline number, and report its flags descriptively.
8. Do not treat drift as inherently a deterioration risk; report the drift metric and the next-period performance change as two separate numbers.
9. Read a generic duplicate/overlap check against the full feature set used by the model, not a single low-cardinality baseline column.

---

# Software and reproducibility

Python 3.13.14; pandas 2.2; numpy 2.4; scikit-learn 1.7.2; lightgbm 4.6.0; scipy 1.17.1; bdp-model-gate 0.6.0 (pinned; PyPI's current release at analysis time was 0.7.0). Random seed 20261015 throughout. All code, the registered analysis plan, and every CSV/JSON backing this report are released at `github.com/chaqBD/BDP-Validation-Paper`.
