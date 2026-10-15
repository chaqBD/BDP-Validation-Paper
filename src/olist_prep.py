"""O1 for Olist: decision time, unit of analysis, eligible population, predictors
available at decision time (Section 5.2, Section 6).

Row = order. Decision time = purchase. Outcome = delivered after the promised date.
"""
import numpy as np
import pandas as pd

from .config import OLIST_DIR, OLIST_PRIMARY_START, OLIST_PRIMARY_END

OUTCOME_COL = "late"
ORDER_ID_COL = "order_id"
CUSTOMER_ID_COL = "customer_unique_id"
TIME_COL = "purchase_ts"

NUMERIC_FEATURES = [
    "promised_days", "n_items", "n_sellers", "n_categories", "total_price",
    "total_freight", "avg_item_price", "total_payment_value", "payment_installments",
    "avg_product_weight_g", "avg_product_volume_cm3", "purchase_month", "purchase_dow",
    "n_prior_orders",
]
CATEGORICAL_FEATURES = ["customer_state", "payment_type_mode", "main_category"]


def load_raw():
    orders = pd.read_csv(OLIST_DIR / "olist_orders_dataset.csv",
                          parse_dates=["order_purchase_timestamp", "order_approved_at",
                                       "order_delivered_carrier_date",
                                       "order_delivered_customer_date",
                                       "order_estimated_delivery_date"])
    customers = pd.read_csv(OLIST_DIR / "olist_customers_dataset.csv")
    items = pd.read_csv(OLIST_DIR / "olist_order_items_dataset.csv")
    payments = pd.read_csv(OLIST_DIR / "olist_order_payments_dataset.csv")
    products = pd.read_csv(OLIST_DIR / "olist_products_dataset.csv")
    return orders, customers, items, payments, products


def _aggregate_items(items, products):
    merged = items.merge(products, on="product_id", how="left")
    merged["volume_cm3"] = (merged["product_length_cm"] * merged["product_height_cm"]
                             * merged["product_width_cm"])
    agg = merged.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        n_sellers=("seller_id", "nunique"),
        n_categories=("product_category_name", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        avg_item_price=("price", "mean"),
        avg_product_weight_g=("product_weight_g", "mean"),
        avg_product_volume_cm3=("volume_cm3", "mean"),
    ).reset_index()
    main_cat = (merged.sort_values("price", ascending=False)
                .drop_duplicates("order_id")[["order_id", "product_category_name"]]
                .rename(columns={"product_category_name": "main_category"}))
    return agg.merge(main_cat, on="order_id", how="left")


def _aggregate_payments(payments):
    agg = payments.groupby("order_id").agg(
        total_payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()
    mode_type = (payments.groupby("order_id")["payment_type"]
                 .agg(lambda s: s.value_counts().idxmax())
                 .reset_index().rename(columns={"payment_type": "payment_type_mode"}))
    return agg.merge(mode_type, on="order_id", how="left")


def audit(orders, items):
    delivered = orders.loc[orders["order_status"] == "delivered"].copy()
    delivered["late"] = (delivered["order_delivered_customer_date"]
                          > delivered["order_estimated_delivery_date"])
    out = {
        "n_orders_total": int(len(orders)),
        "n_delivered": int(len(delivered)),
        "late_rate_delivered": float(delivered["late"].mean()),
        "purchase_min": str(orders["order_purchase_timestamp"].min()),
        "purchase_max": str(orders["order_purchase_timestamp"].max()),
        "n_avg_items_per_order": float(len(items) / orders["order_id"].nunique()),
        "last_delivered_date": str(orders["order_delivered_customer_date"].max()),
    }
    in_window = delivered.loc[(delivered["order_purchase_timestamp"] >= OLIST_PRIMARY_START)
                               & (delivered["order_purchase_timestamp"] <= OLIST_PRIMARY_END)]
    monthly = (in_window.set_index("order_purchase_timestamp")["late"]
               .resample("ME").mean())
    monthly_n = (in_window.set_index("order_purchase_timestamp")["late"]
                 .resample("ME").count())
    reliable = monthly[monthly_n >= 30]
    out["monthly_late_rate_min"] = float(reliable.min())
    out["monthly_late_rate_max"] = float(reliable.max())
    return out, monthly


def build_population(orders, customers, items, payments, products,
                      treat_undelivered_as_late=False, cutoff_override=None):
    """Return (X, y, meta, numeric_cols, categorical_cols) for the Olist primary
    population (Section 5.2) with predictors available at purchase time only.
    """
    orders = orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                           on="customer_id", how="left")

    last_delivery = orders["order_delivered_customer_date"].max()
    cutoff = cutoff_override if cutoff_override is not None else last_delivery - pd.Timedelta(days=14)

    if treat_undelivered_as_late:
        pop = orders.loc[orders["order_status"].isin(["delivered", "shipped", "canceled",
                                                        "unavailable", "invoiced",
                                                        "processing"])].copy()
        pop["late"] = np.where(
            pop["order_delivered_customer_date"].notna(),
            pop["order_delivered_customer_date"] > pop["order_estimated_delivery_date"],
            pop["order_estimated_delivery_date"] < last_delivery,
        )
    else:
        pop = orders.loc[orders["order_status"] == "delivered"].copy()
        # A handful of rows (8 in the 2018-10 snapshot) are marked "delivered" but have
        # no delivered-customer-date -- a known Olist data-quality quirk. Treating a
        # missing date as "not late" would silently misclassify an unobservable outcome
        # (the same risk the proposal flags for DataCo's cancelled rows, Section 5.2), so
        # these are excluded from the eligible population rather than coded either way.
        pop = pop.loc[pop["order_delivered_customer_date"].notna()].copy()
        pop["late"] = pop["order_delivered_customer_date"] > pop["order_estimated_delivery_date"]

    pop = pop.loc[pop["order_estimated_delivery_date"] <= cutoff].copy()
    pop = pop.loc[(pop["order_purchase_timestamp"] >= OLIST_PRIMARY_START)
                  & (pop["order_purchase_timestamp"] <= OLIST_PRIMARY_END)].copy()

    item_agg = _aggregate_items(items, products)
    pay_agg = _aggregate_payments(payments)
    pop = pop.merge(item_agg, on="order_id", how="left")
    pop = pop.merge(pay_agg, on="order_id", how="left")

    pop["promised_days"] = (pop["order_estimated_delivery_date"]
                             - pop["order_purchase_timestamp"]).dt.days
    pop["purchase_month"] = pop["order_purchase_timestamp"].dt.month
    pop["purchase_dow"] = pop["order_purchase_timestamp"].dt.dayofweek

    pop = pop.sort_values("order_purchase_timestamp").reset_index(drop=True)
    first_seen = pop.groupby("customer_unique_id")["order_purchase_timestamp"].rank(method="first")
    pop["n_prior_orders"] = (first_seen - 1).astype(int)

    X = pop[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = pop["late"].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    actual_fulfillment_days = (pop["order_delivered_customer_date"]
                                - pop["order_purchase_timestamp"]).dt.days
    meta = pd.DataFrame({
        "order_id": pop["order_id"].values,
        "customer_id": pop["customer_unique_id"].values,
        "decision_time": pop["order_purchase_timestamp"].values,
        "promised_days": pop["promised_days"].values,
        "customer_state": pop["customer_state"].values,
        "actual_fulfillment_days": actual_fulfillment_days.values,
        # The exact timestamp the outcome became observable, not a day-rounded
        # reconstruction from decision_time + actual_fulfillment_days: Olist records
        # order_delivered_customer_date to the second, so label availability (Section
        # 4.2) should use it directly rather than lose precision to day-rounding.
        "label_known_time": pop["order_delivered_customer_date"].values,
    }).reset_index(drop=True)

    return X, y, meta, NUMERIC_FEATURES, CATEGORICAL_FEATURES
