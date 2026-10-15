"""O1 for DataCo: decision time, unit of analysis, eligible population, predictors
available at decision time (Section 5.2, Section 6).

Row = order line item. Decision time = order placement. Outcome = Late_delivery_risk.
"""
import numpy as np
import pandas as pd

from .config import (DATACO_CSV, DATACO_PRIMARY_START, DATACO_PRIMARY_END,
                      DATACO_SENSITIVITY_START, DATACO_SENSITIVITY_END)

# Predictors known only after the decision (Section 5.2 "Excluded predictors").
LEAKY_COLS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    "shipping date (DateOrders)",
]
PROFIT_COLS = ["Benefit per order", "Order Profit Per Order", "Order Item Profit Ratio"]

# Identifiers, PII and free text: not predictors, kept only for grouping/decision time.
ID_AND_PII_COLS = [
    "Customer Id", "Customer Email", "Customer Fname", "Customer Lname",
    "Customer Password", "Customer Street", "Customer Zipcode", "Order Customer Id",
    "Order Id", "Order Item Id", "Order Item Cardprod Id", "Order Zipcode",
    "Product Card Id", "Product Category Id", "Category Id", "Department Id",
    "Product Description", "Product Image", "Product Name", "Customer City",
    "Order City", "Product Status",
]

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)", "Order Item Discount", "Order Item Discount Rate",
    "Order Item Product Price", "Order Item Quantity", "Sales", "Order Item Total",
    "Product Price", "Latitude", "Longitude", "order_month", "order_dow",
]
CATEGORICAL_FEATURES = [
    "Type", "Category Name", "Customer Segment", "Customer State", "Customer Country",
    "Market", "Order Region", "Order State", "Order Country", "Shipping Mode",
    "Department Name",
]

OUTCOME_COL = "Late_delivery_risk"
ORDER_ID_COL = "Order Id"
CUSTOMER_ID_COL = "Order Customer Id"
TIME_COL = "order_date"


def load_raw():
    df = pd.read_csv(DATACO_CSV, encoding="latin1")
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"])
    df["order_month"] = df["order_date"].dt.month
    df["order_dow"] = df["order_date"].dt.dayofweek
    return df


def audit(df):
    """Reproduce the preliminary signals in Section 6 and return an audit dict."""
    cancelled = df["Delivery Status"] == "Shipping canceled"
    n = len(df)
    out = {
        "n_rows": int(n),
        "n_cancelled": int(cancelled.sum()),
        "pct_cancelled": float(cancelled.mean() * 100),
        "pct_non_cancelled": float((1 - cancelled.mean()) * 100),
        "n_orders": int(df[ORDER_ID_COL].nunique()),
        "mean_items_per_order": float(n / df[ORDER_ID_COL].nunique()),
        "n_customers": int(df[CUSTOMER_ID_COL].nunique()),
        "date_min": str(df["order_date"].min()),
        "date_max": str(df["order_date"].max()),
        "late_rate_overall": float(df[OUTCOME_COL].mean()),
    }
    non_cancelled = df.loc[~cancelled]
    first_class_late = non_cancelled.loc[non_cancelled["Shipping Mode"] == "First Class", OUTCOME_COL]
    out["first_class_late_rate"] = float(first_class_late.mean()) if len(first_class_late) else np.nan

    # Every item in an order shares the same label? (Section 6 "DataCo grouping")
    label_nunique_per_order = non_cancelled.groupby(ORDER_ID_COL)[OUTCOME_COL].nunique()
    out["pct_orders_single_label"] = float((label_nunique_per_order == 1).mean() * 100)
    return out


def build_population(df, include_sensitivity_window=False, include_cancelled=False,
                      include_profit=False):
    """Return (X, y, meta) for the DataCo primary population and feature set.

    meta carries order id, customer id and decision time for grouping / chronological
    splitting / cluster bootstrap, kept out of X so they are never used as features.
    """
    data = df.copy()
    if not include_cancelled:
        data = data.loc[data["Delivery Status"] != "Shipping canceled"].copy()

    start, end = DATACO_PRIMARY_START, DATACO_PRIMARY_END
    if include_sensitivity_window:
        end = DATACO_SENSITIVITY_END
    mask = (data["order_date"] >= start) & (data["order_date"] <= end)
    data = data.loc[mask].copy()

    numeric = list(NUMERIC_FEATURES)
    if include_profit:
        numeric = numeric + PROFIT_COLS

    feature_cols = numeric + CATEGORICAL_FEATURES
    X = data[feature_cols].copy()
    y = data[OUTCOME_COL].astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    meta = pd.DataFrame({
        "order_id": data[ORDER_ID_COL].values,
        "customer_id": data[CUSTOMER_ID_COL].values,
        "decision_time": data["order_date"].values,
        "shipping_mode": data["Shipping Mode"].values,
        "scheduled_days": data["Days for shipment (scheduled)"].values,
        "actual_fulfillment_days": data["Days for shipping (real)"].values,
    }).reset_index(drop=True)

    return X, y, meta, numeric, CATEGORICAL_FEATURES
