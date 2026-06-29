import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

    


# ----------------------------- FILE PATHS ---------------------------------



MODEL_FILE = "xgb_reorder_model.pkl"
PIPELINE_FILE = "reorder_pipeline.pkl"
THRESHOLD_FILE = "best_threshold.pkl"
FEATURES_FILE = "feature_columns.pkl"

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"
RECOMMENDATIONS_FILE = "recommendations.csv"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"

RANDOM_STATE = 42
SAMPLE_SIZE = 20_000
CHUNK_SIZE = 20_000

ID_COLS = ["user_id", "product_id"]

FEATURE_COLS = [
    "up_times_bought",
    "up_last_order",
    "up_first_order",
    "up_avg_cart_pos",

    "u_total_orders",
    "u_total_items",
    "u_avg_day_between_orders",
    "u_avg_basket_size",
    "u_unique_products",
    "u_reorder_ratio",

    "p_purchase_count",
    "p_reorder_rate",
    "p_unique_product",
    "p_avg_cart_pos",

    "up_purchase_rate",
    "up_orders_since_last",
    "up_reorder_density",
    "unique_deversity",
    "loyalty",
]


# ----------------------------- PIPELINE ------------------------------------

def build_pipeline(num_attribs):
    """
    XGBoost does not need scaling.
    So here pipeline only handles missing values.
    """

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    full_pipeline = ColumnTransformer([
        ("nums", num_pipeline, num_attribs)
    ])

    return full_pipeline


# ----------------------------- LOAD DATA -----------------------------------



def load_data():
    orders = pd.read_csv("orders.csv")
    products = pd.read_csv("products.csv")
    order_products__prior = pd.read_csv("order_products__prior.csv")
    order_products__train = pd.read_csv("order_products__train.csv")


    print("orders:", orders.shape)
    print("products:", products.shape)
    print("order_products__prior:", order_products__prior.shape)
    print("order_products__train:", order_products__train.shape)

    return orders, products, order_products__prior, order_products__train


# ----------------------------- FEATURE ENGINEERING --------------------------

def build_features( orders,order_products__prior,order_products__train=None,training=True):
    """
    Creates final feature dataframe.
    If training=True, target column is also created.
    """

    prior_order = orders[orders["eval_set"] == "prior"].copy()

    # take only limited users to avoid memory error
    sample_users = prior_order["user_id"].drop_duplicates().sample(
    n=5000,
    random_state=42
)

    prior_order = prior_order[prior_order["user_id"].isin(sample_users)]

    order_products__prior = order_products__prior[
    order_products__prior["order_id"].isin(prior_order["order_id"])
    ]

    prior_data = prior_order.merge(
    order_products__prior,
    on="order_id",
    how="inner"
    )

    # ---------------- user-product features ----------------

    up_count = (
        prior_data.groupby(["user_id", "product_id"])
        .size()
        .reset_index(name="up_times_bought")
    )

    up_last = (
        prior_data.groupby(["user_id", "product_id"])["order_number"]
        .max()
        .reset_index(name="up_last_order")
    )

    up_first = (
        prior_data.groupby(["user_id", "product_id"])["order_number"]
        .min()
        .reset_index(name="up_first_order")
    )

    up_cart_pos = (
        prior_data.groupby(["user_id", "product_id"])["add_to_cart_order"]
        .mean()
        .reset_index(name="up_avg_cart_pos")
    )

    # ---------------- user features ----------------

    u_total_orders = (
        prior_order.groupby("user_id")["order_number"]
        .max()
        .reset_index(name="u_total_orders")
    )

    u_total_items = (
        prior_data.groupby("user_id")["product_id"]
        .size()
        .reset_index(name="u_total_items")
    )

    u_avg_day_between_orders = (
        prior_data.groupby("user_id")["days_since_prior_order"]
        .mean()
        .reset_index(name="u_avg_day_between_orders")
    )

    u_basket_size = (
        prior_data.groupby(["user_id", "order_number"])["product_id"]
        .count()
        .reset_index(name="u_basket_size")
    )

    u_avg_basket_size = (
        u_basket_size.groupby("user_id")["u_basket_size"]
        .mean()
        .reset_index(name="u_avg_basket_size")
    )

    u_unique_products = (
        prior_data.groupby("user_id")["product_id"]
        .nunique()
        .reset_index(name="u_unique_products")
    )

    u_reorder_ratio = (
        prior_data.groupby("user_id")["reordered"]
        .mean()
        .reset_index(name="u_reorder_ratio")
    )

    # ---------------- product features ----------------

    p_purchase_count = (
        prior_data.groupby("product_id")
        .size()
        .reset_index(name="p_purchase_count")
    )

    p_reorder_rate = (
        prior_data.groupby("product_id")["reordered"]
        .mean()
        .reset_index()
        .rename(columns={"reordered": "p_reorder_rate"})
    )

    p_unique_product = (
        prior_data.groupby("product_id")["user_id"]
        .nunique()
        .reset_index(name="p_unique_product")
    )

    p_avg_cart_pos = (
        prior_data.groupby("product_id")["add_to_cart_order"]
        .mean()
        .reset_index(name="p_avg_cart_pos")
    )

    # ---------------- merge features ----------------

    features = (
        up_count
        .merge(up_last, on=["user_id", "product_id"], how="left")
        .merge(up_first, on=["user_id", "product_id"], how="left")
        .merge(up_cart_pos, on=["user_id", "product_id"], how="left")

        .merge(u_total_orders, on="user_id", how="left")
        .merge(u_total_items, on="user_id", how="left")
        .merge(u_avg_day_between_orders, on="user_id", how="left")
        .merge(u_avg_basket_size, on="user_id", how="left")
        .merge(u_unique_products, on="user_id", how="left")
        .merge(u_reorder_ratio, on="user_id", how="left")

        .merge(p_purchase_count, on="product_id", how="left")
        .merge(p_reorder_rate, on="product_id", how="left")
        .merge(p_unique_product, on="product_id", how="left")
        .merge(p_avg_cart_pos, on="product_id", how="left")
    )

    # ---------------- derived features ----------------

    features["up_purchase_rate"] = (
        features["up_times_bought"] /
        features["u_total_orders"].replace(0, np.nan)
    )

    features["up_orders_since_last"] = (
        features["u_total_orders"] -
        features["up_last_order"]
    )

    features["up_reorder_density"] = (
        features["up_times_bought"] /
        (
            features["up_last_order"] -
            features["up_first_order"] +
            1
        ).replace(0, np.nan)
    )

    features["unique_deversity"] = (
        features["u_unique_products"] /
        features["u_total_items"].replace(0, np.nan)
    )

    features["loyalty"] = (
        features["p_purchase_count"] /
        features["u_unique_products"].replace(0, np.nan)
    )

    features["u_avg_day_between_orders"] = (
        features["u_avg_day_between_orders"].fillna(0)
    )

    if not training:
        return features

    if order_products__train is None:
        raise ValueError("order_products__train is required for training")

    # ---------------- target creation ----------------

    train_orders = orders[orders["eval_set"] == "train"].copy()

    train_data = train_orders.merge(
    order_products__train,
    on="order_id",
    how="left"
)

    train_labels = train_data[["user_id", "product_id"]].dropna().copy()
    train_labels["product_id"] = train_labels["product_id"].astype(features["product_id"].dtype)
    train_labels["target"] = 1

    # Only train users have real labels.
    train_user_ids = train_orders["user_id"].unique()

    features_train = features[
        features["user_id"].isin(train_user_ids)
    ].copy()

    final_df = features_train.merge(
        train_labels,
        on=["user_id", "product_id"],
        how="left"
    )

    final_df["target"] = final_df["target"].fillna(0).astype(int)

    print("final_df:", final_df.shape)
    print("target distribution:")
    print(final_df["target"].value_counts(normalize=True).round(4))

    return final_df


# ----------------------------- HELPERS -------------------------------------

def make_sample(final_df, sample_size=SAMPLE_SIZE):
    """
    Takes stratified sample.
    This keeps same 0/1 target ratio.
    """

    if sample_size is None or sample_size >= len(final_df):
        return final_df.copy()

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=sample_size,
        random_state=RANDOM_STATE
    )

    for _, sample_idx in split.split(final_df, final_df["target"]):
        sampled_df = final_df.iloc[sample_idx].copy()
        return sampled_df

    return final_df.copy()


def find_best_threshold(probs, y_true):
    best_t = 0.5
    best_score = 0

    for t in np.arange(0.10, 0.90, 0.02):
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)

        if score > best_score:
            best_score = score
            best_t = t

    return round(float(best_t), 2), round(float(best_score), 4)


def evaluate_model(model, pipeline, X_test, y_test, threshold=0.5):
    X_test_prepared = pipeline.transform(X_test)

    probs = model.predict_proba(X_test_prepared)[:, 1]
    preds = (probs >= threshold).astype(int)

    print()
    print("=" * 60)
    print(f"MODEL: XGBoost | threshold={threshold}")
    print("=" * 60)

    print(classification_report(y_test, preds, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))

    return probs


def save_feature_importance(model, feature_cols):
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(FEATURE_IMPORTANCE_FILE, index=False)

    print(f"Feature importance saved to {FEATURE_IMPORTANCE_FILE}")


# ----------------------------- PREDICTION ----------------------------------

def predict_dataframe(data, model, pipeline, feature_cols, threshold):
    X = data[feature_cols].copy()

    X_prepared = pipeline.transform(X)

    probs = model.predict_proba(X_prepared)[:, 1]

    output = data[ID_COLS].copy()
    output["reorder_probability"] = probs
    output["predicted_reorder"] = (
        output["reorder_probability"] >= threshold
    ).astype(int)

    return output


def create_recommendations(data, model, pipeline, feature_cols, threshold, products):
    outputs = []

    for start in range(0, len(data), CHUNK_SIZE):
        end = start + CHUNK_SIZE

        chunk = data.iloc[start:end].copy()

        preds = predict_dataframe(
            chunk,
            model,
            pipeline,
            feature_cols,
            threshold
        )

        preds = preds[preds["reorder_probability"] >= threshold]

        outputs.append(preds)

        print(f"Predicted rows {start} to {min(end, len(data))}")

    if outputs:
        recommendations = pd.concat(outputs, ignore_index=True)
    else:
        recommendations = pd.DataFrame(
            columns=ID_COLS + ["reorder_probability", "predicted_reorder"]
        )

    recommendations = recommendations.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="left"
    )

    recommendations = recommendations.sort_values(
        ["user_id", "reorder_probability"],
        ascending=[True, False]
    )

    return recommendations


# ----------------------------- TRAIN ---------------------------------------

def train():
    orders, products, order_products__prior, order_products__train = load_data()

    final_df = build_features(
    orders=orders,
    order_products__prior=order_products__prior,
    order_products__train=order_products__train,
    training=True
    )

    sampled_df = make_sample(final_df, SAMPLE_SIZE)

    X = sampled_df[FEATURE_COLS].copy()
    y = sampled_df["target"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = build_pipeline(FEATURE_COLS)

    X_train_prepared = pipeline.fit_transform(X_train)

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())

    scale_w = neg_count / pos_count if pos_count > 0 else 1

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_w,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss"
    )

    model.fit(X_train_prepared, y_train)

    probs_05 = evaluate_model(
        model,
        pipeline,
        X_test,
        y_test,
        threshold=0.5
    )

    best_t, best_f1 = find_best_threshold(probs_05, y_test)

    print()
    print(f"Best threshold: {best_t} -> F1 = {best_f1}")

    evaluate_model(
        model,
        pipeline,
        X_test,
        y_test,
        threshold=best_t
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    joblib.dump(best_t, THRESHOLD_FILE)
    joblib.dump(FEATURE_COLS, FEATURES_FILE)

    save_feature_importance(model, FEATURE_COLS)

    # Save sample input for later inference.
    input_data = sampled_df.loc[X_test.index, ID_COLS + FEATURE_COLS].copy()
    input_data.to_csv(INPUT_FILE, index=False)

    # Create final product recommendations.
    recommendations = create_recommendations(
        data=final_df[ID_COLS + FEATURE_COLS].copy(),
        model=model,
        pipeline=pipeline,
        feature_cols=FEATURE_COLS,
        threshold=best_t,
        products=products
    )

    recommendations.to_csv(RECOMMENDATIONS_FILE, index=False)

    print()
    print("Model trained successfully")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved pipeline: {PIPELINE_FILE}")
    print(f"Saved input file: {INPUT_FILE}")
    print(f"Saved recommendations: {RECOMMENDATIONS_FILE}")


# ----------------------------- INFERENCE -----------------------------------

def inference():
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    threshold = joblib.load(THRESHOLD_FILE)
    feature_cols = joblib.load(FEATURES_FILE)

    input_data = pd.read_csv(INPUT_FILE)

    predictions = predict_dataframe(
        data=input_data,
        model=model,
        pipeline=pipeline,
        feature_cols=feature_cols,
        threshold=threshold
    )

    
    products = pd.read_csv("products.csv")

    predictions = predictions.merge(
            products[["product_id", "product_name"]],
            on="product_id",
            how="left"
        )

    predictions = predictions.sort_values(
        ["user_id", "reorder_probability"],
        ascending=[True, False]
    )

    predictions.to_csv(OUTPUT_FILE, index=False)

    print("Inference completed")
    print(f"Saved output: {OUTPUT_FILE}")


# ----------------------------- MAIN ----------------------------------------

if __name__ == "__main__":

    if not os.path.exists(MODEL_FILE):
        train()
    else:
        inference()