# 🛒 Instacart Product Reorder Predictor

A machine learning pipeline that predicts which products an Instacart user will reorder in their next order, using XGBoost and 19 hand-crafted features across user, product, and user-product dimensions.

---

## 📌 Problem Statement

Given a user's order history, predict whether a previously purchased product will be reordered in their next order. This is a **binary classification** task (`target = 1` if reordered, `0` otherwise).

---

## 📂 Project Structure

```
instacart-reorder-predictor/
├── notebooks/
│   └── instacart_final.ipynb     # Full EDA + feature engineering + model training
├── src/
│   └── pipeline.py               # Modular train/inference pipeline
├── data/                         # Place Kaggle CSVs here (see below)
├── models/                       # Saved model artifacts (auto-generated)
├── outputs/                      # Predictions & recommendations (auto-generated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

This project uses the **Instacart Market Basket Analysis** dataset from Kaggle.

🔗 **[Download the dataset here](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis)**

After downloading, place the following CSV files inside the `data/` folder:

| File | Description |
|---|---|
| `orders.csv` | Order metadata (user_id, eval_set, day/time info) |
| `products.csv` | Product names and department/aisle IDs |
| `order_products__prior.csv` | Products in all prior orders |
| `order_products__train.csv` | Products in the training (last) order |

> ⚠️ The dataset files are **not included** in this repo due to their size. You must download them separately from Kaggle.

---

## 🧠 Feature Engineering

19 features across 3 groups:

### User–Product Features
| Feature | Description |
|---|---|
| `up_times_bought` | How many times user bought this product |
| `up_last_order` | Order number when last purchased |
| `up_first_order` | Order number when first purchased |
| `up_avg_cart_pos` | Average add-to-cart position |
| `up_purchase_rate` | Purchases / total orders (loyalty rate) |
| `up_orders_since_last` | Orders elapsed since last purchase (recency) |
| `up_reorder_density` | Purchase frequency between first and last order |

### User-Level Features
| Feature | Description |
|---|---|
| `u_total_orders` | Total number of prior orders |
| `u_total_items` | Total items bought across all orders |
| `u_avg_days_between_orders` | Average order frequency |
| `u_avg_basket_size` | Average items per order |
| `u_unique_products` | Unique products ever purchased |
| `u_reorder_ratio` | Proportion of reordered items overall |
| `u_diversity` | Unique products / total items (shopping diversity) |

### Product-Level Features
| Feature | Description |
|---|---|
| `p_purchase_count` | Total purchases of this product |
| `p_reorder_rate` | Fraction of orders that are reorders |
| `p_unique_buyers` | Number of unique users who bought it |
| `p_avg_cart_pos` | Average add-to-cart position across all users |
| `p_loyalty` | Average purchases per unique buyer |

---

## 🤖 Model

- **Algorithm:** XGBoost (`XGBClassifier`)
- **Class imbalance:** handled via `scale_pos_weight`
- **Threshold tuning:** best F1 threshold found via grid search over `[0.10, 0.90]`
- **Preprocessing pipeline:** `SimpleImputer(strategy="median")` via sklearn `ColumnTransformer`

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/instacart-reorder-predictor.git
cd instacart-reorder-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the data
Download the Kaggle dataset (link above) and place all 4 CSVs in the `data/` folder.

### 4. Run training
```bash
cd data
python ../src/pipeline.py
```

> On first run, the model trains and saves artifacts to `models/`. On subsequent runs, it loads the saved model and runs inference.

### 5. Explore the notebook
Open `notebooks/instacart_final.ipynb` for full step-by-step EDA, feature engineering, and model evaluation.

---

## 📁 Output Files

| File | Description |
|---|---|
| `xgb_reorder_model.pkl` | Trained XGBoost model |
| `reorder_pipeline.pkl` | Fitted sklearn preprocessing pipeline |
| `best_threshold.pkl` | Optimal classification threshold |
| `feature_columns.pkl` | List of feature column names |
| `recommendations.csv` | Top reorder predictions per user with product names |
| `feature_importance.csv` | XGBoost feature importances |
| `output.csv` | Raw inference predictions |

---

## 🛠️ Requirements

See `requirements.txt`. Key dependencies:

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

---------------------------------------------------------------