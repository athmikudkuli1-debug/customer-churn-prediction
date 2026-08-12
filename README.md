# Customer Churn Prediction — Beginner Walkthrough

**Goal:** predict whether a customer will leave ("churn") a service, based
on their account history and behavior — and figure out *what* drives churn
so the business can act on it.

This is a **classification** problem: the output is one of two categories,
`Churn` or `No Churn`. That's different from *regression*, where you'd
predict a number (like revenue).

---

## Files in this project

| File | What it does |
|---|---|
| `1_generate_data.py` | Creates a realistic sample dataset (`customer_churn.csv`) with 2,000 customers |
| `2_churn_prediction.py` | The full pipeline: explore data → preprocess → train models → evaluate |
| `customer_churn.csv` | The dataset itself |
| `outputs/` | All charts and the model comparison table |

> **Note on the data:** I generated a *synthetic* dataset that mimics a real
> telecom churn dataset (same kind of columns: contract type, tenure,
> charges, support services, etc.), with realistic churn patterns built in.
> This means you can run the whole pipeline immediately, see real results,
> and later swap in your own company's actual CSV with the same column
> structure — the code doesn't change.

---

## The pipeline, step by step

### 1. Load the data
Just reading the CSV into a pandas DataFrame — the standard first step
for basically any data science project.

### 2. Exploratory Data Analysis (EDA)
**Before touching any model**, look at the data. This step answers:
- What fraction of customers actually churn? (Here: ~29%)
- Which features *look* related to churn? (Contract type, tenure, monthly charges)
- Are there missing values or weird outliers to clean up?

Skipping this is the most common beginner mistake — you can't trust a
model's output if you don't understand the data going in.

See `outputs/1_eda_overview.png`. Key takeaway from this dataset:
month-to-month customers churn far more than customers on 1–2 year
contracts, and short-tenure customers are the highest-risk group.

### 3. Preprocessing
Models only understand numbers. Two key transformations:
- **One-hot encoding**: categorical columns like `Contract`
  ("Month-to-month"/"One year"/"Two year") become separate 0/1 columns.
- **Scaling**: numeric columns like `tenure` (0–72) and `MonthlyCharges`
  (18–120) get rescaled to comparable ranges. This matters for Logistic
  Regression specifically — it's sensitive to feature scale, tree-based
  models like Random Forest aren't.

We also split data into a **training set** (75%) and a **test set** (25%).
The model only ever learns from the training set; the test set simulates
"new customers it's never seen," which is how we honestly measure
real-world performance.

### 4. Train two models
- **Logistic Regression**: a simple, interpretable baseline. Despite the
  name, it's a classification model, not regression — it estimates the
  *probability* a customer churns.
- **Random Forest**: an ensemble of many decision trees voting together.
  Usually captures more complex patterns than logistic regression, and
  conveniently gives us feature importance rankings for free.

### 5. Handling class imbalance (an important lesson)
Only ~29% of customers churn. A lazy model could just predict "No churn"
for everyone and still be ~71% accurate — while catching **zero** actual
churners. That's useless, since the entire point is to flag people about
to leave.

Fix: `class_weight="balanced"` tells both models to pay more attention to
the minority (churn) class. Look at the difference this made in
`outputs/model_comparison.csv` — recall (the % of actual churners the
model correctly catches) jumped substantially once we did this.

### 6. Evaluation metrics — what they mean
| Metric | What it tells you |
|---|---|
| **Accuracy** | % of all predictions that were correct. Misleading alone with imbalanced classes. |
| **Precision** | Of customers we *predicted* would churn, what % actually did? (Avoids wasting retention offers on people who weren't leaving.) |
| **Recall** | Of customers who *actually* churned, what % did we catch? (Usually the most important metric for churn — missing a real churner is costly.) |
| **F1 score** | Balance between precision and recall. |
| **ROC-AUC** | How well the model ranks churners above non-churners overall, across all thresholds. |

See `outputs/2_confusion_matrices.png` for a visual breakdown of correct
vs. incorrect predictions, and `outputs/3_roc_curves.png` for the ROC
comparison.

### 7. Feature importance
`outputs/4_feature_importance.png` shows which features the Random Forest
relied on most. In this dataset: total charges, tenure, monthly charges,
and contract type dominate — which lines up with the EDA findings. This
is often the most business-actionable part of the whole project: it tells
a company *where to intervene* (e.g., "target month-to-month customers in
their first 6 months with a discount to upgrade to annual contracts").

---

## How to run it

```bash
python3 1_generate_data.py
python3 2_churn_prediction.py
```

All charts and the metrics table land in `outputs/`.

---

## Natural next steps to extend this project

1. **Try more models**: Gradient Boosting (XGBoost/LightGBM) usually beats
   Random Forest on tabular data like this.
2. **Hyperparameter tuning**: use `GridSearchCV` to find the best settings
   for Random Forest instead of guessing `max_depth=6`.
3. **Threshold tuning**: instead of the default 0.5 cutoff for "churn or
   not," pick a threshold that matches business cost (e.g. missing a
   churner costs more than a false alarm, so lower the threshold).
4. **Cross-validation**: instead of one train/test split, use k-fold CV
   for a more robust performance estimate.
5. **Real data**: swap `customer_churn.csv` for a real dataset (e.g. the
   IBM Telco Customer Churn dataset on Kaggle) — the pipeline code needs
   almost no changes since the column structure matches.
