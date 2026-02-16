🧠 Regret Prediction Model
Modeling Impulse Spending & Financial Self-Control Using Machine Learning

What if your past behavior could warn you before your next regretful purchase?

This project began from a personal question:
Why do some purchases feel fine… and others feel like regret the next day?

Instead of guessing, I built a full ML pipeline to model behavioral regret patterns using synthetic transaction data inspired by real-world impulse spending behavior.

🚀 Project Overview

This project simulates user transaction data and trains a machine learning model to predict whether a purchase will result in post-purchase regret.

It combines:

Behavioral signals (intent, stress, mood)

Temporal context (late night purchases, weekday vs weekend)

User-relative spending patterns (deviation from historical average)

Interaction features (impulse × stress, late-night × impulse)

The final model achieves:

Validation ROC-AUC: 0.781


With clear, interpretable drivers of regret.

📊 Example Output

Top Feature Importances (Permutation Importance):

is_want

is_high_stress

is_impulse_category

hour

is_low_mood

amount_ratio_user_median

These align strongly with behavioral psychology research on impulsive decision-making.

🏗️ Architecture

The project follows a clean ML pipeline:

Synthetic Data Generation
        ↓
Feature Engineering
        ↓
Time-Aware Train/Validation Split
        ↓
Gradient Boosting Model
        ↓
Permutation Feature Importance
        ↓
Model Artifacts + Metrics

📁 Project Structure
regret-prediction-model/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── gradient_boosting.joblib
│   ├── feature_importance.png
│   └── metrics.json
│
├── src/
│   ├── make_synthetic_data.py
│   ├── build_dataset.py
│   ├── features.py
│   ├── train.py
│   ├── predict_one.py
│   ├── config.py
│   └── utils.py
│
└── README.md

🧪 How to Run
1️⃣ Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyarrow

2️⃣ Generate Synthetic Transaction Data
python -m src.make_synthetic_data


Creates:

transactions.csv

labels.csv

3️⃣ Build Engineered Feature Dataset
python -m src.build_dataset


Creates:

feature_dataset.parquet

Includes:

Temporal features

User deviation metrics

Behavioral flags

Interaction features

4️⃣ Train Model
python -m src.train


Outputs:

Validation ROC-AUC

Permutation Feature Importance

Saved model artifact

5️⃣ Predict on Single Example
python -m src.predict_one


Produces:

Regret probability

Predicted label

Top contributing features

Plain-English explanation

🧠 Feature Engineering

The most impactful features include:

Behavioral Signals

is_want

is_high_stress

is_low_mood

is_impulse_category

User-Relative Spending

amount_z_user

amount_ratio_user_median

log_amount

Temporal Context

hour

is_late_night

minutes_since_prev_txn

Interaction Features

want_x_stress

late_x_impulse

The model learns patterns such as:

High stress + discretionary purchase → higher regret

Late night impulse buys → elevated risk

Spending far above personal baseline → elevated risk

🤖 Model

Model used:

HistGradientBoostingClassifier


Why?

Handles nonlinear relationships

Captures feature interactions

Robust to scaling issues

Strong tabular performance

Key hyperparameters:

learning_rate = 0.06
max_depth = 6
max_iter = 500
min_samples_leaf = 25
l2_regularization = 0.5

📈 Results

Regret rate: ~30%

Validation ROC-AUC: 0.78

Strong behavioral feature dominance

Stable performance across multiple runs

This suggests regret behavior is significantly driven by psychological + contextual signals, not just transaction amount.

🔍 Why This Project Matters

This project demonstrates:

Behavioral modeling

Time-aware train/validation splitting

Robust feature engineering

Gradient boosting on structured data

Permutation-based interpretability

Clean modular ML architecture

It blends:

Personal finance insight

Behavioral economics

Applied machine learning

Interpretable AI

🛠️ Tech Stack

Python 3.11

Pandas

NumPy

Scikit-learn

Matplotlib

PyArrow

Joblib

🚀 Future Improvements

Real-world bank transaction integration

SHAP explainability dashboard

Deployment as a web app

Personalized regret risk notifications

Reinforcement learning for spending behavior optimization

💡 Inspiration

This started from a simple, honest question:

Why do I sometimes regret purchases even when I can afford them?

Instead of guessing, I modeled it.

📬 Connect

If you're interested in:

Behavioral ML

FinTech modeling

Personalization systems

Applied gradient boosting

Feel free to connect or reach out.
