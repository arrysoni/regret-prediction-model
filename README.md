# 🧠 Regret Prediction Model  
### Modeling Impulse Spending & Post-Purchase Regret Using Machine Learning

> What if your past behavior could warn you before your next regretful purchase?

This project began from a personal question:

**Why do some purchases feel fine… and others feel like regret the next day?**

Instead of guessing, I built a complete end-to-end ML pipeline to model regret behavior using synthetic transaction data inspired by real impulse spending patterns.

---

## 🚀 Project Overview

This project simulates user transaction data and trains a machine learning model to predict whether a purchase will result in **post-purchase regret**.

It combines:

- Behavioral signals (intent, stress, mood)
- Temporal context (late night purchases, weekday vs weekend)
- User-relative spending patterns (deviation from historical average)
- Interaction features (impulse × stress, late-night × impulse)

### Final Model Performance


With interpretable behavioral drivers of regret.

---

## 📊 Example Feature Importance (Permutation)

Top drivers of regret:

- `is_want`
- `is_high_stress`
- `is_impulse_category`
- `hour`
- `is_low_mood`
- `amount_ratio_user_median`

These align closely with behavioral psychology research on impulse decision-making.

---

## 🏗️ Architecture

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


---

## 📁 Project Structure

regret-prediction-model/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── artifacts/
│ ├── gradient_boosting.joblib
│ ├── feature_importance.png
│ └── metrics.json
│
├── src/
│ ├── make_synthetic_data.py
│ ├── build_dataset.py
│ ├── features.py
│ ├── train.py
│ ├── predict_one.py
│ ├── config.py
│ └── utils.py
│
└── README.md


---

## 🧪 How to Run

### 1️⃣ Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pyarrow
python -m src.make_synthetic_data
python -m src.build_dataset
python -m src.train
python -m src.predict_one
```


## 🧠 Feature Engineering

### Behavioral Signals
- `is_want`
- `is_high_stress`
- `is_low_mood`
- `is_impulse_category`

### User-Relative Spending
- `amount_z_user`
- `amount_ratio_user_median`
- `log_amount`

### Temporal Context
- `hour`
- `is_late_night`
- `minutes_since_prev_txn`

### Interaction Features
- `want_x_stress`
- `late_x_impulse`

The model learns patterns such as:

- High stress + discretionary purchase → higher regret  
- Late night impulse purchases → elevated risk  
- Spending far above personal baseline → elevated risk  

---

## 🤖 Model

### Model Used
`HistGradientBoostingClassifier`

### Why Gradient Boosting?

- Captures nonlinear relationships  
- Handles tabular data extremely well  
- Learns feature interactions automatically  
- More expressive than logistic regression  

### Key Hyperparameters

```python
learning_rate = 0.06
max_depth = 6
max_iter = 500
min_samples_leaf = 25
l2_regularization = 0.5
```


## 📈 Results

- **Regret rate:** ~30%  
- **Validation ROC-AUC:** ~0.78  
- Strong behavioral signal dominance  
- Stable performance across multiple runs  

This suggests regret behavior is significantly influenced by psychological + contextual signals, not just transaction amount.

---

## 🛠️ Tech Stack

- Python 3.11  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- PyArrow  
- Joblib  

---

## 🔍 What This Demonstrates

- End-to-end ML pipeline design  
- Behavioral feature engineering  
- Time-aware data splitting  
- Gradient boosting on structured data  
- Permutation-based interpretability  
- Modular, production-style code structure  

---

## 🚀 Future Improvements

- Real bank transaction integration  
- SHAP explainability dashboard  
- Deploy as web application  
- Personalized regret risk alerts  
- Reinforcement learning spending optimization  

---

## 💡 Inspiration

This started from a simple question:

> Why do I sometimes regret purchases even when I can afford them?

Instead of guessing, I modeled it.

---

## 📬 Connect

If you're interested in:

- Behavioral ML  
- FinTech modeling  
- Personalization systems  
- Applied gradient boosting  

Feel free to connect or reach out.

