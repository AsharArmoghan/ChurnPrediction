# Churn Prediction System

A production-ready customer churn prediction system using a simplified Random Forest classifier. Predicts churn probability and risk levels via FastAPI backend with a Tailwind + Chart.js dashboard.

## Key Features

- **Single Model Architecture**: Random Forest classifier (100 trees, balanced class weights)
- **High Performance**: 94.9% ROC-AUC, 92.1% accuracy, 96.6% recall
- **Real-time API**: FastAPI with 3 endpoints for predictions and dashboard data
- **Interactive Dashboard**: HTML/Chart.js frontend showing portfolio risk distribution
- **Kaggle Data Source**: Uses `muhammadshahidazeem/customer-churn-dataset` (505K records)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML5, Tailwind CSS, Chart.js (CDN) |
| Backend | FastAPI (Python 3.10+), Uvicorn |
| ML Model | scikit-learn RandomForestClassifier |
| Data Source | Kaggle via `kagglehub` |
| Database | SQLite via SQLAlchemy ORM |
| Preprocessing | StandardScaler, LabelEncoder, SimpleImputer |

---

## Project Structure

```
churnForecasting/
├── requirements.txt          # Dependency list
├── .gitignore                # Excludes artifacts, caches, DB, logs
│
├── advanced_pipeline/        # Minimal data utilities
│   ├── __init__.py           # Exports load_kaggle_full, get_logger
│   ├── kaggle_data.py        # Kaggle dataset loader (churn-only)
│   └── utils.py              # Logging utilities
│
├── simple_churn_model.py     # Standalone Random Forest model
├── models.py                 # SQLAlchemy ORM: Customer table
├── database.py               # Engine / Session / Base / init_db / get_db
├── seed.py                   # Seed 100 sample customers into SQLite
├── main.py                   # FastAPI app + REST endpoints
├── index.html                # Tailwind + Chart.js dashboard
│
├── notebooks/                # Jupyter notebooks
│   ├── 00_exploratory_data_analysis.ipynb  # EDA on Kaggle data
│   └── 01_modeling.ipynb                  # Model training & evaluation
│
├── models/                   # Trained model artifacts
│   └── simple_churn_model.pkl  # Random Forest (14.6 MB)
│
├── database.db               # SQLite file (created at runtime)

```

---

## Setup & Run

> Requires **Python 3.10+** (developed/tested on 3.13).

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (optional - pre-trained model included)

```bash
python simple_churn_model.py
```

This trains a Random Forest on 505K Kaggle records (~2-3 minutes) and saves to `models/simple_churn_model.pkl`.

### 3. Seed the database

```bash
python seed.py
```
Creates `database.db` with 100 sample customers for API testing.

### 4. Start the API

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`

### 5. Launch the dashboard

Open `index.html` directly in a browser. It calls the API at `http://127.0.0.1:8000`.

---

## Data Source

The project uses the Kaggle dataset **`muhammadshahidazeem/customer-churn-dataset`** exclusively.

```python
from advanced_pipeline import load_kaggle_full, load_kaggle_split

# Full dataset (505K records, 11 columns)
df = load_kaggle_full(random_state=42)

# Train/test split (stratified)
train_df, test_df = load_kaggle_split(test_size=0.2, random_state=42)
```

**Schema (11 columns):**
- **ID**: `customer_id`
- **Numeric (7)**: `tenure_months`, `monthly_charges`, `total_charges`, `support_tickets_last_3mo`, `avg_session_duration_min`, `login_frequency_weekly`, `nps_score`
- **Categorical (2)**: `contract_type`, `payment_method`
- **Target**: `churn` (0=retained, 1=churned)

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check & service info |
| GET | `/api/customers` | List all customers |
| GET | `/api/predict/churn/{id}` | Churn probability + risk level |
| GET | `/api/dashboard-summary` | Portfolio risk metrics |

### Example Responses

**GET `/api/predict/churn/CUST0001`**
```json
{
  "customer_id": "CUST0001",
  "churn_probability": 22.35,
  "risk_level": "Low"
}
```

**GET `/api/dashboard-summary`**
```json
{
  "total_customers": 100,
  "high_risk_churn_count": 15,
  "medium_risk_churn_count": 35,
  "low_risk_churn_count": 50,
  "average_churn_risk_pct": 32.45
}
```

**Risk Levels:**
- **High**: ≥70% churn probability
- **Medium**: 40-69%
- **Low**: <40%

---

## Model Details

### Architecture
- **Algorithm**: RandomForestClassifier
- **Trees**: 100
- **Max Depth**: 10
- **Min Samples Split**: 10
- **Min Samples Leaf**: 5
- **Class Weights**: `balanced` (handles slight class imbalance)

### Performance (on 101K test records)

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.9490** |
| Accuracy | 0.9208 |
| Precision | 0.8989 |
| Recall | **0.9659** |
| F1 Score | 0.9312 |

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | support_tickets_last_3mo | 0.3556 |
| 2 | total_charges | 0.2455 |
| 3 | monthly_charges | 0.1863 |
| 4 | nps_score | 0.1102 |
| 5 | contract_type | 0.0799 |

---

## Usage Examples

### Python API
```python
from simple_churn_model import SimpleChurnPredictor

# Load trained model
predictor = SimpleChurnPredictor.load_model('models/simple_churn_model.pkl')

# Predict on new data
import pandas as pd
new_customer = pd.DataFrame([{
    'tenure_months': 12,
    'monthly_charges': 70.0,
    'total_charges': 840.0,
    'support_tickets_last_3mo': 2,
    'avg_session_duration_min': 15.0,
    'login_frequency_weekly': 5,
    'nps_score': 8,
    'contract_type': 'one-year',
    'payment_method': 'credit_card'
}])

prob = predictor.predict_proba(new_customer)[0]
print(f"Churn probability: {prob:.1%}")
```

### cURL
```bash
# Get all customers
curl http://127.0.0.1:8000/api/customers

# Predict churn
curl http://127.0.0.1:8000/api/predict/churn/CUST0001

# Dashboard summary
curl http://127.0.0.1:8000/api/dashboard-summary
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `00_exploratory_data_analysis.ipynb` | Full EDA: distributions, correlations, churn analysis |
| `01_modeling.ipynb` | Training, evaluation, feature importance, ROC curves |

Run with:
```bash
pip install jupyter nbformat nbconvert ipykernel
jupyter notebook notebooks/
```

---

## Architecture Notes

### Class Imbalance
Dataset is naturally balanced (55.5% churn rate, ratio 1.25). Model uses `class_weight='balanced'` in RandomForestClassifier which automatically adjusts weights:
- Class 0 (retained): weight ≈ 1.12
- Class 1 (churned): weight ≈ 0.90

SMOTE is **not required** but available in `imbalanced-learn` if needed.

---

## Dependencies

Key packages (see `requirements.txt`):
- `scikit-learn>=1.9.0`
- `pandas>=3.0.0`
- `numpy>=2.5.0`
- `fastapi>=0.141.0`, `uvicorn>=0.52.0`
- `sqlalchemy>=2.0.0`
- `kagglehub>=1.0.0`
- `imbalanced-learn>=0.14.0`

Optional (for notebooks): `jupyter`, `nbformat`, `nbconvert`, `ipykernel`

---

## License

My License - Feel free to use and modify.