# Healthcare Provider Anomaly Detection & Billing Pattern Analysis

> Detecting fraudulent Medicare billing behavior using a dual-layer ML pipeline — unsupervised anomaly detection + supervised classification on real CMS provider data.

**Stack:** Python · Scikit-learn · XGBoost · Isolation Forest · SMOTE · Pandas · Matplotlib · Seaborn  
**Data:** CMS Medicare Physician & Other Practitioners dataset (2023) — public, real-world billing records  
**Type:** Capstone Project · DePaul University M.S. Data Science

---

## Problem

Medicare fraud costs the US billions annually. Identifying which providers are billing anomalously — versus just treating complex patients — is a needle-in-a-haystack problem at scale. This project builds a pipeline to flag high-risk providers for investigation using both statistical benchmarking and machine learning.

---

## Approach

### Layer 1 — Feature Engineering
Starting from raw CMS billing records, I engineered provider-level features:

| Feature | Description |
|---|---|
| `charge_ratio` | Submitted charge ÷ Medicare allowed amount |
| `charge_diff` | Submitted charge − allowed amount (absolute overbilling signal) |
| `total_services` | Volume of services billed per provider |
| `peer_mean_ratio` | Average charge ratio within the same provider specialty |
| `peer_z_score` | How many standard deviations a provider deviates from their specialty group |

### Layer 2 — Unsupervised Anomaly Detection

**Peer-Group Z-Score Benchmarking**  
Providers are compared within their specialty type (1,000+ types), not against the full dataset. Providers with Z-score > 3 are flagged — they deviate significantly from peers doing the same type of work.

**Isolation Forest**  
Unsupervised anomaly detection with `contamination=0.03` (assuming ~3% of providers are anomalous). Flags providers whose billing feature profile is unusual relative to the full distribution.

Both flags are combined into a single `needs_investigation` label used to train supervised models.

### Layer 3 — Supervised Classification

6 classifiers trained and compared on the labeled dataset:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **Random Forest** | **0.97** | **0.77** | **0.86** ✅ |
| **Gradient Boosting** | **0.98** | **0.75** | **0.85** |
| **Voting Ensemble (RF + GB)** | **0.99** | **0.74** | **0.85** |
| Naive Bayes | 0.55 | 0.67 | 0.60 |
| XGBoost | 0.35 | 0.95 | 0.51 |
| Logistic Regression | 0.35 | 0.89 | 0.50 |
| Decision Tree | 0.32 | 0.92 | 0.47 |

**Class imbalance handling:** SMOTE (Synthetic Minority Oversampling) and Random Oversampling were tested. Random Forest on the original imbalanced data (with appropriate class weighting) achieved the best overall F1.

**Final model selected: Random Forest** — highest F1 (0.86), strong precision-recall balance, no overfitting evidence (training and test performance consistent).

---

## Key Results

- **Random Forest**: F1 = **0.86**, Precision = 0.97, Recall = 0.77
- **Voting Ensemble (RF + GB)**: Precision = **0.99**, F1 = 0.85
- Dual-layer detection (Z-score + Isolation Forest) outperformed either method alone for generating reliable investigation labels
- Peer-group normalization was critical — comparing a cardiologist to a family physician directly would produce misleading anomaly scores

---

## Project Structure

```
healthcare-anomaly-detection/
│
├── capstone-final.ipynb        # Full pipeline notebook
├── README.md
└── data/
    └── README.md               # CMS data source info (data not included — public download link below)
```

---

## Data Source

**CMS Medicare Physician & Other Practitioners — by Provider (2023)**  
Public dataset available at: [data.cms.gov](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider)  
*Raw data not included in repo due to file size (~500MB). Download and place in `/data/` before running the notebook.*

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/karthikkonjeti/healthcare-anomaly-detection.git
cd healthcare-anomaly-detection

# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn

# Launch notebook
jupyter notebook capstone-final.ipynb
```

> Update the `file_path` variable in Cell 2 to point to your local CMS data file.

---

## Lessons Learned

- **Peer-group normalization matters more than the model choice.** A raw anomaly score without specialty-group context flags rural generalists who simply bill more than urban specialists — not actual fraud.
- **Precision vs. recall tradeoff is a business decision.** High recall (XGBoost: 0.95) catches more fraud but floods investigators with false positives. High precision (Ensemble: 0.99) sends fewer cases but nearly all are real. The right choice depends on investigator capacity.
- **SMOTE didn't help much here.** The class imbalance was modest enough that class weighting in Random Forest handled it better than synthetic oversampling.
