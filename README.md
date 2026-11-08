# FIFA 22 Player Value Prediction & Player Clustering

A machine learning project built on the FIFA 22 player dataset. It does two things:

1. **Regression**: compares 6 models that predict a player's market value (`value_eur`) from their in-game attributes.
2. **Clustering**: groups players into 4 tiers (regular, mid-level, strong, star) with K-Means using rating and market value.

The whole analysis lives in one Jupyter notebook: `FIFA_22.ipynb`.

---

## Highlights

- Cleaned and prepared a dataset of **19,239 players with 110 columns**
- Compared **6 regression models** on the same train/test split
- **Gradient Boosting reached R² = 97.7%** on unseen data
- Built **4 player tiers** with K-Means and used the model to classify a new, hypothetical player
- Visualized relationships between rating, value, age, potential and wage

---

## Dataset

- **File**: `players_22.csv` (FIFA 22 complete player dataset, sourced from sofifa.com)
- **Size**: 19,239 rows x 110 columns (after dropping rows with no market value: 19,165 rows)
- **Target**: `value_eur` (market value in euros)

The dataset is **not included** in this repository. Download `players_22.csv` (for example from the "FIFA 22 complete player dataset" on Kaggle) and place it in the project folder next to the notebook.

---

## Methodology

### 1. Data preparation
- Removed rows where `value_eur` is missing
- Kept only numeric columns
- Dropped identifier-like columns (`sofifa_id`, `club_team_id`, `nationality_id`, `club_jersey_number`, `nation_team_id`, `nation_jersey_number`) and `release_clause_eur`
- Filled remaining missing values with each column's median
- Final modeling table: **19,165 rows x 53 numeric columns** (52 features + target)

### 2. Train/test split
- 70% training (13,415 players) / 30% testing (5,750 players)
- `random_state=42` for reproducibility

### 3. Model comparison
Six regressors were trained with default settings and scored with R² on the test set:

| Model | R² score (%) |
|-------|-------------|
| **Gradient Boosting** | **97.7** |
| Random Forest | 96.5 |
| Decision Tree | 93.3 |
| Ridge Regression | 70.8 |
| KNN | 65.6 |
| SVR | -5.4 |

Tree-based ensembles perform best. SVR scores below zero, meaning it does worse than simply predicting the average value. That is expected here because SVR (and KNN) are sensitive to feature scale and no scaling was applied in this comparison.

### 4. Exploratory visualizations
- Overall rating vs. market value
- Age vs. potential
- Weekly wage vs. market value
- Bar chart of the model comparison

### 5. Player clustering (K-Means)
- Features: `overall` and `value_eur`, standardized with `StandardScaler`
- 4 clusters, `random_state=42`
- Clusters are ranked by average rating and named:
  1. Regular players
  2. Mid-level players
  3. Strong players
  4. Star players (TOP)
- A custom example player (overall 100, value 100M EUR) is placed into the **Star players (TOP)** group

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| pandas | Data loading and cleaning |
| matplotlib, seaborn | Visualization |
| scikit-learn | Models, train/test split, scaling, K-Means |
| Jupyter | Interactive notebook |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add players_22.csv to the project folder, then start Jupyter
jupyter notebook FIFA_22.ipynb
```

Run the cells from top to bottom.

---

## Project Structure

```
.
├── FIFA_22.ipynb        # Full analysis
├── players_22.csv       # Dataset (download separately)
├── requirements.txt
└── README.md
```

---

## Possible Improvements

- Scale features before KNN and SVR for a fairer comparison
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) and cross-validation
- Feature importance analysis for the best model
- Predict `log(value_eur)` to handle the strong skew in player values
- Choose the number of K-Means clusters with the elbow method or silhouette score
- Add more clustering features (age, potential, position)

---

## Acknowledgements

- Player data originates from [sofifa.com](https://sofifa.com) (FIFA 22)
- Built with [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/) and [seaborn](https://seaborn.pydata.org/)

---

## License

This project is licensed under the MIT License. The dataset belongs to its original owners and is subject to their terms.
