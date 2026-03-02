# Intelligent Player Churn Prediction

A Streamlit-based ML app that predicts player churn in online games using a Random Forest classifier.

## Overview

The app loads an online gaming behavior dataset, engineers a binary **Churn** label from `EngagementLevel` (Low → churn), preprocesses features, trains a model, and displays evaluation metrics and top risk factors — all through an interactive Streamlit UI.

## Project Structure

```
data/
  online_gaming_behavior_dataset.csv   # Source dataset (~13 columns)
src/
  app.py              # Streamlit entry point – orchestrates the pipeline
  data_handler.py     # Loads CSV, creates churn label, splits data
  preprocessor.py     # StandardScaler + OneHotEncoder fitting & transform
  model.py            # Trains a RandomForestClassifier
  evaluator.py        # Computes Accuracy, Precision, Recall, AUC & feature importance
```

## Features Used

| Numeric | Categorical |
|---|---|
| PlayTimeHours | GameDifficulty |
| SessionsPerWeek | Gender |
| AvgSessionDurationMinutes | |
| AchievementsUnlocked | |

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas scikit-learn

# Run the app
streamlit run src/app.py
```

## How It Works

1. **Data Loading** – Reads the CSV, derives `Churn` from `EngagementLevel`, drops IDs and nulls, and performs an 80/20 stratified train-test split.
2. **Preprocessing** – Scales numeric features with `StandardScaler` and one-hot encodes categorical features.
3. **Training** – Fits a `RandomForestClassifier` (100 estimators, seed 42).
4. **Evaluation** – Displays Accuracy, Precision, Recall, and AUC alongside a bar chart of the top 5 churn risk factors by feature importance.
