# Intelligent Player Churn Prediction

A machine learning web application that predicts player churn in online gaming environments. Built with Streamlit, it allows game analysts and product teams to assess individual player risk in real time and review overall model performance through an interactive dashboard.

---

## Live Demo

Hosted Application: [Demo](https://churnprediction-production.up.railway.app)

Video Walkthrough: [Video](https://genai-demo.dotenv.live/)

---

## Overview

Player churn — the likelihood that a user will stop engaging with a game — is a critical metric for online gaming platforms. This application addresses that problem by training a Random Forest classifier on behavioral and demographic player data. It derives a binary churn label from engagement level, engineers interaction features to improve signal quality, and exposes the model through a two-tab Streamlit interface: one tab for predicting individual player risk and another for inspecting model telemetry.

---

## Project Structure

```
Churn_Prediction/
    data/
        online_gaming_behavior_dataset.csv   # Raw dataset with behavioral and demographic columns
    src/
        app.py              # Streamlit entry point; orchestrates the full pipeline and UI
        data_handler.py     # Loads CSV, engineers the churn label, performs train-test split
        preprocessor.py     # Feature engineering, StandardScaler, and OneHotEncoder
        model.py            # Trains the RandomForestClassifier
        evaluator.py        # Computes evaluation metrics and extracts feature importance
    requirements.txt
    README.md
```

---

## Features

### Predict Player Risk Tab

Users enter a player profile across three sections: engagement metrics, game progress and monetization, and player demographics. On submission, the model returns a churn classification (High Risk or Healthy Engagement) alongside a churn probability score displayed on an interactive gauge chart.

### Model Telemetry Tab

Displays four key performance metrics computed on the held-out test set: Accuracy, Precision, Recall, and AUC-ROC. A horizontal bar chart shows the top feature importances, revealing which behavioral signals drive churn decisions in the current model.

---

## Input Features

### Numeric Features

| Feature | Description |
|---|---|
| Age | Player age in years |
| PlayTimeHours | Total weekly play time in hours |
| InGamePurchases | Whether the player has made in-game purchases (0 or 1) |
| SessionsPerWeek | Number of gaming sessions per week |
| AvgSessionDurationMinutes | Average length of a session in minutes |
| PlayerLevel | Current in-game level of the player |
| AchievementsUnlocked | Total number of achievements earned |

### Categorical Features

| Feature | Options |
|---|---|
| Gender | Male, Female |
| Location | Asia, Europe, USA, Other |
| GameGenre | Action, RPG, Simulation, Sports, Strategy |
| GameDifficulty | Easy, Medium, Hard |

### Engineered Features

In addition to the raw inputs, the preprocessor derives three interaction features before scaling:

- PlayTimePerSession: PlayTimeHours divided by SessionsPerWeek
- AchievementsPerLevel: AchievementsUnlocked divided by PlayerLevel
- TotalWeeklyMinutes: SessionsPerWeek multiplied by AvgSessionDurationMinutes

---

## How It Works

1. Data Loading: The CSV is loaded and a binary churn label is derived from the EngagementLevel column, where Low engagement maps to 1 (churned) and all other levels map to 0. Player IDs and rows with null values are dropped. An 80/20 stratified train-test split is applied to preserve class distribution.

2. Preprocessing: The three engineered features are computed. All numeric features are scaled using StandardScaler. Categorical features are one-hot encoded using OneHotEncoder with unknown-value handling enabled. Preprocessors are fit exclusively on training data and then applied to both splits.

3. Model Training: A RandomForestClassifier is trained with 300 estimators, a maximum depth of 15, balanced class weights to account for class imbalance, and a fixed random seed of 42 for reproducibility.

4. Evaluation: The model is evaluated on the test set and reports Accuracy, Precision, Recall, and AUC-ROC. The top 5 most important features are extracted from the fitted forest and visualized as a ranked bar chart.

---

## Tech Stack

| Layer | Library |
|---|---|
| UI and Application | Streamlit |
| Data Processing | pandas, NumPy |
| Machine Learning | scikit-learn |
| Visualization | Plotly |

---

## Installation and Usage

```bash
# Clone the repository
git clone <repository-url>
cd Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/app.py
```

The application will open in your browser at http://localhost:8501.

---

## Requirements

- Python 3.8 or higher
- pandas >= 1.4.0
- NumPy >= 1.24.0
- scikit-learn >= 1.5.0
- Streamlit >= 1.40.0
- Plotly >= 5.20.0
