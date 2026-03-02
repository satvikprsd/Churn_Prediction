import streamlit as st
from src.data_handler import load_and_split_data
from src.preprocessor import fit_preprocessors, apply_preprocessors
from src.model import train_model
from src.evaluator import evaluate_model, get_feature_importance

st.title("Intelligent Player Churn Prediction")

data_splits, num_feats, cat_feats = load_and_split_data('data/online_gaming_behavior_dataset.csv')
X_train, X_test, y_train, y_test = data_splits

with st.spinner('Preparing data and training model...'):
    scaler, encoder = fit_preprocessors(X_train, num_feats, cat_feats)
    
    X_train_processed = apply_preprocessors(X_train, scaler, encoder, num_feats, cat_feats)
    X_test_processed = apply_preprocessors(X_test, scaler, encoder, num_feats, cat_feats)
    
    model = train_model(X_train_processed, y_train)

st.header("Model Evaluation")
metrics = evaluate_model(model, X_test_processed, y_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
col2.metric("Precision", f"{metrics['Precision']:.3f}")
col3.metric("Recall", f"{metrics['Recall']:.3f}")
col4.metric("AUC", f"{metrics['AUC']:.3f}")

st.header("Top Churn Risk Factors")
feature_names = list(X_train_processed.columns)
importances = get_feature_importance(model, feature_names)
st.bar_chart(importances)