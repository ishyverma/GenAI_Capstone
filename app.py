import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from data_preprocessing import load_and_clean_data, prepare_features
from models import ChurnModels
from evaluation import print_metrics, plot_feature_importance

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Telco Customer Churn Predictor")
st.markdown("### Milestone 1: Logistic Regression + Decision Tree + Full Evaluation")

st.sidebar.header("Instructions")
st.sidebar.info("""
1. **Upload** your CSV (WA_Fn-UseC_-Telco-Customer-Churn.csv)
2. **Auto-trains** 80/20 split models
3. **Shows** predictions, metrics, confusion matrices, insights
4. **Deploy** to Streamlit Cloud for submission
""")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        df = load_and_clean_data(uploaded_file)
        st.dataframe(df.head())
        
        X, y, preprocessor, feature_names = prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train & Evaluate Models", type="primary"):
            models = ChurnModels(X_train, X_test, y_train, y_test)
            models.train_models(feature_names)
            
            log_proba, dt_proba = models.predict_proba(X_test)
            log_pred, dt_pred = models.predict(X_test)
            metrics = models.get_metrics(y_test, log_pred, dt_pred)
            
            st.session_state.models = models
            st.session_state.predictions = {
                'log_proba': log_proba, 'dt_proba': dt_proba,
                'log_pred': log_pred, 'dt_pred': dt_pred,
                'metrics': metrics
            }
        
        if st.session_state.predictions is not None:
            metrics = st.session_state.predictions['metrics']
            st.success("Evaluation Complete!")
            
            st.subheader("Model Performance")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.style.highlight_max(axis=0))
            
            if metrics['Logistic Regression']['F1 Score'] > metrics['Decision Tree']['F1 Score']:
                st.info("**Logistic Regression wins** - Better generalization")
            else:
                st.info("**Decision Tree wins** - Captures non-linear patterns")

    if st.session_state.predictions is not None:
        st.subheader("Confusion Matrices")
        preds = st.session_state.predictions
        y_true = st.session_state.y_test
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            plot_confusion_matrix(y_true, preds['log_pred'], 'Logistic Regression')
        with cm_col2:
            plot_confusion_matrix(y_true, preds['dt_pred'], 'Decision Tree')
    
    with col2:
        st.subheader("Sample Predictions")
        if st.session_state.predictions is not None:
            preds = st.session_state.predictions
            sample_idx = np.random.choice(len(X_test), 5, replace=False)
            sample_df = pd.DataFrame({
                'Customer': sample_idx,
                'True Churn': y_test.iloc[sample_idx].values,
                'Log Proba': preds['log_proba'][sample_idx],
                'Log Pred': preds['log_pred'][sample_idx],
                'DT Proba': preds['dt_proba'][sample_idx],
                'DT Pred': preds['dt_pred'][sample_idx]
            })
            st.dataframe(sample_df)
    
    if st.button("Show Feature Insights"):
        if st.session_state.models is not None:
            dt_imp, lr_coef = st.session_state.models.get_insights()
            plot_feature_importance(st.session_state.models.feature_names, dt_imp, lr_coef)
            st.balloons()
        else:
            st.warning("Please train the models first!")
