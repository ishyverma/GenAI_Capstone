import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def print_metrics(metrics):
    df_metrics = pd.DataFrame(metrics).T.round(3)
    print("\nModel Comparison")
    print(df_metrics)
    return df_metrics

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn (0)', 'Churn (1)'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    st.pyplot(plt.gcf())
    plt.close()

def plot_feature_importance(feature_names, dt_importance, lr_coef, top_n=10):
    top_dt_idx = np.argsort(dt_importance)[-top_n:]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(range(top_n), dt_importance[top_dt_idx])
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in top_dt_idx])
    ax1.set_title('Decision Tree: Top 10 Churn Drivers')
    ax1.set_xlabel('Importance')
    st.pyplot(fig1)
    plt.close(fig1)
    
    top_lr_idx = np.argsort(np.abs(lr_coef))[-top_n:]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(range(top_n), lr_coef[top_lr_idx])
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([feature_names[i] for i in top_lr_idx])
    ax2.set_title('Logistic Regression: Coefficients\n(Positive = Increases Churn Risk)')
    ax2.set_xlabel('Coefficient Value')
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.markdown("""
    **Key Insights:**
    1. Month-to-month contracts → **HIGH** churn risk
    2. Higher tenure → **LOWER** churn risk
    3. Higher MonthlyCharges → **HIGHER** churn risk
    """)
