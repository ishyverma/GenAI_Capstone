import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # dataframer shape
    print(f"Dataset shape: {df.shape}")
    
    # Dropping customer id as it is not helpful at all
    df.drop('customerID', axis=1, inplace=True)
    
    # converting total charges to numeric and filling missing values with median
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_tc = df['TotalCharges'].median()

    # inplace will fill the missing data with the median of the total charges and it will add the data in the original dataframe itself
    df['TotalCharges'].fillna(median_tc, inplace=True)


    yes_no_cols = [
        'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'MultipleLines'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].replace('No internet service', 'No')
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # added 1 to df['tenure'] to avoid division by 0
    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)  
    df['TenureGroup'] = pd.cut(df['tenure'], 
                              bins=[-1, 12, 48, 999], 
                              labels=['New', 'Mid', 'Loyal'])
    
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    return df

def prepare_features(df):
    # deleteing the churn feature from the df to get X or inputs
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']
    categorical_features = ['gender', 'SeniorCitizen', 'Contract', 'InternetService', 
                           'PaymentMethod', 'TenureGroup']
    
    # used column transformer to scale num or numerical data and one hot encoding for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    X_processed = preprocessor.fit_transform(X)
    
    cat_names = list(preprocessor.named_transformers_['cat']
                     .get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_names
    
    return X_processed, y, preprocessor, feature_names
