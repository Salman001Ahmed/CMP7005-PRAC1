import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Title
st.title("📊 Model Building: PM2.5 Prediction using Random Forest")

# Load data (this should be replaced with your actual data source)
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/PRSA_Data_Changping_20130301-20170228.csv")
    df2 = pd.read_csv("data/PRSA_Data_Huairou_20130301-20170228.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.drop(columns=['No', 'year', 'month', 'day', 'hour'])
    df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]
    return df

data = load_data()

# Handle missing data
st.subheader("🧹 Preprocessing")
numeric_cols = data.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

# Encode categorical variables
encoder = LabelEncoder()
data['wd'] = encoder.fit_transform(data['wd'])
data['station'] = encoder.fit_transform(data['station'])

# Standardize numeric columns
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# Define features and target
X = data.drop(columns=['PM2.5', 'datetime'])
y = data['PM2.5']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Evaluation Metrics")
st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
st.write(f"**R-squared (R²):** {r2:.4f}")

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
st.subheader("🔍 Feature Importance")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
ax.set_title("Feature Importance from Random Forest Regressor")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)
