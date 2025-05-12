import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
@st.cache_data
def load_and_prepare():
    df1 = pd.read_csv("data/PRSA_Data_Changping_20130301-20170228.csv")
    df2 = pd.read_csv("data/PRSA_Data_Huairou_20130301-20170228.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.drop(columns=['No', 'year', 'month', 'day', 'hour'])
    df = df[['datetime'] + [col for col in df.columns if col != 'datetime']]
    return df

def count_outliers(data, columns):
    outlier_counts = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower) | (data[col] > upper)]
        outlier_counts[col] = outliers.shape[0]
    return pd.DataFrame(list(outlier_counts.items()), columns=["Pollutant", "Outlier Count"])

st.title("📈 Exploratory Data Analysis")

data = load_and_prepare()
pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

st.subheader("Outlier Counts")
st.dataframe(count_outliers(data, pollutants))

st.subheader("Distributions")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, pol in enumerate(pollutants):
    sns.histplot(data[pol], bins=50, kde=True, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Distribution of {pol}')
st.pyplot(fig)

st.subheader("Boxplots")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, pol in enumerate(pollutants):
    sns.boxplot(y=data[pol], ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Boxplot of {pol}')
st.pyplot(fig)

st.subheader("Wind Direction Count")
fig, ax = plt.subplots(figsize=(10, 4))
sns.countplot(x='wd', data=data, order=data['wd'].value_counts().index, ax=ax)
ax.set_title("Wind Direction")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Pollutant vs TEMP")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, pol in enumerate(pollutants):
    sns.scatterplot(x='TEMP', y=pol, data=data, ax=axes[i//3, i%3], alpha=0.4)
    axes[i//3, i%3].set_title(f'{pol} vs TEMP')
st.pyplot(fig)

st.subheader("Pollutant vs Wind Speed")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, pol in enumerate(pollutants):
    sns.scatterplot(x='WSPM', y=pol, data=data, ax=axes[i//3, i%3], alpha=0.4)
    axes[i//3, i%3].set_title(f'{pol} vs WSPM')
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig = plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(fig)

st.subheader("Pairplot of Pollutants")
st.pyplot(sns.pairplot(data[pollutants]))

st.subheader("Station-wise Average")
station_avg = data.groupby('station')[pollutants].mean().reset_index()
melted = pd.melt(station_avg, id_vars='station', var_name='Pollutant', value_name='Avg Value')

fig = plt.figure(figsize=(12, 6))
sns.barplot(x='station', y='Avg Value', hue='Pollutant', data=melted)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Time Series (Weekly Avg)")
data_ts = data.set_index('datetime')
fig = plt.figure(figsize=(12, 6))
for pol in pollutants:
    weekly = data_ts[pol].resample('W').mean()
    plt.plot(weekly, label=pol)
plt.legend()
plt.title("Weekly Avg of Pollutants")
st.pyplot(fig)
