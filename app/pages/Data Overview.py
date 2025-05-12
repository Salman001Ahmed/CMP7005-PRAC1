import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/PRSA_Data_Changping_20130301-20170228.csv")
    df2 = pd.read_csv("data/PRSA_Data_Huairou_20130301-20170228.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    return df

st.title("📊 Data Overview")

data = load_data()

st.subheader("Basic Info")
st.write(f"Shape of dataset: {data.shape}")
st.dataframe(data.head())

st.subheader("Summary Statistics")
st.write(data.describe())

st.subheader("Missing Values (%)")
missing = data.isna().sum() / len(data) * 100
st.write(missing[missing > 0])

st.subheader("Duplicate Records")
st.write(f"Number of duplicates: {data.duplicated().sum()}")
