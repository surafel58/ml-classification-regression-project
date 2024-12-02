import os
from data_quality_checks import handle_anomalies
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

@st.cache_data
def load_data_from_drive():
    # Google Drive file ID
    benin_file_id = '1gn0wLZF9wa4w3qoi20gKsTiPxew4iLW-'
    sierra_leone_id = '1iMJvrxsQdZK9wQRaJ3GsMTJEQrT2N-n6'
    togo_id = '1h-fzwSQLVYbhNlzQZ0LhEgCIw0_8d2t5'

    benin_url = f'https://drive.google.com/uc?id={benin_file_id}'
    sierra_leone_url = f'https://drive.google.com/uc?id={sierra_leone_id}'
    togo_url = f'https://drive.google.com/uc?id={togo_id}'
    
    # Ensure the directories exist
    os.makedirs(os.path.abspath('../datasets'), exist_ok=True)

    # Local file path where the dataset will be saved
    benin_output = os.path.abspath('../datasets/benin-malanville.csv')
    sierra_leone_output = os.path.abspath('../datasets/sierraleone-bumbuna.csv')
    togo_output = os.path.abspath('../datasets/togo-dapaong_qc.csv')
    
    # Download the file
    gdown.download(benin_url, benin_output, quiet=False)
    gdown.download(sierra_leone_url, sierra_leone_output, quiet=False)
    gdown.download(togo_url, togo_output, quiet=False)
    
    # Load the dataset
    benin_data = pd.read_csv(benin_output)
    sierra_leone_data = pd.read_csv(sierra_leone_output)
    togo_data = pd.read_csv(togo_output)
    return benin_data, sierra_leone_data, togo_data

@st.cache_data
def load_data():
    benin, sierra_leone, togo = load_data_from_drive()

    # Handle anomalies in the dataset
    handle_anomalies(
    benin,
    negative_value_columns=['GHI', 'DNI', 'DHI'],
    outlier_columns=['ModA', 'ModB', 'WS', 'WSgust'],
    lower_percentile=0.05,
    upper_percentile=0.95
    )

    handle_anomalies(
    sierra_leone,
    negative_value_columns=['GHI', 'DNI', 'DHI'],
    outlier_columns=['ModA', 'ModB', 'WS', 'WSgust'],
    lower_percentile=0.05,
    upper_percentile=0.95
    )

    handle_anomalies(
    togo,
    negative_value_columns=['GHI', 'DNI', 'DHI'],
    outlier_columns=['ModA', 'ModB', 'WS', 'WSgust'],
    lower_percentile=0.05,
    upper_percentile=0.95
    )

    return benin, sierra_leone, togo

benin_data, sierra_leone_data, togo_data = load_data()

st.title("Solar Radiation and Environmental Data Dashboard")

# Dataset Selection
dataset_choice = st.selectbox("Choose a dataset to visualize:", ["Benin", "Sierra Leone", "Togo"])

if dataset_choice == "Benin":
    data = benin_data
elif dataset_choice == "Sierra Leone":
    data = sierra_leone_data
else:
    data = togo_data

# Display Raw Data
if st.checkbox("Show raw data"):
    st.write(data.head())

# Interactive Time Series Plot
st.subheader("Time Series Analysis")

data["Timestamp"] = pd.to_datetime(data["Timestamp"])
downsampled_data = data.resample('D', on='Timestamp').mean()  # Downsample to daily data

columns_to_plot = st.multiselect("Select columns to plot:", data.columns[1:], default=["GHI", "DNI", "DHI", "Tamb"])

if len(columns_to_plot) > 0:
    plt.figure(figsize=(10, 6))
    for col in columns_to_plot:
        plt.plot(data["Timestamp"], data[col], label=col)
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Correlation Heatmap
st.subheader("Correlation Heatmap")

# Allow the user to select features
selected_features = st.multiselect(
    "Select features for the correlation heatmap:", 
    options=data.columns.tolist(), 
    default=data.columns.tolist()[:2]
)

# Check if at least two features are selected
if len(selected_features) >= 2:
    # Compute the correlation matrix for the selected features
    corr_matrix = data[selected_features].corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    st.pyplot(plt)
else:
    st.write("Please select at least two features to generate the correlation heatmap.")

# Scatter Matrix (Pair Plot)
st.subheader("Scatter Matrix (Pair Plot)")

st.write("""
Select the features you want to include in the pair plot. This allows you to focus on specific relationships within the dataset.
""")

# Select numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

selected_features = st.multiselect(
    "Select features for pair plot:",
    options=numeric_columns,
    default=None  # default selection
)

if len(selected_features) > 1:
    with st.spinner("Generating pair plot..."):
        pair_plot = sns.pairplot(data[selected_features], diag_kind='kde')
        st.pyplot(pair_plot)
else:
    st.warning("Please select at least two features to create a pair plot.")


# Histogram with Adjustable Bin Size
st.subheader("Histogram with Adjustable Bin Size")

selected_column = st.selectbox("Select a feature for histogram:", numeric_columns)

bin_size = st.slider("Select number of bins:", min_value=10, max_value=100, value=30)

plt.figure(figsize=(10, 6))
plt.hist(data[selected_column], bins=bin_size, color='skyblue', edgecolor='black')
plt.xlabel(selected_column)
plt.ylabel('Frequency')
st.pyplot(plt)