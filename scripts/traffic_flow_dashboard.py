import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(current_dir), 'model', 'traffic_flow', 'model.pkl')
scaler_path = os.path.join(os.path.dirname(current_dir), 'model', 'traffic_flow', 'scaler.pkl')
day_of_week_encoder_path = os.path.join(os.path.dirname(current_dir), 'model', 'traffic_flow', 'day_of_week_encoder.pkl')
time_period_encoder_path = os.path.join(os.path.dirname(current_dir), 'model', 'traffic_flow', 'time_period_encoder.pkl')
y_label_encoder_path = os.path.join(os.path.dirname(current_dir), 'model', 'traffic_flow', 'y_label_encoder.pkl')

# Load the trained model and encoders
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    day_of_week_encoder = joblib.load(day_of_week_encoder_path)
    time_period_encoder = joblib.load(time_period_encoder_path)
    y_label_encoder = joblib.load(y_label_encoder_path)
except FileNotFoundError as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

# Feature Engineering Function
def process_features(data, scaler=None, day_of_week_encoder=None, time_period_encoder=None):
    """
    Function to process features for both training and prediction.
    - Encodes categorical features like 'day_of_week' and 'time_period'.
    - Scales numerical features.
    
    Parameters:
    - data: DataFrame with user input or training data
    - scaler: A fitted scaler for feature scaling
    - day_of_week_encoder: Encoder for day of week
    - time_period_encoder: Encoder for time period
    
    Returns:
    - data_processed: The data with processed features
    """
    
    # Encode 'day_of_week' and 'time_period' if the encoder is provided
    if day_of_week_encoder:
        data['day_of_week'] = day_of_week_encoder.transform(data['day_of_week'])
    if time_period_encoder:
        data['time_period'] = time_period_encoder.transform(data['time_period'])

    # Scale numerical features
    numeric_columns = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total']
    
    if scaler:
        data[numeric_columns] = scaler.transform(data[numeric_columns])
    
    # remove 'Total' from the numeric columns
    data = data.drop(columns=['Total'])
    return data

# Function to handle user input via Streamlit sliders
def get_user_input():
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour", 0, 23, 12)
        minute = st.slider("Minute", 0, 59, 30)
        car_count = st.slider("Car Count", 0, 200, 20)
        bike_count = st.slider("Bike Count", 0, 50, 5)
        bus_count = st.slider("Bus Count", 0, 50, 5)
    
    with col2:
        truck_count = st.slider("Truck Count", 0, 50, 5)
        day_of_week = st.selectbox("Day of the Week", 
                                 ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                  'Friday', 'Saturday', 'Sunday'])
        day_of_month = st.slider("Day of the Month", 1, 31, 15)
        time_period = st.selectbox("Time Period", 
                                 ['Morning', 'Afternoon', 'Evening', 'Night'])

    # Create a DataFrame with the user input
    data = pd.DataFrame({
        'hour': [hour],
        'minute': [minute],
        'CarCount': [car_count],
        'BikeCount': [bike_count],
        'BusCount': [bus_count],
        'TruckCount': [truck_count],
        'Total': [car_count + bike_count + bus_count + truck_count],
        'day_of_week': [day_of_week],
        'day_of_month': [day_of_month],
        'time_period': [time_period]
    })
    
    return data

# Function to make predictions using the model
def make_prediction(user_input):
    # Process the user input using the feature engineering function
    processed_input = process_features(user_input, scaler, day_of_week_encoder, time_period_encoder)
    
    # Make prediction using the processed features
    prediction = model.predict(processed_input)
    
    # Convert numerical prediction back to label
    traffic_situation = y_label_encoder.inverse_transform(prediction)
    
    return traffic_situation[0]

# Main function for the app
def run_app():
    st.title("Traffic Flow Prediction")
    st.write("Predict traffic conditions based on vehicle counts and time information.")
    
    # Get user input
    user_input = get_user_input()
    
    # Make prediction
    traffic_situation = make_prediction(user_input)
    
    # Display prediction in a prominent way
    st.header("Prediction Results")

    st.markdown(
        f"""
            <div style="
                padding: 20px;
                background-color: #f0f2f6;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h2 style="color: #0066cc; margin-bottom: 10px;">Predicted Traffic</h2>
                <p style="font-size: 36px; font-weight: bold; color: #1f1f1f; margin: 0;">
                    {traffic_situation}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add some space
    st.write("")
    st.write("")
    
    # Display input summary
    with st.expander("View Input Summary"):
        st.write("Vehicle Counts:")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🚗 Cars: {user_input['CarCount'].values[0]}")
            st.write(f"🚲 Bikes: {user_input['BikeCount'].values[0]}")
            st.write(f"🚌 Buses: {user_input['BusCount'].values[0]}")
        with col2:
            st.write(f"🚛 Trucks: {user_input['TruckCount'].values[0]}")
            st.write(f"📊 Total: {user_input['Total'].values[0]}")
            st.write(f"⏰ Time: {user_input['hour'].values[0]:02d}:{user_input['minute'].values[0]:02d}")
    
    # Display feature importance
    st.header("Feature Importance")
    
    # Get feature importance from model
    feature_importance = pd.DataFrame({
        'Feature': model.feature_names_in_,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance in Traffic Prediction')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Display plot in Streamlit
    st.pyplot(fig)
    
    # Display feature importance as a table
    with st.expander("View Feature Importance Details"):
        st.dataframe(feature_importance)
# Run the app
if __name__ == '__main__':
    run_app()
