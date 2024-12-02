import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
    data = pd.read_csv(os.path.join(os.path.dirname(current_dir), 'dataset', 'traffic_flow_dataset.csv'))
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

def plot_hourly_pattern():
    """Create an interactive line plot showing traffic patterns by hour."""
    # Extract hour from Time column and group data
    data['hour'] = pd.to_datetime(data['Time']).dt.hour
    hourly_data = data.groupby('hour').agg({
        'CarCount': 'mean',
        'BikeCount': 'mean', 
        'BusCount': 'mean',
        'TruckCount': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['CarCount'], name='Cars', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['BikeCount'], name='Bikes', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['BusCount'], name='Buses', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=hourly_data['hour'], y=hourly_data['TruckCount'], name='Trucks', mode='lines+markers'))
    
    fig.update_layout(
        title='Hourly Traffic Pattern',
        xaxis_title='Hour of Day',
        yaxis_title='Average Vehicle Count',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig)

def plot_vehicle_composition(user_input):
    """Create a pie chart showing vehicle composition."""
    labels = ['Cars', 'Bikes', 'Buses', 'Trucks']
    values = [
        user_input['CarCount'].values[0],
        user_input['BikeCount'].values[0],
        user_input['BusCount'].values[0],
        user_input['TruckCount'].values[0]
    ]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title='Vehicle Composition')
    
    st.plotly_chart(fig)

def plot_weekly_pattern():
    """Create a heatmap showing traffic patterns by day."""
    # Calculate average total traffic for each day
    weekly_data = data.groupby('Day of the week')['Total'].mean()
    
    # Ensure consistent ordering of days
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_data = weekly_data.reindex(days)
    
    fig = go.Figure(data=go.Bar(
        x=days,
        y=weekly_data.values,
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Weekly Traffic Pattern',
        xaxis_title='Day of Week',
        yaxis_title='Average Total Traffic'
    )
    
    st.plotly_chart(fig)

def plot_traffic_distribution():
    """Plot distribution of traffic volumes."""
    fig = go.Figure()
    
    # Create histogram of total traffic volume
    hist_values = data['Total'].values
    
    fig.add_trace(go.Histogram(
        x=hist_values,
        nbinsx=50,
        name='Traffic Distribution',
        opacity=0.75
    ))
    
    fig.update_layout(
        title='Traffic Volume Distribution',
        xaxis_title='Total Vehicle Count',
        yaxis_title='Frequency'
    )
    
    st.plotly_chart(fig)

# Main function for the app
def run_app():
    st.title("Traffic Flow Prediction")
    st.write("Predict traffic conditions based on vehicle counts and time information.")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Traffic Patterns", "Analysis"])
    
    with tab1:
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
        
        # Display vehicle composition
        st.subheader("Current Vehicle Composition")
        plot_vehicle_composition(user_input)
    
    with tab2:
        st.header("Traffic Patterns")
        
        # Hourly pattern
        st.subheader("Hourly Traffic Pattern")
        plot_hourly_pattern()
        
        # Weekly pattern
        st.subheader("Weekly Traffic Pattern")
        plot_weekly_pattern()
    
    with tab3:
        st.header("Traffic Analysis")
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature')
        plt.title('Feature Importance in Traffic Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Traffic distribution
        st.subheader("Traffic Volume Distribution")
        plot_traffic_distribution()
        
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

# Run the app
if __name__ == '__main__':
    run_app()
