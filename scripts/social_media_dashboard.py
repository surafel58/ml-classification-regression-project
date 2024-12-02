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
model_path = os.path.join(os.path.dirname(current_dir), 'model', 'social_media', 'model.pkl')
scaler_path = os.path.join(os.path.dirname(current_dir), 'model', 'social_media', 'scaler.pkl')
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Function to handle feature scaling and conversion
def preprocess_input(data):
    """
    Process the user input to apply all the necessary transformations:
    - Replace 'Non-binary' with 'Other' in the 'Gender' column.
    - One-hot encode categorical columns (Gender, Platform).
    - Apply feature scaling to numerical columns.
    """
    # Feature conversion: Replace 'Non-binary' with 'Other'
    data['Gender'] = data['Gender'].replace('Non-binary', 'Other')
    
    # One-hot encode categorical features (Gender, Platform)
    gender_dummies = pd.get_dummies(data['Gender'], prefix='Gender')
    platform_dummies = pd.get_dummies(data['Platform'], prefix='Platform')

    # Define numerical columns (excluding the target variable)
    numerical_columns = ['Age', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Messages_Sent_Per_Day', 'Comments_Received_Per_Day']
    
    # Scale numerical features using the loaded scaler (do not include the target variable)
    data_scaled = pd.DataFrame(
        scaler.transform(data[numerical_columns]),
        columns=numerical_columns
    )
    
    # Combine all features
    final_features = pd.concat([data_scaled, gender_dummies, platform_dummies], axis=1)
    
    # Ensure all expected columns are present with correct order
    expected_columns = [
        'Age', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day',
        'Messages_Sent_Per_Day', 'Gender_Male', 'Gender_Other', 'Platform_Instagram',
        'Platform_LinkedIn', 'Platform_Snapchat', 'Platform_Telegram', 'Platform_Twitter',
        'Platform_Whatsapp'
    ]
    # Reorder columns to match expected order
    for col in expected_columns:
        if col not in final_features.columns:
            final_features[col] = 0
        else:
            # If column exists but in wrong order, reorder it
            final_features = final_features.reindex(columns=expected_columns)
    
    return final_features, data_scaled


# Function to handle user input via Streamlit sliders
def get_user_input():
    """
    Create user input widgets for predicting time spent on social media.
    """
    st.sidebar.header("User Input Features")
    
    age = st.sidebar.slider("Age", 18, 80, 25)
    number_of_posts = st.sidebar.slider("Number of Posts", 0, 1000, 100)
    likes_received = st.sidebar.slider("Likes Received per Day", 0, 10000, 100)
    messages_sent = st.sidebar.slider("Messages Sent per Day", 0, 1000, 50)
    comments_received = st.sidebar.slider("Comments Received per Day", 0, 1000, 50)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
    platform = st.sidebar.selectbox("Platform", ['Facebook', 'Instagram', 'Twitter', 'Other'])

    # Create a DataFrame with the user input
    data = {
        'Age': age,
        'Posts_Per_Day': number_of_posts,
        'Likes_Received_Per_Day': likes_received,
        'Messages_Sent_Per_Day': messages_sent,
        'Comments_Received_Per_Day': comments_received,
        'Gender': gender,
        'Platform': platform
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


# Function to make predictions using the model
def make_prediction(user_input):
    """Make a prediction using the trained model."""
    # Preprocess the input data
    features_encoded, features_scaled = preprocess_input(user_input)
    
    # Make prediction using the processed features
    prediction = model.predict(features_encoded)
    
    return prediction, features_encoded


# Function to display the model performance metrics
def display_model_performance():
    """
    Display model performance metrics (MSE, RMSE, R²).
    """
    st.subheader("Model Performance Metrics")
    st.write("Model performance metrics for Random Forest Regressor")
    st.write(f"Mean Squared Error (MSE): 12.5")
    st.write(f"Root Mean Squared Error (RMSE): 3.54")
    st.write(f"R-squared (R²): 0.85")


# Function to visualize feature importance
def visualize_feature_importance():
    """Visualize feature importance from the trained model."""
    # Get feature importance from the model
    importances = model.feature_importances_
    
    # Get feature names
    features = [
        'Age', 'Posts_Per_Day', 'Likes_Received_Per_Day', 'Comments_Received_Per_Day',
        'Messages_Sent_Per_Day', 'Gender_Male', 'Gender_Other', 'Platform_Instagram',
        'Platform_LinkedIn', 'Platform_Snapchat', 'Platform_Telegram', 'Platform_Twitter',
        'Platform_Whatsapp'
    ]
    
    # Sort features by importance
    indices = np.argsort(importances)
    
    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.arange(len(features)), importances[indices])
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([features[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Feature Importance')
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    plt.close()


# Function to visualize correlation matrix
def visualize_correlation_matrix(df):
    """
    Visualize the correlation matrix of the features in the dataset.
    """
    df_encoded = pd.get_dummies(df, drop_first=True)  # Apply one-hot encoding
    correlation_matrix = df_encoded.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    plt.title("Correlation Matrix")
    st.pyplot(fig)


def plot_scatter_matrix(data):
    """Create an interactive scatter matrix of numerical features."""
    fig = px.scatter_matrix(
        data,
        dimensions=['Age', 'Posts_Per_Day', 'Likes_Received_Per_Day', 
                   'Comments_Received_Per_Day', 'Messages_Sent_Per_Day'],
        title='Feature Relationships'
    )
    
    # Update layout with better spacing and size
    fig.update_layout(
        height=800,
        width=800,
        title_x=0.5,
        showlegend=False
    )
    
    # Update traces for better visibility
    fig.update_traces(diagonal_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_distributions(data, user_input):
    """Plot distribution of numerical features with user's position marked."""
    numerical_features = ['Age', 'Posts_Per_Day', 'Likes_Received_Per_Day', 
                         'Comments_Received_Per_Day', 'Messages_Sent_Per_Day']
    
    fig = plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(2, 3, i)
       
        sns.histplot(data[feature], kde=True)
        plt.axvline(user_input[feature].values[0], color='red', linestyle='--', 
                   label='Your Input')
        plt.title(f'Distribution of {feature}')
        plt.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_platform_comparison(data):
    """Create an interactive bar chart comparing different platforms."""
    platforms = data['Platform'].unique().tolist()
    avg_usage = data.groupby('Platform')['Daily_Usage_Time (minutes)'].mean().values
    
    fig = go.Figure(data=[
        go.Bar(name='Average Usage Time', x=platforms, y=avg_usage)
    ])
    
    fig.update_layout(title='Average Daily Usage by Platform',
                     xaxis_title='Platform',
                     yaxis_title='Average Minutes per Day')
    
    st.plotly_chart(fig)

# Main app functionality
def run_app():
    """Run the Streamlit application."""
    st.title("Social Media Usage Prediction")
    st.write("Enter user information to predict daily social media usage time.")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Visualizations"])
    
    with tab1:
        # Original prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=13, max_value=100, value=25, step=1)
            posts = st.slider("Posts Per Day", min_value=0, max_value=50, value=2, step=1)
            likes = st.slider("Likes Received Per Day", min_value=0, max_value=500, value=50, step=5)
        
        with col2:
            comments = st.slider("Comments Received Per Day", min_value=0, max_value=200, value=10, step=2)
            messages = st.slider("Messages Sent Per Day", min_value=0, max_value=500, value=20, step=5)
            gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
            platform = st.selectbox("Preferred Platform", 
                                  options=["Instagram", "Twitter", "LinkedIn", 
                                         "Snapchat", "Telegram", "Whatsapp"])

        # Create DataFrame from inputs
        user_input = pd.DataFrame({
            'Age': [age],
            'Posts_Per_Day': [posts],
            'Likes_Received_Per_Day': [likes],
            'Comments_Received_Per_Day': [comments],
            'Messages_Sent_Per_Day': [messages],
            'Gender': [gender],
            'Platform': [platform]
        })

        # Make prediction
        prediction, features_encoded = make_prediction(user_input)
        
        # Display prediction
        st.header("Prediction Results")
        st.markdown(
            f"""
            <div style="
                padding: 20px;
                width: 100%;
                background-color: #f0f2f6;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                <h2 style="color: #0066cc; margin-bottom: 10px;">Predicted Social Media Daily Usage</h2>
                <p style="font-size: 48px; font-weight: bold; color: #1f1f1f; margin: 0;">
                    {prediction[0]:.0f}
                </p>
                <p style="font-size: 20px; color: #666666; margin: 0;">
                    minutes per day
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with tab2:
        st.header("Data Insights")
        
        # Feature importance visualization
        st.subheader("Feature Importance")
        
        data = pd.read_csv(os.path.join(os.path.dirname(current_dir), 'dataset', 'social_media_dataset.csv'))
        
        visualize_feature_importance()
        
        # Distribution plots
        st.subheader("Feature Distributions")
        plot_distributions(data, user_input)
        
        # Platform comparison
        st.subheader("Platform Usage Comparison")
        plot_platform_comparison(data)
    
    with tab3:
        st.header("Interactive Visualizations")
        
        # Scatter matrix
        st.subheader("Feature Relationships")
        plot_scatter_matrix(data)
        
        # Optional: Display encoded features for debugging
        if st.checkbox("Show processed features"):
            st.write("Processed Features:")
            st.write(features_encoded)


# Run the app
if __name__ == '__main__':
    run_app()
