# ML Classification and Regression Projects

This repository contains two machine learning projects:

1. Traffic Flow Pattern Analysis (Classification)
2. Social Media Usage Prediction (Regression)

# Social Media Daily Usage Predictor

A machine learning application that predicts a user's daily social media usage time based on their social media behavior and demographics.

🔗 **Live Demo**: [Social Media Usage Predictor](https://social-media-daily-usage.streamlit.app/)

## Overview

This application uses a Random Forest model to predict how many minutes a user spends on social media daily. The prediction is based on various factors including age, posting frequency, engagement metrics, and platform preferences.

## Features

- Interactive sliders for easy input adjustment
- Real-time predictions
- Support for multiple social media platforms
- Feature importance visualization
- Clean, user-friendly interface

## Input Features

- **Age**: User's age (13-100 years)
- **Posts Per Day**: Number of posts made daily (0-50)
- **Likes Received**: Average likes received per day (0-500)
- **Comments Received**: Average comments received per day (0-200)
- **Messages Sent**: Number of messages sent daily (0-500)
- **Gender**: Male/Female/Other
- **Platform**: Choice of major social media platforms
  - Instagram
  - Twitter
  - LinkedIn
  - Snapchat
  - Telegram
  - WhatsApp

## Technology Stack

- Python
- Streamlit for web interface
- Scikit-learn for machine learning
- Pandas for data processing
- Matplotlib for visualization

## Usage

1. Visit the [live demo](https://social-media-daily-usage.streamlit.app/)
2. Adjust the sliders and select your preferences
3. View the predicted daily usage time instantly
4. Explore feature importance to understand what factors influence the prediction

## Model Information

The prediction is made using a Random Forest Regressor trained on social media usage data. The model takes into account both numerical features (like age and engagement metrics) and categorical features (like gender and platform preference).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

# Traffic Flow Pattern Analysis and Prediction

A machine learning application that predicts traffic flow patterns based on vehicle counts, time information, and other relevant factors.

🔗 **Live Demo**: [Traffic Flow Pattern Predictor](https://traffic-flow-pattern.streamlit.app/)

## Overview

This application uses a machine learning classification model to predict traffic patterns (Low, Medium, High, Heavy) based on real-time vehicle counts and temporal features.

## Features

- Interactive sliders for vehicle counts and time inputs
- Real-time traffic pattern predictions
- Support for multiple vehicle types:
  - Cars
  - Bikes
  - Buses
  - Trucks
- Time-based analysis
- Feature importance visualization
- Clean, user-friendly interface

## Input Features

- **Vehicle Counts**:
  - Cars (0-200)
  - Bikes (0-50)
  - Buses (0-50)
  - Trucks (0-50)
- **Time Information**:
  - Hour (0-23)
  - Minute (0-59)
  - Day of Week
  - Day of Month (1-31)
- **Time Period**: Morning/Afternoon/Evening/Night

## Technology Stack

- Python
- Streamlit for web interface
- Scikit-learn for machine learning
- Pandas for data processing
- Matplotlib and Seaborn for visualization
- Joblib for model serialization

## Usage

1. Visit the [live demo](https://traffic-flow-pattern.streamlit.app/)
2. Adjust the vehicle count sliders
3. Set the time-related parameters
4. View the predicted traffic pattern instantly
5. Explore feature importance to understand prediction factors

## Model Information

The prediction is made using a machine learning classifier trained on traffic flow data. The model considers both vehicle count features and temporal features to classify traffic patterns.
