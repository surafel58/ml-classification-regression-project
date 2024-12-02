# ml-classification-regression-project

This repository contains machine learning models to predict social media usage and classify traffic flow patterns. The regression model predicts the amount of time users spend on social media, while the classification model categorizes traffic into low, medium, high, and heavy.

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
