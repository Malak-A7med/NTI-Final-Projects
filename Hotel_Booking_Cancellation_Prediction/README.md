# 🏨 Hotel Booking Cancellation Predictor

## Overview

Hotel Booking Cancellation Predictor is a Machine Learning web application that predicts whether a hotel reservation is likely to be canceled based on booking information provided by the user.

The application is powered by a trained **XGBoost Classifier** and deployed using **Streamlit**. It provides an interactive dashboard that displays the prediction result along with the cancellation probability.

---

## Features

- Predict hotel booking cancellation status.
- Display cancellation probability using an interactive gauge chart.
- Simple and user-friendly interface.
- Automatic feature engineering from the selected arrival date.
- Interactive booking summary.
- Real-time prediction using a trained XGBoost model.

---

## Machine Learning Pipeline

The model was developed following a complete machine learning workflow:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
  - Standard Scaling
  - One-Hot Encoding
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Deployment using Streamlit

---

## Input Features

The prediction is based on several booking attributes, including:

- Number of Adults
- Number of Children
- Weekend Nights
- Week Nights
- Meal Plan
- Room Type
- Lead Time
- Arrival Date
- Market Segment
- Parking Requirement
- Repeated Guest
- Previous Cancellations
- Previous Completed Bookings
- Average Price per Room
- Number of Special Requests

---

## Model Performance

| Metric | Score |
|---------|------:|
| Training Accuracy | 92.0% |
| Test Accuracy | 91% |
| Precision | 92% |
| Recall | 95% |
| F1-Score | 93% |
| ROC-AUC | 96.2% |

The model achieved strong predictive performance while maintaining good generalization between the training and testing datasets.

---

## Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Plotly

---

## Project Structure

```
Hotel_Cancellation_Predictor/
│
├── app.py
├── Hotel_Booking_Prediction.ipynb
├── hotel_model.pkl
├── preprocessor.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

## Application Preview

The application allows users to enter booking information through an interactive interface and instantly receive:

- Booking cancellation prediction
- Cancellation probability
- Interactive dashboard
- Booking summary

---

## Author

**Malak Ahmed**
