# 🏨 Hotel Booking Cancellation Predictor

## 🚀 Live Demo

Try the deployed application here:

🔗 **Streamlit App:** [https://hotelbookingcancellationprediction2772026.streamlit.app/](https://aiprojects-ak3mwwdjdxqvsviftkjmrx.streamlit.app/)

---

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

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering:
  - Created new meaningful features to improve model performance.
- Data Visualization
- Data Preprocessing:
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
| Training Accuracy | 92% |
| Test Accuracy | 91% |
| Precision | 92% |
| Recall | 95% |
| F1-Score | 93% |
| ROC-AUC | 96.2% |

The model achieved strong predictive performance while maintaining good generalization between training and testing datasets.

---

## Technologies Used

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Joblib


---
## Model Development and Comparison

13 classification models were trained and compared, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- SVM
- KNN
- Naive Bayes
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- AdaBoost
- HistGradient Boosting

Models were evaluated using Accuracy, Precision, Recall, F1-Score, Balanced Accuracy, and ROC-AUC.

**XGBoost Classifier** was selected as the final model due to its best overall performance.

---

## Application Preview

The application allows users to enter booking information through an interactive interface and instantly receive:

- Booking cancellation prediction.
- Cancellation probability.
- Interactive probability visualization.
- Booking summary.

---

## Deployment

The application was deployed using **Streamlit Cloud**.

🔗 **Live Application:**  
[https://hotelbookingcancellationprediction2772026.streamlit.app/](https://aiprojects-ak3mwwdjdxqvsviftkjmrx.streamlit.app/)

---

## Author

**Malak Ahmed**
