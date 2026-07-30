# Student Performance Prediction

A machine learning project designed to predict student academic performance using supervised learning models. This project implements predictive systems for two critical tasks: classifying pass/fail status and predicting final grades.

---

## 📋 Project Overview

This project leverages machine learning algorithms to provide insights into student performance. By analyzing various academic indicators, the system can:

1. **Predict Pass/Fail Status**: Binary classification to determine if a student will pass or fail
2. **Predict Final Grade**: Regression model to estimate the student's final grade

### Key Features
- Multiple model evaluation (classification and regression)
- Optimized SVM models for both prediction tasks
- Interactive web interface via Streamlit
- Comprehensive data analysis and visualization

---

## 🎯 Objectives

### Task 1: Pass/Fail Classification
- **Goal**: Binary classification to predict whether a student will pass or fail
- **Models Evaluated**: 
  - Logistic Regression
  - Decision Tree (DT)
  - Random Forest (RF)
  - Support Vector Machine (SVM) ✅ **Selected**
- **Result**: SVM achieved the best performance and was selected as the final model

### Task 2: Final Grade Prediction
- **Goal**: Regression model to predict the student's final grade
- **Models Evaluated**:
  - Linear Regression
  - Decision Tree (DT)
  - Support Vector Machine (SVM) ✅ **Selected**
- **Result**: SVM demonstrated superior performance in grade prediction and was selected as the final model

Detailed accuracy metrics and performance comparisons are available in the ML.ipynb notebook.

---

## 🛠️ Technologies & Libraries

- **Python 3.x**
- **Scikit-learn**: Machine learning models and preprocessing
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Model serialization
- **Streamlit**: Interactive web application

---

## 📁 Project Structure

```
Student Performance Predictiion/
├── ML.ipynb                          # Main notebook with model development and evaluation
├── svm_pass_model.joblib            # Trained SVM model for pass/fail prediction
├── svm_final_grade_model.joblib     # Trained SVM model for grade prediction
├── data.txt                          # Dataset source reference
├── summary.png                       # Models Summary 
└── README.md                         # Project documentation
```

---

## 📈 Data Source

The project uses the **300K Student Performance Prediction Dataset** from Kaggle:
- **Source**: [Student Performance Prediction Dataset](https://www.kaggle.com/datasets/rhythmghai/300k-student-performance-prediction-dataset)
- **Size**: 300,000+ student records
- **Features**: Academic and performance indicators

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit joblib
```

### Running the Project

1. **Explore the Analysis**: Open the Jupyter notebook
   ```bash
   jupyter notebook ML.ipynb
   ```
---

## 📱 Interactive Application

The project includes a fully functional Streamlit web application for real-time predictions:

**Access the Live Application**: [Machine Learning Project - Streamlit](https://machine-learning---project-6pqvtojrvisumxepx8chay.streamlit.app/)

### Features:
- User-friendly interface for input
- Real-time predictions for both tasks
- Model performance visualization
- Results interpretation and insights

---

## 📊 Key Results

The project successfully evaluated multiple machine learning algorithms:

- **Best Classification Model**: Support Vector Machine (SVM)
  - Superior accuracy in binary pass/fail prediction
  
- **Best Regression Model**: Support Vector Machine (SVM)
  - Excellent performance in final grade estimation
  - Better generalization compared to linear and tree-based models

All results and detailed metrics are documented in the ML.ipynb notebook and visualized in summary.png.

---

## 📝 Model Details

### Pass/Fail Prediction Model
- **File**: `svm_pass_model.joblib`
- **Algorithm**: Support Vector Machine (SVM)
- **Task Type**: Binary Classification
- **Output**: Pass (1) or Fail (0)

### Final Grade Prediction Model
- **File**: `svm_final_grade_model.joblib`
- **Algorithm**: Support Vector Machine (SVM)
- **Task Type**: Regression
- **Output**: Predicted numerical grade

---

## 🔍 Notebook Highlights

The ML.ipynb notebook includes:
- Data loading and exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Model training and evaluation for multiple algorithms
- Comparative analysis of different models
- Selection rationale for final models
- Performance metrics and visualizations
- Results summary

---

## 📄 License

This project is open source and available for educational purposes.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! Feel free to open an issue or submit a pull request.

---

## 🔗 Links

- **GitHub Repository**: [AI_Projects](https://github.com/Malak-A7med/AI_Projects)
- **Student Performance Project**: [Student Performance Predictiion](https://github.com/Malak-A7med/AI_Projects/tree/main/Student%20Performance%20Predictiion)
- **Interactive App**: [Streamlit Application](https://machine-learning---project-6pqvtojrvisumxepx8chay.streamlit.app/)
- **Dataset**: [Kaggle - Student Performance Dataset](https://www.kaggle.com/datasets/rhythmghai/300k-student-performance-prediction-dataset)

---

**Last Updated**: May 6, 2026
