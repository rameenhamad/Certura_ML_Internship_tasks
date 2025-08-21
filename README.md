## Certura ML Internship – Machine Learning Tasks

During my Certura Machine Learning Internship, I completed three end-to-end ML tasks using Python, Scikit-learn, Pandas, NumPy, and Matplotlib. Each task focused on building, training, and evaluating models with proper preprocessing pipelines.

# Task 1: California Housing Price Prediction
## Goal:
Predict house prices from the California Housing dataset.
## Approach:
  - Applied PowerTransformer on input features (X) for normalization.
  - Applied FunctionTransformer to take log of target (y).
  - Used StandardScaler for scaling numerical features.
  - Trained a Linear Regression model.
  - Evaluated performance using RMSE (Root Mean Square Error).
## Results:
- RMSE on actual y: 0.60
- RMSE on log-transformed y: 0.17

# Task 2: Email Spam Detection (NLP)
## Goal: 
Build a text classification model to detect spam emails.
## Approach:
  - Performed data cleaning using regex and string functions.
  - Encoded labels (y) using LabelEncoder.
  - Converted text to numerical features using TF-IDF Vectorizer.
  - Handled class imbalance with SMOTE (Synthetic Minority Oversampling Technique).
  - Trained a Multinomial Naive Bayes model.
  - Built a Pipeline combining preprocessing and modeling steps.
  - Evaluated using Accuracy Score.
## Features:
Pipeline can predict on unseen email text with consistent preprocessing.

# Task 3: Feature Importance Visualization (Titanic Dataset)
## Goal: 
Identify and visualize important features in survival prediction.
## Approach:
  - Performed feature engineering:
  - Dropped irrelevant/missing columns.
  - Filled missing values.
  - Created new columns (e.g., extracted Title from passenger names).
  - Encoded categorical columns.
  - Applied train-test split and Power Transformation.
  - Trained a Random Forest Classifier.
  - Extracted and visualized Feature Importances using Matplotlib.

# Tools & Libraries Used
Python
Scikit-learn (sklearn)
Pandas
NumPy
Matplotlib

## What i gained throughout this Internship

This internship project helped me gain hands-on experience in:
  - Building end-to-end ML pipelines
  - Applying data preprocessing & feature engineering
  - Working with regression, classification, and ensemble models
  - Evaluating models with metrics like RMSE and Accuracy
  - Visualizing feature importance for model interpretability
