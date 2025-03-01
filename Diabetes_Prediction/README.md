# Diabetes Detection Model

## Overview
This project aims to predict the likelihood of diabetes using a machine learning model. The model is trained on a dataset containing various features such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level. The model achieves an accuracy of 97%.

## Features
- **gender**: The gender of the individual (e.g., Male, Female).
- **age**: The age of the individual.
- **hypertension**: Whether the individual has hypertension (e.g., Yes, No).
- **heart_disease**: Whether the individual has heart disease (e.g., Yes, No).
- **smoking_history**: The smoking history of the individual (e.g., Never, Current, Former).
- **bmi**: The Body Mass Index of the individual.
- **HbA1c_level**: The HbA1c level of the individual.
- **blood_glucose_level**: The blood glucose level of the individual.

## Model
The models used for this prediction are Logistic Regression and Random Forest Classifier. It was trained on a dataset containing the above features and achieved an accuracy of 97%.

## Requirements
- Python 3.x
- Scikit-learn
- Pandas
- Numpy

## Model Evaluation
The model was evaluated using various metrics such as accuracy, precision, recall, and F1-score. The accuracy of the model is 97%.
<br>
But taken False Negative Rate as the metrics as we don't want people who have diabetes are incorrectly predicted as not having diabetes.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
