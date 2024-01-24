# Disease Prediction from Symptoms

## Overview
This repository hosts a machine learning project aimed at predicting diseases from a set of symptoms. The project applies several classification techniques, such as Decision Tree, Random Forest, SVM, and XGBoost, to a dataset of symptoms and their corresponding diseases.

## Dataset
The dataset contains 132 symptoms as features and a target variable for prognosis, mapping to 42 different diseases. It is split into two CSV files: one for training the models and the other for testing their performance. The features underwent preprocessing and feature selection to identify the most relevant for disease classification.

## Top 5 Features for each Target varaible
  ![image](https://github.com/tanzealist/Disease-Prediction-Benchmarking/assets/114698958/31d2351a-2bca-4a02-9259-d4d01d11f213)

## Number of target class variables
  ![image](https://github.com/tanzealist/Disease-Prediction-Benchmarking/assets/114698958/c1b955fb-4fdd-45e2-92d2-a505143ee99b)

## Data Insights
## Top Features for Disease Prediction

| Target Label          | Top 5 Contributing Features                                 |
|-----------------------|-------------------------------------------------------------|
| Chronic Cholestasis   | `malaise`, `chest_pain`, `excessive_hunger`, `dizziness`, `blurred_and_distorted_vision` |
| Drug Reaction         | `irritability`, `muscle_pain`, `loss_of_balance`, `swelling_joints`, `stiff_neck`       |
| Fungal Infection      | `vomiting`, `chills`, `skin_rash`, `joint_pain`, `itching`                              |
| GERD                  | `nausea`, `loss_of_appetite`, `abdominal_pain`, `yellowing_of_eyes`, `yellowish_skin`   |
| Peptic Ulcer Disease  | `family_history`, `painful_walking`, `red_sore_around_nose`, `stomach_bleeding`, `coma` |
| Allergy               | `fatigue`, `high_fever`, `headache`, `sweating`, `cough`                                |



## Methodology
1. **Data Preprocessing**: The raw data was cleaned and preprocessed to prepare for the analysis. This included handling missing values, normalizing, and encoding categorical variables.
2. **Exploratory Data Analysis (EDA)**: Univariate and multivariate analyses were performed to understand the relationships between features and the prognosis.
3. **Feature Selection**: Recursive Feature Elimination (RFE) was utilized to reduce the number of features, focusing on those most impactful for predicting the outcome.
4. **Model Training**: The models were trained on the training dataset, using cross-validation techniques to ensure robustness.
5. **Model Evaluation**: The trained models were evaluated on a separate testing dataset. The performance metrics include accuracy, precision, recall, and F1-score.

## Models Implemented
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

## Requirements
This project uses the following Python libraries:
- Collections
- Matplotlib
- NumPy
- Pandas
- Seaborn
- Scikit-learn
- Warnings
- XGBoost

## Results and Comparison
The following images show the comparison of all models based on their performance metrics and feature importances as assessed by mutual information:

## Conclusions

- **Model Performance and Comparision**: All models, namely Decision Tree (DT), Random Forest (RF), SVM Linear, and XGBoost, have an almost identical accuracy score on the test data, indicating that they are performing equally well in predicting the target variable.

 ![Screenshot 2024-01-24 at 1 29 09 AM](https://github.com/tanzealist/Disease-Prediction-Benchmarking/assets/114698958/44f085ff-2ccf-4615-bbb4-726a723ea08e)

- **Overfitting**: The overfitting score of SVM is lowest, except for XGBoost, which has an overfitting score of 3.086.This indicates that DT, RF, and SVM Linear models are not overfitting, but XGBoost is severely overfitting.
  
  ![image](https://github.com/tanzealist/Disease-Prediction-Benchmarking/assets/114698958/458b0b80-7216-4565-aca4-8424cd3ac6da)

- **Training Accuracy**: All models achieved a training accuracy of 1.0, indicating perfect fitting to the training data. However, this may not necessarily translate to performance on unseen data.
  
- **Model Complexity**: The DT model is the simplest, with RF and XGBoost being more complex. SVM Linear has intermediate complexity. Balancing model complexity and performance is crucial to prevent overfitting and ensure good generalization.

- **Model Selection**: Although all models perform similarly on the given data, XGBoost's overfitting indicates it may not generalize well. Therefore, careful evaluation using appropriate metrics is important before selecting the final model.

## Future Work
- Further hyperparameter tuning could improve model performances.
- Investigating additional features and engineering new ones may provide better insights.
- Expanding the dataset could enhance the model's ability to generalize to new data.

## Contact
If you have any questions or would like to contribute to the project, please contact Tanuj Verma at verma.tanu@northeastern.edu.


