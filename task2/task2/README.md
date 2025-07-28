# End-to-End ML Pipeline (Churn Prediction)

## 1. Objective of the Task
The objective of this task is to build an end-to-end machine learning pipeline to predict customer churn. The pipeline covers all steps from data loading and preprocessing to model selection, evaluation, and export, providing a reproducible workflow for binary classification problems.

## 2. Methodology / Approach
- **Data Loading & Exploration:**
  - The dataset (`churn.csv`) is loaded and explored for missing values and target distribution.
- **Preprocessing:**
  - Numerical features: Imputed using the mean and scaled with StandardScaler.
  - Categorical features: Imputed with the most frequent value and one-hot encoded.
  - Preprocessing steps are combined using a `ColumnTransformer`.
- **Model Selection & Training:**
  - Two models are considered: Logistic Regression and Random Forest.
  - Hyperparameters are tuned using `GridSearchCV` with cross-validation.
  - The pipeline is wrapped using `sklearn.Pipeline` for reproducibility.
- **Evaluation:**
  - The best model is evaluated on a held-out test set.
  - Metrics reported: accuracy, precision, recall, f1-score.
- **Export:**
  - The best pipeline is saved using `joblib` for future inference.

## 3. Key Results or Observations (Outputs)
- **Best Model:** Logistic Regression with C=0.1
- **Best Cross-Validation Accuracy:** 0.8024
- **Test Set Accuracy:** 0.7984
- **Classification Report:**
  - Precision, recall, and f1-score are higher for the 'No' (non-churn) class than for the 'Yes' (churn) class, indicating some class imbalance.
- **Pipeline Export:**
  - The best pipeline is saved as `churn_best_pipeline.joblib` for easy reuse.

## 4. What is `joblib`?
`joblib` is a Python library for efficient serialization of large data objects. In this project, it is used to save ("dump") the entire trained machine learning pipeline—including preprocessing steps and the best model—to a file (`churn_best_pipeline.joblib`). This allows you to reload the pipeline later and make predictions on new data without retraining.

## 5. Libraries Used and Why
- **pandas**: For data loading, manipulation, and exploration.
- **numpy**: For numerical operations and array handling.
- **scikit-learn (sklearn)**: Provides tools for preprocessing, model building, pipelines, hyperparameter tuning (GridSearchCV), and evaluation metrics.
- **joblib**: For saving and loading the trained pipeline efficiently.
- **os**: For file path and existence checks.

---

This pipeline provides a robust template for binary classification tasks and can be adapted to other domains by changing the dataset and target variable. 