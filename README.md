# Churn Prediction Model Comparison

This repository compares various machine learning models to predict customer churn. The models included in this project are:

- **Gaussian Naive Bayes (GaussianNB)**
- **Bernoulli Naive Bayes (BernoulliNB)**
- **Logistic Regression (LogisticRegression)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree Classifier (DecisionTreeClassifier)**
- **Random Forest Classifier (RandomForestClassifier)**
- **Linear Support Vector Classifier (LinearSVC)**
- **Support Vector Classifier (SVC)**

The goal is to evaluate which machine learning model performs best for predicting customer churn, using various performance metrics and visualizations.

## Project Overview

Customer churn prediction is crucial for businesses to identify customers who are likely to leave. This project compares different classification algorithms to see which one best predicts churn using various evaluation metrics:

- **F1-Score**
- **AUC (Area Under Curve)**
- **Log-Loss**
- **ROC-AUC**
- **Time taken for training and evaluation**

The models were tuned using cross-validation, and results are visualized with bar plots and ROC curves to make comparisons easier.

## Models Implemented

1. **Gaussian Naive Bayes (GaussianNB)**
2. **Bernoulli Naive Bayes (BernoulliNB)**
3. **Logistic Regression (LogisticRegression)**
4. **K-Nearest Neighbors (KNN)**
5. **Decision Tree Classifier (DecisionTreeClassifier)**
6. **Random Forest Classifier (RandomForestClassifier)**
7. **Linear Support Vector Classifier (LinearSVC)**
8. **Support Vector Classifier (SVC)**

## Key Features

- **Hyperparameter Tuning:** Each model underwent hyperparameter tuning to optimize performance (e.g., optimal `C` for Logistic Regression, optimal `K` for KNN, etc.).
  
- **Evaluation Metrics:**
  - **F1-Score:** Measures the balance between precision and recall for a classification model.
  - **AUC (Area Under Curve):** A metric that evaluates the model's ability to distinguish between classes.
  - **Log-Loss:** Quantifies how well the predicted probabilities match the true class labels.
  - **ROC-AUC:** A graphical representation of a model's performance, showing the trade-off between true positive and false positive rates.

- **Visualizations:** 
  - Bar plots comparing F1-score, AUC, Log-Loss, ROC-AUC, and time taken for each model.
  - ROC curves for each model to visualize performance.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `seaborn`

## Usage

1. **Data Preprocessing:** 
   Ensure the dataset is preprocessed correctly, with features properly encoded and split into training and test sets (e.g., using `train_test_split` from `scikit-learn`).
   
2. **Model Training and Evaluation:** 
   Run the script to evaluate and compare the performance of each model. The results will include key metrics like F1-Score, AUC, Log-Loss, and ROC-AUC.

```bash
python churn_prediction_comparison.py
```

3. **Results Visualization:**
   After running the models, the script will generate:
   - **Bar plots** comparing each model's F1-Score, AUC, Log-Loss, and time taken.
   - **ROC curves** for each model to visually evaluate classification performance.

## Example Results

```text
"""""" Logistic Regression """"
F1-Score: 0.85,   AUC: 0.91,   Log-Loss: 0.34,   ROC-AUC: 0.92
Time taken: 2.4s

"""""" Random Forest Classifier """"
F1-Score: 0.83,   AUC: 0.89,   Log-Loss: 0.38,   ROC-AUC: 0.90
Time taken: 4.6s

"""""" Support Vector Classifier """"
F1-Score: 0.80,   AUC: 0.85,   Log-Loss: 0.42,   ROC-AUC: 0.88
Time taken: 3.1s
```

## Results Visualizations

The following visualizations are generated:

- **F1-Score Comparison**: Bar plot comparing the F1-Scores of all models.
- **AUC Comparison**: Bar plot comparing the AUC values of the models.
- **Log-Loss Comparison**: Bar plot comparing the Log-Loss for each model.
- **ROC-AUC Comparison**: Bar plot comparing the ROC-AUC scores.
- **Model Training Time Comparison**: Bar plot showing the time taken for each model to train and evaluate.

## Contributing

Feel free to fork the repository and submit pull requests to improve the project. If you encounter any bugs or have suggestions, please open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
