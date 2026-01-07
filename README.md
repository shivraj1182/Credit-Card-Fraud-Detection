# Credit Card Fraud Detection

A comprehensive machine learning solution for credit card fraud detection using pattern recognition and anomaly detection. This project implements multiple classification models with techniques to handle imbalanced datasets and includes detailed exploratory data analysis.

## Project Overview

Credit card fraud is a critical issue affecting financial institutions and customers worldwide. This project develops and compares various machine learning models to accurately identify fraudulent transactions while minimizing false positives. The solution addresses the inherent class imbalance in fraud detection datasets using SMOTE (Synthetic Minority Over-sampling Technique).

### Key Features

- Data preprocessing and feature scaling
- Exploratory Data Analysis (EDA) with visualizations
- Handling imbalanced datasets using SMOTE
- Multiple classification models:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Model evaluation with multiple metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Score
  - Confusion Matrix and Classification Report
- Cross-validation for robust evaluation
- Feature importance analysis

## Installation

### Requirements
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/shivraj1182/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses credit card transaction data with the following characteristics:
- Highly imbalanced dataset (frauds are rare compared to legitimate transactions)
- Features include transaction amount, time, and anonymized user/merchant information
- Binary classification: Legitimate (0) vs Fraudulent (1)

## Usage

### Basic Usage

```python
from fraud_detection import FraudDetectionModel
import pandas as pd

# Load data
df = pd.read_csv('creditcard.csv')

# Initialize model
model = FraudDetectionModel()

# Preprocess data
X_train, X_test, y_train, y_test = model.preprocess_data(df)

# Train models
model.train_logistic_regression(X_train, y_train)
model.train_random_forest(X_train, y_train)
model.train_xgboost(X_train, y_train)

# Evaluate models
results = model.evaluate_all_models(X_test, y_test)
print(results)

# Get predictions
predictions = model.predict(X_test)
```

## Machine Learning Models

### 1. Logistic Regression
- Baseline linear classification model
- Provides interpretable coefficients
- Fast training and prediction
- Good for understanding feature importance

### 2. Random Forest
- Ensemble method using multiple decision trees
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance rankings
- Good at capturing complex patterns

### 3. XGBoost
- Gradient boosting classifier
- Sequential ensemble approach
- Highly tunable hyperparameters
- Excellent performance on imbalanced datasets
- Fast and scalable

## Data Preprocessing

1. **Feature Scaling**: StandardScaler normalization for numerical features
2. **Handling Imbalance**: SMOTE technique applied only to training data
3. **Train-Test Split**: 80-20 split with stratification
4. **Missing Values**: Handled appropriately based on data exploration

## Model Evaluation Metrics

### Accuracy
Ratio of correct predictions to total predictions. Less useful for imbalanced datasets.

### Precision
Ratio of true positives to predicted positives. Important when false positives are costly.

### Recall (Sensitivity)
Ratio of true positives to actual positives. Critical for catching fraudulent transactions.

### F1-Score
Harmonic mean of precision and recall. Balances both metrics.

### ROC-AUC
Area under the Receiver Operating Characteristic curve. Evaluates model across all classification thresholds.

## Key Techniques

### SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic samples from minority class
- Prevents overfitting to majority class
- Applied only to training data to avoid data leakage

### Cross-Validation
- K-fold cross-validation for robust performance estimation
- Helps identify model stability and generalization
- Prevents overfitting to specific train-test split

### Feature Engineering
- Statistical features from transaction data
- Time-based features (hour, day of week, etc.)
- Scaled transaction amounts
- Normalized user behavior patterns

## Project Structure

```
Credit-Card-Fraud-Detection/
├── fraud_detection.py      # Main fraud detection model class
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── .gitignore            # Git ignore file
├── LICENSE               # MIT License
└── data/                 # Data directory (optional)
    └── creditcard.csv    # Dataset
```

## Technologies & Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models, preprocessing, metrics
- **xgboost**: Gradient boosting classifier
- **imbalanced-learn**: SMOTE for handling imbalanced data
- **matplotlib & seaborn**: Data visualization
- **jupyter**: Interactive notebooks for analysis

## Performance Comparison

The project includes comparative analysis of:
- Model training time
- Prediction accuracy
- Precision and recall trade-offs
- ROC-AUC scores
- F1-scores across different thresholds

## Handling Class Imbalance

Fraud detection datasets are typically highly imbalanced (1-2% fraud rate). This project addresses imbalance through:

1. **SMOTE**: Synthetic oversampling of minority class
2. **Class Weights**: Adjusting model training to penalize minority class misclassification
3. **Threshold Tuning**: Adjusting decision threshold based on business requirements
4. **Appropriate Metrics**: Using precision, recall, and F1-score instead of accuracy alone

## Hyperparameter Tuning

The project includes hyperparameter optimization for:
- Logistic Regression: Regularization parameter, solver type
- Random Forest: Number of trees, max depth, min samples split
- XGBoost: Learning rate, max depth, subsample ratio, colsample bytree

## Future Enhancements

- Deep learning models (Neural Networks, LSTM)
- Anomaly detection algorithms (Isolation Forest, One-Class SVM)
- Real-time prediction pipeline
- Model deployment using Flask/FastAPI
- Hyperparameter optimization with GridSearchCV
- Feature selection techniques
- Ensemble methods combining multiple models
- Explainability analysis (SHAP, LIME)
- Model monitoring and retraining strategies

## Results & Performance

The project provides:
- Model comparison matrices
- ROC curves visualization
- Confusion matrices for each model
- Feature importance plots
- Performance metrics summary
- Cross-validation results

## Author

Shivraj (shivraj1182)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Imbalanced-learn (SMOTE): https://imbalanced-learn.org/
- Credit Card Fraud Detection Kaggle Dataset: https://www.kaggle.com/

## Disclaimer

This project is for educational purposes. Real-world fraud detection systems require additional security measures, compliance with financial regulations, and continuous model monitoring.

## Contact

For questions or suggestions, please open an issue on the GitHub repository.
