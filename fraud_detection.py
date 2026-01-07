"""Credit Card Fraud Detection Model"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """ML Model for Credit Card Fraud Detection"""
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        return df
    def preprocess_data(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test, y_train, y_test
    def handle_imbalance(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X_train, y_train)
    def train_models(self, X_train, y_train):
        self.models['LR'] = LogisticRegression(max_iter=1000).fit(X_train, y_train)
        self.models['RF'] = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        self.models['XGB'] = XGBClassifier(n_estimators=100).fit(X_train, y_train)
    def evaluate(self, X_test, y_test):
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"{name} - ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    model = FraudDetectionModel()
    print("Fraud Detection Model Ready")
