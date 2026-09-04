"""
Simple Churn Prediction Model
==============================

A baseline churn prediction system using industry-standard practices:
- Single Random Forest classifier
- Standard preprocessing (imputation, scaling, encoding)
- Basic feature engineering
- Clear evaluation metrics

This implementation prioritizes simplicity and interpretability over complexity.
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')


class SimpleChurnPredictor:
    """
    A straightforward churn prediction model using Random Forest.
    
    This implementation follows standard ML practices:
    1. Data loading and validation
    2. Feature preprocessing
    3. Model training with cross-validation
    4. Evaluation with standard metrics
    5. Model persistence
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the churn predictor.
        
        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []
        self.feature_importances = {}
        
        self.is_fitted = False
        
    def load_data(self, data_source: str = 'kaggle') -> pd.DataFrame:
        """
        Load data from available sources.
        
        Parameters
        ----------
        data_source : str
            'kaggle' to use the Kaggle dataset loader
        
        Returns
        -------
        pd.DataFrame
            Complete dataset with churn target
        """
        print("=" * 70)
        print("LOADING DATA")
        print("=" * 70)
        
        if data_source == 'kaggle':
            from advanced_pipeline.kaggle_data import load_kaggle_full
            df = load_kaggle_full(random_state=self.random_state)
            print(f"✓ Loaded Kaggle dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        # Basic data validation
        if 'churn' not in df.columns:
            raise ValueError("Dataset must contain 'churn' target column")
        
        print(f"✓ Churn distribution: {df['churn'].value_counts().to_dict()}")
        print(f"✓ Churn rate: {df['churn'].mean():.2%}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        """
        print("\n" + "=" * 70)
        print("FEATURE PREPARATION")
        print("=" * 70)
        
        # Separate features and target
        y = df['churn'].copy()
        X = df.drop(columns=['churn', 'customer_id'], errors='ignore')
        
        # Drop the synthetic demand column (not needed for churn prediction)
        if 'next_month_demand' in X.columns:
            X = X.drop(columns=['next_month_demand'])
        
        # Identify feature types
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"✓ Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"✓ Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        # Check for missing values
        missing = X.isnull().sum()
        if missing.any():
            print(f"\n⚠ Missing values detected:")
            for col in missing[missing > 0].index:
                print(f"  - {col}: {missing[col]} ({missing[col]/len(X):.1%})")
        
        return X, y
    
    def preprocess(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Preprocess features with standard transformations.
        
        Steps:
        1. Impute missing values
        2. Encode categorical variables
        3. Scale numeric features
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        fit : bool
            If True, fit transformers; if False, use existing transformers
        
        Returns
        -------
        np.ndarray
            Preprocessed feature matrix
        """
        X = X.copy()
        
        # Handle numeric features
        if self.numeric_features:
            if fit:
                X[self.numeric_features] = self.numeric_imputer.fit_transform(
                    X[self.numeric_features]
                )
            else:
                X[self.numeric_features] = self.numeric_imputer.transform(
                    X[self.numeric_features]
                )
        
        # Handle categorical features
        if self.categorical_features:
            if fit:
                X[self.categorical_features] = self.categorical_imputer.fit_transform(
                    X[self.categorical_features]
                )
            else:
                X[self.categorical_features] = self.categorical_imputer.transform(
                    X[self.categorical_features]
                )
            
            # Label encoding for categorical variables
            for col in self.categorical_features:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Scale numeric features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = X.columns.tolist()
        
        return X_scaled
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        test_size : float
            Proportion of data to use for testing
        
        Returns
        -------
        dict
            Training results including metrics and splits
        """
        print("\n" + "=" * 70)
        print("MODEL TRAINING")
        print("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        print(f"✓ Train churn rate: {y_train.mean():.2%}")
        print(f"✓ Test churn rate: {y_test.mean():.2%}")
        
        # Preprocess
        print("\n[Preprocessing]")
        X_train_processed = self.preprocess(X_train, fit=True)
        X_test_processed = self.preprocess(X_test, fit=False)
        print(f"✓ Preprocessed shape: {X_train_processed.shape}")
        
        # Initialize Random Forest with balanced class weights
        print("\n[Training Random Forest]")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train_processed, y_train)
        print("✓ Model training complete")
        
        # Cross-validation
        print("\n[Cross-Validation]")
        cv_scores = cross_val_score(
            self.model, X_train_processed, y_train,
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        print(f"✓ 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importances
        self.feature_importances = dict(
            zip(self.feature_names, self.model.feature_importances_)
        )
        
        self.is_fitted = True
        
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'cv_scores': cv_scores
        }
    
    def evaluate(self, X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features
        y_test : pd.Series
            Test target
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")
        
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Print results
        print("\n[Classification Metrics]")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n[Confusion Matrix]")
        print(f"  True Negatives:  {cm[0, 0]}")
        print(f"  False Positives: {cm[0, 1]}")
        print(f"  False Negatives: {cm[1, 0]}")
        print(f"  True Positives:  {cm[1, 1]}")
        
        # Top features
        print("\n[Top 10 Important Features]")
        top_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for i, (feat, importance) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feat:30s} {importance:.4f}")
        
        metrics['confusion_matrix'] = cm
        metrics['classification_report'] = classification_report(y_test, y_pred)
        metrics['feature_importances'] = self.feature_importances
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn for new data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Churn predictions (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities for new data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        
        Returns
        -------
        np.ndarray
            Churn probabilities for class 1
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
        
        X_processed = self.preprocess(X, fit=False)
        return self.model.predict_proba(X_processed)[:, 1]
    
    def save_model(self, filepath: str = 'models/simple_churn_model.pkl') -> str:
        """
        Save the trained model to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        
        Returns
        -------
        str
            Path where model was saved
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'numeric_imputer': self.numeric_imputer,
            'categorical_imputer': self.categorical_imputer,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'random_state': self.random_state,
            'trained_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_bundle, filepath)
        print(f"\n✓ Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SimpleChurnPredictor':
        """
        Load a trained model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
        
        Returns
        -------
        SimpleChurnPredictor
            Loaded model instance
        """
        model_bundle = joblib.load(filepath)
        
        predictor = cls(random_state=model_bundle['random_state'])
        predictor.model = model_bundle['model']
        predictor.scaler = model_bundle['scaler']
        predictor.label_encoders = model_bundle['label_encoders']
        predictor.numeric_imputer = model_bundle['numeric_imputer']
        predictor.categorical_imputer = model_bundle['categorical_imputer']
        predictor.numeric_features = model_bundle['numeric_features']
        predictor.categorical_features = model_bundle['categorical_features']
        predictor.feature_names = model_bundle['feature_names']
        predictor.feature_importances = model_bundle['feature_importances']
        predictor.is_fitted = True
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"✓ Trained on: {model_bundle.get('trained_date', 'Unknown')}")
        
        return predictor


def main():
    """
    Main execution function - complete training pipeline.
    """
    print("\n" + "=" * 70)
    print("SIMPLE CHURN PREDICTION MODEL")
    print("Baseline Implementation using Random Forest")
    print("=" * 70)
    
    # Initialize predictor
    predictor = SimpleChurnPredictor(random_state=42)
    
    # Load data
    df = predictor.load_data(data_source='kaggle')
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Train model
    results = predictor.train(X, y, test_size=0.2)
    
    # Evaluate
    metrics = predictor.evaluate(results['X_test'], results['y_test'])
    
    # Save model
    model_path = predictor.save_model('models/simple_churn_model.pkl')
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"✓ Model Type: Random Forest Classifier")
    print(f"✓ Features: {len(predictor.feature_names)}")
    print(f"✓ Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Model saved: {model_path}")
    print("=" * 70)
    
    return predictor, metrics


if __name__ == '__main__':
    predictor, metrics = main()
