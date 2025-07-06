"""
Iris Flower Classification Module

This module provides a complete machine learning pipeline for classifying Iris flowers
into three species (Setosa, Versicolor, Virginica) based on morphological measurements.

Author: Auto-generated from Jupyter notebook analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configuration
DEFAULT_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'model_params': {
        'random_state': 42
    },
    'visualization_params': {
        'figure_size': (6, 4),
        'colormap': 'Blues'
    }
}

class IrisClassifier:
    """
    Complete Iris flower classification system.
    
    This class encapsulates the entire machine learning pipeline including
    data loading, preprocessing, training, evaluation, and prediction.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Iris classifier.
        
        Args:
            config (dict): Configuration parameters (optional)
        """
        self.config = config or DEFAULT_CONFIG
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.species_names = None
        
    def load_iris_data(self, file_path):
        """
        Load the Iris dataset from CSV file.
        
        Args:
            file_path (str): Path to the Iris CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset with columns:
                - SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
                
        Example:
            classifier = IrisClassifier()
            df = classifier.load_iris_data('/path/to/Iris.csv')
            print(df.head())
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Store species names for later use
            self.species_names = df['Species'].unique()
            
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def perform_eda(self, df):
        """
        Perform exploratory data analysis on the Iris dataset.
        
        Args:
            df (pandas.DataFrame): The Iris dataset
            
        Returns:
            dict: Dictionary containing:
                - 'info': Dataset information
                - 'describe': Summary statistics
                - 'shape': Dataset dimensions
                - 'species_counts': Count of each species
                
        Example:
            classifier = IrisClassifier()
            eda_results = classifier.perform_eda(df)
            print(eda_results['describe'])
        """
        results = {
            'shape': df.shape,
            'describe': df.describe(),
            'species_counts': df['Species'].value_counts(),
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes
        }
        
        print("Dataset Information:")
        print(f"Shape: {results['shape']}")
        print(f"Missing values: {results['missing_values'].sum()}")
        print(f"Species distribution:\n{results['species_counts']}")
        
        return results
    
    def preprocess_data(self, df, test_size=None, random_state=None):
        """
        Preprocess the Iris dataset for machine learning.
        
        Args:
            df (pandas.DataFrame): The Iris dataset
            test_size (float): Proportion of dataset for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, scaler)
                - X_train: Training features (scaled)
                - X_test: Testing features (scaled) 
                - y_train: Training labels
                - y_test: Testing labels
                - scaler: Fitted StandardScaler object
                
        Example:
            classifier = IrisClassifier()
            X_train, X_test, y_train, y_test, scaler = classifier.preprocess_data(df)
        """
        test_size = test_size or self.config['test_size']
        random_state = random_state or self.config['random_state']
        
        # Validate dataset
        if 'Species' not in df.columns:
            raise ValueError("Dataset must contain 'Species' column")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'Species' and col != 'Id']
        X = df[feature_cols]
        y = df['Species']
        
        # Validate features
        if X.shape[1] != 4:
            print(f"Warning: Expected 4 features, got {X.shape[1]}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Store scaler for later use
        self.scaler = scaler
        
        print(f"Data preprocessed: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
        
        return X_train, X_test, y_train, y_test, scaler
    
    def train_model(self, X_train, y_train, model_type='logistic_regression', **kwargs):
        """
        Train a machine learning model on the Iris dataset.
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (pandas.Series): Training labels
            model_type (str): Type of model ('logistic_regression', 'random_forest', 'svm')
            **kwargs: Additional model parameters
            
        Returns:
            sklearn model: Trained model
            
        Example:
            classifier = IrisClassifier()
            model = classifier.train_model(X_train, y_train, model_type='random_forest')
        """
        random_state = kwargs.get('random_state', self.config['random_state'])
        
        # Select model
        if model_type == 'logistic_regression':
            model = LogisticRegression(random_state=random_state, **kwargs)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(random_state=random_state, **kwargs)
        elif model_type == 'svm':
            model = SVC(random_state=random_state, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store model
        self.model = model
        self.is_trained = True
        
        print(f"{model_type.title()} model trained successfully!")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model's performance.
        
        Args:
            model: Trained scikit-learn model
            X_test (numpy.ndarray): Testing features
            y_test (pandas.Series): True testing labels
            
        Returns:
            dict: Dictionary containing:
                - 'predictions': Model predictions
                - 'accuracy': Model accuracy score
                - 'confusion_matrix': Confusion matrix
                - 'classification_report': Detailed classification report
                
        Example:
            classifier = IrisClassifier()
            results = classifier.evaluate_model(model, X_test, y_test)
            print(f"Accuracy: {results['accuracy']:.3f}")
        """
        y_pred = model.predict(X_test)
        
        results = {
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'classification_report_str': classification_report(y_test, y_pred)
        }
        
        print(f"Model Accuracy: {results['accuracy']:.3f}")
        print(f"Classification Report:\n{results['classification_report_str']}")
        
        return results
    
    def predict_iris_species(self, sepal_length, sepal_width, petal_length, petal_width, 
                           model=None, scaler=None):
        """
        Predict Iris species for given measurements.
        
        Args:
            sepal_length (float): Sepal length in cm
            sepal_width (float): Sepal width in cm  
            petal_length (float): Petal length in cm
            petal_width (float): Petal width in cm
            model: Trained model (optional, uses self.model if None)
            scaler: Fitted scaler (optional, uses self.scaler if None)
            
        Returns:
            str: Predicted species name
            
        Example:
            classifier = IrisClassifier()
            species = classifier.predict_iris_species(5.1, 3.5, 1.4, 0.2)
            print(f"Predicted species: {species}")
        """
        # Use instance variables if not provided
        model = model or self.model
        scaler = scaler or self.scaler
        
        if model is None or scaler is None:
            raise ValueError("Model and scaler must be provided or trained first")
        
        # Validate inputs
        validate_input_features(sepal_length, sepal_width, petal_length, petal_width)
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        return prediction[0]
    
    def predict_proba_iris_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict probabilities for each Iris species.
        
        Args:
            sepal_length (float): Sepal length in cm
            sepal_width (float): Sepal width in cm  
            petal_length (float): Petal length in cm
            petal_width (float): Petal width in cm
            
        Returns:
            dict: Dictionary with species names and their probabilities
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        # Validate inputs
        validate_input_features(sepal_length, sepal_width, petal_length, petal_width)
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create result dictionary
        species_proba = dict(zip(self.model.classes_, probabilities))
        
        return species_proba

# Standalone functions for backward compatibility and modular usage

def load_iris_data(file_path):
    """
    Load the Iris dataset from CSV file.
    
    Args:
        file_path (str): Path to the Iris CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset
        
    Example:
        df = load_iris_data('/path/to/Iris.csv')
        print(df.head())
    """
    classifier = IrisClassifier()
    return classifier.load_iris_data(file_path)

def perform_eda(df):
    """
    Perform exploratory data analysis on the Iris dataset.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        
    Returns:
        dict: EDA results
        
    Example:
        eda_results = perform_eda(df)
        print(eda_results['describe'])
    """
    classifier = IrisClassifier()
    return classifier.perform_eda(df)

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the Iris dataset for machine learning.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        test_size (float): Proportion of dataset for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
            
    Example:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    """
    classifier = IrisClassifier()
    return classifier.preprocess_data(df, test_size, random_state)

def train_model(X_train, y_train, model_type='logistic_regression', random_state=42):
    """
    Train a machine learning model on the Iris dataset.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (pandas.Series): Training labels
        model_type (str): Type of model ('logistic_regression', 'random_forest', 'svm')
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        sklearn model: Trained model
        
    Example:
        model = train_model(X_train, y_train)
    """
    classifier = IrisClassifier()
    return classifier.train_model(X_train, y_train, model_type, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model's performance.
    
    Args:
        model: Trained scikit-learn model
        X_test (numpy.ndarray): Testing features
        y_test (pandas.Series): True testing labels
        
    Returns:
        dict: Evaluation results
        
    Example:
        results = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {results['accuracy']}")
    """
    classifier = IrisClassifier()
    return classifier.evaluate_model(model, X_test, y_test)

def predict_iris_species(model, scaler, sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict Iris species for given measurements.
    
    Args:
        model: Trained logistic regression model
        scaler: Fitted StandardScaler object
        sepal_length (float): Sepal length in cm
        sepal_width (float): Sepal width in cm  
        petal_length (float): Petal length in cm
        petal_width (float): Petal width in cm
        
    Returns:
        str: Predicted species name
        
    Example:
        species = predict_iris_species(model, scaler, 5.1, 3.5, 1.4, 0.2)
        print(f"Predicted species: {species}")
    """
    classifier = IrisClassifier()
    return classifier.predict_iris_species(sepal_length, sepal_width, petal_length, petal_width, model, scaler)

# Visualization functions

def create_pairplot(df, save_path=None):
    """
    Create pairplot to visualize feature relationships by species.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        save_path (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: Pairplot figure
        
    Example:
        fig = create_pairplot(df)
        plt.show()
    """
    fig = sns.pairplot(df, hue='Species', height=2.5)
    fig.fig.suptitle('Iris Dataset - Feature Relationships by Species', y=1.02)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pairplot saved to {save_path}")
    
    return fig

def plot_confusion_matrix(conf_matrix, species_names, save_path=None):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        species_names (list): List of species names
        save_path (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix heatmap
        
    Example:
        plot_confusion_matrix(results['confusion_matrix'], df['Species'].unique())
        plt.show()
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Iris Species Classification')
    plt.xlabel('Predicted Species')
    plt.ylabel('Actual Species')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt.gcf()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        save_path (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Iris Classification")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance Score')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return plt.gcf()

# Extended functionality

def train_multiple_models(X_train, y_train):
    """
    Train multiple models for comparison.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (pandas.Series): Training labels
        
    Returns:
        dict: Dictionary of trained models
        
    Example:
        models = train_multiple_models(X_train, y_train)
        print(models.keys())
    """
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'svm': SVC(random_state=42, probability=True)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"{name.title()} model trained")
        
    return models

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation on the model.
    
    Args:
        model: Scikit-learn model
        X (numpy.ndarray): Features
        y (pandas.Series): Labels
        cv (int): Number of cross-validation folds
        
    Returns:
        dict: Cross-validation results
        
    Example:
        cv_results = cross_validate_model(model, X_scaled, y)
        print(f"Mean accuracy: {cv_results['mean_score']:.3f}")
    """
    scores = cross_val_score(model, X, y, cv=cv)
    
    results = {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'cv_folds': cv
    }
    
    print(f"Cross-validation results ({cv} folds):")
    print(f"Mean accuracy: {results['mean_score']:.3f} (+/- {results['std_score'] * 2:.3f})")
    
    return results

def validate_input_features(sepal_length, sepal_width, petal_length, petal_width):
    """
    Validate input features for prediction.
    
    Args:
        sepal_length, sepal_width, petal_length, petal_width: Feature values
        
    Raises:
        ValueError: If any feature is negative or unrealistic
        
    Returns:
        bool: True if all inputs are valid
        
    Example:
        validate_input_features(5.1, 3.5, 1.4, 0.2)  # Returns True
    """
    features = [sepal_length, sepal_width, petal_length, petal_width]
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Check for negative values
    if any(val < 0 for val in features):
        raise ValueError("All measurements must be positive")
    
    # Check for unrealistic values (basic sanity check)
    if sepal_length > 10 or sepal_width > 10:
        raise ValueError("Sepal measurements seem unrealistic (>10 cm)")
        
    if petal_length > 10 or petal_width > 10:
        raise ValueError("Petal measurements seem unrealistic (>10 cm)")
    
    # Check for extremely small values
    if any(val < 0.1 for val in features):
        print("Warning: Some measurements are very small (<0.1 cm)")
        
    return True

# Note: validate_input_features is available as a standalone function

if __name__ == "__main__":
    # Example usage
    print("Iris Classifier Module Loaded Successfully!")
    print("Available classes: IrisClassifier")
    print("Available functions: load_iris_data, perform_eda, preprocess_data, train_model, evaluate_model, predict_iris_species")
    print("Visualization functions: create_pairplot, plot_confusion_matrix, plot_feature_importance")
    print("Extended functions: train_multiple_models, cross_validate_model, validate_input_features")