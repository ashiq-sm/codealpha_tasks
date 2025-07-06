# Iris Flower Classification - API Documentation

## Overview

This project implements a machine learning pipeline for classifying Iris flowers into three species (Setosa, Versicolor, and Virginica) based on their morphological measurements. The implementation uses Python with scikit-learn for machine learning, pandas for data manipulation, and matplotlib/seaborn for visualization.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Dataset Information](#dataset-information)
3. [Core Components](#core-components)
4. [Functions & Methods](#functions--methods)
5. [Usage Examples](#usage-examples)
6. [Model Performance](#model-performance)
7. [Visualization Components](#visualization-components)
8. [Extending the Project](#extending-the-project)

## Installation & Setup

### Dependencies

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Installation Command

```bash
pip install pandas seaborn matplotlib scikit-learn
```

### Verification

Run the following to verify all libraries are properly installed:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('All necessary libraries are imported successfully.')
```

## Dataset Information

### Dataset Structure

- **Source**: Iris.csv (150 samples, 5 columns)
- **Features**: 4 numerical features
  - `SepalLengthCm`: Sepal length in centimeters
  - `SepalWidthCm`: Sepal width in centimeters  
  - `PetalLengthCm`: Petal length in centimeters
  - `PetalWidthCm`: Petal width in centimeters
- **Target**: `Species` (categorical)
  - Iris-setosa
  - Iris-versicolor
  - Iris-virginica

### Data Loading

```python
# Load the dataset
df = pd.read_csv('/kaggle/input/iriscsv/Iris.csv')
```

**Returns**: pandas.DataFrame with shape (150, 5)

## Core Components

### 1. Data Loading Component

**Purpose**: Load and preview the Iris dataset

```python
def load_iris_data(file_path):
    """
    Load the Iris dataset from CSV file.
    
    Args:
        file_path (str): Path to the Iris CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataset with columns:
            - SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
            
    Example:
        df = load_iris_data('/path/to/Iris.csv')
        print(df.head())
    """
    df = pd.read_csv(file_path)
    return df
```

### 2. Exploratory Data Analysis Component

**Purpose**: Analyze dataset characteristics and relationships

```python
def perform_eda(df):
    """
    Perform exploratory data analysis on the Iris dataset.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        
    Returns:
        dict: Dictionary containing:
            - 'info': Dataset information
            - 'describe': Summary statistics
            - 'shape': Dataset dimensions
            
    Example:
        eda_results = perform_eda(df)
        print(eda_results['info'])
    """
    results = {
        'info': df.info(),
        'describe': df.describe(),
        'shape': df.shape
    }
    return results
```

### 3. Data Preprocessing Component

**Purpose**: Prepare data for machine learning model training

```python
def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the Iris dataset for machine learning.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        test_size (float): Proportion of dataset for testing (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
            - X_train: Training features (scaled)
            - X_test: Testing features (scaled) 
            - y_train: Training labels
            - y_test: Testing labels
            - scaler: Fitted StandardScaler object
            
    Example:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    """
    # Separate features and target
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, scaler
```

### 4. Model Training Component

**Purpose**: Train logistic regression classifier

```python
def train_model(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression model on the Iris dataset.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (pandas.Series): Training labels
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        sklearn.linear_model.LogisticRegression: Trained model
        
    Example:
        model = train_model(X_train, y_train)
    """
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model
```

### 5. Model Evaluation Component

**Purpose**: Evaluate model performance using multiple metrics

```python
def evaluate_model(model, X_test, y_test):
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
        results = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {results['accuracy']}")
    """
    y_pred = model.predict(X_test)
    
    results = {
        'predictions': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return results
```

## Functions & Methods

### Prediction Function

```python
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
    # Create feature array
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return prediction[0]
```

## Usage Examples

### Complete Workflow Example

```python
# 1. Load data
df = pd.read_csv('/path/to/Iris.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# 2. Perform EDA
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 3. Preprocess data
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# 4. Train model
model = train_model(X_train, y_train)
print("Model trained successfully!")

# 5. Evaluate model
results = evaluate_model(model, X_test, y_test)
print(f"\nModel Accuracy: {results['accuracy']}")
print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
print(f"\nClassification Report:\n{results['classification_report']}")

# 6. Make predictions
species = predict_iris_species(model, scaler, 5.1, 3.5, 1.4, 0.2)
print(f"Predicted species: {species}")
```

### Individual Component Usage

```python
# Load and explore data
df = load_iris_data('/path/to/Iris.csv')
eda_results = perform_eda(df)

# Train model with custom parameters
X_train, X_test, y_train, y_test, scaler = preprocess_data(df, test_size=0.3)
model = train_model(X_train, y_train)

# Evaluate performance
evaluation = evaluate_model(model, X_test, y_test)
```

## Model Performance

### Expected Results

- **Accuracy**: 100% (1.0) on test set
- **Precision**: 1.00 for all classes
- **Recall**: 1.00 for all classes  
- **F1-Score**: 1.00 for all classes

### Confusion Matrix Structure

```
              Predicted
           Setosa  Versicolor  Virginica
Actual Setosa     10         0          0
    Versicolor     0         9          0  
    Virginica      0         0         11
```

## Visualization Components

### 1. Pairplot Visualization

```python
def create_pairplot(df):
    """
    Create pairplot to visualize feature relationships by species.
    
    Args:
        df (pandas.DataFrame): The Iris dataset
        
    Returns:
        matplotlib.figure.Figure: Pairplot figure
        
    Example:
        fig = create_pairplot(df)
        plt.show()
    """
    return sns.pairplot(df, hue='Species')
```

### 2. Confusion Matrix Heatmap

```python
def plot_confusion_matrix(conf_matrix, species_names):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        conf_matrix (numpy.ndarray): Confusion matrix
        species_names (list): List of species names
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix heatmap
        
    Example:
        plot_confusion_matrix(results['confusion_matrix'], df['Species'].unique())
        plt.show()
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_names, yticklabels=species_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return plt.gcf()
```

## Extending the Project

### Adding New Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_multiple_models(X_train, y_train):
    """
    Train multiple models for comparison.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (pandas.Series): Training labels
        
    Returns:
        dict: Dictionary of trained models
    """
    models = {
        'logistic_regression': LogisticRegression(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'svm': SVC(random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

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
    """
    scores = cross_val_score(model, X, y, cv=cv)
    
    return {
        'scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std()
    }
```

## Error Handling

### Input Validation

```python
def validate_input_features(sepal_length, sepal_width, petal_length, petal_width):
    """
    Validate input features for prediction.
    
    Args:
        sepal_length, sepal_width, petal_length, petal_width: Feature values
        
    Raises:
        ValueError: If any feature is negative or unrealistic
        
    Returns:
        bool: True if all inputs are valid
    """
    if any(val < 0 for val in [sepal_length, sepal_width, petal_length, petal_width]):
        raise ValueError("All measurements must be positive")
        
    if sepal_length > 10 or sepal_width > 10:
        raise ValueError("Sepal measurements seem unrealistic")
        
    if petal_length > 10 or petal_width > 10:
        raise ValueError("Petal measurements seem unrealistic")
        
    return True
```

## Configuration

### Default Parameters

```python
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
```

## License & Attribution

This implementation is based on the classic Iris dataset introduced by Ronald Fisher in 1936. The dataset is in the public domain and commonly used for machine learning education and testing.

---

*Generated using the Iris Flower Classification API Documentation Generator*