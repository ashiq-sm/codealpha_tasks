"""
Comprehensive Usage Examples for Iris Classification API

This file demonstrates how to use all the documented APIs, functions, and components
for the Iris flower classification project.

Run this file to see the complete workflow in action.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iris_classifier import (
    IrisClassifier,
    load_iris_data, 
    perform_eda,
    preprocess_data,
    train_model,
    evaluate_model,
    predict_iris_species,
    create_pairplot,
    plot_confusion_matrix,
    plot_feature_importance,
    train_multiple_models,
    cross_validate_model,
    validate_input_features
)

def example_1_basic_workflow():
    """
    Example 1: Basic workflow using standalone functions
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Workflow with Standalone Functions")
    print("=" * 60)
    
    # Note: Using sample data since actual file path may not exist
    # In real usage, replace with actual file path
    print("Creating sample Iris dataset...")
    
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 150
    
    # Generate sample features
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    # Generate species labels
    species = ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
    
    # Create DataFrame
    df = pd.DataFrame({
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width,
        'Species': species
    })
    
    print(f"Sample dataset created with shape: {df.shape}")
    print("\nDataset preview:")
    print(df.head())
    
    # Perform EDA
    print("\n1. Performing Exploratory Data Analysis...")
    eda_results = perform_eda(df)
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # Train model
    print("\n3. Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\n4. Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Make predictions
    print("\n5. Making predictions...")
    test_measurements = [5.1, 3.5, 1.4, 0.2]  # Typical Setosa measurements
    species = predict_iris_species(model, scaler, *test_measurements)
    print(f"Predicted species for {test_measurements}: {species}")
    
    return df, model, scaler, results

def example_2_class_based_workflow():
    """
    Example 2: Object-oriented workflow using IrisClassifier class
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Object-Oriented Workflow with IrisClassifier")
    print("=" * 60)
    
    # Initialize classifier
    classifier = IrisClassifier()
    
    # Create sample data (same as example 1)
    np.random.seed(42)
    n_samples = 150
    
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    species = ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
    
    df = pd.DataFrame({
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width,
        'Species': species
    })
    
    print("Using IrisClassifier class for complete workflow...")
    
    # Perform EDA
    print("\n1. Performing EDA...")
    eda_results = classifier.perform_eda(df)
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = classifier.preprocess_data(df)
    
    # Train model (trying different model types)
    print("\n3. Training different models...")
    
    # Logistic Regression
    lr_model = classifier.train_model(X_train, y_train, 'logistic_regression')
    lr_results = classifier.evaluate_model(lr_model, X_test, y_test)
    
    # Random Forest
    rf_model = classifier.train_model(X_train, y_train, 'random_forest', n_estimators=100)
    rf_results = classifier.evaluate_model(rf_model, X_test, y_test)
    
    # SVM
    svm_model = classifier.train_model(X_train, y_train, 'svm', probability=True)
    svm_results = classifier.evaluate_model(svm_model, X_test, y_test)
    
    # Compare accuracies
    print(f"\nModel Comparison:")
    print(f"Logistic Regression Accuracy: {lr_results['accuracy']:.3f}")
    print(f"Random Forest Accuracy: {rf_results['accuracy']:.3f}")
    print(f"SVM Accuracy: {svm_results['accuracy']:.3f}")
    
    # Make predictions with probability
    print("\n4. Making predictions with probabilities...")
    test_measurements = [6.5, 3.0, 5.5, 1.8]  # Typical Virginica measurements
    
    # Set the model to the best performing one for predictions
    classifier.model = rf_model
    classifier.scaler = scaler
    
    species = classifier.predict_iris_species(*test_measurements)
    probabilities = classifier.predict_proba_iris_species(*test_measurements)
    
    print(f"Predicted species for {test_measurements}: {species}")
    print("Prediction probabilities:")
    for sp, prob in probabilities.items():
        print(f"  {sp}: {prob:.3f}")
    
    return classifier, df

def example_3_visualization():
    """
    Example 3: Using visualization components
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Visualization Components")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 150
    
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    species = ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
    
    df = pd.DataFrame({
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width,
        'Species': species
    })
    
    # Train a model for visualizations
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    rf_model = train_model(X_train, y_train, 'random_forest')
    results = evaluate_model(rf_model, X_test, y_test)
    
    print("Creating visualizations...")
    
    # 1. Pairplot
    print("\n1. Creating pairplot...")
    try:
        fig1 = create_pairplot(df)
        print("Pairplot created successfully!")
        plt.close()  # Close to prevent display in script
    except Exception as e:
        print(f"Pairplot creation failed: {e}")
    
    # 2. Confusion Matrix
    print("\n2. Creating confusion matrix...")
    try:
        fig2 = plot_confusion_matrix(results['confusion_matrix'], df['Species'].unique())
        print("Confusion matrix plot created successfully!")
        plt.close()  # Close to prevent display in script
    except Exception as e:
        print(f"Confusion matrix plot creation failed: {e}")
    
    # 3. Feature Importance (only for tree-based models)
    print("\n3. Creating feature importance plot...")
    feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    try:
        fig3 = plot_feature_importance(rf_model, feature_names)
        print("Feature importance plot created successfully!")
        plt.close()  # Close to prevent display in script
    except Exception as e:
        print(f"Feature importance plot creation failed: {e}")

def example_4_advanced_features():
    """
    Example 4: Advanced features - multiple models and cross-validation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Advanced Features")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 150
    
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    species = ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
    
    df = pd.DataFrame({
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width,
        'Species': species
    })
    
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # 1. Train multiple models
    print("\n1. Training multiple models for comparison...")
    models = train_multiple_models(X_train, y_train)
    
    # Evaluate all models
    print("\nEvaluating all models...")
    for name, model in models.items():
        results = evaluate_model(model, X_test, y_test)
        print(f"{name.title()} - Accuracy: {results['accuracy']:.3f}")
    
    # 2. Cross-validation
    print("\n2. Performing cross-validation...")
    X = scaler.transform(df.drop(['Species'], axis=1))
    y = df['Species']
    
    for name, model in models.items():
        cv_results = cross_validate_model(model, X, y, cv=5)
        print(f"\n{name.title()} Cross-validation:")
        print(f"  Mean accuracy: {cv_results['mean_score']:.3f} (+/- {cv_results['std_score'] * 2:.3f})")

def example_5_input_validation():
    """
    Example 5: Input validation and error handling
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Input Validation and Error Handling")
    print("=" * 60)
    
    print("Testing input validation...")
    
    # Valid inputs
    print("\n1. Testing valid inputs:")
    try:
        validate_input_features(5.1, 3.5, 1.4, 0.2)
        print("✓ Valid inputs accepted")
    except ValueError as e:
        print(f"✗ Unexpected error: {e}")
    
    # Invalid inputs - negative values
    print("\n2. Testing negative values:")
    try:
        validate_input_features(-1.0, 3.5, 1.4, 0.2)
        print("✗ Should have failed!")
    except ValueError as e:
        print(f"✓ Correctly rejected negative values: {e}")
    
    # Invalid inputs - unrealistic values
    print("\n3. Testing unrealistic values:")
    try:
        validate_input_features(15.0, 3.5, 1.4, 0.2)
        print("✗ Should have failed!")
    except ValueError as e:
        print(f"✓ Correctly rejected unrealistic values: {e}")
    
    # Warning for very small values
    print("\n4. Testing very small values (should warn but accept):")
    try:
        validate_input_features(0.05, 0.05, 0.05, 0.05)
        print("✓ Small values accepted with warning")
    except ValueError as e:
        print(f"✗ Unexpected rejection: {e}")

def example_6_complete_production_workflow():
    """
    Example 6: Complete production-ready workflow
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Complete Production-Ready Workflow")
    print("=" * 60)
    
    # Initialize classifier with custom configuration
    config = {
        'test_size': 0.25,
        'random_state': 123,
        'model_params': {'random_state': 123},
        'visualization_params': {'figure_size': (8, 6), 'colormap': 'viridis'}
    }
    
    classifier = IrisClassifier(config)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 150
    
    sepal_length = np.random.normal(5.8, 0.8, n_samples)
    sepal_width = np.random.normal(3.1, 0.4, n_samples)
    petal_length = np.random.normal(3.8, 1.8, n_samples)
    petal_width = np.random.normal(1.2, 0.8, n_samples)
    
    species = ['Iris-setosa'] * 50 + ['Iris-versicolor'] * 50 + ['Iris-virginica'] * 50
    
    df = pd.DataFrame({
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width,
        'Species': species
    })
    
    print("Running complete production workflow...")
    
    # 1. Data validation
    print("\n1. Validating dataset...")
    if df.isnull().sum().sum() > 0:
        print("⚠️  Warning: Dataset contains missing values")
    else:
        print("✓ Dataset validation passed")
    
    # 2. EDA
    eda_results = classifier.perform_eda(df)
    
    # 3. Preprocessing
    X_train, X_test, y_train, y_test, scaler = classifier.preprocess_data(df)
    
    # 4. Model selection and training
    print("\n4. Model selection and training...")
    models = train_multiple_models(X_train, y_train)
    
    # Select best model based on cross-validation
    X_scaled = scaler.transform(df.drop(['Species'], axis=1))
    y = df['Species']
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        cv_results = cross_validate_model(model, X_scaled, y, cv=5)
        if cv_results['mean_score'] > best_score:
            best_score = cv_results['mean_score']
            best_model = model
            best_name = name
    
    print(f"\n✓ Best model selected: {best_name} (CV Score: {best_score:.3f})")
    
    # 5. Final evaluation
    classifier.model = best_model
    classifier.scaler = scaler
    final_results = classifier.evaluate_model(best_model, X_test, y_test)
    
    # 6. Production predictions
    print("\n6. Making production predictions...")
    
    # Batch predictions
    test_cases = [
        [5.1, 3.5, 1.4, 0.2],  # Expected: Setosa
        [7.0, 3.2, 4.7, 1.4],  # Expected: Versicolor  
        [6.3, 3.3, 6.0, 2.5],  # Expected: Virginica
    ]
    
    for i, measurements in enumerate(test_cases, 1):
        try:
            species = classifier.predict_iris_species(*measurements)
            if hasattr(classifier.model, 'predict_proba'):
                probabilities = classifier.predict_proba_iris_species(*measurements)
                max_prob = max(probabilities.values())
                print(f"Test case {i}: {measurements} → {species} (confidence: {max_prob:.3f})")
            else:
                print(f"Test case {i}: {measurements} → {species}")
        except Exception as e:
            print(f"Test case {i}: Failed - {e}")
    
    print(f"\n✓ Production workflow completed successfully!")
    print(f"Model ready for deployment with {final_results['accuracy']:.1%} accuracy")

def main():
    """
    Run all examples to demonstrate the complete API
    """
    print("IRIS CLASSIFICATION API - COMPREHENSIVE USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        # Run all examples
        example_1_basic_workflow()
        example_2_class_based_workflow() 
        example_3_visualization()
        example_4_advanced_features()
        example_5_input_validation()
        example_6_complete_production_workflow()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nThe Iris Classification API provides:")
        print("✓ Complete machine learning pipeline")
        print("✓ Multiple model types (Logistic Regression, Random Forest, SVM)")
        print("✓ Comprehensive evaluation metrics")
        print("✓ Data visualization capabilities")
        print("✓ Input validation and error handling")
        print("✓ Both functional and object-oriented interfaces")
        print("✓ Production-ready workflow")
        
    except Exception as e:
        print(f"\n❌ Error during example execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()