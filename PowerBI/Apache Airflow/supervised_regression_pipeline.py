"""
Supervised learning regression pipeline
=======================================

A supervised learning project using regression to predict target values.
This script demonstrates the complete machine learning pipeline including data loading,
preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

Author: ML
Date: July 2025
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SupervisedRegressionPipeline:
    """
    A comprehensive class for supervised learning regression tasks.
    """
    
    def __init__(self, dataset_type='california_housing'):
        """
        Initialize the regression pipeline.
        
        Args:
            dataset_type (str): Type of dataset to use ('california_housing' or 'fish')
        """
        self.dataset_type = dataset_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_pipeline = None
        self.best_params = None
        self.results = {}
        
    def load_data(self):
        """
        Load and split data into training and testing sets.
        """
        print("=" * 60)
        print("STEP 1: DATA LOADING AND SPLITTING")
        print("=" * 60)
        
        if self.dataset_type == 'california_housing':
            # Load California Housing dataset
            housing = fetch_california_housing(as_frame=True)
            X, y = housing.data, housing.target
            print("Dataset: California Housing")
            print(f"Features: {list(X.columns)}")
            print(f"Target: Median house value")
            
        elif self.dataset_type == 'fish':
            # Load Fish dataset if available
            try:
                df = pd.read_csv('Dataset/Fish.csv')
                X = df.drop('Weight', axis=1)
                y = df['Weight']
                print("Dataset: Fish Weight Prediction")
                print(f"Features: {list(X.columns)}")
                print(f"Target: Weight")
                
                # Handle categorical features for fish dataset
                from sklearn.preprocessing import LabelEncoder
                if 'Species' in X.columns:
                    le = LabelEncoder()
                    X['Species'] = le.fit_transform(X['Species'])
                    
            except FileNotFoundError:
                print("Fish dataset not found, using California Housing dataset instead.")
                housing = fetch_california_housing(as_frame=True)
                X, y = housing.data, housing.target
                self.dataset_type = 'california_housing'
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nDataset shape: {X.shape}")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {X.shape[1]}")
        
        # Basic statistics
        print(f"\nTarget variable statistics:")
        print(f"Mean: {y.mean():.2f}")
        print(f"Std: {y.std():.2f}")
        print(f"Min: {y.min():.2f}")
        print(f"Max: {y.max():.2f}")
        
        return X, y
    
    def build_pipeline(self):
        """
        Build the machine learning pipeline with preprocessing, feature engineering, and model.
        """
        print("\n" + "=" * 60)
        print("STEP 2: BUILDING MACHINE LEARNING PIPELINE")
        print("=" * 60)
        
        # Create a pipeline with preprocessing, feature engineering, and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Preprocessing: Feature scaling
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),  # Feature Engineering
            ('feature_selection', SelectKBest(score_func=f_regression, k=5)),  # Feature Selection
            ('regressor', LinearRegression())  # Model training
        ])
        
        print("Pipeline components:")
        print("1. StandardScaler - Feature scaling")
        print("2. PolynomialFeatures - Feature engineering (polynomial features)")
        print("3. SelectKBest - Feature selection (top K features)")
        print("4. LinearRegression - Regression model")
        
        return pipeline
    
    def hyperparameter_tuning(self, pipeline):
        """
        Perform hyperparameter tuning using GridSearchCV.
        """
        print("\n" + "=" * 60)
        print("STEP 3: HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Define multiple algorithms to test
        param_grids = [
            # Linear Regression with polynomial features
            {
                'poly_features__degree': [1, 2, 3],
                'feature_selection__k': [5, 8, 10, 15],
                'regressor': [LinearRegression()],
                'regressor__fit_intercept': [True, False]
            },
            # Ridge Regression
            {
                'poly_features__degree': [1, 2],
                'feature_selection__k': [5, 8, 10],
                'regressor': [Ridge()],
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            # Lasso Regression
            {
                'poly_features__degree': [1, 2],
                'feature_selection__k': [5, 8, 10],
                'regressor': [Lasso()],
                'regressor__alpha': [0.1, 1.0, 10.0]
            },
            # Random Forest (no polynomial features needed)
            {
                'poly_features__degree': [1],  # Keep degree=1 for RF
                'feature_selection__k': [5, 8, 10],
                'regressor': [RandomForestRegressor(random_state=42)],
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [5, 10, None]
            }
        ]
        
        best_score = float('-inf')
        best_pipeline = None
        best_params = None
        
        for i, param_grid in enumerate(param_grids):
            print(f"\nTesting parameter grid {i+1}/{len(param_grids)}...")
            regressor_name = type(param_grid['regressor'][0]).__name__
            print(f"Algorithm: {regressor_name}")
            
            # Perform GridSearchCV for each parameter grid
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=5, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")
        
        self.best_pipeline = best_pipeline
        self.best_params = best_params
        
        print(f"\n🏆 OVERALL BEST RESULTS:")
        print(f"Best CV Score: {best_score:.4f}")
        print(f"Best Algorithm: {type(best_params['regressor']).__name__}")
        print(f"Best Parameters: {best_params}")
        
        return best_pipeline, best_params
    
    def evaluate_model(self):
        """
        Evaluate the best model on training and test sets.
        """
        print("\n" + "=" * 60)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 60)
        
        if self.best_pipeline is None:
            raise ValueError("Model has not been trained yet. Run hyperparameter_tuning first.")
        
        # Make predictions on both training and test sets
        y_train_pred = self.best_pipeline.predict(self.X_train)
        y_test_pred = self.best_pipeline.predict(self.X_test)
        
        # Calculate metrics for training set
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # Calculate metrics for test set
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Store results
        self.results = {
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'y_test_pred': y_test_pred
        }
        
        print("📊 MODEL PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"Training Set:")
        print(f"  Mean Squared Error (MSE): {train_mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {train_rmse:.2f}")
        print(f"  Mean Absolute Error (MAE): {train_mae:.2f}")
        print(f"  R-squared (R²): {train_r2:.4f}")
        
        print(f"\nTest Set:")
        print(f"  Mean Squared Error (MSE): {test_mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {test_rmse:.2f}")
        print(f"  Mean Absolute Error (MAE): {test_mae:.2f}")
        print(f"  R-squared (R²): {test_r2:.4f}")
        
        # Check for overfitting
        overfitting_threshold = 0.1
        r2_diff = train_r2 - test_r2
        
        print(f"\n🔍 MODEL ANALYSIS:")
        print(f"R² difference (Train - Test): {r2_diff:.4f}")
        
        if r2_diff > overfitting_threshold:
            print("⚠️  Potential overfitting detected!")
            print("   Consider reducing model complexity or getting more data.")
        else:
            print("✅ Good model generalization!")
        
        if test_r2 > 0.8:
            print("🎯 Excellent model performance!")
        elif test_r2 > 0.6:
            print("👍 Good model performance!")
        elif test_r2 > 0.4:
            print("⚡ Moderate model performance!")
        else:
            print("📈 Room for improvement in model performance!")
        
        return self.results
    
    def plot_results(self):
        """
        Create visualizations for model evaluation.
        """
        print("\n" + "=" * 60)
        print("STEP 5: VISUALIZATION")
        print("=" * 60)
        
        if not self.results:
            raise ValueError("Model has not been evaluated yet. Run evaluate_model first.")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.y_test, self.results['y_test_pred'], alpha=0.7, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.results["test_r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.y_test - self.results['y_test_pred']
        axes[0, 1].scatter(self.results['y_test_pred'], residuals, alpha=0.7, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison
        metrics = ['MSE', 'RMSE', 'MAE', 'R²']
        train_values = [self.results['train_mse'], self.results['train_rmse'], 
                       self.results['train_mae'], self.results['train_r2']]
        test_values = [self.results['test_mse'], self.results['test_rmse'], 
                      self.results['test_mae'], self.results['test_r2']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Training vs Test Performance')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Visualizations created and saved as 'model_evaluation_results.png'")
    
    def save_model(self, filename='best_regression_model.pkl'):
        """
        Save the trained model to disk.
        """
        if self.best_pipeline is None:
            raise ValueError("Model has not been trained yet.")
        
        joblib.dump(self.best_pipeline, filename)
        print(f"\n💾 Model saved as '{filename}'")
        
        # Save model information
        model_info = {
            'dataset_type': self.dataset_type,
            'best_params': self.best_params,
            'test_r2_score': self.results.get('test_r2', 'Not evaluated'),
            'test_rmse': self.results.get('test_rmse', 'Not evaluated')
        }
        
        import json
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"📋 Model information saved as 'model_info.json'")
    
    def run_full_pipeline(self):
        """
        Run the complete machine learning pipeline.
        """
        print("🚀 STARTING SUPERVISED LEARNING REGRESSION PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load data
            X, y = self.load_data()
            
            # Step 2: Build pipeline
            pipeline = self.build_pipeline()
            
            # Step 3: Hyperparameter tuning
            best_pipeline, best_params = self.hyperparameter_tuning(pipeline)
            
            # Step 4: Evaluate model
            results = self.evaluate_model()
            
            # Step 5: Create visualizations
            self.plot_results()
            
            # Step 6: Save model
            self.save_model()
            
            print("\n" + "=" * 70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            
            return self.best_pipeline, self.results
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {str(e)}")
            raise


def main():
    """
    Main function to run the supervised learning regression pipeline.
    """
    # Initialize and run the pipeline
    pipeline = SupervisedRegressionPipeline(dataset_type='california_housing')
    best_model, results = pipeline.run_full_pipeline()
    
    # Example of making predictions on new data
    print("\n" + "=" * 60)
    print("EXAMPLE: MAKING PREDICTIONS ON NEW DATA")
    print("=" * 60)
    
    # Create some example new data (using first few samples from test set)
    new_data = pipeline.X_test.iloc[:5]
    predictions = best_model.predict(new_data)
    actual_values = pipeline.y_test.iloc[:5].values
    
    print("Sample predictions:")
    for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
        print(f"Sample {i+1}: Predicted = {pred:.2f}, Actual = {actual:.2f}, "
              f"Error = {abs(pred - actual):.2f}")


if __name__ == "__main__":
    main()
