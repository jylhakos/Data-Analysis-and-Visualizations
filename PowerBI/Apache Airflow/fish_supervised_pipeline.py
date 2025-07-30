"""
Fish weight prediction supervised learning pipeline
=================================================

A comprehensive supervised learning project using regression to predict fish weight
based on species and physical measurements. This script demonstrates the complete 
machine learning pipeline including data loading, preprocessing, feature engineering, 
model training, hyperparameter tuning, and evaluation specifically for Fish.csv dataset.

Based on fish_regression.py implementation with enhanced features.

Author: Fish ML Project Team
Date: July 2025
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FishWeightPredictionPipeline:
    """
    A comprehensive class for fish weight prediction using supervised learning regression.
    """
    
    def __init__(self):
        """Initialize the fish weight prediction pipeline."""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.best_pipeline = None
        self.best_params = None
        self.results = {}
        
    def load_and_explore_data(self):
        """Load Fish.csv and perform exploratory data analysis."""
        print("=" * 60)
        print("STEP 1: FISH DATASET LOADING AND EXPLORATION")
        print("=" * 60)
        
        try:
            # Load the Fish dataset
            self.df = pd.read_csv('Dataset/Fish.csv')
            
            print(f"Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            print(f"Features: {list(self.df.columns)}")
            
            # Basic dataset information
            print(f"\nDataset Overview:")
            print(f"   Total fish samples: {len(self.df)}")
            print(f"   Features: {len(self.df.columns) - 1}")
            print(f"   Target: Weight (grams)")
            
            # Species information
            species_counts = self.df['Species'].value_counts()
            print(f"\n🐟 Species Distribution:")
            for species, count in species_counts.items():
                print(f"   {species}: {count} samples")
            
            # Weight statistics
            print(f"\n⚖️  Weight Statistics:")
            print(f"   Mean: {self.df['Weight'].mean():.1f}g")
            print(f"   Std: {self.df['Weight'].std():.1f}g")
            print(f"   Min: {self.df['Weight'].min():.1f}g")
            print(f"   Max: {self.df['Weight'].max():.1f}g")
            
            # Check for missing values
            missing_values = self.df.isnull().sum()
            if missing_values.sum() == 0:
                print(f"\nNo missing values found")
            else:
                print(f"\n⚠️  Missing values found:")
                for col, missing in missing_values.items():
                    if missing > 0:
                        print(f"   {col}: {missing}")
            
            # Physical measurements statistics
            print(f"\nPhysical Measurements:")
            numeric_cols = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
            for col in numeric_cols:
                if col in self.df.columns:
                    print(f"   {col}: {self.df[col].mean():.2f} ± {self.df[col].std():.2f} cm")
            
            return self.df
            
        except FileNotFoundError:
            print("❌ Error: Fish.csv not found in Dataset/ directory")
            print("Please ensure the Fish.csv file is in the Dataset/ folder")
            raise
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            raise
    
    def preprocess_data(self):
        """Preprocess fish data for machine learning."""
        print("\n" + "=" * 60)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 60)
        
        if self.df is None:
            raise ValueError("Dataset not loaded. Run load_and_explore_data() first.")
        
        # Separate features and target
        X = self.df.drop('Weight', axis=1)
        y = self.df['Weight']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Handle categorical features (Species)
        print(f"\nEncoding categorical features...")
        self.label_encoder = LabelEncoder()
        X_processed = X.copy()
        X_processed['Species'] = self.label_encoder.fit_transform(X['Species'])
        
        # Display species encoding
        species_mapping = dict(zip(self.label_encoder.classes_, 
                                 self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"Species encoding:")
        for species, code in species_mapping.items():
            print(f"   {species} -> {code}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=X['Species']
        )
        
        print(f"\nData Split:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Training ratio: {len(self.X_train)/len(X_processed)*100:.1f}%")
        
        # Display species distribution in splits
        train_species = X.iloc[self.X_train.index]['Species'].value_counts()
        test_species = X.iloc[self.X_test.index]['Species'].value_counts()
        
        print(f"\n🐟 Species distribution in training set:")
        for species in train_species.index:
            print(f"   {species}: {train_species[species]} samples")
        
        return X_processed, y
    
    def build_pipeline(self):
        """Build machine learning pipeline for fish weight prediction."""
        print("\n" + "=" * 60)
        print("STEP 3: BUILDING FISH WEIGHT PREDICTION PIPELINE")
        print("=" * 60)
        
        # Create a pipeline optimized for fish weight prediction
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Scale physical measurements
            ('poly_features', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ('feature_selection', SelectKBest(score_func=f_regression, k=15)),  # Select best features
            ('regressor', LinearRegression())  # Regression model
        ])
        
        print("Pipeline components:")
        print("1. StandardScaler - Normalize physical measurements")
        print("2. PolynomialFeatures - Create interaction features between measurements")
        print("3. SelectKBest - Select most relevant features for weight prediction")
        print("4. Regressor - Fish weight prediction model")
        
        return pipeline
    
    def hyperparameter_tuning(self, pipeline):
        """Perform hyperparameter tuning for fish weight prediction."""
        print("\n" + "=" * 60)
        print("STEP 4: HYPERPARAMETER TUNING FOR FISH WEIGHT PREDICTION")
        print("=" * 60)
        
        # Define parameter grids optimized for fish weight prediction
        param_grids = [
            # Linear Regression with different polynomial degrees
            {
                'poly_features__degree': [1, 2],
                'feature_selection__k': [10, 15, 20],
                'regressor': [LinearRegression()],
                'regressor__fit_intercept': [True, False]
            },
            # Ridge Regression for handling overfitting
            {
                'poly_features__degree': [1, 2],
                'feature_selection__k': [10, 15, 20],
                'regressor': [Ridge()],
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            },
            # Lasso Regression for feature selection
            {
                'poly_features__degree': [1, 2],
                'feature_selection__k': [10, 15],
                'regressor': [Lasso()],
                'regressor__alpha': [0.1, 1.0, 10.0]
            },
            # Random Forest for non-linear relationships
            {
                'poly_features__degree': [1],  # RF doesn't need polynomial features
                'feature_selection__k': [10, 15],
                'regressor': [RandomForestRegressor(random_state=42)],
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [5, 10, 15, None],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        ]
        
        best_score = float('-inf')
        best_pipeline = None
        best_params = None
        
        for i, param_grid in enumerate(param_grids):
            print(f"\nTesting parameter grid {i+1}/{len(param_grids)}...")
            regressor_name = type(param_grid['regressor'][0]).__name__
            print(f"Algorithm: {regressor_name}")
            
            # Perform GridSearchCV
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
        
        print(f"\nOVERALL RESULTS:")
        print(f"Best CV Score: {best_score:.4f}")
        print(f"Best Algorithm: {type(best_params['regressor']).__name__}")
        print(f"Best Parameters: {best_params}")
        
        return best_pipeline, best_params
    
    def evaluate_model(self):
        """Evaluate fish weight prediction model."""
        print("\n" + "=" * 60)
        print("STEP 5: FISH WEIGHT PREDICTION MODEL EVALUATION")
        print("=" * 60)
        
        if self.best_pipeline is None:
            raise ValueError("Model has not been trained yet. Run hyperparameter_tuning first.")
        
        # Make predictions
        y_train_pred = self.best_pipeline.predict(self.X_train)
        y_test_pred = self.best_pipeline.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        test_mape = np.mean(np.abs((self.y_test - y_test_pred) / self.y_test)) * 100
        
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
            'test_mape': test_mape,
            'y_test_pred': y_test_pred
        }
        
        print("📊 FISH WEIGHT PREDICTION PERFORMANCE:")
        print("-" * 50)
        print(f"Training Set:")
        print(f"  Mean Squared Error (MSE): {train_mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {train_rmse:.2f}g")
        print(f"  Mean Absolute Error (MAE): {train_mae:.2f}g")
        print(f"  R-squared (R²): {train_r2:.4f}")
        
        print(f"\nTest Set:")
        print(f"  Mean Squared Error (MSE): {test_mse:.2f}")
        print(f"  Root Mean Squared Error (RMSE): {test_rmse:.2f}g")
        print(f"  Mean Absolute Error (MAE): {test_mae:.2f}g")
        print(f"  R-squared (R²): {test_r2:.4f}")
        print(f"  Mean Absolute Percentage Error (MAPE): {test_mape:.2f}%")
        
        # Performance assessment specific to fish weight prediction
        print(f"\n🔍 FISH WEIGHT PREDICTION ANALYSIS:")
        overfitting_threshold = 0.1
        r2_diff = train_r2 - test_r2
        print(f"R² difference (Train - Test): {r2_diff:.4f}")
        
        if r2_diff > overfitting_threshold:
            print("⚠️  Potential overfitting detected!")
        else:
            print("Good model generalization!")
        
        if test_r2 > 0.9:
            print("Outstanding fish weight prediction!")
        elif test_r2 > 0.85:
            print("Excellent fish weight prediction!")
        elif test_r2 > 0.7:
            print("Good fish weight prediction!")
        elif test_r2 > 0.5:
            print("⚡ Moderate fish weight prediction!")
        else:
            print("Fish weight prediction needs improvement!")
        
        # Business-relevant insights
        print(f"\nBUSINESS INSIGHTS:")
        print(f"   Average prediction error: {test_mae:.1f}g")
        print(f"   Prediction accuracy: {(1 - test_mape/100)*100:.1f}%")
        print(f"   Model explains {test_r2*100:.1f}% of weight variance")
        
        return self.results
    
    def plot_results(self):
        """Create visualizations for fish weight prediction evaluation."""
        print("\n" + "=" * 60)
        print("STEP 6: FISH WEIGHT PREDICTION VISUALIZATION")
        print("=" * 60)
        
        if not self.results:
            raise ValueError("Model has not been evaluated yet. Run evaluate_model first.")
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fish Weight Prediction Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.y_test, self.results['y_test_pred'], alpha=0.7, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Weight (g)')
        axes[0, 0].set_ylabel('Predicted Weight (g)')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.results["test_r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.y_test - self.results['y_test_pred']
        axes[0, 1].scatter(self.results['y_test_pred'], residuals, alpha=0.7, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Weight (g)')
        axes[0, 1].set_ylabel('Residuals (g)')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        axes[0, 2].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Residuals (g)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Residuals')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Weight distribution by species
        species_test = self.df.iloc[self.X_test.index]['Species']
        unique_species = species_test.unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_species)))
        
        for i, species in enumerate(unique_species):
            species_mask = species_test == species
            species_actual = self.y_test[species_mask]
            species_pred = self.results['y_test_pred'][species_mask]
            axes[1, 0].scatter(species_actual, species_pred, 
                             alpha=0.7, label=species, color=colors[i])
        
        axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Weight (g)')
        axes[1, 0].set_ylabel('Predicted Weight (g)')
        axes[1, 0].set_title('Predictions by Species')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error by weight range
        weight_ranges = pd.cut(self.y_test, bins=5)
        error_by_range = []
        range_labels = []
        
        for range_val in weight_ranges.cat.categories:
            mask = weight_ranges == range_val
            if mask.sum() > 0:
                range_errors = np.abs(self.y_test[mask] - self.results['y_test_pred'][mask])
                error_by_range.append(range_errors.mean())
                range_labels.append(f"{range_val.left:.0f}-{range_val.right:.0f}g")
        
        axes[1, 1].bar(range_labels, error_by_range, alpha=0.7, color='skyblue')
        axes[1, 1].set_xlabel('Weight Range')
        axes[1, 1].set_ylabel('Mean Absolute Error (g)')
        axes[1, 1].set_title('Prediction Error by Weight Range')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance metrics comparison
        metrics = ['RMSE (g)', 'MAE (g)', 'R²', 'MAPE (%)']
        train_values = [self.results['train_rmse'], self.results['train_mae'], 
                       self.results['train_r2'], 0]  # MAPE not calculated for training
        test_values = [self.results['test_rmse'], self.results['test_mae'], 
                      self.results['test_r2'], self.results['test_mape']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, train_values, width, label='Training', alpha=0.8, color='lightblue')
        axes[1, 2].bar(x + width/2, test_values, width, label='Test', alpha=0.8, color='lightcoral')
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Values')
        axes[1, 2].set_title('Training vs Test Performance')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(metrics)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fish_weight_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Fish weight prediction visualizations created and saved as 'fish_weight_prediction_results.png'")
    
    def make_sample_predictions(self):
        """Make sample predictions on different fish species."""
        print("\n" + "=" * 60)
        print("STEP 7: SAMPLE FISH WEIGHT PREDICTIONS")
        print("=" * 60)
        
        if self.best_pipeline is None:
            raise ValueError("Model has not been trained yet.")
        
        # Get sample fish from each species in test set
        test_indices = self.X_test.index
        species_samples = {}
        
        for species in self.df['Species'].unique():
            species_test_fish = self.df.loc[test_indices][self.df.loc[test_indices]['Species'] == species]
            if len(species_test_fish) > 0:
                sample_fish = species_test_fish.iloc[0]
                species_samples[species] = sample_fish
        
        print("🐟 Sample fish weight predictions:")
        print("-" * 60)
        
        for species, fish in species_samples.items():
            # Prepare features for prediction
            features = fish.drop('Weight')
            features_encoded = features.copy()
            features_encoded['Species'] = self.label_encoder.transform([features['Species']])[0]
            
            # Make prediction
            predicted_weight = self.best_pipeline.predict([features_encoded.values])[0]
            actual_weight = fish['Weight']
            error = abs(predicted_weight - actual_weight)
            error_pct = (error / actual_weight) * 100
            
            print(f"{species:<15}: "
                  f"Actual={actual_weight:>6.1f}g, "
                  f"Predicted={predicted_weight:>6.1f}g, "
                  f"Error={error:>5.1f}g ({error_pct:>4.1f}%)")
    
    def save_model(self, filename='fish_weight_prediction_model.pkl'):
        """Save the trained fish weight prediction model."""
        if self.best_pipeline is None:
            raise ValueError("Model has not been trained yet.")
        
        # Save model and preprocessing components
        model_package = {
            'model': self.best_pipeline,
            'label_encoder': self.label_encoder,
            'best_params': self.best_params,
            'results': self.results,
            'feature_names': list(self.X_train.columns)
        }
        
        joblib.dump(model_package, filename)
        print(f"\nFish weight prediction model saved as '{filename}'")
        
        # Save model information
        model_info = {
            'model_type': 'fish_weight_prediction',
            'dataset': 'Fish.csv',
            'best_algorithm': type(self.best_params['regressor']).__name__,
            'best_params': self.best_params,
            'test_r2_score': self.results.get('test_r2', 'Not evaluated'),
            'test_rmse': self.results.get('test_rmse', 'Not evaluated'),
            'test_mape': self.results.get('test_mape', 'Not evaluated'),
            'species_encoded': dict(zip(self.label_encoder.classes_, 
                                      self.label_encoder.transform(self.label_encoder.classes_)))
        }
        
        import json
        with open('fish_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"Model information saved as 'fish_model_info.json'")
    
    def run_full_pipeline(self):
        """Run the complete fish weight prediction pipeline."""
        print("🐟 STARTING FISH WEIGHT PREDICTION PIPELINE")
        print("=" * 70)
        
        try:
            # Step 1: Load and explore data
            self.load_and_explore_data()
            
            # Step 2: Preprocess data
            X, y = self.preprocess_data()
            
            # Step 3: Build pipeline
            pipeline = self.build_pipeline()
            
            # Step 4: Hyperparameter tuning
            best_pipeline, best_params = self.hyperparameter_tuning(pipeline)
            
            # Step 5: Evaluate model
            results = self.evaluate_model()
            
            # Step 6: Create visualizations
            self.plot_results()
            
            # Step 7: Sample predictions
            self.make_sample_predictions()
            
            # Step 8: Save model
            self.save_model()
            
            print("\n" + "=" * 70)
            print("✅ FISH WEIGHT PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"Final Model Performance:")
            print(f"   R² Score: {results['test_r2']:.4f}")
            print(f"   RMSE: {results['test_rmse']:.2f}g")
            print(f"   MAE: {results['test_mae']:.2f}g")
            print(f"   MAPE: {results['test_mape']:.2f}%")
            
            return self.best_pipeline, self.results
            
        except Exception as e:
            print(f"\n❌ Error in fish weight prediction pipeline: {str(e)}")
            raise


def main():
    """Main function to run the fish weight prediction pipeline."""
    # Initialize and run the pipeline
    pipeline = FishWeightPredictionPipeline()
    best_model, results = pipeline.run_full_pipeline()
    
    print("\n" + "=" * 60)
    print("🐟 FISH WEIGHT PREDICTION SUMMARY")
    print("=" * 60)
    print(f"The model can predict fish weight with:")
    print(f"  • {results['test_r2']*100:.1f}% accuracy (R²)")
    print(f"  • Average error of {results['test_mae']:.1f} grams")
    print(f"  • {results['test_mape']:.1f}% mean percentage error")
    print(f"\nThis model is ready for production use in fish weight estimation!")


if __name__ == "__main__":
    main()
