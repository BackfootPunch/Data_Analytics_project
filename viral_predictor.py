import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ViralHitPredictor:
    def __init__(self, genre_data_file: str = 'genre_evolution.csv', 
                 sentiment_data_file: str = 'sentiment_analysis.csv'):
        """Initialize the viral hit predictor."""
        self.genre_df = pd.read_csv(genre_data_file)
        try:
            self.sentiment_df = pd.read_csv(sentiment_data_file)
        except FileNotFoundError:
            print("Warning: sentiment_analysis.csv not found. Will proceed without sentiment features.")
            self.sentiment_df = None
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare and engineer features for the model."""
        print("Preparing data for viral hit prediction...")
        
        # Define virality threshold (top 15% popularity as viral)
        popularity_threshold = self.genre_df['popularity'].quantile(0.85)
        self.genre_df['is_viral'] = (self.genre_df['popularity'] > popularity_threshold).astype(int)
        
        print(f"Viral threshold set at popularity > {popularity_threshold:.1f}")
        print(f"Viral tracks: {self.genre_df['is_viral'].sum()} / {len(self.genre_df)} ({self.genre_df['is_viral'].mean()*100:.1f}%)")
        
        # Extract year from release date
        self.genre_df['release_date'] = pd.to_datetime(self.genre_df['release_date'])
        self.genre_df['release_year'] = self.genre_df['release_date'].dt.year
        self.genre_df['release_month'] = self.genre_df['release_date'].dt.month
        
        # Create decade numeric feature
        decade_mapping = {'1980s': 1980, '2000s': 2000, '2020s': 2020}
        self.genre_df['decade_numeric'] = self.genre_df['decade'].map(decade_mapping)
        
        # Audio feature engineering
        self.create_audio_features()
        
        # Merge with sentiment data if available
        if self.sentiment_df is not None:
            self.merge_sentiment_features()
        
        # Select features for modeling
        self.select_features()
        
        print(f"Final dataset shape: {self.X.shape}")
        print(f"Features: {self.feature_names}")
    
    def create_audio_features(self):
        """Create additional audio-based features."""
        # Normalize tempo and loudness if not already done
        if 'tempo_normalized' not in self.genre_df.columns:
            self.genre_df['tempo_normalized'] = (
                (self.genre_df['tempo'] - self.genre_df['tempo'].min()) / 
                (self.genre_df['tempo'].max() - self.genre_df['tempo'].min())
            )
        
        if 'loudness_normalized' not in self.genre_df.columns:
            self.genre_df['loudness_normalized'] = (
                (self.genre_df['loudness'] - self.genre_df['loudness'].min()) / 
                (self.genre_df['loudness'].max() - self.genre_df['loudness'].min())
            )
        
        # Create composite features
        self.genre_df['energy_danceability'] = self.genre_df['energy'] * self.genre_df['danceability']
        self.genre_df['valence_energy'] = self.genre_df['valence'] * self.genre_df['energy']
        self.genre_df['acoustic_energy_ratio'] = self.genre_df['acousticness'] / (self.genre_df['energy'] + 0.001)
        
        # Create categorical features for key and mode
        self.genre_df['key_mode'] = self.genre_df['key'].astype(str) + '_' + self.genre_df['mode'].astype(str)
        
        # Create "danceability index" (combination of multiple features)
        self.genre_df['danceability_index'] = (
            self.genre_df['danceability'] * 0.4 +
            self.genre_df['energy'] * 0.3 +
            self.genre_df['valence'] * 0.2 +
            self.genre_df['tempo_normalized'] * 0.1
        )
    
    def merge_sentiment_features(self):
        """Merge sentiment analysis features."""
        # Merge on track_id if available, otherwise on track name and artist
        if 'id' in self.genre_df.columns and 'track_id' in self.sentiment_df.columns:
            merged = self.genre_df.merge(
                self.sentiment_df[['track_id', 'polarity', 'subjectivity', 'word_count', 'unique_words']],
                left_on='id', right_on='track_id', how='left'
            )
        else:
            # Fallback to name and artist matching
            merged = self.genre_df.merge(
                self.sentiment_df[['track_name', 'artist', 'polarity', 'subjectivity', 'word_count', 'unique_words']],
                left_on=['name', 'artist'], right_on=['track_name', 'artist'], how='left'
            )
        
        # Fill missing sentiment values with neutral values
        sentiment_columns = ['polarity', 'subjectivity', 'word_count', 'unique_words']
        for col in sentiment_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(merged[col].median())
        
        self.genre_df = merged
        print(f"Merged sentiment data. Missing sentiment for {merged[sentiment_columns].isnull().any(axis=1).sum()} tracks")
    
    def select_features(self):
        """Select and prepare features for modeling."""
        # Audio features
        audio_features = [
            'danceability', 'energy', 'speechiness', 'acousticness',
            'instrumentalness', 'liveness', 'valence', 'tempo_normalized', 'loudness_normalized'
        ]
        
        # Engineered audio features
        engineered_features = [
            'energy_danceability', 'valence_energy', 'acoustic_energy_ratio', 'danceability_index'
        ]
        
        # Temporal features
        temporal_features = ['release_year', 'release_month', 'decade_numeric']
        
        # Sentiment features (if available)
        sentiment_features = []
        if self.sentiment_df is not None:
            sentiment_features = ['polarity', 'subjectivity', 'word_count', 'unique_words']
            sentiment_features = [f for f in sentiment_features if f in self.genre_df.columns]
        
        # Categorical features
        categorical_features = ['genre', 'key_mode']
        
        # Combine all features
        feature_columns = audio_features + engineered_features + temporal_features + sentiment_features
        
        # Remove any features that don't exist in the dataframe
        feature_columns = [f for f in feature_columns if f in self.genre_df.columns]
        
        # Prepare feature matrix
        X_numeric = self.genre_df[feature_columns].copy()
        
        # Handle categorical features
        X_categorical = pd.DataFrame()
        for cat_feature in categorical_features:
            if cat_feature in self.genre_df.columns:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    encoded = self.label_encoders[cat_feature].fit_transform(self.genre_df[cat_feature].astype(str))
                else:
                    encoded = self.label_encoders[cat_feature].transform(self.genre_df[cat_feature].astype(str))
                
                X_categorical[f'{cat_feature}_encoded'] = encoded
        
        # Combine numeric and categorical features
        self.X = pd.concat([X_numeric, X_categorical], axis=1)
        self.y = self.genre_df['is_viral']
        self.feature_names = list(self.X.columns)
        
        # Handle any remaining missing values
        self.X = self.X.fillna(self.X.median())
        
        print(f"Selected {len(self.feature_names)} features for modeling")
    
    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """Train the Random Forest model with hyperparameter tuning."""
        print("Training viral hit prediction model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        # Use a smaller grid for faster training
        param_grid_small = {
            'n_estimators': [200],
            'max_depth': [15, None],
            'min_samples_split': [5],
            'min_samples_leaf': [2],
            'class_weight': ['balanced']
        }
        
        print("Performing hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=random_state)
        grid_search = GridSearchCV(
            rf, param_grid_small, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nTest Set Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store test data for SHAP analysis
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'best_params': grid_search.best_params_,
            'feature_importance': feature_importance
        }
    
    def explain_with_shap(self, num_samples: int = 100) -> shap.TreeExplainer:
        """Generate SHAP explanations for the model."""
        print("Generating SHAP explanations...")
        
        if self.model is None:
            raise ValueError("Model must be trained before generating SHAP explanations")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values for a sample of test data
        sample_size = min(num_samples, len(self.X_test))
        sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
        X_sample = self.X_test[sample_indices]
        
        shap_values = explainer.shap_values(X_sample)
        
        # If binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (viral)
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.title('SHAP Feature Importance for Viral Hit Prediction')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create waterfall plot for first sample
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            explainer.expected_value[1], shap_values[0], X_sample[0], 
            feature_names=self.feature_names, show=False
        )
        plt.title('SHAP Waterfall Plot - Example Prediction')
        plt.tight_layout()
        plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate mean absolute SHAP values for feature ranking
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap_value': mean_shap_values
        }).sort_values('mean_shap_value', ascending=False)
        
        print("Top 10 Features by SHAP Importance:")
        print(shap_importance.head(10))
        
        # Save SHAP importance
        shap_importance.to_csv('shap_feature_importance.csv', index=False)
        
        print("SHAP analysis completed. Plots saved as:")
        print("- shap_summary_plot.png")
        print("- shap_waterfall_plot.png")
        print("- shap_feature_importance.csv")
        
        return explainer
    
    def predict_virality(self, track_features: Dict) -> Dict:
        """Predict virality for a new track."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input to DataFrame
        feature_df = pd.DataFrame([track_features])
        
        # Handle categorical encoding
        for cat_feature, encoder in self.label_encoders.items():
            original_feature = cat_feature.replace('_encoded', '')
            if original_feature in feature_df.columns:
                try:
                    encoded_value = encoder.transform([str(feature_df[original_feature].iloc[0])])[0]
                    feature_df[cat_feature] = encoded_value
                except ValueError:
                    # Handle unseen categories
                    feature_df[cat_feature] = 0
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in feature_df.columns:
                feature_df[feature] = 0  # Default value for missing features
        
        # Select and order features
        feature_df = feature_df[self.feature_names]
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_df)
        
        # Make prediction
        probability = self.model.predict_proba(feature_scaled)[0, 1]
        prediction = self.model.predict(feature_scaled)[0]
        
        return {
            'viral_probability': probability,
            'is_viral_prediction': bool(prediction),
            'confidence': max(probability, 1 - probability)
        }
    
    def save_model(self, filename: str = 'viral_predictor.pkl'):
        """Save the trained model and preprocessing objects."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename: str = 'viral_predictor.pkl'):
        """Load a pre-trained model."""
        model_data = joblib.load(filename)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filename}")
    
    def create_performance_report(self, results: Dict):
        """Create a detailed performance report."""
        report = []
        report.append("=== VIRAL HIT PREDICTOR PERFORMANCE REPORT ===\n")
        
        report.append(f"Model Type: Random Forest Classifier")
        report.append(f"Test Accuracy: {results['accuracy']:.3f}")
        report.append(f"ROC AUC Score: {results['roc_auc']:.3f}")
        report.append(f"Best Parameters: {results['best_params']}")
        report.append("")
        
        report.append("TOP 10 MOST IMPORTANT FEATURES:")
        for idx, row in results['feature_importance'].head(10).iterrows():
            report.append(f"{row['feature']}: {row['importance']:.4f}")
        report.append("")
        
        # Dataset statistics
        viral_count = self.y.sum()
        total_count = len(self.y)
        report.append("DATASET STATISTICS:")
        report.append(f"Total tracks: {total_count}")
        report.append(f"Viral tracks: {viral_count} ({viral_count/total_count*100:.1f}%)")
        report.append(f"Non-viral tracks: {total_count - viral_count} ({(total_count - viral_count)/total_count*100:.1f}%)")
        report.append("")
        
        # Feature statistics
        report.append("FEATURE SUMMARY:")
        report.append(f"Total features used: {len(self.feature_names)}")
        report.append(f"Audio features: {len([f for f in self.feature_names if f in ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo_normalized', 'loudness_normalized']])}")
        report.append(f"Engineered features: {len([f for f in self.feature_names if f in ['energy_danceability', 'valence_energy', 'acoustic_energy_ratio', 'danceability_index']])}")
        
        with open('viral_predictor_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("Performance report saved as viral_predictor_report.txt")


def main():
    """Main function to train the viral hit predictor."""
    try:
        # Initialize predictor
        predictor = ViralHitPredictor()
        
        # Train model
        results = predictor.train_model()
        
        # Generate SHAP explanations
        explainer = predictor.explain_with_shap()
        
        # Save model
        predictor.save_model()
        
        # Create performance report
        predictor.create_performance_report(results)
        
        print("\nViral hit predictor training completed!")
        print("Generated files:")
        print("- viral_predictor.pkl")
        print("- shap_summary_plot.png")
        print("- shap_waterfall_plot.png")
        print("- shap_feature_importance.csv")
        print("- viral_predictor_report.txt")
        
        # Example prediction
        print("\n=== EXAMPLE PREDICTION ===")
        example_track = {
            'danceability': 0.7,
            'energy': 0.8,
            'speechiness': 0.1,
            'acousticness': 0.2,
            'instrumentalness': 0.1,
            'liveness': 0.3,
            'valence': 0.6,
            'tempo_normalized': 0.7,
            'loudness_normalized': 0.8,
            'energy_danceability': 0.56,
            'valence_energy': 0.48,
            'acoustic_energy_ratio': 0.25,
            'danceability_index': 0.69,
            'release_year': 2023,
            'release_month': 6,
            'decade_numeric': 2020,
            'genre': 'k-pop',
            'key_mode': '1_1'
        }
        
        if predictor.sentiment_df is not None:
            example_track.update({
                'polarity': 0.2,
                'subjectivity': 0.6,
                'word_count': 150,
                'unique_words': 80
            })
        
        prediction = predictor.predict_virality(example_track)
        print(f"Example track viral probability: {prediction['viral_probability']:.3f}")
        print(f"Predicted as viral: {prediction['is_viral_prediction']}")
        print(f"Confidence: {prediction['confidence']:.3f}")
        
    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}")
        print("Please run data_collection.py and nlp_analysis.py first.")
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()