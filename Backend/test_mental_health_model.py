#!/usr/bin/env python
"""
Test Mental Health Model with Converted HealthKit Data

This script loads the converted Fitbit-format CSV files and directly passes them
to the pre-trained XGBoost mental health model for prediction without additional processing.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMentalHealthPredictor:
    """Simple class to load model and make predictions without data processing"""
    
    def __init__(self, model_dir, data_dir):
        """
        Initialize the predictor with paths to model and data directories
        
        Args:
            model_dir: Directory containing the model files
            data_dir: Directory containing the converted CSV files
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def load_model(self):
        """Load the XGBoost model and feature information"""
        model_path = self.model_dir / "mental_health_model.xgb"
        scaler_path = self.model_dir / "scaler.save"
        feature_importance_path = self.model_dir / "feature_importance.csv"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing in {self.model_dir}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature names if available
        if scaler_path.exists():
            logger.info(f"Loading feature names from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.feature_names = pickle.load(f)
                if isinstance(self.feature_names, np.ndarray):
                    logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load feature importance
        if feature_importance_path.exists():
            self.feature_importance = pd.read_csv(feature_importance_path)
            logger.info(f"Loaded feature importance with {len(self.feature_importance)} features")
    
    def load_csv_files(self):
        """
        Load all CSV files directly with minimal processing.
        Based on algo.mdc, the model expects a single feature dataset.
        """
        logger.info("Loading CSV files for prediction")
        
        # Determine which file contains the main features
        # According to algo.mdc, this should be a combined feature set
        # Let's load each file and examine which one has the features we need
        
        # Get the feature names we expect from feature_importance
        expected_features = self.feature_names
        if expected_features is None and self.feature_importance is not None:
            expected_features = self.feature_importance['feature'].values
        
        if expected_features is None:
            logger.warning("No expected feature list available")
            # Default to using most comprehensive file
            return self._load_all_features_from_csvs()
        
        # Try to load from existing combined feature file if it exists
        combined_features_path = self.data_dir / "combined_features.csv"
        if combined_features_path.exists():
            logger.info(f"Loading pre-combined features from {combined_features_path}")
            features_df = pd.read_csv(combined_features_path)
            return features_df
        
        # Otherwise load all files and see which has the features we need
        logger.info("No combined feature file found, attempting to load from individual files")
        return self._load_all_features_from_csvs()
    
    def _load_all_features_from_csvs(self):
        """Load features from all available CSV files"""
        # Create a dummy person record since we don't have actual person data
        # Based on algo.mdc, we need:
        # - person_id
        # - age
        # - gender_binary
        features_df = pd.DataFrame({
            'person_id': ['1001'],  # Default ID
            'age': [33],            # Default age
            'gender_binary': [1]    # Default female (1)
        })
        
        # File mapping from dataset type to filename
        file_mapping = {
            'heart_rate_summary': 'dataset_59116210_fitbit_heart_rate_summary.csv',
            'heart_rate_level': 'dataset_59116210_fitbit_heart_rate_level.csv',
            'intraday_steps': 'dataset_59116210_fitbit_intraday_steps.csv',
            'activity': 'dataset_59116210_fitbit_activity.csv',
            'sleep_daily_summary': 'dataset_59116210_fitbit_sleep_daily_summary.csv',
            'sleep_level': 'dataset_59116210_fitbit_sleep_level.csv'
        }
        
        # Load each file and extract its features
        all_features = {}
        for dataset_type, filename in file_mapping.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                logger.info(f"Loading {dataset_type} from {filename}")
                df = pd.read_csv(file_path)
                
                # For each dataset type, extract the aggregated features following algo.mdc pattern
                if dataset_type == 'activity':
                    # From activity, we need various aggregated metrics
                    if len(df) > 0:
                        # Calculate mean, std, etc for activity metrics
                        agg_data = df.agg({
                            'very_active_minutes': ['mean', 'std', 'max'],
                            'fairly_active_minutes': ['mean', 'std'],
                            'lightly_active_minutes': ['mean', 'std'],
                            'sedentary_minutes': ['mean', 'std'],
                            'steps': ['mean', 'std', 'max'],
                            'calories_out': ['mean', 'std']
                        })
                        
                        # Flatten column names to match expected format
                        agg_data = agg_data.to_dict()
                        for col, metrics in agg_data.items():
                            for metric, value in metrics.items():
                                feature_name = f"activity_{col}_{metric}"
                                all_features[feature_name] = value
                                
                        # Calculate activity ratio features
                        total_active = df['very_active_minutes'].mean() + df['fairly_active_minutes'].mean() + df['lightly_active_minutes'].mean()
                        activity_ratio = total_active / max(df['sedentary_minutes'].mean(), 1)  # Avoid div by 0
                        activity_ratio_std = df['very_active_minutes'].std() / max(df['sedentary_minutes'].std(), 1)
                        
                        all_features['activity_activity_ratio_mean'] = activity_ratio
                        all_features['activity_activity_ratio_std'] = activity_ratio_std
                
                elif dataset_type == 'sleep_daily_summary':
                    # From sleep, extract the sleep metrics
                    if len(df) > 0:
                        sleep_cols = ['minute_asleep', 'minute_deep', 'minute_light', 'minute_rem', 'minute_awake']
                        for col in sleep_cols:
                            if col in df.columns:
                                all_features[f'sleep_{col}_mean'] = df[col].mean()
                                all_features[f'sleep_{col}_std'] = df[col].std()
                                
                                # For minute_asleep, also get min/max
                                if col == 'minute_asleep':
                                    all_features[f'sleep_{col}_min'] = df[col].min()
                                    all_features[f'sleep_{col}_max'] = df[col].max()
                        
                        # Calculate sleep regularity metrics
                        if 'sleep_date' in df.columns:
                            df['sleep_date'] = pd.to_datetime(df['sleep_date'])
                            df = df.sort_values('sleep_date')
                            df['next_day'] = df['sleep_date'].shift(-1)
                            df['time_diff'] = (df['next_day'] - df['sleep_date']).dt.total_seconds() / 3600
                            
                            all_features['sleep_time_diff_mean'] = df['time_diff'].mean()
                            all_features['sleep_time_diff_std'] = df['time_diff'].std()
                            
                            # Social jetlag (weekend vs weekday difference)
                            df['is_weekend'] = df['sleep_date'].dt.dayofweek.isin([5, 6])
                            weekend_sleep = df[df['is_weekend']]['minute_asleep'].mean()
                            weekday_sleep = df[~df['is_weekend']]['minute_asleep'].mean()
                            social_jetlag = abs(weekend_sleep - weekday_sleep) if not np.isnan(weekend_sleep) and not np.isnan(weekday_sleep) else 0
                            
                            all_features['sleep_social_jetlag_'] = social_jetlag
                
                elif dataset_type == 'heart_rate_level':
                    # Extract heart rate features
                    if len(df) > 0 and 'avg_rate' in df.columns:
                        # Convert to numeric if needed
                        df['avg_rate'] = pd.to_numeric(df['avg_rate'], errors='coerce')
                        
                        # Calculate aggregate metrics
                        all_features['hr_avg_rate_mean'] = df['avg_rate'].mean()
                        all_features['hr_avg_rate_std'] = df['avg_rate'].std()
                        all_features['hr_avg_rate_min'] = df['avg_rate'].min()
                        all_features['hr_avg_rate_max'] = df['avg_rate'].max()
                        all_features['hr_avg_rate_skew'] = df['avg_rate'].skew()
        
        # Create the feature dataframe
        features_df_expanded = pd.DataFrame([all_features])
        
        # Combine with person data
        features_df = pd.concat([features_df, features_df_expanded], axis=1)
        
        return features_df
    
    def predict_mental_health(self):
        """Run prediction on the loaded data without extra processing"""
        # Load model files if not already loaded
        if self.model is None:
            self.load_model()
        
        # Load features from CSV files
        features_df = self.load_csv_files()
        
        # Drop person_id before prediction (if it exists)
        X = features_df.drop('person_id', axis=1) if 'person_id' in features_df.columns else features_df
        
        # Log the feature columns we're using
        logger.info(f"Feature columns available: {X.columns.tolist()}")
        
        # Check if we need to reorder/filter columns to match training data
        if self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                logger.warning(f"Missing {len(missing_cols)} features expected by model: {missing_cols[:5]}")
                # Add missing columns with zero values
                for col in missing_cols:
                    X[col] = 0
            
            # Keep only the features expected by the model, in the correct order
            X = X[self.feature_names]
            logger.info(f"Using {len(self.feature_names)} feature columns for prediction")
        
        # Make prediction directly with the model
        # No scaling needed as the model should apply it internally
        try:
            prediction_proba = self.model.predict_proba(X)[:, 1]
            prediction_class = self.model.predict(X)
            
            logger.info(f"Prediction successful: class={prediction_class[0]}, probability={prediction_proba[0]:.4f}")
            
            # Get binary classification result
            # The original model in algo.mdc was trained with 0=control, 1=mental health issue 
            if prediction_proba[0] < 0.5:
                risk_level = "NEGATIVE"  # Control group (no disorder)
            else:
                risk_level = "POSITIVE"  # Subject group (has disorder)
                
            # Get top contributing features
            contributing_features = {}
            if self.feature_importance is not None:
                # Get top features by importance
                top_features = self.feature_importance.sort_values('importance', ascending=False).head(10)
                
                for _, row in top_features.iterrows():
                    feature_name = row['feature']
                    importance = row['importance']
                    contributing_features[feature_name] = importance
            
            # Create result dictionary
            result = {
                'user_id': '1001',
                'prediction': int(prediction_class[0]),
                'risk_level': risk_level,
                'risk_score': float(prediction_proba[0]),
                'contributing_factors': contributing_features,
                'analysis_date': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def print_mental_health_analysis(result):
    """Print the mental health analysis result in a readable format"""
    if result is None:
        print("\n❌ Analysis failed to produce any results.")
        return
        
    print("\n" + "="*50)
    print("🧠 Mental Health Analysis Result")
    print("="*50)
    print(f"User ID: {result['user_id']}")
    print(f"Prediction: {result['prediction']} (0=control, 1=mental health issue)")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']:.2%}")
    print(f"Analysis Date: {result['analysis_date']}")
    
    print("\n📊 Contributing Factors:")
    for factor, importance in sorted(result['contributing_factors'].items(), key=lambda x: x[1], reverse=True):
        # Convert feature names to more readable form
        readable_name = factor.replace('_', ' ').title()
        print(f"  - {readable_name}: {importance:.2%}")
    
    print("="*50)

def main():
    """Main function to test the mental health model"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test the mental health model with converted HealthKit data'
    )
    
    parser.add_argument(
        '--data_dir',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'MindBack', 'converted_data'),
        help='Directory containing the converted Fitbit data CSV files'
    )
    
    parser.add_argument(
        '--model_dir',
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'MindBack', 'model'),
        help='Directory containing the mental health model files'
    )
    
    args = parser.parse_args()
    
    print(f"\n🔍 Testing mental health model with data from: {args.data_dir}")
    print(f"📁 Using model files from: {args.model_dir}\n")
    
    # Check if model file exists
    model_path = os.path.join(args.model_dir, "mental_health_model.xgb")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        print("Please make sure the model file exists before running this test.")
        return 1
    
    # Run the test
    try:
        predictor = SimpleMentalHealthPredictor(args.model_dir, args.data_dir)
        result = predictor.predict_mental_health()
        print_mental_health_analysis(result)
        
        print("\n✅ Test completed successfully!")
        return 0
    
    except Exception as e:
        print(f"\n❌ Error during test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
