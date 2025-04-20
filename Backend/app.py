from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import logging
from pathlib import Path
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MindFull API",
    description="API for MindWatch iOS app to analyze mental health based on HealthKit data",
    version="1.0.0"
)

# Set up CORS middleware (to allow requests from iOS app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model paths - EXACTLY as in test_mental_health_model.py
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "mental_health_model.xgb"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

# Pydantic models for request/response validation
class HealthKitData(BaseModel):
    heartRate: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    activeEnergy: List[Dict[str, Any]]
    sleep: List[Dict[str, Any]]
    workout: List[Dict[str, Any]]
    distance: List[Dict[str, Any]]
    basalEnergy: List[Dict[str, Any]]
    flightsClimbed: List[Dict[str, Any]]
    userInfo: Dict[str, Any] = {
        "personId": "1001",
        "age": 33,
        "genderBinary": 1  # Default: female
    }

class AnalysisResult(BaseModel):
    userId: str
    prediction: int
    riskLevel: str
    riskScore: float
    contributingFactors: Dict[str, float]
    analysisDate: str

# Helper class for mental health prediction
class MentalHealthPredictor:
    def __init__(self):
        """Load the mental health prediction model and feature importance data"""
        
        # Initialize Mental Health Predictor - EXACTLY as in test_mental_health_model.py
        logger.info("Initializing Mental Health Predictor...")
        
        # Load the model - EXACTLY as in test_mental_health_model.py
        model_path = os.path.join(os.path.dirname(__file__), "model", "mental_health_model.xgb")
        logger.info(f"Loading model from: {model_path}")
        
        try:
            # Use pickle.load instead of joblib.load to match test_mental_health_model.py
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load mental health model: {e}")
        
        # Load feature importance - EXACTLY as in test_mental_health_model.py
        try:
            importance_path = os.path.join(os.path.dirname(__file__), "model", "feature_importance.csv")
            logger.info(f"Loading feature importance from: {importance_path}")
            
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)
                logger.info(f"Feature importance loaded with {len(self.feature_importance)} features")
            else:
                logger.warning("Feature importance file not found, will not include importance data")
                self.feature_importance = None
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            self.feature_importance = None
        
        # Load feature names - EXACTLY as in test_mental_health_model.py
        try:
            feature_names_path = os.path.join(os.path.dirname(__file__), "model", "feature_names.json")
            logger.info(f"Loading feature names from: {feature_names_path}")
            
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names file not found, will use all available features")
                self.feature_names = None
        except Exception as e:
            logger.error(f"Error loading feature names: {e}")
            self.feature_names = None
            
        logger.info("Mental Health Predictor initialized successfully")
    
    def transform_data(self, health_data: HealthKitData) -> pd.DataFrame:
        """Transform HealthKit data to the format expected by the model"""
        # Create base dataframe with user info - EXACTLY as in test_mental_health_model.py
        features_df = pd.DataFrame({
            'person_id': [health_data.userInfo.get('personId', '1001')],
            'age': [health_data.userInfo.get('age', 33)],
            'gender_binary': [health_data.userInfo.get('genderBinary', 1)]
        })
        
        # Initialize all_features dictionary to collect aggregated metrics
        all_features = {}
        
        # Process heart rate data - EXACTLY matching test_mental_health_model.py
        heart_rate_df = pd.DataFrame(health_data.heartRate)
        if not heart_rate_df.empty and 'value' in heart_rate_df.columns:
            # Convert to numeric if needed
            heart_rate_df['value'] = pd.to_numeric(heart_rate_df['value'], errors='coerce')
            
            # Calculate aggregate metrics
            all_features['hr_avg_rate_mean'] = heart_rate_df['value'].mean()
            all_features['hr_avg_rate_std'] = heart_rate_df['value'].std()
            all_features['hr_avg_rate_min'] = heart_rate_df['value'].min()
            all_features['hr_avg_rate_max'] = heart_rate_df['value'].max()
            all_features['hr_avg_rate_skew'] = heart_rate_df['value'].skew()
        
        # Process step data - EXACTLY matching test_mental_health_model.py
        steps_df = pd.DataFrame(health_data.steps)
        if not steps_df.empty and 'value' in steps_df.columns:
            steps_df['value'] = pd.to_numeric(steps_df['value'], errors='coerce')
            steps_df['date'] = pd.to_datetime(steps_df['endDate']).dt.date
            
            # Aggregate by day
            daily_steps = steps_df.groupby('date')['value'].sum().reset_index()
            
            # Match exactly the feature naming in test_mental_health_model.py
            all_features['activity_steps_mean'] = daily_steps['value'].mean()
            all_features['activity_steps_std'] = daily_steps['value'].std() 
            all_features['activity_steps_max'] = daily_steps['value'].max()
        
        # Process active energy data - EXACTLY matching test_mental_health_model.py
        energy_df = pd.DataFrame(health_data.activeEnergy)
        if not energy_df.empty and 'value' in energy_df.columns:
            energy_df['value'] = pd.to_numeric(energy_df['value'], errors='coerce')
            energy_df['date'] = pd.to_datetime(energy_df['endDate']).dt.date
            
            # Aggregate by day
            daily_energy = energy_df.groupby('date')['value'].sum().reset_index()
            
            # Match exactly the feature naming in test_mental_health_model.py
            all_features['activity_calories_out_mean'] = daily_energy['value'].mean()
            all_features['activity_calories_out_std'] = daily_energy['value'].std()
        
        # Process sleep data - EXACTLY matching test_mental_health_model.py
        sleep_df = pd.DataFrame(health_data.sleep)
        if not sleep_df.empty and 'value' in sleep_df.columns:
            # Convert sleep values as needed
            sleep_df['value'] = pd.to_numeric(sleep_df['value'], errors='coerce')
            sleep_df['date'] = pd.to_datetime(sleep_df['startDate']).dt.date
            
            # Aggregate by day
            daily_sleep = sleep_df.groupby('date')['value'].sum().reset_index()
            
            # EXACTLY match the features in test_mental_health_model.py
            all_features['sleep_minute_asleep_mean'] = daily_sleep['value'].mean()
            all_features['sleep_minute_asleep_std'] = daily_sleep['value'].std()
            all_features['sleep_minute_asleep_min'] = daily_sleep['value'].min()
            all_features['sleep_minute_asleep_max'] = daily_sleep['value'].max()
            
            # Calculate sleep regularity metrics EXACTLY as in test file
            if len(daily_sleep) > 1:
                daily_sleep = daily_sleep.sort_values('date')
                daily_sleep['next_day'] = daily_sleep['date'].shift(-1)
                # Critical: Convert to datetime objects before calculating difference
                daily_sleep['time_diff'] = (pd.to_datetime(daily_sleep['next_day']) - 
                                           pd.to_datetime(daily_sleep['date'])).dt.total_seconds() / 3600
                
                all_features['sleep_time_diff_mean'] = daily_sleep['time_diff'].mean()
                all_features['sleep_time_diff_std'] = daily_sleep['time_diff'].std()
                
                # Social jetlag calculation EXACTLY as in test_mental_health_model.py
                daily_sleep['is_weekend'] = pd.to_datetime(daily_sleep['date']).dt.dayofweek.isin([5, 6])
                weekend_sleep = daily_sleep[daily_sleep['is_weekend']]['value'].mean()
                weekday_sleep = daily_sleep[~daily_sleep['is_weekend']]['value'].mean()
                
                # CRITICAL: Use this exact condition matching the test file
                social_jetlag = abs(weekend_sleep - weekday_sleep) if not np.isnan(weekend_sleep) and not np.isnan(weekday_sleep) else 0
                all_features['sleep_social_jetlag_'] = social_jetlag
        
        # Process workout data - EXACTLY matching test_mental_health_model.py logic
        workout_df = pd.DataFrame(health_data.workout)
        if not workout_df.empty:
            # In test_mental_health_model.py, duration is used directly
            workout_df['duration'] = pd.to_numeric(workout_df['value'], errors='coerce')
            all_features['activity_very_active_minutes_mean'] = workout_df['duration'].mean()
            all_features['activity_very_active_minutes_std'] = workout_df['duration'].std()
            all_features['activity_very_active_minutes_max'] = workout_df['duration'].max()
        
        # Add remaining activity data EXACTLY as in test_mental_health_model.py
        # If we don't have direct workout data, estimate these activity metrics
        if 'activity_very_active_minutes_mean' not in all_features:
            all_features['activity_very_active_minutes_mean'] = 0
            all_features['activity_very_active_minutes_std'] = 0
            all_features['activity_very_active_minutes_max'] = 0
            
        # Estimate values for fairly active and lightly active minutes
        # EXACTLY as in test_mental_health_model.py
        all_features['activity_fairly_active_minutes_mean'] = all_features.get('activity_very_active_minutes_mean', 0) * 0.5
        all_features['activity_fairly_active_minutes_std'] = all_features.get('activity_very_active_minutes_std', 0) * 0.5
        
        if 'activity_steps_mean' in all_features:
            all_features['activity_lightly_active_minutes_mean'] = all_features['activity_steps_mean'] / 100
            all_features['activity_lightly_active_minutes_std'] = all_features['activity_steps_std'] / 100
        else:
            all_features['activity_lightly_active_minutes_mean'] = 0
            all_features['activity_lightly_active_minutes_std'] = 0
        
        # Calculate total active minutes and sedentary minutes
        # EXACTLY as in test_mental_health_model.py
        total_active_minutes = (
            all_features.get('activity_very_active_minutes_mean', 0) +
            all_features.get('activity_fairly_active_minutes_mean', 0) +
            all_features.get('activity_lightly_active_minutes_mean', 0)
        )
        
        # Sedentary minutes (1440 = minutes in a day) EXACTLY as in test file
        all_features['activity_sedentary_minutes_mean'] = 1440 - total_active_minutes
        all_features['activity_sedentary_minutes_std'] = all_features.get('activity_very_active_minutes_std', 20)
        
        # Activity ratio - use max() to avoid division by zero
        # CRITICAL: This max() function is exactly how it's done in test_mental_health_model.py
        activity_ratio = total_active_minutes / max(all_features['activity_sedentary_minutes_mean'], 1)
        activity_ratio_std = all_features.get('activity_very_active_minutes_std', 0) / max(all_features['activity_sedentary_minutes_std'], 1)
        
        all_features['activity_activity_ratio_mean'] = activity_ratio
        all_features['activity_activity_ratio_std'] = activity_ratio_std
        
        # Create the feature dataframe EXACTLY as in test_mental_health_model.py
        features_df_expanded = pd.DataFrame([all_features])
        
        # Combine with person data
        features_df = pd.concat([features_df, features_df_expanded], axis=1)
        
        # For debugging, print all features used
        logger.info(f"Final feature set: {list(features_df.columns)}")
        
        return features_df
    
    def predict(self, health_data: HealthKitData) -> AnalysisResult:
        """Process health data and make a prediction"""
        # Transform HealthKit data to features
        features_df = self.transform_data(health_data)
        
        # Log the input data shape
        logger.info(f"Input feature dataframe shape: {features_df.shape}")
        
        # Drop person_id before prediction (if it exists) - EXACTLY as in test_mental_health_model.py
        X = features_df.drop('person_id', axis=1) if 'person_id' in features_df.columns else features_df
        
        # Log the feature columns for debugging
        logger.info(f"Feature columns being used: {X.columns.tolist()}")
        
        # Check if we need to reorder/filter columns to match training data - EXACTLY as in test file
        if self.feature_names is not None:
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                logger.warning(f"Missing {len(missing_cols)} features expected by model: {missing_cols[:5]}...")
                # Add missing columns with zero values
                for col in missing_cols:
                    X[col] = 0
            
            # Keep only the features expected by the model, in the correct order
            X = X[self.feature_names]
            logger.info(f"Using {len(self.feature_names)} feature columns for prediction")
        
        # Make prediction with the model - EXACTLY as in test_mental_health_model.py
        prediction_proba = self.model.predict_proba(X)[:, 1]
        prediction_class = self.model.predict(X)
        
        # Log prediction details
        logger.info(f"Prediction result: class={prediction_class[0]}, probability={prediction_proba[0]:.4f}")
        
        # Get binary classification result - EXACTLY as in test_mental_health_model.py
        if prediction_proba[0] < 0.5:
            risk_level = "NEGATIVE"  # Control group (no disorder)
        else:
            risk_level = "POSITIVE"  # Subject group (has disorder)
            
        # Get top contributing features - EXACTLY as in test_mental_health_model.py
        contributing_features = {}
        if self.feature_importance is not None:
            # Get top features by importance
            top_features = self.feature_importance.sort_values('importance', ascending=False).head(10)
            
            for _, row in top_features.iterrows():
                feature_name = row['feature']
                importance = row['importance']
                contributing_features[feature_name] = importance
        
        # Create result - EXACTLY as in test_mental_health_model.py format
        result = AnalysisResult(
            userId=health_data.userInfo.get('personId', '1001'),
            prediction=int(prediction_class[0]),
            riskLevel=risk_level,
            riskScore=float(prediction_proba[0]),
            contributingFactors=contributing_features,
            analysisDate=datetime.now().isoformat()
        )
        
        # Log the risk assessment result
        logger.info(f"Analysis complete. Risk level: {result.riskLevel}, Score: {result.riskScore:.2f}")
        
        return result

# Lazy-load our predictor
@app.on_event("startup")
async def startup_db_client():
    app.predictor = MentalHealthPredictor()
    logger.info("Mental health predictor initialized")

# Get predictor instance
def get_predictor():
    return app.predictor

# Routes
@app.get("/")
async def root():
    return {"message": "MindFull API is running", "status": "OK"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_health_data(
    health_data: HealthKitData,
    predictor: MentalHealthPredictor = Depends(get_predictor)
):
    try:
        logger.info(f"Received health data analysis request")
        result = predictor.predict(health_data)
        logger.info(f"Analysis complete. Risk level: {result.riskLevel}, Score: {result.riskScore:.2f}")
        return result
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 