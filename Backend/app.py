from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib
import time
import uuid

# Import Supabase client
from supabase_client import SupabaseClient

# Import HealthKit extractor for standardized data transformation
from health_extractor import HealthKitExtractor

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO), 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application settings
class Settings:
    """Application settings loaded from environment variables"""
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_service_key = os.environ.get("SUPABASE_ANON_KEY", "")

# Initialize settings
settings = Settings()

# Dictionary to store user-specific predictor instances
user_predictors = {}

# Create a custom filter to limit logging of large data structures
class DataSizeFilter(logging.Filter):
    def filter(self, record):
        if record.levelno < logging.WARNING and isinstance(record.msg, str):
            # Replace large data structures in log messages with summaries
            if "[" in record.msg and "]" in record.msg and len(record.msg) > 500:
                # This is a crude way to detect array-like data in logs
                record.msg = record.msg[:250] + "... [truncated for brevity]"
        return True

# Add the filter to the logger
logger.addFilter(DataSizeFilter())

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

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Define model paths - EXACTLY as in test_mental_health_model.py
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "mental_health_model.xgb"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "feature_importance.csv"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"

# Authentication models
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInfo(BaseModel):
    id: str
    email: EmailStr

class TokenData(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    user: UserInfo

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
    activity: List[Dict[str, Any]] = []
    userInfo: Dict[str, Any] = {
        "userId": "unknown",
        "personId": "unknown",
        "age": 33,
        "genderBinary": 1
    }

class UserProfileData(BaseModel):
    userId: str
    gender: str
    birthdate: str
    age: int

class AnalysisResult(BaseModel):
    user_id: str
    prediction: int
    risk_level: str
    risk_score: float
    contributing_factors: Dict[str, Any]
    analysis_date: str

class UserAuth(BaseModel):
    userId: str
    authToken: str

class AnalysisRequest(BaseModel):
    userId: str
    forceRun: bool = False

# Helper class for mental health prediction - EXACTLY matching test_mental_health_model.py
class MentalHealthPredictor:
    def __init__(self):
        """Load the mental health prediction model and feature importance data"""
        logger.info("Initializing Mental Health Predictor...")
        
        # Load the model - EXACTLY as in test_mental_health_model.py
        model_path = os.path.join(os.path.dirname(__file__), "model", "mental_health_model.xgb")
        logger.info(f"Loading model from: {model_path}")
        
        # Initialize feature lists
        self.model_feature_names = None
        self.feature_importances = None
        self.expected_feature_count = None
        
        # Initialize the HealthKit extractor for data transformation
        self.extractor = HealthKitExtractor()
        logger.info("HealthKit extractor initialized")
        
        try:
            # Use joblib.load instead of pickle.load
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Get the expected feature count from the model if possible
            if hasattr(self.model, 'n_features_'):
                self.expected_feature_count = self.model.n_features_
                logger.info(f"Model explicitly expects {self.expected_feature_count} features (n_features_)")
            elif hasattr(self.model, 'n_features_in_'):
                self.expected_feature_count = self.model.n_features_in_
                logger.info(f"Model explicitly expects {self.expected_feature_count} features (n_features_in_)")
            elif hasattr(self.model, 'feature_importances_'):
                self.expected_feature_count = len(self.model.feature_importances_)
                logger.info(f"Model implicitly expects {self.expected_feature_count} features (from feature_importances_)")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load mental health model: {e}")
        
        # Load feature importance - EXACTLY as in test_mental_health_model.py
        feature_importance_path = os.path.join(os.path.dirname(__file__), "model", "feature_importance.csv")
        logger.info(f"Loading feature importance from: {feature_importance_path}")
        try:
            if os.path.exists(feature_importance_path):
                self.feature_importances = pd.read_csv(feature_importance_path)
                logger.info(f"Feature importance loaded with {len(self.feature_importances)} features")
                
                # Extract feature names from feature importance CSV
                # We prioritize this since it has all 38 features the model expects
                self.model_feature_names = self.feature_importances['feature'].tolist()
                logger.info(f"Using feature names from importance file: {len(self.model_feature_names)} features")
                
                # If we expect a certain number of features, verify we have them all
                if self.expected_feature_count is not None:
                    if len(self.model_feature_names) != self.expected_feature_count:
                        logger.warning(
                            f"Feature count mismatch in feature_importance.csv. "
                            f"Expected {self.expected_feature_count}, got {len(self.model_feature_names)}."
                        )
                        # Only add dummy features if absolutely necessary
                        if len(self.model_feature_names) < self.expected_feature_count:
                            missing_count = self.expected_feature_count - len(self.model_feature_names)
                            logger.warning(f"Adding {missing_count} dummy features to match model requirements")
                            for i in range(missing_count):
                                self.model_feature_names.append(f"dummy_feature_{i}")
                            logger.info(f"Updated feature count: {len(self.model_feature_names)}")
            else:
                logger.warning("Feature importance file not found, will check for feature_names.json instead")
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            logger.warning("Will check for feature_names.json instead")
        
        # Only load feature_names.json if we still don't have feature names
        if self.model_feature_names is None:
            feature_names_path = os.path.join(os.path.dirname(__file__), "model", "feature_names.json")
            logger.info(f"Loading feature names from: {feature_names_path}")
            try:
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        self.model_feature_names = json.load(f)
                    logger.info(f"Feature names loaded from JSON: {len(self.model_feature_names)} features")
                    
                    # If model expects more features than we have, add dummy features
                    if self.expected_feature_count is not None and len(self.model_feature_names) < self.expected_feature_count:
                        missing_count = self.expected_feature_count - len(self.model_feature_names)
                        logger.warning(f"Adding {missing_count} dummy features to match model requirements")
                        for i in range(missing_count):
                            dummy_name = f"dummy_feature_{i}"
                            if dummy_name not in self.model_feature_names:
                                self.model_feature_names.append(dummy_name)
                        logger.info(f"Updated feature count: {len(self.model_feature_names)}")
                else:
                    logger.warning("Feature names file not found, cannot determine feature list")
            except Exception as e:
                logger.error(f"Error loading feature names: {e}")
        
        # Final check to ensure we have the feature names
        if self.model_feature_names is None:
            logger.warning("No feature names available from any source, this will likely cause prediction errors")
        else:
            logger.info(f"Final feature set has {len(self.model_feature_names)} features: {self.model_feature_names[:5]}...")
            logger.info(f"Complete list of expected features: {self.model_feature_names}")
            
        logger.info("Mental Health Predictor initialized successfully")
    
    def transform_data(self, health_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Transform Supabase data (in Fitbit format) for model input.
        This follows the same pattern as the training code to ensure consistency.
        """
        logger.info("Transforming Supabase data using Fitbit format...")
        
        try:
            # Get user information - similar to process_person_data in training
            user_id = health_data["userInfo"].get("personId", "unknown")
            age = health_data["userInfo"].get("age", 33)
            gender_binary = health_data["userInfo"].get("genderBinary", 1)
            
            # Create base person dataframe
            features_df = pd.DataFrame({
                'person_id': [user_id],
                'age': [age],
                'gender_binary': [gender_binary]
            })
            
            # Process sleep features - similar to process_sleep_features in training
            sleep_features = {}
            if health_data.get("sleep") and len(health_data["sleep"]) > 0:
                try:
                    sleep_df = pd.DataFrame(health_data["sleep"])
                    
                    # Calculate basic sleep metrics
                    sleep_metrics = {
                        'sleep_minute_asleep_mean': sleep_df['minute_asleep'].mean(),
                        'sleep_minute_asleep_std': sleep_df['minute_asleep'].std(),
                        'sleep_minute_asleep_min': sleep_df['minute_asleep'].min(),
                        'sleep_minute_asleep_max': sleep_df['minute_asleep'].max()
                    }
                    
                    # Process different sleep types if available
                    for sleep_type in ['minute_deep', 'minute_light', 'minute_rem', 'minute_awake']:
                        if sleep_type in sleep_df.columns:
                            sleep_metrics[f'sleep_{sleep_type}_mean'] = sleep_df[sleep_type].mean()
                            sleep_metrics[f'sleep_{sleep_type}_std'] = sleep_df[sleep_type].std()
                    
                    # Sleep regularity if enough data points
                    if len(sleep_df) > 1 and 'sleep_date' in sleep_df.columns:
                        sleep_df['sleep_date'] = pd.to_datetime(sleep_df['sleep_date'])
                        sleep_df = sleep_df.sort_values('sleep_date')
                        sleep_df['next_day'] = sleep_df['sleep_date'].shift(-1)
                        sleep_df['time_diff'] = (sleep_df['next_day'] - sleep_df['sleep_date']).dt.total_seconds() / 3600
                        
                        if not sleep_df['time_diff'].isnull().all():
                            sleep_metrics['sleep_time_diff_std'] = sleep_df['time_diff'].std()
                            sleep_metrics['sleep_time_diff_mean'] = sleep_df['time_diff'].mean()
                    
                        # Calculate weekend vs weekday differences (social jetlag)
                        if len(sleep_df) >= 7:
                            sleep_df['is_weekend'] = sleep_df['sleep_date'].dt.dayofweek.isin([5, 6])
                            weekend_stats = sleep_df[sleep_df['is_weekend']]['minute_asleep'].mean()
                            weekday_stats = sleep_df[~sleep_df['is_weekend']]['minute_asleep'].mean()
                            
                            if not np.isnan(weekend_stats) and not np.isnan(weekday_stats):
                                sleep_metrics['sleep_social_jetlag_'] = abs(weekend_stats - weekday_stats)
                    
                    sleep_features.update(sleep_metrics)
                except Exception as e:
                    logger.warning(f"Error processing sleep features: {e}")
            
            # Process activity features - similar to process_activity_features in training
            activity_features = {}
            if health_data.get("activity") and len(health_data["activity"]) > 0:
                try:
                    # Create dataframe from activity data
                    activity_df = pd.DataFrame(health_data["activity"])
                    logger.info(f"Activity dataframe columns: {list(activity_df.columns)}")
                    
                    # Process calories out features
                    if 'calories_out' in activity_df.columns:
                        activity_features['activity_calories_out_mean'] = activity_df['calories_out'].mean()
                        activity_features['activity_calories_out_std'] = activity_df['calories_out'].std()
                    
                    # Process steps features if available in activity data
                    if 'steps' in activity_df.columns:
                        activity_features['activity_steps_mean'] = activity_df['steps'].mean()
                        activity_features['activity_steps_std'] = activity_df['steps'].std()
                        activity_features['activity_steps_max'] = activity_df['steps'].max()
                    
                    # Process activity minutes features
                    for metric in ['very_active_minutes', 'fairly_active_minutes', 'lightly_active_minutes', 'sedentary_minutes']:
                        if metric in activity_df.columns:
                            activity_features[f'activity_{metric}_mean'] = activity_df[metric].mean()
                            activity_features[f'activity_{metric}_std'] = activity_df[metric].std()
                            
                            if metric == 'very_active_minutes':
                                activity_features[f'activity_{metric}_max'] = activity_df[metric].max()
                    
                    # Calculate activity ratio if we have both active and sedentary minutes
                    if all(col in activity_df.columns for col in ['very_active_minutes', 'fairly_active_minutes', 'lightly_active_minutes', 'sedentary_minutes']):
                        activity_df['total_active_minutes'] = (
                            activity_df['very_active_minutes'] + 
                            activity_df['fairly_active_minutes'] + 
                            activity_df['lightly_active_minutes']
                        )
                        activity_df['activity_ratio'] = activity_df['total_active_minutes'] / (activity_df['sedentary_minutes'] + 1)
                        
                        activity_features['activity_activity_ratio_mean'] = activity_df['activity_ratio'].mean()
                        activity_features['activity_activity_ratio_std'] = activity_df['activity_ratio'].std()
                except Exception as e:
                    logger.warning(f"Error processing activity features: {e}")
            
            # Process steps data separately if not available in activity data
            if 'activity_steps_mean' not in activity_features and health_data.get("steps") and len(health_data["steps"]) > 0:
                try:
                    steps_df = pd.DataFrame(health_data["steps"])
                    if 'steps' in steps_df.columns:
                        activity_features['activity_steps_mean'] = steps_df['steps'].mean()
                        activity_features['activity_steps_std'] = steps_df['steps'].std()
                        activity_features['activity_steps_max'] = steps_df['steps'].max()
                    elif 'sum_steps' in steps_df.columns:
                        activity_features['activity_steps_mean'] = steps_df['sum_steps'].mean()
                        activity_features['activity_steps_std'] = steps_df['sum_steps'].std()
                        activity_features['activity_steps_max'] = steps_df['sum_steps'].max()
                    elif 'value' in steps_df.columns:
                        activity_features['activity_steps_mean'] = steps_df['value'].mean()
                        activity_features['activity_steps_std'] = steps_df['value'].std()
                        activity_features['activity_steps_max'] = steps_df['value'].max()
                except Exception as e:
                    logger.warning(f"Error processing steps features: {e}")
            
            # Process heart rate data - similar to process_heart_rate_features in training
            hr_features = {}
            if health_data.get("heartRate") and len(health_data["heartRate"]) > 0:
                try:
                    hr_df = pd.DataFrame(health_data["heartRate"])
                    
                    # Check which column has the heart rate values
                    hr_column = None
                    if 'avg_rate' in hr_df.columns:
                        hr_column = 'avg_rate'
                    elif 'value' in hr_df.columns:
                        hr_column = 'value'
                    
                    if hr_column:
                        # Ensure numeric values
                        hr_df[hr_column] = pd.to_numeric(hr_df[hr_column], errors='coerce')
                        
                        # Calculate heart rate metrics
                        hr_features['hr_avg_rate_mean'] = hr_df[hr_column].mean()
                        hr_features['hr_avg_rate_std'] = hr_df[hr_column].std()
                        hr_features['hr_avg_rate_min'] = hr_df[hr_column].min()
                        hr_features['hr_avg_rate_max'] = hr_df[hr_column].max()
                        
                        # Calculate skew if we have scipy
                        try:
                            from scipy.stats import skew
                            hr_features['hr_avg_rate_skew'] = skew(hr_df[hr_column].dropna())
                        except (ImportError, ValueError):
                            # Use a placeholder or calculate approximation if scipy not available
                            logger.warning("Scipy not available or insufficient data for skew calculation")
                            hr_features['hr_avg_rate_skew'] = 0
                except Exception as e:
                    logger.warning(f"Error processing heart rate features: {e}")
            
            # Combine all features
            all_features = {}
            all_features.update(sleep_features)
            all_features.update(activity_features)
            all_features.update(hr_features)
            
            # Create expanded features dataframe
            features_df_expanded = pd.DataFrame([all_features])
            
            # Combine user info with extracted features
            features_df = pd.concat([features_df, features_df_expanded], axis=1)
            
            # Fill missing values with means for numeric columns only
            numeric_columns = features_df.select_dtypes(include=['number']).columns
            features_df[numeric_columns] = features_df[numeric_columns].fillna(features_df[numeric_columns].mean())
            
            # Fill any remaining NaN values with zeros
            features_df = features_df.fillna(0)
            
            logger.info(f"Features extracted successfully: shape={features_df.shape}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error transforming Supabase data: {e}", exc_info=True)
            # Fall back to minimal dataframe with demographic data only
            return pd.DataFrame({
                'person_id': [user_id] if 'user_id' in locals() else ['unknown'],
                'age': [age] if 'age' in locals() else [33],
                'gender_binary': [gender_binary] if 'gender_binary' in locals() else [1]
            })
    
    def predict(self, health_data: Dict[str, Any]) -> AnalysisResult:
        """
        Predict mental health status from Supabase health data in Fitbit format.
        Returns structured result with prediction and contributing factors.
        """
        logger.info("Making mental health prediction...")
        # Extract user ID from input data
        user_id = health_data.get("userInfo", {}).get("personId", "unknown")
        
        # First, check if we have enough data
        data_quality_score, data_quality_message = self._calculate_data_quality(health_data)
        
        # Transform data for model input
        input_df = self.transform_data(health_data)
        
        # Drop person_id as it's not a model feature
        if 'person_id' in input_df.columns:
            input_df = input_df.drop('person_id', axis=1)
        
        logger.info(f"Input feature dataframe shape: {input_df.shape}")
        logger.info(f"Generated features: {list(input_df.columns)}")
        
        # Check if we have the expected number of features
        expected_feature_count = 38  # Based on the training model
        if input_df.shape[1] != expected_feature_count:
            logger.warning(f"Expected {expected_feature_count} features but got {input_df.shape[1]}")
            
            # Get the missing features
            missing_features = [f for f in self.model_feature_names if f not in input_df.columns]
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}")
                
                # Add missing features with zero values
                for feat in missing_features:
                    input_df[feat] = 0
        
        # Ensure we have the features in the right order
        if self.model_feature_names is not None:
            # Only keep the features we expect
            common_features = [f for f in self.model_feature_names if f in input_df.columns]
            # Add zeros for missing ones
            missing_features = [f for f in self.model_feature_names if f not in input_df.columns]
            
            # Reindex to match training feature order
            if common_features:
                input_df = input_df[common_features]
            
            # Add missing features with zero values
            for feat in missing_features:
                input_df[feat] = 0
                
            # Final reordering to match model expectations
            input_df = input_df[self.model_feature_names]
        
        # Make prediction
        try:
            # Predict probability
            prediction_proba = self.model.predict_proba(input_df)[:, 1][0]
            prediction_class = 1 if prediction_proba >= 0.5 else 0
            
            logger.info(f"Prediction: class={prediction_class}, probability={prediction_proba:.4f}")
            
            # Map prediction to risk level
            if prediction_class == 0:
                risk_level = "LOW"
            else:
                risk_level = "HIGH"
            
            # Get feature importance from the model
            feature_importance = {}
            # Top 5 contributing factors
            if hasattr(self, 'feature_importances') and self.feature_importances is not None:
                top_features = self.feature_importances.sort_values('importance', ascending=False).head(5)
                for index, row in top_features.iterrows():
                    # Convert feature names to more readable format
                    feature_name = row['feature'].replace('_', ' ').title()
                    feature_importance[feature_name] = float(row['importance'])
            
            # Create analysis result
            result = AnalysisResult(
                user_id=user_id,
                prediction=prediction_class,
                risk_level=risk_level,
                risk_score=float(prediction_proba),
                contributing_factors={
                    "feature_importance": feature_importance,
                    "data_quality": {
                        "score": data_quality_score,
                        "message": data_quality_message
                    }
                },
                analysis_date=datetime.now().isoformat()
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            # Return default result with error information
            return AnalysisResult(
                user_id=user_id,
                prediction=0,
                risk_level="UNKNOWN",
                risk_score=0.0,
                contributing_factors={
                    "error": str(e),
                    "data_quality": {
                        "score": data_quality_score,
                        "message": data_quality_message
                    }
                },
                analysis_date=datetime.now().isoformat()
            )

    def _calculate_data_quality(self, health_data: Dict[str, Any]) -> Tuple[int, str]:
        """
        Calculate a data quality score based on the completeness of the data
        Returns a tuple of (score, message)
        """
        # Define weights for different data types
        weights = {
            'heartRate': 25,
            'sleep': 25,
            'steps': 20,
            'activity': 30
        }
        
        quality_score = 0
        
        # Check heart rate data
        if health_data.get("heartRate") and len(health_data["heartRate"]) > 0:
            quality_score += weights['heartRate']
        
        # Check sleep data
        if health_data.get("sleep") and len(health_data["sleep"]) > 0:
            quality_score += weights['sleep']
            
        # Check steps data
        if health_data.get("steps") and len(health_data["steps"]) > 0:
            quality_score += weights['steps']
            
        # Check activity data
        if health_data.get("activity") and len(health_data["activity"]) > 0:
            quality_score += weights['activity']
        
        # Generate a message based on the data quality score
        # Count how many required data types are present
        required_types = ['heartRate', 'sleep', 'steps']
        present_types = sum(1 for data_type in required_types if health_data.get(data_type) and len(health_data[data_type]) > 0)
        
        if present_types == 0:
            data_quality_message = "Analysis not possible. No required data types present."
        elif present_types < len(required_types):
            data_quality_message = f"Analysis based on {present_types}/{len(required_types)} required data types."
        else:
            data_quality_message = "Analysis based on all required data types."
            
        # Add quality score information
        if quality_score >= 70:
            data_quality_message += " Data quality is good."
        else:
            data_quality_message += " Data quality needs improvement."
        
        return quality_score, data_quality_message

# Lazy-load our predictor and Supabase client
@app.on_event("startup")
async def startup_db_client():
    app.predictor = MentalHealthPredictor()
    logger.info("Mental health predictor initialized")
    
    try:
        app.supabase = SupabaseClient()
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error(f"Error initializing Supabase client: {e}")
        # Continue without Supabase client - it will be created on first use

# Get predictor instance
def get_predictor():
    return app.predictor

# Get Supabase client instance
def get_supabase():
    if not hasattr(app, 'supabase'):
        app.supabase = SupabaseClient()
    return app.supabase

# Helper to validate authorization
async def validate_auth(request: Request, auth_token: Optional[str] = Header(None)):
    """Validate authentication token with Supabase"""
    if not auth_token:
        # Check if token is in query params
        query_params = request.query_params
        if 'auth_token' in query_params:
            auth_token = query_params['auth_token']
    
    if not auth_token:
        # Check if Authorization header exists
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            auth_token = auth_header.replace('Bearer ', '')
    
    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    
    # Validate token with Supabase
    supabase = get_supabase()
    validation_result = supabase.get_user_from_token(auth_token)
    
    if not validation_result.get('success', False):
        logger.error(f"Invalid auth token: {validation_result.get('message', 'Unknown error')}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Return user info from the token
    return {
        "user_id": validation_result['user_id'],
        "email": validation_result['email']
    }

# Authentication routes
@app.post("/register", response_model=TokenData)
async def register(user_data: UserRegister, supabase: SupabaseClient = Depends(get_supabase)):
    """Register a new user"""
    try:
        result = supabase.register_user(user_data.email, user_data.password)
        
        if not result.get("success", False):
            logger.error(f"Registration failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get("message", "Registration failed"))
        
        # For now, we'll construct a placeholder token since we don't get session info from register_user
        # In a production app, you'd want to modify register_user to return proper session data
        # or immediately call login_user after registration
        login_result = supabase.login_user(user_data.email, user_data.password)
        
        if not login_result.get("success", False):
            logger.error(f"Auto-login after registration failed: {login_result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=400, detail="Registration successful but login failed")
        
        return TokenData(
            access_token=login_result["access_token"],
            refresh_token="", # Not returned in our simplified version
            expires_at=int(time.time()) + 3600, # Placeholder expiry, 1 hour from now
            user=UserInfo(
                id=result["user_id"],
                email=user_data.email
            )
        )
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login", response_model=TokenData)
async def login(user_data: UserLogin, supabase: SupabaseClient = Depends(get_supabase)):
    """Log in an existing user"""
    try:
        result = supabase.login_user(user_data.email, user_data.password)
        
        if not result.get("success", False):
            logger.error(f"Login failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(status_code=400, detail=result.get("message", "Login failed"))
        
        return TokenData(
            access_token=result["access_token"],
            refresh_token="", # Not returned in our simplified version
            expires_at=int(time.time()) + 3600, # Placeholder expiry, 1 hour from now
            user=UserInfo(
                id=result["user_id"],
                email=user_data.email
            )
        )
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@app.post("/validate-token")
async def validate_token(auth_info: dict = Depends(validate_auth)):
    """Validate a token and return user information"""
    return {
        "valid": True,
        "user_id": auth_info["user_id"],
        "email": auth_info["email"]
    }

@app.post("/logout")
async def logout(auth_info: dict = Depends(validate_auth), supabase: SupabaseClient = Depends(get_supabase)):
    """Log out a user"""
    try:
        result = supabase.logout_user(auth_info["user_id"])
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logout failed: {str(e)}")

# Routes
@app.get("/")
async def root():
    return {"message": "MindFull API is running", "status": "OK"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_health_data(
    health_data: HealthKitData,
    predictor: MentalHealthPredictor = Depends(get_predictor),
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    try:
        logger.info(f"Received health data analysis request")
        
        # Log received data statistics without printing actual data points
        data_counts = {
            "heartRate": len(health_data.heartRate),
            "steps": len(health_data.steps),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout),
            "activity": getattr(health_data, "activity", 0) if hasattr(health_data, "activity") else 0
        }
        logger.info(f"Received data counts: {data_counts}")
        
        # Set the user ID from auth in the health data
        user_id = auth_info["user_id"]
        health_data.userInfo["personId"] = user_id
        health_data.userInfo["userId"] = user_id
        
        # Step 1: Check existing data completeness before storing new data
        logger.info(f"Checking existing data completeness for user {user_id}")
        completeness_check = supabase.check_data_completeness(user_id, days=60, min_coverage_percentage=60)
        logger.info(f"Data completeness check: {completeness_check['complete']}, Quality Score: {completeness_check.get('quality_score', 0)}%")
        
        # Get latest timestamps per data type to determine what needs uploading
        latest_timestamps = supabase.get_latest_data_timestamp_per_type(user_id)
        
        # Filter and prepare uploaded data based on what's new
        has_new_data = False
        data_counts_before = {
            "heartRate": len(health_data.heartRate),
            "steps": len(health_data.steps),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout),
            "activity": getattr(health_data, "activity", 0) if hasattr(health_data, "activity") else 0
        }
        
        # If all timestamps are None, we should upload all data
        if all(timestamp is None for timestamp in latest_timestamps.values()):
            logger.info(f"No existing data found for user {user_id}, uploading all data")
            has_new_data = True
        else:
            # Filter heart rate data
            if latest_timestamps.get('heartRate') and health_data.heartRate:
                last_date = latest_timestamps['heartRate']
                health_data.heartRate = [
                    record for record in health_data.heartRate 
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')).replace(tzinfo=None) > last_date
                ]
                if len(health_data.heartRate) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.heartRate)} new heart rate records after {last_date}")
            elif health_data.heartRate:
                # No existing heart rate data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.heartRate)} new heart rate records (no previous data)")
            
            # Filter steps data
            if latest_timestamps.get('steps') and health_data.steps:
                last_date = latest_timestamps['steps']
                health_data.steps = [
                    record for record in health_data.steps 
                    if datetime.fromisoformat(record['endDate'].replace('Z', '+00:00')).replace(tzinfo=None) > last_date
                ]
                if len(health_data.steps) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.steps)} new step records after {last_date}")
            elif health_data.steps:
                # No existing steps data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.steps)} new step records (no previous data)")
            
            # Filter sleep data
            if latest_timestamps.get('sleep') and health_data.sleep:
                last_date = latest_timestamps['sleep']
                health_data.sleep = [
                    record for record in health_data.sleep 
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')).replace(tzinfo=None) > last_date
                ]
                if len(health_data.sleep) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.sleep)} new sleep records after {last_date}")
            elif health_data.sleep:
                # No existing sleep data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.sleep)} new sleep records (no previous data)")
            
            # Filter workout data
            if latest_timestamps.get('workout') and health_data.workout:
                last_date = latest_timestamps['workout']
                health_data.workout = [
                    record for record in health_data.workout 
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')).replace(tzinfo=None) > last_date
                ]
                if len(health_data.workout) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.workout)} new workout records after {last_date}")
            elif health_data.workout:
                # No existing workout data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.workout)} new workout records (no previous data)")
        
        # Log what was filtered out
        data_counts_after = {
            "heartRate": len(health_data.heartRate),
            "steps": len(health_data.steps),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout),
            "activity": getattr(health_data, "activity", 0) if hasattr(health_data, "activity") else 0
        }
        
        logger.info(f"Data counts before filtering: {data_counts_before}")
        logger.info(f"Data counts after filtering: {data_counts_after}")
        
        # Step 2: Store health data in Supabase (only if we have new data)
        if has_new_data:
            # Convert any list values to their length before summing
            safe_count_values = []
            for value in data_counts_after.values():
                if isinstance(value, list):
                    safe_count_values.append(len(value))
                else:
                    safe_count_values.append(value)
            
            total_records = sum(safe_count_values)
            logger.info(f"Storing {total_records} new health data records for user {user_id}")
            
            try:
                # Pass health_data.dict() as first parameter and user_id as second parameter
                # to match the expected signature in supabase_client.py
                store_success = supabase.store_health_data(health_data.dict(), user_id)
                if not store_success:
                    logger.warning(f"Some health data may not have been stored properly for user {user_id}")
            except Exception as e:
                logger.warning(f"Error storing health data: {str(e)}")
                # Continue even if storage fails - we can still attempt analysis with the current data
            
            # Wait a moment for data to be properly stored and indexed
            time.sleep(1)
            
            # Update data completeness after storing new data
            updated_completeness = supabase.check_data_completeness(user_id, days=60, min_coverage_percentage=60)
            logger.info(f"Updated data completeness: {updated_completeness['complete']}, Quality Score: {updated_completeness.get('quality_score', 0)}%")
            
            if updated_completeness.get('quality_score', 0) > completeness_check.get('quality_score', 0):
                logger.info(f"Data quality improved from {completeness_check.get('quality_score', 0)}% to {updated_completeness.get('quality_score', 0)}%")
        else:
            logger.info(f"No new data to store for user {user_id}")
        
        # Step 3: Check for missing data types and detailed coverage
        missing_data_info = supabase.check_missing_data_types(user_id, days=60)
        
        # Format missing data info for logging
        missing_types = [data_type for data_type, is_missing in missing_data_info.get("missingDataTypes", {}).items() if is_missing]
        if missing_types:
            logger.warning(f"Missing data types for user {user_id}: {', '.join(missing_types)}")
            
            # Check coverage for available data
            coverage_summary = missing_data_info.get("dataCoverageSummary", {})
            logger.info(f"Data coverage percentages: " + 
                      ", ".join([f"{dt}: {cov.get('coverage_percentage', 0)}%" 
                                for dt, cov in coverage_summary.items()]))
        
        # Step 4: Fetch the most recent 60 days of data from Supabase
        logger.info(f"Fetching 60 days of health data from Supabase for user {user_id}")
        supabase_health_data = supabase.get_health_data_for_analysis(user_id, days=60)
        
        # Convert Supabase data to HealthKitData model
        supabase_health_model = HealthKitData(
            heartRate=supabase_health_data["heartRate"],
            steps=supabase_health_data["steps"],
            sleep=supabase_health_data["sleep"],
            workout=supabase_health_data["workout"],
            distance=supabase_health_data["distance"],
            basalEnergy=supabase_health_data["basalEnergy"],
            flightsClimbed=supabase_health_data["flightsClimbed"],
            userInfo=supabase_health_data["userInfo"],
            # Include activity data - this will be used for activeEnergy since we consolidated the data
            activity=supabase_health_data["activity"],
            activeEnergy=supabase_health_data["activity"]  # Use activity data for activeEnergy for backward compatibility
        )
        
        # Log the data counts for analysis without printing actual data
        logger.info(f"Using Supabase data for analysis:")
        for data_type, data_list in supabase_health_data.items():
            if isinstance(data_list, list):
                logger.info(f"  - {data_type}: {len(data_list)} records")
        
        # Check if we're still missing data after the upload
        if (len(supabase_health_model.heartRate) == 0 and 
            len(supabase_health_model.steps) == 0 and 
            len(supabase_health_model.sleep) == 0 and 
            len(supabase_health_model.workout) == 0 and
            len(supabase_health_data["activity"]) == 0):
            
            logger.warning("No health data found in Supabase after uploading. Using uploaded data directly.")
            # Fall back to the uploaded data if nothing in Supabase
            supabase_health_model = health_data
        
        # Check if we have enough data for analysis - require at least one primary data type
        required_data_types = ["heartRate", "steps", "sleep"]
        available_required = {
            "heartRate": len(supabase_health_model.heartRate) > 0,
            "steps": len(supabase_health_model.steps) > 0, 
            "sleep": len(supabase_health_model.sleep) > 0
        }
        
        available_data_types = [dt for dt, has_data in available_required.items() if has_data]
        missing_data_types = [dt for dt, has_data in available_required.items() if not has_data]
        
        logger.info(f"Available data types: {available_data_types}")
        logger.info(f"Missing data types: {missing_data_types}")
        
        if not available_data_types:
            error_msg = "Insufficient health data for analysis. At least one of heart rate, steps, or sleep data is required."
            logger.error(error_msg)
            
            # Get detailed missing data info for better error message
            missing_data_info = supabase.check_missing_data_types(user_id, days=60)
            coverage_summary = missing_data_info.get("dataCoverageSummary", {})
            missing_details = {
                dt: {
                    "coverage": cov.get("coverage_percentage", 0),
                    "missing_days": cov.get("missing_days", 60)
                } for dt, cov in coverage_summary.items() if dt in required_data_types
            }
            
            # Return a structured error response that still matches the AnalysisResult format
            # so the iOS app can display a meaningful message
            return AnalysisResult(
                user_id=user_id,
                prediction=0,
                risk_level="INSUFFICIENT_DATA",
                risk_score=0.0,
                contributing_factors={
                    "error": 1.0,
                    "message": error_msg,
                    "missingDataTypes": {dt: 1.0 for dt in missing_data_types},
                    "availableDataTypes": {dt: 1.0 for dt in available_data_types}
                },
                analysis_date=datetime.now().isoformat()
            )
        
        # Step 5: Run the prediction using data from Supabase
        logger.info(f"Running prediction using Supabase data for user {user_id}")
        result = predictor.predict(supabase_health_model.dict())
        
        # Step 6: Store analysis result in Supabase
        logger.info(f"Storing analysis result for user {user_id}")
        
        # Create a result dictionary with fields that match the Supabase schema
        result_dict = {
            "user_id": result.user_id,
            "prediction": result.prediction,
            "riskLevel": result.risk_level,
            "riskScore": result.risk_score,
            "contributingFactors": result.contributing_factors,
            "analysis_date": result.analysis_date
        }
        
        # Add data quality information to the result
        data_quality = {
            "qualityScore": updated_completeness.get('quality_score', 0) if 'updated_completeness' in locals() else completeness_check.get('quality_score', 0),
            "dataTypes": {
                "heartRate": len(supabase_health_model.heartRate) > 0,
                "steps": len(supabase_health_model.steps) > 0,
                "sleep": len(supabase_health_model.sleep) > 0,
                "workout": len(supabase_health_model.workout) > 0,
                "activity": len(supabase_health_model.activity) > 0
            },
            "message": f"Analysis based on {len(available_data_types)}/{len(required_data_types)} required data types."
        }
        
        # Add the data quality information to contributing factors
        if result_dict["contributingFactors"] is None:
            result_dict["contributingFactors"] = {}
        result_dict["contributingFactors"]["dataQuality"] = data_quality
        
        try:
            logger.info(f"Saving analysis result to Supabase for user {user_id}")
            save_success = supabase.save_analysis_result(user_id, result_dict)
            if save_success:
                logger.info("Successfully saved analysis result to database.")
            else:
                logger.warning("Failed to save analysis result to database. Continuing anyway.")
        except Exception as e:
            logger.error(f"Error during analysis result saving: {e}", exc_info=True)
            # Don't fail the request just because saving failed
        
        logger.info(f"Analysis complete. Risk level: {result.risk_level}, Score: {result.risk_score:.2f}")
        
        # Return the result object directly - the iOS app expects this format
        return result
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/check-analysis")
async def check_analysis(
    request: AnalysisRequest,
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """Check if a new analysis should be run for the user"""
    try:
        user_id = request.userId
        
        # Check if we should run a new analysis
        if request.forceRun:
            should_run = True
        else:
            should_run = supabase.should_run_analysis(user_id)
        
        # Get the latest analysis result
        latest_analysis = supabase.get_latest_analysis(user_id)
        
        return {
            "user_id": user_id,
            "shouldRunAnalysis": should_run,
            "latestAnalysis": latest_analysis,
            "message": "New analysis needed" if should_run else "Latest analysis is still valid"
        }
    except Exception as e:
        logger.error(f"Error checking analysis status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check analysis status: {str(e)}")

@app.get("/latest-analysis/{user_id}")
async def get_latest_analysis(
    request: Request,
    user_id: str,
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """Get the latest analysis result for a user"""
    try:
        # Ensure the user can only access their own data
        if user_id != auth_info["user_id"] and user_id != "me":
            raise HTTPException(status_code=403, detail="Not authorized to access this user's data")
        
        # If user_id is "me", use the authenticated user's ID
        if user_id == "me":
            user_id = auth_info["user_id"]
            
        # Get the latest analysis result
        latest_analysis = supabase.get_latest_analysis(user_id)
        
        if not latest_analysis:
            logger.info(f"No analysis results found for user {user_id}")
            
            # For iOS app compatibility, if this is a direct request from the app,
            # return a properly formatted empty result instead of a custom object
            user_agent = request.headers.get("user-agent", "").lower()
            if "cfnetwork" in user_agent or "darwin" in user_agent:
                return AnalysisResult(
                    user_id=user_id,
                    prediction=0,
                    risk_level="NO_DATA",
                    risk_score=0.0,
                    contributing_factors={},
                    analysis_date=datetime.now().isoformat()
                )
                
            return {
                "user_id": user_id,
                "hasAnalysis": False,
                "message": "No analysis results found for this user"
            }
        
        # Ensure all required fields are present for the iOS app
        required_fields = ["risk_score", "risk_level", "analysis_date", "contributing_factors"]
        for field in required_fields:
            if field not in latest_analysis:
                logger.warning(f"Missing required field '{field}' in analysis result")
                if field == "contributing_factors":
                    latest_analysis[field] = {}
                elif field == "risk_score":
                    latest_analysis[field] = 0.0
                elif field == "risk_level":
                    latest_analysis[field] = "UNKNOWN"
                elif field == "analysis_date":
                    latest_analysis[field] = datetime.now().isoformat()
        
        # Ensure the user_id is consistent with the requested user
        latest_analysis["user_id"] = user_id
        
        logger.info(f"Returning analysis result: {latest_analysis}")
        
        # For iOS app compatibility, if this is a direct request from the app,
        # return the analysis result directly, not wrapped in a container
        user_agent = request.headers.get("user-agent", "").lower()
        if "cfnetwork" in user_agent or "darwin" in user_agent:
            # This is likely a request from the iOS app
            logger.info("iOS app detected, returning direct format")
            return latest_analysis
        
        # Otherwise return the wrapped format
        return {
            "user_id": user_id,
            "hasAnalysis": True,
            "analysis": latest_analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get latest analysis: {str(e)}")

@app.get("/latest-data-timestamp/{user_id}")
async def get_latest_data_timestamp(
    user_id: str,
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """Get the latest data timestamp for a user"""
    try:
        # Ensure the user can only access their own data
        if user_id != auth_info["user_id"] and user_id != "me":
            raise HTTPException(status_code=403, detail="Not authorized to access this user's data")
        
        # If user_id is "me", use the authenticated user's ID
        if user_id == "me":
            user_id = auth_info["user_id"]
            
        # Get the latest data timestamp
        latest_timestamp = supabase.get_latest_data_timestamp(user_id)
        
        if not latest_timestamp:
            return {
                "user_id": user_id,
                "hasData": False,
                "latestTimestamp": datetime.now().isoformat(),
                "message": "No health data found for this user"
            }
        
        return {
            "user_id": user_id,
            "hasData": True,
            "latestTimestamp": latest_timestamp.isoformat(),
            "message": f"Latest data timestamp: {latest_timestamp.isoformat()}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest data timestamp: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get latest data timestamp: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint (no auth required)"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/data-quality/{user_id}")
async def check_data_quality(
    user_id: str,
    days: int = Query(60, description="Number of days to check for data"),
    min_coverage: int = Query(70, description="Minimum coverage percentage required"),
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """
    Detailed diagnostic endpoint for data quality and completeness.
    Validates data completeness and provides guidance for improving data quality.
    """
    try:
        # Ensure the user can only access their own data
        if user_id != auth_info["user_id"] and user_id != "me":
            raise HTTPException(status_code=403, detail="Not authorized to access this user's data")
        
        # If user_id is "me", use the authenticated user's ID
        if user_id == "me":
            user_id = auth_info["user_id"]
        
        # Get a detailed data completeness check
        completeness_check = supabase.check_data_completeness(user_id, days, min_coverage)
        
        # Get per-type data timestamps
        latest_timestamps = supabase.get_latest_data_timestamp_per_type(user_id)
        
        # Convert datetime objects to strings for JSON serialization
        timestamp_strings = {
            data_type: timestamp.isoformat() if timestamp else None
            for data_type, timestamp in latest_timestamps.items()
        }
        
        # Generate specific feedback based on data quality
        feedback = []
        improvement_suggestions = []
        
        # Check if data is complete
        if completeness_check.get("complete", False):
            feedback.append(f"Data quality is good! Quality score: {completeness_check.get('quality_score', 0)}%")
        else:
            feedback.append(f"Data quality needs improvement. Quality score: {completeness_check.get('quality_score', 0)}%")
            
            # Add specific feedback for missing data types
            missing_required = completeness_check.get("missing_required", [])
            if missing_required:
                feedback.append(f"Missing required data types: {', '.join(missing_required)}")
                
                for data_type in missing_required:
                    if data_type == "heartRate":
                        improvement_suggestions.append("Wear your Apple Watch consistently to capture heart rate data")
                    elif data_type == "steps":
                        improvement_suggestions.append("Carry your iPhone or wear your Apple Watch to capture step data")
                    elif data_type == "sleep":
                        improvement_suggestions.append("Wear your Apple Watch during sleep or use a sleep tracking app")
            
            # Check if there are specific problematic time periods
            missing_timelines = completeness_check.get("missing_timelines", {})
            if missing_timelines:
                for data_type, ranges in missing_timelines.items():
                    if len(ranges) > 0:
                        # Find the longest gap
                        longest_gap = max(ranges, key=lambda x: x.get("days", 0))
                        feedback.append(f"Largest gap in {data_type} data: {longest_gap.get('days', 0)} days " +
                                     f"from {longest_gap.get('start', 'unknown')} to {longest_gap.get('end', 'unknown')}")
        
        # Construct response with all the details
        response = {
            "user_id": user_id,
            "dataQualityScore": completeness_check.get("quality_score", 0),
            "dataComplete": completeness_check.get("complete", False),
            "dataCompleteness": {
                data_type: {
                    "complete": result.get("complete", False),
                    "coverage": result.get("coverage_percentage", 0),
                    "availableDays": result.get("available_days", 0),
                    "lastUpdated": timestamp_strings.get(data_type, None)
                }
                for data_type, result in completeness_check.get("coverage_results", {}).items()
            },
            "feedback": feedback,
            "improvementSuggestions": improvement_suggestions,
            "details": {
                "missingTimelines": completeness_check.get("missing_timelines", {}),
                "latestTimestamps": timestamp_strings,
                "timeRange": completeness_check.get("time_range", {})
            }
        }
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check data quality: {str(e)}")

@app.get("/check-missing-data/{user_id}")
async def check_missing_data(
    user_id: str,
    days: int = Query(60, description="Number of days to check for data"),
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """Check which data types are missing in Supabase for the last N days"""
    try:
        # Ensure the user can only access their own data
        if user_id != auth_info["user_id"] and user_id != "me":
            raise HTTPException(status_code=403, detail="Not authorized to access this user's data")
        
        # If user_id is "me", use the authenticated user's ID
        if user_id == "me":
            user_id = auth_info["user_id"]
            
        # Check missing data types
        missing_data_info = supabase.check_missing_data_types(user_id, days)
        
        return {
            "user_id": user_id,
            "checkPeriodDays": days,
            "startDate": (datetime.now() - timedelta(days=days)).isoformat(),
            "endDate": datetime.now().isoformat(),
            **missing_data_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking missing data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check missing data: {str(e)}")

@app.get("/data-collection-guidance")
async def data_collection_guidance(
    auth_info: dict = Depends(validate_auth)
):
    """
    Provides guidance on what data to collect for an accurate mental health analysis.
    """
    user_id = auth_info["user_id"]
    
    # Required data types with descriptions and collection methods
    data_types = {
        "heartRate": {
            "name": "Heart Rate",
            "importance": "High",
            "description": "Heart rate variability is a key indicator of mental health and stress levels.",
            "collection_methods": [
                "Wear your Apple Watch consistently throughout the day",
                "Ensure the watch is snug but comfortable for accurate readings",
                "Enable background heart rate monitoring in the Watch app"
            ],
            "minimum_requirement": "At least 8 hours of heart rate data daily"
        },
        "steps": {
            "name": "Steps",
            "importance": "High",
            "description": "Physical activity levels correlate strongly with mental wellbeing.",
            "collection_methods": [
                "Carry your iPhone or wear your Apple Watch throughout the day",
                "Enable Motion & Fitness tracking in your Privacy settings"
            ],
            "minimum_requirement": "Daily step counts for most days of the week"
        },
        "sleep": {
            "name": "Sleep",
            "importance": "High",
            "description": "Sleep patterns are critical indicators of mental health status.",
            "collection_methods": [
                "Wear your Apple Watch during sleep",
                "Set up Sleep tracking in the Health app",
                "Create a Sleep Schedule in the Health app"
            ],
            "minimum_requirement": "At least 5 nights of sleep data per week"
        },
        "activity": {
            "name": "Active Energy",
            "importance": "Medium",
            "description": "Calories burned during activity help measure exercise intensity and frequency.",
            "collection_methods": [
                "Wear your Apple Watch during exercise",
                "Use the Workout app to track specific activities"
            ],
            "minimum_requirement": "Data for most active days"
        },
        "workout": {
            "name": "Workouts",
            "importance": "Medium",
            "description": "Structured exercise has specific benefits for mental health.",
            "collection_methods": [
                "Use the Workout app on your Apple Watch to track exercises",
                "Remember to end workouts when complete"
            ],
            "minimum_requirement": "Track any intentional exercise sessions"
        }
    }
    
    # Get user's current data quality to provide personalized guidance
    try:
        supabase = get_supabase()
        completeness_check = supabase.check_data_completeness(user_id, days=60, min_coverage_percentage=70)
        latest_timestamps = supabase.get_latest_data_timestamp_per_type(user_id)
        
        # Add status to each data type based on user's data
        for data_type, info in data_types.items():
            coverage_results = completeness_check.get("coverage_results", {})
            if data_type in coverage_results:
                result = coverage_results[data_type]
                info["status"] = "Complete" if result.get("complete", False) else "Incomplete"
                info["coverage_percentage"] = result.get("coverage_percentage", 0)
                info["available_days"] = result.get("available_days", 0)
                
                # Add last updated timestamp
                timestamp = latest_timestamps.get(data_type)
                info["last_updated"] = timestamp.isoformat() if timestamp else None
            else:
                info["status"] = "No Data"
                info["coverage_percentage"] = 0
                info["available_days"] = 0
                info["last_updated"] = None
        
        # Generate personalized recommendations
        recommendations = []
        if completeness_check.get("quality_score", 0) < 70:
            missing_required = completeness_check.get("missing_required", [])
            for data_type in missing_required:
                if data_type in data_types:
                    recommendations.append({
                        "priority": "High",
                        "data_type": data_types[data_type]["name"],
                        "action": f"Start collecting {data_types[data_type]['name']} data using the methods described"
                    })
            
            # Add recommendations for improving existing but incomplete data
            for data_type, result in coverage_results.items():
                if data_type in data_types and not result.get("complete", False) and data_type not in missing_required:
                    recommendations.append({
                        "priority": "Medium",
                        "data_type": data_types[data_type]["name"],
                        "action": f"Improve your {data_types[data_type]['name']} data collection consistency"
                    })
        else:
            recommendations.append({
                "priority": "Low",
                "data_type": "All",
                "action": "Continue your current data collection habits - you're doing great!"
            })
        
        return {
            "data_types": data_types,
            "quality_score": completeness_check.get("quality_score", 0),
            "personalized_recommendations": recommendations,
            "general_tips": [
                "Consistency is more important than perfect data - try to collect data most days",
                "Wear your Apple Watch consistently for best results",
                "Make sure your Apple Health permissions are enabled for all tracking",
                "Sync your Apple Watch regularly with your iPhone",
                "Ensure your Apple Watch has sufficient battery life for continuous tracking"
            ]
        }
    
    except Exception as e:
        # Return the guidance without personalization if there's an error
        logger.warning(f"Error getting personalized guidance: {e}")
        return {
            "data_types": data_types,
            "general_tips": [
                "Consistency is more important than perfect data - try to collect data most days",
                "Wear your Apple Watch consistently for best results",
                "Make sure your Apple Health permissions are enabled for all tracking",
                "Sync your Apple Watch regularly with your iPhone",
                "Ensure your Apple Watch has sufficient battery life for continuous tracking"
            ]
        }

@app.post("/update-profile")
async def update_profile(
    profile_data: UserProfileData,
    supabase: SupabaseClient = Depends(get_supabase),
    auth_info: dict = Depends(validate_auth)
):
    """Update user profile with gender and birth date"""
    try:
        # Ensure user can only update their own profile
        if profile_data.userId != auth_info["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to update this user's profile")
        
        user_id = auth_info["user_id"]
        
        # Store profile data in Supabase
        success = supabase.update_user_profile(
            user_id=user_id,
            gender=profile_data.gender,
            birthdate=profile_data.birthdate,
            age=profile_data.age
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update profile")
        
        return {
            "user_id": user_id,
            "success": True,
            "message": "Profile updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")

@app.post("/api/analyze_health_data", response_model=AnalysisResult)
async def analyze_health_data(token: TokenData):
    """
    Analyze user health data to predict mental health indicators.
    
    This endpoint:
    1. Validates the user authentication token
    2. Retrieves health data from Supabase directly without transformations
    3. Processes the data using the mental health algorithm
    4. Returns the analysis results
    """
    try:
        logger.info(f"Analyzing health data for user: {token.sub}")
        start_time = time.time()
        
        # Get user ID from token
        user_id = token.sub
        
        # Create predictor instance if not already cached
        if user_id not in user_predictors:
            user_predictors[user_id] = MentalHealthPredictor()
            
        predictor = user_predictors[user_id]
        
        # Get health data directly from Supabase without format conversions
        supabase_client = SupabaseClient()
        health_data_dict = supabase_client.get_health_data_for_analysis(user_id)
        
        # Convert dictionary to HealthKitData object
        health_data = HealthKitData(**health_data_dict)
        
        # Log data counts
        logger.info(f"Retrieved health data for analysis: HR={len(health_data.heartRate or [])}, " 
                    f"Sleep={len(health_data.sleep or [])}, Activity={len(health_data.activity or [])}")
        
        # Analyze the health data
        if any([
            len(health_data.heartRate or []) == 0,
            len(health_data.sleep or []) == 0,
            len(health_data.activity or []) == 0,
            len(health_data.steps or []) == 0
        ]):
            logger.warning(f"Missing data for at least one health metric: user_id={user_id}")
            
        # Use the predictor to transform data and make predictions
        analysis_result = predictor.predict(health_data.dict())
        
        # Log success
        logger.info(f"Analysis completed successfully in {time.time() - start_time:.2f}s")
        
        # Return the analysis results
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing health data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze health data: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Get API settings from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # Run the API
    uvicorn.run("app:app", host=host, port=port, reload=debug) 