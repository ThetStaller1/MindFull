from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Body, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Optional, Any
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
        self.feature_names = None
        self.feature_importance = None
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
                self.feature_importance = pd.read_csv(feature_importance_path)
                logger.info(f"Feature importance loaded with {len(self.feature_importance)} features")
                
                # Extract feature names from feature importance CSV
                # We prioritize this since it has all 38 features the model expects
                self.feature_names = self.feature_importance['feature'].tolist()
                logger.info(f"Using feature names from importance file: {len(self.feature_names)} features")
                
                # If we expect a certain number of features, verify we have them all
                if self.expected_feature_count is not None:
                    if len(self.feature_names) != self.expected_feature_count:
                        logger.warning(
                            f"Feature count mismatch in feature_importance.csv. "
                            f"Expected {self.expected_feature_count}, got {len(self.feature_names)}."
                        )
                        # Only add dummy features if absolutely necessary
                        if len(self.feature_names) < self.expected_feature_count:
                            missing_count = self.expected_feature_count - len(self.feature_names)
                            logger.warning(f"Adding {missing_count} dummy features to match model requirements")
                            for i in range(missing_count):
                                self.feature_names.append(f"dummy_feature_{i}")
                            logger.info(f"Updated feature count: {len(self.feature_names)}")
            else:
                logger.warning("Feature importance file not found, will check for feature_names.json instead")
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            logger.warning("Will check for feature_names.json instead")
        
        # Only load feature_names.json if we still don't have feature names
        if self.feature_names is None:
            feature_names_path = os.path.join(os.path.dirname(__file__), "model", "feature_names.json")
            logger.info(f"Loading feature names from: {feature_names_path}")
            try:
                if os.path.exists(feature_names_path):
                    with open(feature_names_path, 'r') as f:
                        self.feature_names = json.load(f)
                    logger.info(f"Feature names loaded from JSON: {len(self.feature_names)} features")
                    
                    # If model expects more features than we have, add dummy features
                    if self.expected_feature_count is not None and len(self.feature_names) < self.expected_feature_count:
                        missing_count = self.expected_feature_count - len(self.feature_names)
                        logger.warning(f"Adding {missing_count} dummy features to match model requirements")
                        for i in range(missing_count):
                            dummy_name = f"dummy_feature_{i}"
                            if dummy_name not in self.feature_names:
                                self.feature_names.append(dummy_name)
                        logger.info(f"Updated feature count: {len(self.feature_names)}")
                else:
                    logger.warning("Feature names file not found, cannot determine feature list")
            except Exception as e:
                logger.error(f"Error loading feature names: {e}")
        
        # Final check to ensure we have the feature names
        if self.feature_names is None:
            logger.warning("No feature names available from any source, this will likely cause prediction errors")
        else:
            logger.info(f"Final feature set has {len(self.feature_names)} features: {self.feature_names[:5]}...")
            
        logger.info("Mental Health Predictor initialized successfully")
    
    def transform_data(self, health_data: HealthKitData) -> pd.DataFrame:
        """Transform HealthKit data to the format expected by the model using the standardized extractor"""
        logger.info("Transforming health data using HealthKit extractor...")
        
        try:
            # Use the HealthKit extractor to process the data consistently with test code
            features_df = self.extractor.process_healthkit_data(
                health_data,
                person_id=health_data.userInfo.get('personId', '1001'),
                age=health_data.userInfo.get('age', 33),
                gender_binary=health_data.userInfo.get('genderBinary', 1)
            )
            
            # Log the feature information without dumping large dataframes
            logger.info(f"Features extracted successfully: shape={features_df.shape}")
            if logger.level <= logging.DEBUG:  # Only log detailed feature info at DEBUG level
                logger.debug(f"Feature columns: {list(features_df.columns)[:10]}...")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error transforming health data: {e}", exc_info=True)
            
            # Fall back to the old transformation method if the new one fails
            logger.warning("Falling back to original transformation method")
            
            # Create base dataframe with user info
            fallback_features_df = pd.DataFrame({
                'person_id': [health_data.userInfo.get('personId', '1001')],
                'age': [health_data.userInfo.get('age', 33)],
                'gender_binary': [health_data.userInfo.get('genderBinary', 1)]
            })
            
            # Add dummy values for required features
            for feature in self.feature_names:
                if feature not in fallback_features_df.columns and feature != 'person_id':
                    fallback_features_df[feature] = 0
            
            logger.warning(f"Created fallback feature dataframe with {len(fallback_features_df.columns)} columns")
            return fallback_features_df
    
    def predict(self, health_data: HealthKitData) -> AnalysisResult:
        """Process health data and make a prediction"""
        # Transform HealthKit data to features
        features_df = self.transform_data(health_data)
        
        # Log the input data shape
        logger.info(f"Input feature dataframe shape: {features_df.shape}")
        
        # Drop person_id before prediction (if it exists) - EXACTLY as in test_mental_health_model.py
        X = features_df.drop('person_id', axis=1) if 'person_id' in features_df.columns else features_df
        
        # Log summarized feature information instead of full columns
        feature_summary = f"{len(X.columns)} features (first 5: {X.columns[:5].tolist()}...)"
        logger.info(f"Using {feature_summary}")
        
        # Check if we need to reorder/filter columns to match training data
        if self.feature_names is not None:
            # Add missing columns expected by the model
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            if missing_cols:
                missing_count = len(missing_cols)
                logger.warning(f"Missing {missing_count} features expected by model")
                if logger.level <= logging.DEBUG and missing_cols:  # Only log at DEBUG level
                    logger.debug(f"First 5 missing features: {missing_cols[:5]}...")
                
                # Add missing columns with zero values
                for col in missing_cols:
                    X[col] = 0
            
            # Keep only the features expected by the model, in the correct order
            try:
                X = X[self.feature_names]
                logger.info(f"Using {len(self.feature_names)} feature columns for prediction")
            except KeyError as e:
                logger.error(f"KeyError when selecting features: {e}")
                # Handle case where feature names might be wrong
                # Get common features between what we have and what we need
                common_features = [col for col in self.feature_names if col in X.columns]
                logger.info(f"Falling back to {len(common_features)} common features")
                X = X[common_features]
                # Add any remaining needed features as zeros
                for col in set(self.feature_names) - set(common_features):
                    X[col] = 0
                # Reorder columns to match feature_names
                X = X[self.feature_names]
            
            # Check if model expects specific number of features
            if self.expected_feature_count is not None:
                if X.shape[1] != self.expected_feature_count:
                    logger.warning(f"Feature count mismatch. Model expects {self.expected_feature_count}, got {X.shape[1]}. Adjusting...")
                    
                    if X.shape[1] < self.expected_feature_count:
                        # Add dummy features with zeros if needed
                        missing_feature_count = self.expected_feature_count - X.shape[1]
                        for i in range(missing_feature_count):
                            feature_name = f"dummy_feature_{i}"
                            if feature_name not in X.columns:
                                X[feature_name] = 0
                        logger.info(f"Added {missing_feature_count} dummy features. New shape: {X.shape}")
                    elif X.shape[1] > self.expected_feature_count:
                        # Too many features - need to remove some
                        # This should not happen with proper feature_names, but just in case
                        logger.warning(f"Too many features ({X.shape[1]}), trimming to {self.expected_feature_count}")
                        X = X.iloc[:, :self.expected_feature_count]
        
        # Make prediction with the model - EXACTLY as in test_mental_health_model.py
        try:
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
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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
    validation_result = supabase.validate_token(auth_token)
    
    if not validation_result.get('valid', False):
        logger.error(f"Invalid auth token: {validation_result.get('error', 'Unknown error')}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Return user info from the token
    return {
        "user_id": validation_result['user']['id'],
        "email": validation_result['user']['email']
    }

# Authentication routes
@app.post("/register", response_model=TokenData)
async def register(user_data: UserRegister, supabase: SupabaseClient = Depends(get_supabase)):
    """Register a new user"""
    try:
        result = supabase.register_user(user_data.email, user_data.password)
        
        if "error" in result:
            logger.error(f"Registration failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TokenData(
            access_token=result["session"]["access_token"],
            refresh_token=result["session"]["refresh_token"],
            expires_at=result["session"]["expires_at"],
            user=UserInfo(
                id=result["user"]["id"],
                email=result["user"]["email"]
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
        
        if "error" in result:
            logger.error(f"Login failed: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        
        return TokenData(
            access_token=result["session"]["access_token"],
            refresh_token=result["session"]["refresh_token"],
            expires_at=result["session"]["expires_at"],
            user=UserInfo(
                id=result["user"]["id"],
                email=result["user"]["email"]
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
            "activeEnergy": len(health_data.activeEnergy),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout),
            "basalEnergy": len(health_data.basalEnergy),
            "flightsClimbed": len(health_data.flightsClimbed),
            "distance": len(health_data.distance)
        }
        logger.info(f"Received data counts: {data_counts}")
        
        # Set the user ID from auth in the health data
        user_id = auth_info["user_id"]
        health_data.userInfo["personId"] = user_id
        
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
            "activeEnergy": len(health_data.activeEnergy),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout)
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
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')) > last_date
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
                    if datetime.fromisoformat(record['endDate'].replace('Z', '+00:00')) > last_date
                ]
                if len(health_data.steps) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.steps)} new step records after {last_date}")
            elif health_data.steps:
                # No existing steps data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.steps)} new step records (no previous data)")
            
            # Filter active energy data
            if latest_timestamps.get('activeEnergy') and health_data.activeEnergy:
                last_date = latest_timestamps['activeEnergy']
                health_data.activeEnergy = [
                    record for record in health_data.activeEnergy 
                    if datetime.fromisoformat(record['endDate'].replace('Z', '+00:00')) > last_date
                ]
                if len(health_data.activeEnergy) > 0:
                    has_new_data = True
                    logger.info(f"Found {len(health_data.activeEnergy)} new active energy records after {last_date}")
            elif health_data.activeEnergy:
                # No existing active energy data but we have new data
                has_new_data = True
                logger.info(f"Found {len(health_data.activeEnergy)} new active energy records (no previous data)")
            
            # Filter sleep data
            if latest_timestamps.get('sleep') and health_data.sleep:
                last_date = latest_timestamps['sleep']
                health_data.sleep = [
                    record for record in health_data.sleep 
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')) > last_date
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
                    if datetime.fromisoformat(record['startDate'].replace('Z', '+00:00')) > last_date
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
            "activeEnergy": len(health_data.activeEnergy),
            "sleep": len(health_data.sleep),
            "workout": len(health_data.workout)
        }
        
        logger.info(f"Data counts before filtering: {data_counts_before}")
        logger.info(f"Data counts after filtering: {data_counts_after}")
        
        # Step 2: Store health data in Supabase (only if we have new data)
        if has_new_data:
            logger.info(f"Storing {sum(data_counts_after.values())} new health data records for user {user_id}")
            
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
            activeEnergy=supabase_health_data["activeEnergy"],
            sleep=supabase_health_data["sleep"],
            workout=supabase_health_data["workout"],
            distance=supabase_health_data["distance"],
            basalEnergy=supabase_health_data["basalEnergy"],
            flightsClimbed=supabase_health_data["flightsClimbed"],
            userInfo=supabase_health_data["userInfo"]
        )
        
        # Log the data counts for analysis without printing actual data
        logger.info(f"Using Supabase data for analysis:")
        for data_type, data_list in supabase_health_data.items():
            if isinstance(data_list, list):
                logger.info(f"  - {data_type}: {len(data_list)} records")
        
        # Check if we're still missing data after the upload
        if (len(supabase_health_model.heartRate) == 0 and 
            len(supabase_health_model.steps) == 0 and 
            len(supabase_health_model.activeEnergy) == 0 and 
            len(supabase_health_model.sleep) == 0 and 
            len(supabase_health_model.workout) == 0):
            
            logger.warning("No health data found in Supabase after uploading. Using uploaded data directly.")
            # Fall back to the uploaded data if nothing in Supabase
            supabase_health_model = health_data
        
        # Check if we have enough data for analysis - require at least one primary data type
        required_data_types = ["heartRate", "steps", "sleep"]
        available_required = [
            "heartRate" if len(supabase_health_model.heartRate) > 0 else None,
            "steps" if len(supabase_health_model.steps) > 0 else None,
            "sleep" if len(supabase_health_model.sleep) > 0 else None
        ]
        available_required = [dt for dt in available_required if dt is not None]
        
        if not available_required:
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
            
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": error_msg,
                    "required_data_types": required_data_types,
                    "available_data_types": available_required,
                    "missing_details": missing_details,
                    "suggestion": "Please sync more health data from your Apple Watch and try again."
                }
            )
        
        # Step 5: Run the prediction using data from Supabase
        logger.info(f"Running prediction using Supabase data for user {user_id}")
        result = predictor.predict(supabase_health_model)
        
        # Step 6: Store analysis result in Supabase
        logger.info(f"Storing analysis result for user {user_id}")
        supabase.store_analysis_result(result.dict())
        
        # Add data coverage information to the result
        result_dict = result.dict()
        result_dict["dataQuality"] = {
            "qualityScore": completeness_check.get("quality_score", 0),
            "dataTypes": {
                "heartRate": len(supabase_health_model.heartRate) > 0,
                "steps": len(supabase_health_model.steps) > 0,
                "activeEnergy": len(supabase_health_model.activeEnergy) > 0,
                "sleep": len(supabase_health_model.sleep) > 0,
                "workout": len(supabase_health_model.workout) > 0
            },
            "message": f"Analysis based on {len(available_required)}/{len(required_data_types)} required data types."
        }
        
        logger.info(f"Analysis complete. Risk level: {result.riskLevel}, Score: {result.riskScore:.2f}")
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
            "userId": user_id,
            "shouldRunAnalysis": should_run,
            "latestAnalysis": latest_analysis,
            "message": "New analysis needed" if should_run else "Latest analysis is still valid"
        }
    except Exception as e:
        logger.error(f"Error checking analysis status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to check analysis status: {str(e)}")

@app.get("/latest-analysis/{user_id}")
async def get_latest_analysis(
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
            return {
                "userId": user_id,
                "hasAnalysis": False,
                "message": "No analysis results found for this user"
            }
        
        return {
            "userId": user_id,
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
                "userId": user_id,
                "hasData": False,
                "latestTimestamp": datetime.now().isoformat(),
                "message": "No health data found for this user"
            }
        
        return {
            "userId": user_id,
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
            "userId": user_id,
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
            "userId": user_id,
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
        "activeEnergy": {
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