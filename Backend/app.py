from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header, Body, Request
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

# Import Supabase client
from supabase_client import SupabaseClient

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
        
        try:
            # Use joblib.load instead of pickle.load
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
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
            else:
                logger.warning("Feature importance file not found, will not include importance data")
                self.feature_importance = None
        except Exception as e:
            logger.error(f"Error loading feature importance: {e}")
            self.feature_importance = None
        
        # Load feature names - EXACTLY as in test_mental_health_model.py
        feature_names_path = os.path.join(os.path.dirname(__file__), "model", "feature_names.json")
        logger.info(f"Loading feature names from: {feature_names_path}")
        try:
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
        
        # Set the user ID from auth in the health data
        user_id = auth_info["user_id"]
        health_data.userInfo["personId"] = user_id
        
        # Step 1: Store health data in Supabase
        logger.info(f"Storing health data for user {user_id}")
        if not supabase.store_health_data(user_id, health_data.dict()):
            logger.warning(f"Some health data may not have been stored properly for user {user_id}")
        
        # Step 2: Fetch the most recent 60 days of data from Supabase
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
        
        # Log the data counts for analysis
        logger.info(f"Using Supabase data for analysis:")
        logger.info(f"  - Heart Rate: {len(supabase_health_model.heartRate)} records")
        logger.info(f"  - Steps: {len(supabase_health_model.steps)} records")
        logger.info(f"  - Active Energy: {len(supabase_health_model.activeEnergy)} records")
        logger.info(f"  - Sleep: {len(supabase_health_model.sleep)} records")
        logger.info(f"  - Workout: {len(supabase_health_model.workout)} records")
        
        # Step 3: Run the prediction using data from Supabase
        logger.info(f"Running prediction using Supabase data for user {user_id}")
        result = predictor.predict(supabase_health_model)
        
        # Step 4: Store analysis result in Supabase
        logger.info(f"Storing analysis result for user {user_id}")
        supabase.store_analysis_result(result.dict())
        
        logger.info(f"Analysis complete. Risk level: {result.riskLevel}, Score: {result.riskScore:.2f}")
        return result
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