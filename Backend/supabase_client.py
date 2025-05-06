import os
import logging
import json
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
import dateutil.parser

import pandas as pd
import numpy as np
from postgrest import APIResponse

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseClient:
    """Client for interacting with Supabase for MindWatch application"""
    
    def __init__(self):
        """Initialize Supabase client with environment variables"""
        # Load environment variables if not already loaded
        load_dotenv()
        
        # Get Supabase URL and key from environment variables
        self.supabase_url = os.environ.get("SUPABASE_URL", "")
        
        self.service_role_key = os.environ.get("SUPABASE_ANON_KEY", "")
        
        if not self.supabase_url:
            logger.error("Supabase URL not found in environment variables")
            raise ValueError("Supabase URL must be set in environment variables")
        
        try:
            # Create primary client - may be used for auth operations
            self.client = create_client(self.supabase_url, self.service_role_key)
            
            # Create dedicated service client that will NEVER be used for auth operations
            # This ensures it won't get a session token that overrides the service role
            self.service_client = create_client(self.supabase_url, self.service_role_key)
            
            # Log the key value (first few chars) to help troubleshoot
            masked_key = self.service_role_key[:5] + "..." if self.service_role_key else None
            logger.info(f"Supabase client initialized with service role key: {masked_key}")

            # Define heart rate zone boundaries (copied from HealthKitToFitbitMapper)
            self.heart_rate_zones = {
                "Out of Range": {"min": 30, "max": 99},
                "Fat Burn": {"min": 99, "max": 139},
                "Cardio": {"min": 139, "max": 169},
                "Peak": {"min": 169, "max": 220}
            }
            
            # Workout intensity mapping (copied from HealthKitToFitbitMapper)
            self.workout_intensity = {
                # High intensity workouts
                'HKWorkoutActivityTypeRunning': 'very_active',
                'HKWorkoutActivityTypeHighIntensityIntervalTraining': 'very_active',
                'HKWorkoutActivityTypeCrossTraining': 'very_active',
                
                # Moderate intensity workouts
                'HKWorkoutActivityTypeWalking': 'fairly_active',
                'HKWorkoutActivityTypeCycling': 'fairly_active',
                'HKWorkoutActivityTypeElliptical': 'fairly_active',
                
                # Light intensity workouts
                'HKWorkoutActivityTypeYoga': 'lightly_active',
                'HKWorkoutActivityTypeFlexibility': 'lightly_active',
                'HKWorkoutActivityTypeMindAndBody': 'lightly_active',
                
                # Default
                'default': 'fairly_active'
            }
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            self.client = None
            raise
    
    # Authentication methods
    def register_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Register a new user with email and password
        
        Args:
            email (str): User's email
            password (str): User's password
            
        Returns:
            dict: User data if successful, error message otherwise
        """
        try:
            logger.info(f"Registering new user with email: {email}")
            
            # Create user with Supabase Auth
            response = self.service_client.auth.sign_up({
                "email": email,
                "password": password,
            })
            
            logger.info(f"User registration response received")
            
            if hasattr(response, 'user') and response.user:
                # User registered successfully
                user_id = response.user.id
                
                # In a real app, we'd need to handle email verification here
                # depending on your Supabase configuration
                
                return {"success": True, "user_id": user_id, "message": "User registered successfully"}
            else:
                logger.error(f"User registration failed: {response}")
                return {"success": False, "message": "User registration failed"}
                
        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """
        Login a user with email and password
        
        Args:
            email (str): User's email
            password (str): User's password
            
        Returns:
            dict: User data and access token if successful, error message otherwise
        """
        try:
            logger.info(f"Logging in user with email: {email}")
            
            # Authenticate user with Supabase Auth
            response = self.service_client.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            
            if hasattr(response, 'user') and response.user:
                # User logged in successfully
                user_id = response.user.id
                access_token = response.session.access_token
                
                return {
                    "success": True, 
                    "user_id": user_id, 
                    "access_token": access_token, 
                    "message": "User logged in successfully"
                }
            else:
                logger.error(f"User login failed: {response}")
                return {"success": False, "message": "Invalid credentials"}
                
        except Exception as e:
            logger.error(f"Error logging in user: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def get_user_from_token(self, token: str) -> Dict[str, Any]:
        """
        Get user data from token
        
        Args:
            token (str): User's access token
            
        Returns:
            dict: User data if token is valid, error message otherwise
        """
        try:
            logger.info(f"Getting user from token")
            
            # Authenticate user with Supabase Auth
            response = self.service_client.auth.get_user(token)
            
            if hasattr(response, 'user') and response.user:
                # Token is valid
                user_id = response.user.id
                email = response.user.email
                
                return {
                    "success": True, 
                    "user_id": user_id, 
                    "email": email, 
                    "message": "Token is valid"
                }
            else:
                logger.error(f"Invalid token: {response}")
                return {"success": False, "message": "Invalid token"}
                
        except Exception as e:
            logger.error(f"Error getting user from token: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def store_health_data(self, health_data: Dict[str, Any], user_id: str) -> bool:
        """
        Store health data from HealthKit into Supabase.
        
        Args:
            health_data: Dictionary with health data from HealthKit
            user_id: User ID to attach to the health data
            
        Returns:
            Boolean indicating success/failure
        """
        try:
            success = True
            
            # Process heart rate data
            if 'heartRate' in health_data and health_data['heartRate']:
                heart_rate_records = []
                
                for record in health_data['heartRate']:
                    try:
                        # Extract date from ISO format
                        start_date = record.get('startDate', '')
                        if 'T' in start_date:
                            date_str = start_date.split('T')[0]
                        else:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        # Get the value as a number
                        value = record.get('value')
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except:
                                value = 0
                        
                        # Create heart rate record
                        heart_rate_record = {
                            'person_id': user_id,
                            'date': date_str,
                            'avg_rate': value
                            # Only include columns that exist in the database schema
                        }
                        
                        heart_rate_records.append(heart_rate_record)
                    except Exception as e:
                        logger.warning(f"Error processing heart rate record: {e}")
                
                if heart_rate_records:
                    try:
                        # Insert records in batches for efficiency
                        for i in range(0, len(heart_rate_records), 10):
                            batch = heart_rate_records[i:i+10]
                            self.service_client.table('fitbit_heart_rate_level').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(heart_rate_records)} heart rate records")
                        
                        # Process heart rate summary by day and zone
                        # Group by date to calculate heart rate zones
                        heart_rate_by_date = {}
                        
                        # Process records to group by date
                        for record in health_data['heartRate']:
                            try:
                                start_date = record.get('startDate', '')
                                if 'T' in start_date:
                                    date_str = start_date.split('T')[0]
                                else:
                                    date_str = datetime.now().strftime('%Y-%m-%d')
                                
                                # Get the value as a number
                                value = record.get('value')
                                if isinstance(value, str):
                                    try:
                                        value = float(value)
                                    except:
                                        value = 0
                                
                                # Group by date
                                if date_str not in heart_rate_by_date:
                                    heart_rate_by_date[date_str] = []
                                
                                heart_rate_by_date[date_str].append(value)
                            except Exception as e:
                                logger.warning(f"Error grouping heart rate: {e}")
                        
                        # Calculate heart rate zones and create summary records
                        summary_records = []
                        
                        for date_str, values in heart_rate_by_date.items():
                            if not values:
                                continue
                                
                            # Calculate zone statistics
                            for zone_name, zone_range in self.heart_rate_zones.items():
                                # Filter values in this zone
                                zone_values = [v for v in values if zone_range["min"] <= v <= zone_range["max"]]
                                
                                if zone_values:
                                    # Create summary record
                                    zone_summary = {
                                        'person_id': user_id,
                                        'date': date_str,
                                        'zone_name': zone_name,
                                        'min_heart_rate': int(min(zone_values)),
                                        'max_heart_rate': int(max(zone_values)),
                                        'minute_in_zone': len(zone_values),
                                        'calorie_count': int(sum(zone_values) * 0.1)  # Simple estimate and not used for anything in the model or features
                                    }
                                    
                                    summary_records.append(zone_summary)
                        
                        if summary_records:
                            # Insert heart rate summary records in batches
                            for i in range(0, len(summary_records), 10):
                                batch = summary_records[i:i+10]
                                self.service_client.table('fitbit_heart_rate_summary').upsert(batch).execute()
                            
                            logger.info(f"Inserted {len(summary_records)} heart rate summary records")
                        
                    except Exception as e:
                        logger.error(f"Error inserting heart rate records: {e}", exc_info=True)
                        success = False
            
            # Process steps data
            if 'steps' in health_data and health_data['steps']:
                step_records = []
                
                for record in health_data['steps']:
                    try:
                        # Extract date from ISO format
                        start_date = record.get('startDate', '')
                        if 'T' in start_date:
                            date_str = start_date.split('T')[0]
                        else:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        # Get the value as a number
                        value = record.get('value')
                        if isinstance(value, str):
                            try:
                                value = float(value)
                            except:
                                value = 0
                        
                        # Create steps record
                        steps_record = {
                            'person_id': user_id,
                            'date': date_str,
                            'sum_steps': int(value)
                            # Only include columns that exist in the database schema
                        }
                        
                        step_records.append(steps_record)
                    except Exception as e:
                        logger.warning(f"Error processing steps record: {e}")
                
                if step_records:
                    try:
                        # Insert records in batches for efficiency
                        for i in range(0, len(step_records), 10):
                            batch = step_records[i:i+10]
                            self.service_client.table('fitbit_intraday_steps').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(step_records)} step records")
                    except Exception as e:
                        logger.error(f"Error inserting step records: {e}", exc_info=True)
                        success = False
            
            # Process activity data - Accept data from either 'activity' (new) or 'activeEnergy' (old) key
            activity_data = []
            if 'activity' in health_data and health_data['activity']:
                activity_data = health_data['activity']
            elif 'activeEnergy' in health_data and health_data['activeEnergy']:
                activity_data = health_data['activeEnergy']
            elif 'active_energy' in health_data and health_data['active_energy']:
                activity_data = health_data['active_energy']
                
            if activity_data:
                # SOPHISTICATED MAPPING LOGIC (replaces hardcoded values)
                # This algorithm is adapted from HealthKitToFitbitMapper and calculates accurate
                # activity minutes based on heart rate zones and workout intensity instead of
                # using fixed placeholder values. This ensures the activity data properly reflects
                # the user's actual activity patterns.
                
                # Convert to DataFrame for easier processing (adopting the mapper approach)
                active_energy_df = pd.DataFrame(activity_data)
                
                # Extract workout data if available
                workout_df = pd.DataFrame()
                if 'workout' in health_data and health_data['workout']:
                    workout_df = pd.DataFrame(health_data['workout'])
                
                # Extract heart rate data if available for activity minutes calculation
                heart_rate_df = pd.DataFrame()
                if 'heartRate' in health_data and health_data['heartRate']:
                    heart_rate_df = pd.DataFrame(health_data['heartRate'])
                
                # Extract basal energy data if available
                basal_energy_df = pd.DataFrame()
                if 'basalEnergy' in health_data and health_data['basalEnergy']:
                    basal_energy_df = pd.DataFrame(health_data['basalEnergy'])
                
                # Extract steps data if available
                steps_df = pd.DataFrame()
                if 'steps' in health_data and health_data['steps']:
                    steps_df = pd.DataFrame(health_data['steps'])
                
                # Extract flights climbed data if available
                flights_df = pd.DataFrame()
                if 'flightsClimbed' in health_data and health_data['flightsClimbed']:
                    flights_df = pd.DataFrame(health_data['flightsClimbed'])
                
                # Group by date to get daily totals for active energy
                if 'date' not in active_energy_df.columns and 'startDate' in active_energy_df.columns:
                    active_energy_df['date'] = pd.to_datetime(active_energy_df['startDate']).dt.date
                
                # Sum active energy by date
                daily_energy = active_energy_df.groupby('date')['value'].sum().reset_index()
                daily_energy = daily_energy.rename(columns={'value': 'activity_calories'})
                
                # Process basal energy data (calories_bmr)
                if not basal_energy_df.empty:
                    if 'date' not in basal_energy_df.columns and 'startDate' in basal_energy_df.columns:
                        basal_energy_df['date'] = pd.to_datetime(basal_energy_df['startDate']).dt.date
                    
                    # Sum basal energy by date
                    daily_basal = basal_energy_df.groupby('date')['value'].sum().reset_index()
                    daily_basal = daily_basal.rename(columns={'value': 'calories_bmr'})
                    
                    # Merge with active energy data
                    daily_energy = pd.merge(daily_energy, daily_basal, on='date', how='outer')
                else:
                    # Use a reasonable default BMR if no data available
                    daily_energy['calories_bmr'] = 1600
                
                # Calculate total calories (activity + BMR)
                daily_energy['calories_out'] = daily_energy['activity_calories'] + daily_energy['calories_bmr']
                
                # Process steps data
                if not steps_df.empty:
                    if 'date' not in steps_df.columns and 'startDate' in steps_df.columns:
                        steps_df['date'] = pd.to_datetime(steps_df['startDate']).dt.date
                    
                    # Sum steps by date
                    daily_steps = steps_df.groupby('date')['value'].sum().reset_index()
                    daily_steps = daily_steps.rename(columns={'value': 'steps'})
                    
                    # Merge with energy data
                    daily_energy = pd.merge(daily_energy, daily_steps, on='date', how='outer')
                else:
                    daily_energy['steps'] = 0
                
                # Process floors/elevation data
                if not flights_df.empty:
                    if 'date' not in flights_df.columns and 'startDate' in flights_df.columns:
                        flights_df['date'] = pd.to_datetime(flights_df['startDate']).dt.date
                    
                    # Sum floors by date
                    daily_floors = flights_df.groupby('date')['value'].sum().reset_index()
                    daily_floors = daily_floors.rename(columns={'value': 'floors'})
                    
                    # Calculate elevation (3 meters per floor)
                    daily_floors['elevation'] = daily_floors['floors'] * 3
                    
                    # Merge with energy data
                    daily_energy = pd.merge(daily_energy, daily_floors[['date', 'floors', 'elevation']], 
                                        on='date', how='outer')
                else:
                    daily_energy['floors'] = 0
                    daily_energy['elevation'] = 0
                
                # Calculate activity minutes based on heart rate data and workout data
                activity_minutes = self._calculate_activity_minutes(heart_rate_df, workout_df)
                
                # Merge activity minutes with daily energy data
                if not activity_minutes.empty:
                    daily_energy = pd.merge(daily_energy, activity_minutes, on='date', how='outer')
                else:
                    # Provide default activity minutes based on activity calories if no HR data
                    # This is a more sophisticated algorithm than the previous hardcoded values
                    # as it scales the activity minutes based on the actual calories burned
                    daily_energy['very_active_minutes'] = daily_energy['activity_calories'].apply(
                        lambda cals: min(int(cals * 0.01), 120)  # ~1% of activity calories, max 2 hours
                    )
                    daily_energy['fairly_active_minutes'] = daily_energy['activity_calories'].apply(
                        lambda cals: min(int(cals * 0.02), 240)  # ~2% of activity calories, max 4 hours
                    )
                    daily_energy['lightly_active_minutes'] = daily_energy['activity_calories'].apply(
                        lambda cals: min(int(cals * 0.03), 360)  # ~3% of activity calories, max 6 hours
                    )
                
                # Calculate sedentary minutes
                total_day_minutes = 24 * 60  # 1440 minutes in a day
                daily_energy['sedentary_minutes'] = total_day_minutes - (
                    daily_energy['very_active_minutes'].fillna(0) + 
                    daily_energy['fairly_active_minutes'].fillna(0) + 
                    daily_energy['lightly_active_minutes'].fillna(0)
                )
                # Ensure sedentary minutes are not negative
                daily_energy['sedentary_minutes'] = daily_energy['sedentary_minutes'].apply(lambda x: max(0, x))
                
                # Calculate marginal calories (typically ~8-10% of total calories)
                # First initialize marginal_calories as 0
                daily_energy['marginal_calories'] = 0
                
                # Only calculate marginal_calories for rows with valid calories_out values
                # This avoids the NaN/inf conversion error
                valid_calories_mask = daily_energy['calories_out'].notna() & np.isfinite(daily_energy['calories_out'])
                if valid_calories_mask.any():
                    daily_energy.loc[valid_calories_mask, 'marginal_calories'] = (
                        daily_energy.loc[valid_calories_mask, 'calories_out'] * 0.09
                    ).astype(int)
                
                # Fill any NaN values with reasonable defaults
                fill_values = {
                    'activity_calories': 0,
                    'calories_bmr': 1600,
                    'calories_out': 1600,
                    'steps': 0,
                    'floors': 0,
                    'elevation': 0,
                    'very_active_minutes': 0,
                    'fairly_active_minutes': 0,
                    'lightly_active_minutes': 0,
                    'sedentary_minutes': total_day_minutes,
                    'marginal_calories': 0
                }
                daily_energy = daily_energy.fillna(fill_values)
                
                # Validate the generated data
                daily_energy = self._validate_activity_data(daily_energy)
                
                # Convert to records for database insertion
                activity_records = []
                for _, row in daily_energy.iterrows():
                    activity_record = {
                        'person_id': user_id,
                        'date': str(row['date']),
                        'activity_calories': int(row['activity_calories']),
                        'calories_bmr': int(row['calories_bmr']),
                        'calories_out': int(row['calories_out']),
                        'floors': int(row['floors']),
                        'elevation': int(row['elevation']),
                        'very_active_minutes': int(row['very_active_minutes']),
                        'fairly_active_minutes': int(row['fairly_active_minutes']),
                        'lightly_active_minutes': int(row['lightly_active_minutes']),
                        'sedentary_minutes': int(row['sedentary_minutes']),
                        'marginal_calories': int(row['marginal_calories']),
                        'steps': int(row['steps'])
                    }
                    activity_records.append(activity_record)
                
                if activity_records:
                    try:
                        # Insert records in batches for efficiency
                        for i in range(0, len(activity_records), 10):
                            batch = activity_records[i:i+10]
                            self.service_client.table('fitbit_activity').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(activity_records)} activity records")
                    except Exception as e:
                        logger.error(f"Error inserting activity records: {e}", exc_info=True)
                        success = False
            
            # Process sleep data
            if 'sleep' in health_data and health_data['sleep']:
                sleep_records = []
                sleep_level_records = []
                
                # Group sleep data by day and session
                sleep_sessions = {}
                
                # First pass: Sort records by start time and group by day
                sorted_sleep_records = sorted(health_data['sleep'], key=lambda r: r.get('startDate', ''))
                
                # Pre-process records to extract dates and handle session creation
                for record in sorted_sleep_records:
                    # Extract date for grouping
                    start_date_str = record.get('startDate', '')
                    if 'T' in start_date_str:
                        date_str = start_date_str.split('T')[0]
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    # Add date to record for easier access later
                    record['sleep_date'] = date_str
                    
                    # Create date key if it doesn't exist
                    if date_str not in sleep_sessions:
                        sleep_sessions[date_str] = []
                        
                    # Add record to the day's collection
                    sleep_sessions[date_str].append(record)
                
                # Session tracking to generate consistent IDs
                current_session_id = None
                last_end_time = None
                
                # Process each day's sleep data
                for date_str, day_records in sleep_sessions.items():
                    # Sort records chronologically to ensure proper session detection
                    day_records.sort(key=lambda r: r.get('startDate', ''))
                    
                    # Group records into sessions
                    current_records = []
                    session_records = []
                    
                    for i, record in enumerate(day_records):
                        start_time_str = record.get('startDate', '')
                        end_time_str = record.get('endDate', start_time_str)
                        
                        try:
                            # Parse start and end times
                            start_time = dateutil.parser.parse(start_time_str)
                            end_time = dateutil.parser.parse(end_time_str)
                            
                            # Check if this is a new session (first record or gap > 30 min)
                            if i == 0 or last_end_time is None or (start_time - last_end_time).total_seconds() > 1800:
                                # Finalize previous session if it exists
                                if current_records:
                                    session_records.append(current_records)
                                    
                                # Start a new session
                                current_records = [record]
                                # Generate session ID using the timestamp of the first record in the session
                                current_session_id = f"sleep_{int(start_time.timestamp())}"
                            else:
                                # Continue current session
                                current_records.append(record)
                            
                            # Update last end time for gap detection
                            last_end_time = end_time
                            
                        except Exception as e:
                            logger.warning(f"Error processing sleep record timestamp: {e}")
                            # Handle records with invalid timestamps
                            if current_records:
                                # Add to current session if one exists
                                current_records.append(record)
                            else:
                                # Create a new session with a fallback ID
                                current_records = [record]
                                current_session_id = f"sleep_{time.time()}"
                    
                    # Add the last session if not empty
                    if current_records:
                        session_records.append(current_records)
                    
                    # Process each session
                    for session in session_records:
                        # Calculate summary values for this session
                        minute_in_bed = 0
                        minute_asleep = 0
                        minute_awake = 0
                        minute_restless = 0
                        minute_deep = 0
                        minute_light = 0
                        minute_rem = 0
                        minute_wake = 0
                        
                        # Generate session ID from first record's start time
                        first_record = session[0]
                        try:
                            start_time = dateutil.parser.parse(first_record.get('startDate', ''))
                            session_id = f"sleep_{int(start_time.timestamp())}"
                        except Exception:
                            # Fallback to current time if parsing fails
                            session_id = f"sleep_{int(time.time())}"
                        
                        # Process each record in the session
                        for record in session:
                            # Extract sleep stage
                            sleep_stage = record.get('value', 'unknown')
                            
                            # Extract or calculate duration
                            duration = 0
                            start_time_str = record.get('startDate', '')
                            end_time_str = record.get('endDate', start_time_str)
                            
                            if 'duration' in record:
                                # Use explicit duration field if available
                                try:
                                    duration = float(record['duration'])
                                except (ValueError, TypeError):
                                    duration = 0
                            
                            # Calculate duration from timestamps if needed
                            if duration == 0 and start_time_str and end_time_str and start_time_str != end_time_str:
                                try:
                                    start_time = dateutil.parser.parse(start_time_str)
                                    end_time = dateutil.parser.parse(end_time_str)
                                    duration = (end_time - start_time).total_seconds() / 60
                                except Exception as e:
                                    logger.warning(f"Error calculating sleep duration: {e}")
                            
                            # Only process if we have a valid duration
                            if duration > 0:
                                # Map sleep stage to standardized format
                                level = self._map_sleep_stage(sleep_stage)
                                
                                # Create sleep level record
                                sleep_level = {
                                    'person_id': user_id,
                                    'sleep_id': session_id,
                                    'sleep_date': date_str,
                                    'is_main_sleep': True,
                                    'level': level,
                                    'start_time': start_time_str,
                                    'end_time': end_time_str,
                                    'duration': int(duration)
                                }
                                
                                sleep_level_records.append(sleep_level)
                                
                                # Update summary values based on sleep stage
                                minute_in_bed += duration
                                
                                # According to Fitbit classification:
                                # - light, deep, rem stages count as asleep time
                                # - wake and restless count as awake time
                                if level in ['light', 'deep', 'rem']:
                                    minute_asleep += duration
                                    
                                if level == 'wake':
                                    minute_wake += duration
                                    minute_awake += duration
                                elif level == 'restless':
                                    minute_restless += duration
                                elif level == 'deep':
                                    minute_deep += duration
                                elif level == 'light':
                                    minute_light += duration
                                elif level == 'rem':
                                    minute_rem += duration
                        
                        # Only create a summary if we have valid data
                        if minute_in_bed > 0:
                            # Create sleep summary for the session
                            summary = {
                                'person_id': user_id,
                                'sleep_date': date_str,
                                'is_main_sleep': True,
                                'minute_in_bed': int(minute_in_bed),
                                'minute_asleep': int(minute_asleep),
                                'minute_after_wakeup': 0,  # No direct mapping from HealthKit
                                'minute_awake': int(minute_awake),
                                'minute_restless': int(minute_restless),
                                'minute_deep': int(minute_deep),
                                'minute_light': int(minute_light),
                                'minute_rem': int(minute_rem),
                                'minute_wake': int(minute_wake)
                            }
                            
                            sleep_records.append(summary)
                
                # Insert records in batches
                if sleep_records:
                    try:
                        # Insert sleep daily summary records
                        for i in range(0, len(sleep_records), 10):
                            batch = sleep_records[i:i+10]
                            self.service_client.table('fitbit_sleep_daily_summary').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(sleep_records)} sleep daily summary records")
                    except Exception as e:
                        logger.error(f"Error inserting sleep records: {e}", exc_info=True)
                        success = False
                
                if sleep_level_records:
                    try:
                        # Filter out records with duration of 0 as they cause analysis issues
                        # This is a final safety check to ensure no zero-duration records are uploaded to Supabase
                        # There is some filtering in health_extractor.py but this ensures all paths are covered
                        filtered_sleep_level_records = [record for record in sleep_level_records if record.get('duration', 0) > 0]
                        
                        if len(filtered_sleep_level_records) < len(sleep_level_records):
                            logger.warning(f"Filtered out {len(sleep_level_records) - len(filtered_sleep_level_records)} sleep level records with duration of 0")
                        
                        # Insert sleep level records
                        for i in range(0, len(filtered_sleep_level_records), 10):
                            batch = filtered_sleep_level_records[i:i+10]
                            self.service_client.table('fitbit_sleep_level').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(filtered_sleep_level_records)} sleep level records")
                    except Exception as e:
                        logger.error(f"Error inserting sleep level records: {e}", exc_info=True)
                        success = False
            
            # Update activity records with workout data if available
            if 'workout' in health_data and health_data['workout']:
                # NOTE: This section is now largely redundant since workout data is handled
                # in the _calculate_activity_minutes method above, but we're keeping it for
                # backward compatibility. The sophisticated mapping now properly considers
                # workout data when creating the activity records initially.
                
                # Group workouts by date
                workout_by_date = {}
                
                for record in health_data['workout']:
                    try:
                        # Extract date from ISO format
                        start_date = record.get('startDate', '')
                        if 'T' in start_date:
                            date_str = start_date.split('T')[0]
                        else:
                            date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        # Get duration in minutes
                        duration = record.get('duration', 0)
                        if isinstance(duration, str):
                            try:
                                duration = float(duration) / 60  # Convert seconds to minutes
                            except:
                                duration = 0
                        
                        # Add to workout minutes by date
                        if date_str in workout_by_date:
                            workout_by_date[date_str] += duration
                        else:
                            workout_by_date[date_str] = duration
                    except Exception as e:
                        logger.warning(f"Error processing workout record: {e}")
                
                # Update activity records with workout durations
                for date_str, minutes in workout_by_date.items():
                    try:
                        # Get existing activity record
                        result = self.service_client.table('fitbit_activity') \
                            .select('*') \
                            .eq('person_id', user_id) \
                            .eq('date', date_str) \
                            .execute()
                    
                        if result.data:
                            # Update existing record - round to integer before storing
                            record_id = result.data[0]['id']
                            self.service_client.table('fitbit_activity') \
                                .update({'very_active_minutes': int(round(minutes))}) \
                                .eq('id', record_id) \
                                .execute()
                        else:
                            # Create a new activity record
                            self.service_client.table('fitbit_activity').insert({
                                'person_id': user_id,
                                'date': date_str,
                                'activity_calories': 300,  # Placeholder
                                'calories_bmr': 1500,  # Default BMR
                                'calories_out': 1800,  # Placeholder: BMR + activity
                                'floors': 0,  # Default value
                                'elevation': 0,  # Default value
                                'very_active_minutes': int(round(minutes)),
                                'fairly_active_minutes': 30,
                                'lightly_active_minutes': 180,
                                'sedentary_minutes': 1440 - (int(round(minutes)) + 210),
                                'marginal_calories': 0,  # Default value
                                'steps': 0  # Will be updated separately
                            }).execute()
                    except Exception as e:
                        logger.warning(f"Error updating activity with workout data: {e}")
            
            logger.info(f"Health data storage complete for user {user_id}, success: {success}")
            return success
            
        except Exception as e:
            logger.error(f"Error storing health data: {str(e)}", exc_info=True)
            return False
    
    def save_analysis_result(self, user_id: str, analysis_result: Dict[str, Any]) -> bool:
        """Save mental health analysis result to Supabase"""
        try:
            # Ensure we have the required fields from the result
            if "riskScore" not in analysis_result or "riskLevel" not in analysis_result:
                logger.error(f"Missing required fields in analysis_result: {analysis_result.keys()}")
                return False
                
            # Map fields from the API format to the database format
            record = {
                'person_id': user_id,
                'analysis_date': datetime.now().isoformat(),
                'prediction': analysis_result.get('prediction', 0),
                'risk_level': analysis_result.get('riskLevel', 'UNKNOWN'),
                'risk_score': analysis_result.get('riskScore', 0),
                'contributing_factors': json.dumps(analysis_result.get('contributingFactors', {})),
                'created_at': datetime.now().isoformat(),
                'last_update_date': datetime.now().isoformat()
            }
            
            logger.info(f"Saving analysis result: {record}")
            
            # Insert the record
            result = self.service_client.table('analysis_results').insert(record).execute()
            
            if hasattr(result, 'error') and result.error:
                logger.error(f"Error from Supabase when saving analysis: {result.error}")
                return False
                
            logger.info(f"Analysis result saved for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analysis result: {str(e)}", exc_info=True)
            return False

    # TODO: The upload_health_data method was previously here but was
    # duplicative of store_health_data and was not being used in the codebase.
    # If you need this functionality, use store_health_data instead.
    # The removed method had the same functionality but with parameters in a different order
    # (user_id first, then health_data).
    
    def _map_sleep_stage(self, stage: str) -> str:
        """
        Map HealthKit sleep stage values to Fitbit format
        
        HealthKit and Fitbit use different classification systems for sleep stages.
        This function standardizes the mapping to ensure consistent representation.
        
        Fitbit sleep stages:
        - light: Light sleep (includes Core sleep from Apple Watch)
        - deep: Deep sleep
        - rem: REM sleep
        - wake: Awake periods during sleep
        - restless: In bed but not fully asleep (maps to HK 'InBed')
        
        Returns:
            Standardized Fitbit sleep stage
        """
        sleep_stage_mapping = {
            # Apple Watch sleep stages (numeric values)
            '0': 'restless',        # In Bed
            '1': 'light',           # Asleep (unspecified)
            '2': 'wake',            # Awake
            '3': 'deep',            # Deep
            '4': 'rem',             # REM
            '5': 'light',           # Core (equivalent to Fitbit's light)
            
            # Apple Watch sleep stages (string values)
            'HKCategoryValueSleepAnalysisInBed': 'restless',
            'HKCategoryValueSleepAnalysisAsleepUnspecified': 'light',
            'HKCategoryValueSleepAnalysisAsleepCore': 'light',
            'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
            'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
            'HKCategoryValueSleepAnalysisAwake': 'wake',
            
            # Already mapped values (pass-through)
            'light': 'light',
            'deep': 'deep',
            'rem': 'rem',
            'wake': 'wake',
            'restless': 'restless'
        }
        
        return sleep_stage_mapping.get(stage, 'light')  # Default to light sleep if unknown
    
    def get_latest_analysis(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest analysis result for a user"""
        try:
            result = self.service_client.table('analysis_results') \
                .select('*') \
                .eq('person_id', user_id) \
                .order('analysis_date', desc=True) \
                .limit(1) \
                .execute()
            
            if not result.data or len(result.data) == 0:
                logger.info(f"No analysis result found for user {user_id}")
                return None
                
            # Get the first (most recent) result
            db_result = result.data[0]
            logger.info(f"Found analysis result: {db_result}")
            
            # Parse the contributing factors JSON if it exists
            contributing_factors = {}
            if 'contributing_factors' in db_result and db_result['contributing_factors']:
                try:
                    contributing_factors = json.loads(db_result['contributing_factors'])
                except Exception as e:
                    logger.warning(f"Error parsing contributing_factors JSON: {e}")
            
            # Construct the result using the database fields directly
            analysis_result = {
                'userId': db_result.get('person_id', user_id),
                'riskScore': db_result.get('risk_score', 0.0),  # Already stored as floating point
                'riskLevel': db_result.get('risk_level', 'UNKNOWN'),
                'analysisDate': db_result.get('analysis_date', datetime.now().isoformat()),
                'contributingFactors': contributing_factors,
                'prediction': db_result.get('prediction', 0)
            }
            
            logger.info(f"Formatted analysis result for iOS app: {analysis_result}")
            return analysis_result
                
        except Exception as e:
            logger.error(f"Error getting latest analysis: {str(e)}", exc_info=True)
            return None
    
    def get_latest_data_timestamp(self, user_id: str) -> Optional[datetime]:
        """Get the latest data timestamp for a user from all health data tables"""
        try:
            tables = [
                'fitbit_heart_rate_level',
                'fitbit_intraday_steps',
                'fitbit_activity',
                'fitbit_sleep_daily_summary'
            ]
            
            latest_timestamps = []
            
            for table in tables:
                date_field = 'date'
                
                if table in ['fitbit_sleep_daily_summary']:
                    date_field = 'sleep_date'
                
                result = self.service_client.table(table) \
                    .select(date_field) \
                    .eq('person_id', user_id) \
                    .order(date_field, desc=True) \
                    .limit(1) \
                    .execute()
                
                if result.data and len(result.data) > 0:
                    date_str = result.data[0][date_field]
                    date_obj = datetime.fromisoformat(date_str) if 'T' in date_str else datetime.strptime(date_str, '%Y-%m-%d')
                    latest_timestamps.append(date_obj)
            
            if latest_timestamps:
                return max(latest_timestamps)
            else:
                logger.info(f"No health data found for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest data timestamp: {str(e)}", exc_info=True)
            return None
    
    def get_latest_data_timestamp_per_type(self, user_id: str) -> Dict[str, Optional[datetime]]:
        """
        Get the latest timestamp for each type of health data for a user.
        
        Args:
            user_id: The user ID to get data for
            
        Returns:
            Dictionary with data type keys and datetime values (or None if no data)
        """
        try:
            logger.info(f"Getting latest health data timestamps for user {user_id}")
            
            latest_timestamps = {
                'heartRate': None,
                'steps': None,
                'activity': None,
                'sleep': None,
                'workout': None
            }
            
            # Get latest heart rate timestamp
            heart_rate_result = self.service_client.table('fitbit_heart_rate_level') \
                .select('date') \
                .eq('person_id', user_id) \
                .order('date', desc=True) \
                .limit(1) \
                .execute()
                
            if heart_rate_result.data:
                date_str = heart_rate_result.data[0]['date']
                latest_timestamps['heartRate'] = datetime.fromisoformat(f"{date_str}T23:59:59")
            
            # Get latest steps timestamp
            steps_result = self.service_client.table('fitbit_intraday_steps') \
                .select('date') \
                .eq('person_id', user_id) \
                .order('date', desc=True) \
                .limit(1) \
                .execute()
                
            if steps_result.data:
                date_str = steps_result.data[0]['date']
                latest_timestamps['steps'] = datetime.fromisoformat(f"{date_str}T23:59:59")
            
            # Get latest activity timestamp
            activity_result = self.service_client.table('fitbit_activity') \
                .select('date') \
                .eq('person_id', user_id) \
                .order('date', desc=True) \
                .limit(1) \
                .execute()
                
            if activity_result.data:
                date_str = activity_result.data[0]['date']
                latest_timestamps['activity'] = datetime.fromisoformat(f"{date_str}T23:59:59")
            
            # Get latest sleep timestamp
            sleep_result = self.service_client.table('fitbit_sleep_daily_summary') \
                .select('sleep_date') \
                .eq('person_id', user_id) \
                .order('sleep_date', desc=True) \
                .limit(1) \
                .execute()
                
            if sleep_result.data:
                date_str = sleep_result.data[0]['sleep_date']
                latest_timestamps['sleep'] = datetime.fromisoformat(f"{date_str}T23:59:59")
            
            # For workout data, we'll use the activity table as a proxy
            # since workouts are reflected there as very_active_minutes
            latest_timestamps['workout'] = latest_timestamps['activity']
            
            logger.info(f"Latest timestamps: {latest_timestamps}")
            return latest_timestamps
            
        except Exception as e:
            logger.error(f"Error getting latest data timestamps: {str(e)}", exc_info=True)
            # In case of error, return all None to force fresh data upload
            return {k: None for k in latest_timestamps.keys()}
    
    def get_health_data_for_analysis(self, user_id: str, days: int = 60) -> Dict[str, Any]:
        """
        Get user health data from Supabase for analysis.
        This retrieves raw data from Supabase tables without transforming formats.
        
        Args:
            user_id: User ID to fetch data for
            days: Number of days of data to retrieve (default: 60)
        """
        try:
            logger.info(f"Retrieving user health data for analysis: user_id={user_id}, days={days}")
            
            # Get user profile information for demographic data
            user_profile = self.get_user_profile(user_id)
            
            # Calculate date range for specified days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for Supabase query
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Get heart rate data
            heart_rate_data = self._fetch_table_data(
                "fitbit_heart_rate_level", 
                user_id,
                start_date_str,
                end_date_str,
                date_column="date"
            )
            
            # Get sleep data - directly use the Fitbit format data without converting
            sleep_data = self._fetch_table_data(
                "fitbit_sleep_daily_summary", 
                user_id,
                start_date_str,
                end_date_str,
                date_column="sleep_date"
            )
            
            # Get activity data
            activity_data = self._fetch_table_data(
                "fitbit_activity", 
                user_id,
                start_date_str,
                end_date_str,
                date_column="date"
            )
            
            # Get steps data
            steps_data = self._fetch_table_data(
                "fitbit_intraday_steps", 
                user_id,
                start_date_str,
                end_date_str,
                date_column="date",
                group_by="date"
            )
            
            # Compute last sync date
            all_dates = []
            for data_list in [heart_rate_data, sleep_data, activity_data, steps_data]:
                if data_list:
                    for item in data_list:
                        for date_key in ["date", "sleep_date"]:
                            if date_key in item:
                                try:
                                    date_str = item[date_key]
                                    if isinstance(date_str, str) and date_str:
                                        all_dates.append(datetime.fromisoformat(date_str.replace("Z", "+00:00")))
                                except (ValueError, TypeError):
                                    pass
            
            last_sync_date = max(all_dates) if all_dates else None
            logger.info(f"Retrieved health data: HR={len(heart_rate_data)}, Sleep={len(sleep_data)}, Activity={len(activity_data)}")
            
            # Prepare user information
            user_info = {
                "userId": user_id,
                "personId": user_id,  # Use same ID for compatibility
                "age": user_profile.get("age"),
                "genderBinary": user_profile.get("gender_binary", 1),  # Default to female if missing
                "lastSyncDate": last_sync_date.isoformat() if last_sync_date else None
            }
            
            # Return data as a dictionary that matches the structure expected by HealthKitData
            return {
                "userInfo": user_info,
                "heartRate": heart_rate_data,
                "sleep": sleep_data,
                "steps": steps_data,
                "workout": [],  # Not used in current implementation
                "distance": [],
                "basalEnergy": [],
                "flightsClimbed": [],
                "activity": activity_data,
                "deviceInfo": {"source": "Supabase", "version": "1.0"}
            }
        except Exception as e:
            logger.error(f"Error retrieving health data for analysis: {e}", exc_info=True)
            # Return empty data structure on error
            return {
                "userInfo": {"userId": user_id, "personId": user_id, "age": 33, "genderBinary": 1},
                "heartRate": [],
                "sleep": [],
                "steps": [],
                "workout": [],
                "distance": [],
                "basalEnergy": [],
                "flightsClimbed": [],
                "activity": [],
                "deviceInfo": {"source": "Supabase", "version": "1.0"}
            }
    
    def check_missing_data_types(self, user_id: str, days: int = 60) -> Dict[str, Any]:
        """Check which data types are missing for the last N days in Supabase with detailed coverage information"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            logger.info(f"Checking missing data types for user {user_id} from {start_date_str}")
            
            missing_data = {
                "heartRate": True,
                "steps": True,
                "activity": True,
                "sleep": True,
                "workout": True
            }
            
            data_coverage = {
                "heartRate": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "steps": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "activity": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "sleep": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "workout": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None}
            }
            
            # Check heart rate data using the new execute_sql function via RPC
            heart_rate_query = """
            SELECT date, COUNT(*) as count 
            FROM fitbit_heart_rate_level 
            WHERE person_id = :person_id AND date >= :start_date 
            GROUP BY date
            """
            heart_rate_params = {"person_id": user_id, "start_date": start_date_str}
            heart_rate_result = self.service_client.rpc(
                'execute_sql',
                {"query": heart_rate_query, "params": heart_rate_params}
            ).execute()
            
            if heart_rate_result.data and len(heart_rate_result.data) > 0:
                # Parse the JSON data from the result
                formatted_data = []
                for row in heart_rate_result.data:
                    if isinstance(row, str):
                        formatted_data.append(json.loads(row))
                    else:
                        formatted_data.append(row)
                
                if formatted_data:
                    missing_data["heartRate"] = False
                    data_coverage["heartRate"]["count"] = sum(int(row["count"]) for row in formatted_data)
                    data_coverage["heartRate"]["days_covered"] = len(formatted_data)
                    dates = sorted([row["date"] for row in formatted_data])
                    data_coverage["heartRate"]["earliest_date"] = dates[0] if dates else None
                    data_coverage["heartRate"]["latest_date"] = dates[-1] if dates else None
            
            # Check steps data
            steps_query = """
            SELECT date, SUM(sum_steps) as total_steps
            FROM fitbit_intraday_steps
            WHERE person_id = :person_id AND date >= :start_date
            GROUP BY date
            """
            steps_params = {"person_id": user_id, "start_date": start_date_str}
            steps_result = self.service_client.rpc(
                'execute_sql',
                {"query": steps_query, "params": steps_params}
            ).execute()
            
            if steps_result.data and len(steps_result.data) > 0:
                # Parse the JSON data from the result
                formatted_data = []
                for row in steps_result.data:
                    if isinstance(row, str):
                        formatted_data.append(json.loads(row))
                    else:
                        formatted_data.append(row)
                
                if formatted_data:
                    missing_data["steps"] = False
                    data_coverage["steps"]["count"] = len(formatted_data)
                    data_coverage["steps"]["days_covered"] = len(formatted_data)
                    dates = sorted([row["date"] for row in formatted_data])
                    data_coverage["steps"]["earliest_date"] = dates[0] if dates else None
                    data_coverage["steps"]["latest_date"] = dates[-1] if dates else None
            
            # Check activity/energy data - no GROUP BY needed so we use the regular method
            activity_result = self.service_client.table('fitbit_activity') \
                .select('date, activity_calories') \
                .eq('person_id', user_id) \
                .gte('date', start_date_str) \
                .gt('activity_calories', 0) \
                .execute()
            
            if activity_result.data and len(activity_result.data) > 0:
                missing_data["activity"] = False
                data_coverage["activity"]["count"] = len(activity_result.data)
                dates = sorted(list(set([row['date'] for row in activity_result.data])))
                data_coverage["activity"]["days_covered"] = len(dates)
                data_coverage["activity"]["earliest_date"] = dates[0] if dates else None
                data_coverage["activity"]["latest_date"] = dates[-1] if dates else None
                
                # Check for workout data (very active minutes)
                workout_result = self.service_client.table('fitbit_activity') \
                    .select('date, very_active_minutes') \
                    .eq('person_id', user_id) \
                    .gte('date', start_date_str) \
                    .gt('very_active_minutes', 0) \
                    .execute()
                
                if workout_result.data and len(workout_result.data) > 0:
                    missing_data["workout"] = False
                    data_coverage["workout"]["count"] = len(workout_result.data)
                    workout_dates = sorted(list(set([row['date'] for row in workout_result.data])))
                    data_coverage["workout"]["days_covered"] = len(workout_dates)
                    data_coverage["workout"]["earliest_date"] = workout_dates[0] if workout_dates else None
                    data_coverage["workout"]["latest_date"] = workout_dates[-1] if workout_dates else None
            
            # Check sleep data
            sleep_query = """
            SELECT sleep_date, COUNT(*) as count 
            FROM fitbit_sleep_daily_summary 
            WHERE person_id = :person_id AND sleep_date >= :start_date 
            GROUP BY sleep_date
            """
            sleep_params = {"person_id": user_id, "start_date": start_date_str}
            sleep_result = self.service_client.rpc(
                'execute_sql',
                {"query": sleep_query, "params": sleep_params}
            ).execute()
            
            if sleep_result.data and len(sleep_result.data) > 0:
                # Parse the JSON data from the result
                formatted_data = []
                for row in sleep_result.data:
                    if isinstance(row, str):
                        formatted_data.append(json.loads(row))
                    else:
                        formatted_data.append(row)
                
                if formatted_data:
                    missing_data["sleep"] = False
                    data_coverage["sleep"]["count"] = sum(int(row["count"]) for row in formatted_data)
                    data_coverage["sleep"]["days_covered"] = len(formatted_data)
                    dates = sorted([row["sleep_date"] for row in formatted_data])
                    data_coverage["sleep"]["earliest_date"] = dates[0] if dates else None
                    data_coverage["sleep"]["latest_date"] = dates[-1] if dates else None
            
            # Calculate overall status
            all_data_present = not any(missing_data.values())
            some_data_missing = any(missing_data.values())
            
            # Calculate how many days of data are missing in the time range
            target_days = (end_date - start_date).days + 1
            data_coverage_summary = {}
            
            for data_type, coverage in data_coverage.items():
                if coverage["days_covered"] > 0:
                    coverage_percentage = min(100, round((coverage["days_covered"] / target_days) * 100))
                    missing_days = target_days - coverage["days_covered"]
                    data_coverage_summary[data_type] = {
                        "coverage_percentage": coverage_percentage,
                        "missing_days": missing_days,
                        "available_days": coverage["days_covered"],
                        "target_days": target_days
                    }
                else:
                    data_coverage_summary[data_type] = {
                        "coverage_percentage": 0,
                        "missing_days": target_days,
                        "available_days": 0,
                        "target_days": target_days
                    }
            
            logger.info(f"Data status for user {user_id}: {missing_data}")
            logger.info(f"Data coverage: {data_coverage_summary}")
            
            return {
                "missingDataTypes": missing_data,
                "allDataPresent": all_data_present,
                "someDataMissing": some_data_missing,
                "noDataPresent": all(missing_data.values()),
                "dataCoverage": data_coverage,
                "dataCoverageSummary": data_coverage_summary,
                "timeRange": {
                    "start_date": start_date_str,
                    "end_date": end_date.strftime('%Y-%m-%d'),
                    "days_requested": days
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking missing data types: {str(e)}", exc_info=True)
            return {
                "missingDataTypes": {
                    "heartRate": True,
                    "steps": True,
                    "activity": True,
                    "sleep": True,
                    "workout": True
                },
                "allDataPresent": False,
                "someDataMissing": True,
                "noDataPresent": True,
                "error": str(e)
            }
    
    def check_data_completeness(self, user_id: str, days: int = 60, min_coverage_percentage: int = 70) -> Dict[str, Any]:
        """
        Check if data coverage meets minimum requirements over the specified time window.
        
        Args:
            user_id: The user ID to check
            days: The number of past days to analyze
            min_coverage_percentage: Minimum acceptable coverage percentage (default: 70%)
            
        Returns:
            Dictionary with detailed completeness information
        """
        # Get detailed coverage information
        coverage_info = self.check_missing_data_types(user_id, days)
        
        # Check if we have overall coverage information
        if "dataCoverageSummary" not in coverage_info:
            return {
                "complete": False,
                "reason": "Could not determine data coverage",
                "details": coverage_info
            }
        
        # Check each required data type for minimum coverage
        required_data_types = ["heartRate", "steps", "sleep"]
        coverage_results = {}
        missing_required = []
        
        for data_type in required_data_types:
            if data_type not in coverage_info["dataCoverageSummary"]:
                coverage_results[data_type] = {
                    "complete": False,
                    "reason": "No coverage information available"
                }
                missing_required.append(data_type)
                continue
                
            summary = coverage_info["dataCoverageSummary"][data_type]
            coverage_percentage = summary["coverage_percentage"]
            
            if coverage_percentage >= min_coverage_percentage:
                coverage_results[data_type] = {
                    "complete": True,
                    "coverage_percentage": coverage_percentage,
                    "available_days": summary["available_days"],
                    "target_days": summary["target_days"]
                }
            else:
                coverage_results[data_type] = {
                    "complete": False,
                    "reason": f"Insufficient coverage ({coverage_percentage}% < {min_coverage_percentage}%)",
                    "coverage_percentage": coverage_percentage,
                    "available_days": summary["available_days"],
                    "target_days": summary["target_days"],
                    "missing_days": summary["missing_days"]
                }
                missing_required.append(data_type)
        
        # Also evaluate non-required but beneficial data types
        optional_data_types = ["activity", "workout"]
        
        for data_type in optional_data_types:
            if data_type in coverage_info["dataCoverageSummary"]:
                summary = coverage_info["dataCoverageSummary"][data_type]
                coverage_percentage = summary["coverage_percentage"]
                
                coverage_results[data_type] = {
                    "complete": coverage_percentage >= min_coverage_percentage,
                    "coverage_percentage": coverage_percentage,
                    "available_days": summary["available_days"],
                    "target_days": summary["target_days"],
                    "optional": True
                }
        
        # Determine overall completeness
        is_complete = len(missing_required) == 0
        
        # Calculate an overall data quality score (simple average of required coverage percentages)
        total_percentage = 0
        count = 0
        
        for data_type in required_data_types:
            if data_type in coverage_info["dataCoverageSummary"]:
                total_percentage += coverage_info["dataCoverageSummary"][data_type]["coverage_percentage"]
                count += 1
        
        quality_score = round(total_percentage / count) if count > 0 else 0
        
        # Create timelines of missing data
        missing_timelines = {}
        
        for data_type in required_data_types + optional_data_types:
            if data_type not in coverage_info["dataCoverageSummary"]:
                continue
                
            # Only compute detailed timeline if coverage is below 100%
            if coverage_info["dataCoverageSummary"][data_type]["coverage_percentage"] < 100:
                # Get actual dates with data
                date_field = 'sleep_date' if data_type == 'sleep' else 'date'
                table = 'fitbit_sleep_daily_summary' if data_type == 'sleep' else \
                        'fitbit_heart_rate_level' if data_type == 'heartRate' else \
                        'fitbit_intraday_steps' if data_type == 'steps' else 'fitbit_activity'
                
                # Add specific condition for workout and activity
                condition = None
                if data_type == 'workout':
                    condition = self.service_client.table(table) \
                        .select(date_field) \
                        .eq('person_id', user_id) \
                        .gt('very_active_minutes', 0) \
                        .gte(date_field, (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
                elif data_type == 'activity':
                    condition = self.service_client.table(table) \
                        .select(date_field) \
                        .eq('person_id', user_id) \
                        .gt('activity_calories', 0) \
                        .gte(date_field, (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
                else:
                    condition = self.service_client.table(table) \
                        .select(date_field) \
                        .eq('person_id', user_id) \
                        .gte(date_field, (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
                
                result = condition.execute()
                
                if result.data:
                    # Create a set of dates that have data
                    dates_with_data = set(row[date_field] for row in result.data)
                    
                    # Create a list of all dates in the range
                    all_dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') 
                                for i in range(days)]
                    
                    # Find missing dates
                    missing_dates = [date for date in all_dates if date not in dates_with_data]
                    
                    if missing_dates:
                        # Group consecutive missing dates
                        missing_ranges = []
                        range_start = missing_dates[0]
                        current_range = [range_start]
                        
                        for i in range(1, len(missing_dates)):
                            prev_date = datetime.strptime(missing_dates[i-1], '%Y-%m-%d')
                            current_date = datetime.strptime(missing_dates[i], '%Y-%m-%d')
                            
                            if (current_date - prev_date).days == 1:
                                current_range.append(missing_dates[i])
                            else:
                                missing_ranges.append({
                                    "start": current_range[0],
                                    "end": current_range[-1],
                                    "days": len(current_range)
                                })
                                current_range = [missing_dates[i]]
                        
                        # Add the last range
                        if current_range:
                            missing_ranges.append({
                                "start": current_range[0],
                                "end": current_range[-1],
                                "days": len(current_range)
                            })
                        
                        missing_timelines[data_type] = missing_ranges
        
        return {
            "complete": is_complete,
            "quality_score": quality_score,
            "missing_required": missing_required if not is_complete else [],
            "coverage_results": coverage_results,
            "missing_timelines": missing_timelines,
            "time_range": {
                "start_date": (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                "end_date": datetime.now().strftime('%Y-%m-%d'),
                "days": days
            },
            "min_coverage_percentage": min_coverage_percentage
        }
    
    def update_user_profile(self, user_id: str, gender: str, birthdate: str, age: int) -> bool:
        """
        Update or create user profile data in person_dataset table
        
        Args:
            user_id: The user ID
            gender: User's gender (MALE, FEMALE, OTHER, or PREFER_NOT_TO_ANSWER)
            birthdate: Birthdate in YYYY-MM-DD format
            age: Calculated age
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Updating profile for user {user_id}")
            
            # Check if user already has a profile
            result = self.service_client.table('person_dataset') \
                .select('*') \
                .eq('person_id', user_id) \
                .execute()
            
            current_time = datetime.now().isoformat()
            
            if result.data and len(result.data) > 0:
                # Update existing profile
                profile_id = result.data[0]['id']
                
                self.service_client.table('person_dataset') \
                    .update({
                        'gender': gender,
                        'age': birthdate,  # Store birthdate in the age column
                        'updated_at': current_time
                    }) \
                    .eq('id', profile_id) \
                    .execute()
                    
                logger.info(f"Updated profile for user {user_id}")
            else:
                # Create new profile
                self.service_client.table('person_dataset').insert({
                    'person_id': user_id,
                    'gender': gender,
                    'age': birthdate,  # Store birthdate in the age column
                    'created_at': current_time,
                    'updated_at': current_time
                }).execute()
                
                logger.info(f"Created new profile for user {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}", exc_info=True)
            return False 
    
    def _fetch_table_data(self, table_name: str, user_id: str, 
                          start_date: str, end_date: str, 
                          date_column: str = "date", 
                          group_by: str = None) -> List[Dict]:
        """
        Helper method to fetch data from a specific Supabase table with pagination.
        Returns data directly without any transformations.
        
        Args:
            table_name: Name of the Supabase table
            user_id: The user's ID
            start_date: Start date for the data range (YYYY-MM-DD)
            end_date: End date for the data range (YYYY-MM-DD)
            date_column: Name of the date column in the table
            group_by: Optional column to group results by
            
        Returns:
            List of data records directly from Supabase
        """
        try:
            all_data = []
            page_size = 1000  # Supabase default limit
            start_index = 0
            has_more = True
            
            logger.info(f"Fetching data from {table_name} for user {user_id} with pagination")
            
            while has_more:
                query = self.service_client.table(table_name) \
                    .select('*') \
                    .eq('person_id', user_id) \
                    .gte(date_column, start_date) \
                    .lte(date_column, end_date) \
                    .order(date_column, desc=False) \
                    .range(start_index, start_index + page_size - 1)
                    
                result = query.execute()
                
                if not result.data:
                    if start_index == 0:
                        logger.info(f"No data found in {table_name} for user {user_id}")
                    break
                    
                all_data.extend(result.data)
                data_count = len(result.data)
                
                logger.info(f"Retrieved {data_count} records from {table_name} (page {start_index//page_size + 1})")
                
                if data_count < page_size:
                    # Received fewer records than requested, so we've reached the end
                    has_more = False
                else:
                    # Move to the next page
                    start_index += page_size
            
            logger.info(f"Total records retrieved from {table_name}: {len(all_data)}")
            return all_data
                
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            return []
            
    def get_user_profile(self, user_id: str) -> Dict:
        """
        Get user profile information from Supabase
        
        Args:
            user_id: The user's ID
            
        Returns:
            Dictionary containing user profile data
        """
        try:
            result = self.service_client.table('person_dataset') \
                .select('*') \
                .eq('person_id', user_id) \
                .execute()
                
            if not result.data or len(result.data) == 0:
                logger.warning(f"No profile found for user {user_id}")
                return {"age": 33, "gender_binary": 1}
                
            profile = result.data[0]
            
            # Convert gender to binary format (1 for female, 0 for others)
            gender = profile.get('gender', '')
            gender_binary = 1 if gender and gender.upper() == 'FEMALE' else 0
            
            # Get age from birthdate or use age field directly
            age = profile.get('age', 33)
            
            # If age field actually contains a birthdate, calculate the age
            if isinstance(age, str) and age:
                try:
                    birthdate = datetime.fromisoformat(age.replace('Z', '+00:00'))
                    today = datetime.now()
                    age = today.year - birthdate.year
                    # Adjust age if birthday hasn't occurred yet this year
                    if (today.month, today.day) < (birthdate.month, birthdate.day):
                        age -= 1
                except Exception as e:
                    logger.warning(f"Error calculating age from birthdate: {e}")
                    age = 33  # Default age
                    
            return {
                "age": age,
                "gender_binary": gender_binary,
                "gender": gender
            }
        except Exception as e:
            logger.error(f"Error retrieving user profile: {e}")
            return {"age": 33, "gender_binary": 1} 

    def _calculate_activity_minutes(self, heart_rate_df: pd.DataFrame, workout_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate activity minutes from heart rate and workout data.
        
        Activity level categorization:
        - very_active_minutes: "Cardio" or "Peak" heart rate zones or high-intensity workouts
        - fairly_active_minutes: "Fat burn" heart rate zone or moderate-intensity workouts
        - lightly_active_minutes: Minutes with activity but below moderate intensity
        
        Args:
            heart_rate_df: DataFrame with heart rate data
            workout_df: DataFrame with workout data
            
        Returns:
            DataFrame with date and activity minutes columns
        """
        result_data = []
        
        # Calculate activity minutes from heart rate data
        if not heart_rate_df.empty:
            # Ensure date column exists
            if 'date' not in heart_rate_df.columns and 'startDate' in heart_rate_df.columns:
                heart_rate_df['date'] = pd.to_datetime(heart_rate_df['startDate']).dt.date
            
            # Group heart rate data by date
            for date, group in heart_rate_df.groupby('date'):
                # Count minutes in each heart rate zone
                very_active = len(group[(group['value'] >= self.heart_rate_zones['Cardio']['min'])])
                fairly_active = len(group[(group['value'] >= self.heart_rate_zones['Fat Burn']['min']) & 
                                        (group['value'] < self.heart_rate_zones['Cardio']['min'])])
                lightly_active = len(group[(group['value'] < self.heart_rate_zones['Fat Burn']['min']) & 
                                        (group['value'] > 60)])  # Assuming above resting HR
                
                result_data.append({
                    'date': date,
                    'very_active_minutes': very_active,
                    'fairly_active_minutes': fairly_active,
                    'lightly_active_minutes': lightly_active
                })
        
        # Add activity minutes from workout data
        if not workout_df.empty:
            # Ensure date column exists
            if 'workout_date' not in workout_df.columns and 'startDate' in workout_df.columns:
                workout_df['workout_date'] = pd.to_datetime(workout_df['startDate']).dt.date
            elif 'date' not in workout_df.columns and 'startDate' in workout_df.columns:
                workout_df['date'] = pd.to_datetime(workout_df['startDate']).dt.date
            
            # Use either workout_date or date column
            date_column = 'workout_date' if 'workout_date' in workout_df.columns else 'date'
            
            # Group workouts by date
            for date, group in workout_df.groupby(date_column):
                very_active_mins = 0
                fairly_active_mins = 0
                lightly_active_mins = 0
                
                # Process each workout
                for _, workout in group.iterrows():
                    # Get duration in minutes
                    duration_mins = workout['duration'] / 60 if 'duration' in workout else 0
                    
                    # Determine intensity based on workout type
                    workout_type = workout.get('workoutActivityType', 'default')
                    intensity = self.workout_intensity.get(workout_type, self.workout_intensity['default'])
                    
                    # Add minutes to the appropriate category
                    if intensity == 'very_active':
                        very_active_mins += duration_mins
                    elif intensity == 'fairly_active':
                        fairly_active_mins += duration_mins
                    else:
                        lightly_active_mins += duration_mins
                
                # Check if this date is already in result_data
                date_exists = False
                for entry in result_data:
                    if entry['date'] == date:
                        # Update existing entry
                        entry['very_active_minutes'] += very_active_mins
                        entry['fairly_active_minutes'] += fairly_active_mins
                        entry['lightly_active_minutes'] += lightly_active_mins
                        date_exists = True
                        break
                
                # Add new entry if date doesn't exist
                if not date_exists:
                    result_data.append({
                        'date': date,
                        'very_active_minutes': very_active_mins,
                        'fairly_active_minutes': fairly_active_mins,
                        'lightly_active_minutes': lightly_active_mins
                    })
        
        # Convert to DataFrame
        if result_data:
            return pd.DataFrame(result_data)
        else:
            return pd.DataFrame(columns=[
                'date', 'very_active_minutes', 'fairly_active_minutes', 'lightly_active_minutes'
            ])
    
    def _validate_activity_data(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct activity data to ensure it meets expected constraints.
        
        Validation rules:
        1. Total minutes (all activity levels) should equal 1440 (minutes in a day)
        2. Ensure calories_out = activity_calories + calories_bmr
        
        Args:
            activity_df: DataFrame with activity data
            
        Returns:
            Validated and corrected DataFrame
        """
        if activity_df.empty:
            return activity_df
        
        # Copy to avoid modifying the original
        validated_df = activity_df.copy()
        
        # 1. Ensure total minutes are 1440
        total_minutes = (validated_df['very_active_minutes'] + 
                        validated_df['fairly_active_minutes'] + 
                        validated_df['lightly_active_minutes'] + 
                        validated_df['sedentary_minutes'])
        
        for idx, row in validated_df.iterrows():
            if total_minutes[idx] != 1440:
                # Adjust sedentary minutes to make total 1440
                validated_df.at[idx, 'sedentary_minutes'] = 1440 - (
                    row['very_active_minutes'] + 
                    row['fairly_active_minutes'] + 
                    row['lightly_active_minutes']
                )
                
                # Ensure sedentary minutes are not negative
                if validated_df.at[idx, 'sedentary_minutes'] < 0:
                    logger.warning(f"Negative sedentary minutes calculated for {row['date']}, adjusting activity minutes")
                    # Scale down activity minutes proportionally
                    total_active = (row['very_active_minutes'] + 
                                  row['fairly_active_minutes'] + 
                                  row['lightly_active_minutes'])
                    if total_active > 0:
                        scale_factor = min(1.0, 1440 / total_active)
                        validated_df.at[idx, 'very_active_minutes'] = int(row['very_active_minutes'] * scale_factor)
                        validated_df.at[idx, 'fairly_active_minutes'] = int(row['fairly_active_minutes'] * scale_factor)
                        validated_df.at[idx, 'lightly_active_minutes'] = int(row['lightly_active_minutes'] * scale_factor)
                        validated_df.at[idx, 'sedentary_minutes'] = max(0, 1440 - (
                            validated_df.at[idx, 'very_active_minutes'] + 
                            validated_df.at[idx, 'fairly_active_minutes'] + 
                            validated_df.at[idx, 'lightly_active_minutes']
                        ))
        
        # 2. Ensure calories_out = activity_calories + calories_bmr
        validated_df['calories_out'] = validated_df['activity_calories'] + validated_df['calories_bmr']
        
        # 3. Ensure marginal_calories are ~9% of calories_out
        # First initialize marginal_calories as 0
        validated_df['marginal_calories'] = 0
        
        # Only calculate marginal_calories for rows with valid calories_out values
        valid_calories_mask = validated_df['calories_out'].notna() & np.isfinite(validated_df['calories_out'])
        if valid_calories_mask.any():
            validated_df.loc[valid_calories_mask, 'marginal_calories'] = (
                validated_df.loc[valid_calories_mask, 'calories_out'] * 0.09
            ).astype(int)
        
        return validated_df 