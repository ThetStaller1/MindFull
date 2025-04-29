import os
import logging
import json
from typing import Dict, List, Any, Optional
from supabase import create_client, Client
from datetime import datetime, timedelta
from dotenv import load_dotenv

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
        Store health data from HealthKit to Supabase tables.
        
        Args:
            health_data: Dictionary with HealthKit data arrays for each type
            user_id: The user ID to store data for
            
        Returns:
            Boolean indicating success
        """
        try:
            logger.info(f"Storing HealthKit data for user {user_id}")
            
            # Track storage success
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
            
            # Process active energy data for activity tracking
            if 'activeEnergy' in health_data and health_data['activeEnergy']:
                activity_records = []
                
                for record in health_data['activeEnergy']:
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
                        
                        # Create activity record
                        activity_record = {
                            'person_id': user_id,
                            'date': date_str,
                            'activity_calories': int(value),
                            'calories_bmr': 1500,  # Default basal metabolic rate
                            'calories_out': int(value) + 1500,  # Activity calories + BMR
                            'floors': 0,  # Default value
                            'elevation': 0,  # Default value
                            'very_active_minutes': 30,  # Placeholder, refined if workout data exists
                            'fairly_active_minutes': 30,  # Placeholder
                            'lightly_active_minutes': 60,  # Placeholder
                            'sedentary_minutes': 1440 - 120,  # Rest of day
                            'marginal_calories': 0,  # Default value
                            'steps': 0  # Will be updated separately
                        }
                        
                        activity_records.append(activity_record)
                    except Exception as e:
                        logger.warning(f"Error processing active energy record: {e}")
                
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
                
                for record in health_data['sleep']:
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
                        
                        # Create sleep record - ensure all duration fields are integers
                        sleep_record = {
                            'person_id': user_id,
                            'sleep_date': date_str,
                            'is_main_sleep': True,  # Default to main sleep
                            'minute_in_bed': int(float(value) + 30),  # Asleep + awake, ensure integer
                            'minute_asleep': int(float(value)),  # Ensure integer
                            'minute_after_wakeup': 10,  # Default placeholder
                            'minute_awake': 20,  # Placeholder estimate
                            'minute_restless': 10,  # Placeholder estimate
                            'minute_deep': int(float(value) * 0.2),  # Ensure integer
                            'minute_light': int(float(value) * 0.5),  # Ensure integer
                            'minute_rem': int(float(value) * 0.3),  # Ensure integer
                            'minute_wake': 20  # Same as minute_awake for consistency
                        }
                        
                        sleep_records.append(sleep_record)
                    except Exception as e:
                        logger.warning(f"Error processing sleep record: {e}")
                
                if sleep_records:
                    try:
                        # Insert records in batches for efficiency
                        for i in range(0, len(sleep_records), 10):
                            batch = sleep_records[i:i+10]
                            self.service_client.table('fitbit_sleep_daily_summary').upsert(batch).execute()
                    
                        logger.info(f"Inserted {len(sleep_records)} sleep records")
                    except Exception as e:
                        logger.error(f"Error inserting sleep records: {e}", exc_info=True)
                        success = False
            
            # Update activity records with workout data if available
            if 'workout' in health_data and health_data['workout']:
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
                            # Update existing record
                            record_id = result.data[0]['id']
                            self.service_client.table('fitbit_activity') \
                                .update({'very_active_minutes': minutes}) \
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
                                'very_active_minutes': int(minutes),
                                'fairly_active_minutes': 30,
                                'lightly_active_minutes': 180,
                                'sedentary_minutes': 1440 - (int(minutes) + 210),
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

    def upload_health_data(self, user_id: str, health_data: Dict[str, Any]) -> bool:
        """
        Upload health data from Apple Health to Supabase.
        
        Args:
            user_id: The user ID to upload data for
            health_data: Dictionary containing Apple Health data
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            success = True
            
            # Upload person dataset if provided
            if 'userInfo' in health_data and health_data['userInfo']:
                user_info = health_data['userInfo']
                
                # Default values
                gender = user_info.get('gender', 'UNKNOWN')
                age = user_info.get('age', '')
                
                # For backward compatibility: old data might have genderBinary
                if not gender or gender == 'UNKNOWN':
                    gender_binary = user_info.get('genderBinary', 1)
                    gender = 'FEMALE' if gender_binary == 1 else 'MALE'
                
                # Check if age is a string date or a number
                age_str = str(age)
                if age_str.isdigit():
                    # Convert age number to a birthdate string
                    birth_year = datetime.now().year - int(age_str)
                    age = f"{birth_year}-01-01T00:00:00Z"
                
                # Create person record
                person_record = {
                    'person_id': user_id,
                    'gender': gender,
                    'age': age,
                    'update_date': datetime.now().isoformat()
                }
                
                try:
                    # Upsert person record
                    self.service_client.table('person_dataset').upsert(person_record).execute()
                    logger.info(f"Updated person record for {user_id}")
                except Exception as e:
                    logger.error(f"Error upserting person record: {e}", exc_info=True)
                    success = False
            
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
                    except Exception as e:
                        logger.error(f"Error inserting heart rate records: {e}", exc_info=True)
                        success = False
            
            # Process step data
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
                        
                        # Create step record
                        step_record = {
                            'person_id': user_id,
                            'date': date_str,
                            'sum_steps': int(value)
                            # Only include columns that exist in the database schema
                        }
                        
                        step_records.append(step_record)
                    except Exception as e:
                        logger.warning(f"Error processing step record: {e}")
                
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

            # Process activity data
            if 'activeEnergy' in health_data and health_data['activeEnergy']:
                activity_records = []
                
                for record in health_data['activeEnergy']:
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
                        
                        # Create activity record (starting with just calories)
                        activity_record = {
                            'person_id': user_id,
                            'date': date_str,
                            'activity_calories': int(value),
                            'very_active_minutes': 0,  # Default until workout data is processed
                            'fairly_active_minutes': 0,
                            'lightly_active_minutes': 0,
                            'sedentary_minutes': 1440  # Default to full day
                        }
                        
                        activity_records.append(activity_record)
                    except Exception as e:
                        logger.warning(f"Error processing activity record: {e}")
                
                # Process workout data to update activity records
                if 'workout' in health_data and health_data['workout']:
                    for record in health_data['workout']:
                        try:
                            # Extract date from ISO format
                            start_date = record.get('startDate', '')
                            if 'T' in start_date:
                                date_str = start_date.split('T')[0]
                            else:
                                continue  # Skip if we can't get a date
                            
                            # Get the duration in minutes
                            duration = record.get('duration', 0)
                            if isinstance(duration, str):
                                try:
                                    duration = float(duration)
                                except:
                                    duration = 0
                            
                            # Convert to minutes
                            minutes = int(duration / 60)
                            
                            # Find matching record by date
                            for activity_record in activity_records:
                                if activity_record['date'] == date_str:
                                    # Update active minutes
                                    activity_record['very_active_minutes'] += minutes
                                    # Reduce sedentary minutes
                                    activity_record['sedentary_minutes'] = max(0, activity_record['sedentary_minutes'] - minutes)
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing workout record: {e}")
                
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
                
                for record in health_data['sleep']:
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
                        
                        # Create sleep record
                        sleep_record = {
                            'person_id': user_id,
                            'sleep_date': date_str,
                            'is_main_sleep': True,  # Default to main sleep
                            'minute_in_bed': int(value) + 30,  # Asleep + awake
                            'minute_asleep': int(value),
                            'minute_after_wakeup': 10,  # Default placeholder
                            'minute_awake': 20,  # Placeholder estimate
                            'minute_restless': 10,  # Placeholder estimate
                            'minute_deep': int(value) * 0.2,  # Placeholder: ~20% of sleep is deep
                            'minute_light': int(value) * 0.5,  # Placeholder: ~50% of sleep is light
                            'minute_rem': int(value) * 0.3,  # Placeholder: ~30% of sleep is REM
                            'minute_wake': 20  # Same as minute_awake for consistency
                        }
                        
                        sleep_records.append(sleep_record)
                    except Exception as e:
                        logger.warning(f"Error processing sleep record: {e}")
                
                if sleep_records:
                    try:
                        # Insert records in batches for efficiency
                        for i in range(0, len(sleep_records), 10):
                            batch = sleep_records[i:i+10]
                            self.service_client.table('fitbit_sleep_daily_summary').upsert(batch).execute()
                        
                        logger.info(f"Inserted {len(sleep_records)} sleep records")
                    except Exception as e:
                        logger.error(f"Error inserting sleep records: {e}", exc_info=True)
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error uploading health data: {str(e)}", exc_info=True)
            return False
    
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
                'activeEnergy': None,
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
                latest_timestamps['activeEnergy'] = datetime.fromisoformat(f"{date_str}T23:59:59")
            
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
            latest_timestamps['workout'] = latest_timestamps['activeEnergy']
            
            logger.info(f"Latest timestamps: {latest_timestamps}")
            return latest_timestamps
            
        except Exception as e:
            logger.error(f"Error getting latest data timestamps: {str(e)}", exc_info=True)
            # In case of error, return all None to force fresh data upload
            return {k: None for k in latest_timestamps.keys()}
    
    def get_health_data_for_analysis(self, user_id: str, days: int = 60, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Get health data for analysis from all health data tables for a user
        
        Args:
            user_id: The user ID
            days: Number of days of data to retrieve
            batch_size: Number of records to fetch per batch (pagination size)
            
        Returns:
            Dictionary with all health data in a format compatible with the mental health model
        """
        try:
            logger.info(f"Getting health data for analysis from user {user_id}")
            
            # Calculate the date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Initialize data structure for results
            health_data = {
                "heartRate": [],
                "steps": [],
                "activeEnergy": [],
                "sleep": [],
                "workout": [],
                "distance": [],
                "basalEnergy": [],
                "flightsClimbed": [],
                "userInfo": {
                    "userId": user_id,
                    "age": 33,  # Default value, will be updated if profile data is found
                    "genderBinary": 1  # Default value (female), will be updated if profile data is found
                }
            }
            
            # Get profile data
            try:
                profile_result = self.service_client.table('person_dataset') \
                    .select('*') \
                    .eq('person_id', user_id) \
                    .execute()
                
                if profile_result.data and len(profile_result.data) > 0:
                    profile = profile_result.data[0]
                    
                    # Get gender and convert to binary (1 for female, 0 for others)
                    gender = profile.get('gender', '')
                    gender_binary = 1 if gender == 'FEMALE' else 0
                    
                    # Get age from birthdate
                    birthdate_str = profile.get('age', '')  # Stored in the 'age' column
                    if birthdate_str:
                        try:
                            birthdate = datetime.fromisoformat(birthdate_str.replace('Z', '+00:00'))
                            age = end_date.year - birthdate.year
                            # Adjust age if birthday hasn't occurred yet this year
                            if (end_date.month, end_date.day) < (birthdate.month, birthdate.day):
                                age -= 1
                        except Exception as e:
                            logger.warning(f"Error calculating age from birthdate: {e}")
                            age = 33  # Default age
                    else:
                        age = 33  # Default age
                    
                    # Update userInfo with profile data
                    health_data["userInfo"].update({
                        "age": age,
                        "genderBinary": gender_binary,
                        "gender": gender
                    })
                    
                    logger.info(f"Retrieved profile for user {user_id}: age={age}, gender={gender}, genderBinary={gender_binary}")
                else:
                    logger.warning(f"No profile found for user {user_id}, using default values")
            except Exception as e:
                logger.warning(f"Error retrieving user profile: {e}")
            
            # Format dates for Supabase queries
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch heart rate data with pagination
            def fetch_heart_rate_data():
                offset = 0
                total_records = 0
                has_more = True
                
                while has_more:
                    heart_rate_result = self.service_client.table('fitbit_heart_rate_level') \
                        .select('*') \
                        .eq('person_id', user_id) \
                        .gte('date', start_date_str) \
                        .range(offset, offset + batch_size - 1) \
                        .execute()
                        
                    if heart_rate_result.data:
                        for entry in heart_rate_result.data:
                            date_str = entry['date']
                            date_time = datetime.strptime(f"{date_str} 12:00:00", '%Y-%m-%d %H:%M:%S')
                            iso_date = date_time.isoformat() + 'Z'
                            
                            health_data["heartRate"].append({
                                "type": "HKQuantityTypeIdentifierHeartRate",
                                "startDate": iso_date,
                                "endDate": iso_date,
                                "value": entry['avg_rate'],
                                "unit": "count/min"
                            })
                        
                        records_fetched = len(heart_rate_result.data)
                        total_records += records_fetched
                        offset += batch_size
                        
                        # Check if we have more records to fetch
                        has_more = records_fetched == batch_size
                    else:
                        has_more = False
                
                return total_records
            
            # Fetch step data with pagination
            def fetch_step_data():
                offset = 0
                total_records = 0
                has_more = True
                
                while has_more:
                    steps_result = self.service_client.table('fitbit_intraday_steps') \
                        .select('*') \
                        .eq('person_id', user_id) \
                        .gte('date', start_date_str) \
                        .range(offset, offset + batch_size - 1) \
                        .execute()
                        
                    if steps_result.data:
                        for entry in steps_result.data:
                            date_str = entry['date']
                            date_time = datetime.strptime(f"{date_str} 23:59:59", '%Y-%m-%d %H:%M:%S')
                            iso_date = date_time.isoformat() + 'Z'
                            
                            health_data["steps"].append({
                                "type": "HKQuantityTypeIdentifierStepCount",
                                "startDate": iso_date,
                                "endDate": iso_date,
                                "value": entry['sum_steps'],
                                "unit": "count"
                            })
                        
                        records_fetched = len(steps_result.data)
                        total_records += records_fetched
                        offset += batch_size
                        
                        # Check if we have more records to fetch
                        has_more = records_fetched == batch_size
                    else:
                        has_more = False
                
                return total_records
            
            # Fetch activity data with pagination
            def fetch_activity_data():
                offset = 0
                total_records = 0
                has_more = True
                
                while has_more:
                    activity_result = self.service_client.table('fitbit_activity') \
                        .select('*') \
                        .eq('person_id', user_id) \
                        .gte('date', start_date_str) \
                        .range(offset, offset + batch_size - 1) \
                        .execute()
                        
                    if activity_result.data:
                        for entry in activity_result.data:
                            date_str = entry['date']
                            date_time = datetime.strptime(f"{date_str} 23:59:59", '%Y-%m-%d %H:%M:%S')
                            iso_date = date_time.isoformat() + 'Z'
                            
                            # Active energy
                            health_data["activeEnergy"].append({
                                "type": "HKQuantityTypeIdentifierActiveEnergyBurned",
                                "startDate": iso_date,
                                "endDate": iso_date,
                                "value": entry['activity_calories'],
                                "unit": "kcal"
                            })
                            
                            # Workout data
                            if entry['very_active_minutes'] > 0:
                                duration_seconds = entry['very_active_minutes'] * 60
                                
                                health_data["workout"].append({
                                    "type": "HKWorkoutTypeIdentifier",
                                    "startDate": iso_date,
                                    "endDate": iso_date,
                                    "duration": duration_seconds,
                                    "workoutActivityType": 37,
                                    "totalEnergyBurned": entry['activity_calories'],
                                    "value": duration_seconds
                                })
                        
                        records_fetched = len(activity_result.data)
                        total_records += records_fetched
                        offset += batch_size
                        
                        # Check if we have more records to fetch
                        has_more = records_fetched == batch_size
                    else:
                        has_more = False
                
                return total_records
            
            # Fetch sleep data with pagination
            def fetch_sleep_data():
                offset = 0
                total_records = 0
                has_more = True
                
                while has_more:
                    sleep_result = self.service_client.table('fitbit_sleep_daily_summary') \
                        .select('*') \
                        .eq('person_id', user_id) \
                        .gte('sleep_date', start_date_str) \
                        .range(offset, offset + batch_size - 1) \
                        .execute()
                        
                    if sleep_result.data:
                        for entry in sleep_result.data:
                            date_str = entry['sleep_date']
                            date_time = datetime.strptime(f"{date_str} 22:00:00", '%Y-%m-%d %H:%M:%S')
                            iso_date = date_time.isoformat() + 'Z'
                            
                            end_time = date_time + timedelta(minutes=entry['minute_asleep'])
                            iso_end_date = end_time.isoformat() + 'Z'
                            
                            health_data["sleep"].append({
                                "type": "HKCategoryTypeIdentifierSleepAnalysis",
                                "startDate": iso_date,
                                "endDate": iso_end_date,
                                "value": entry['minute_asleep'],
                                "unit": "min"
                            })
                        
                        records_fetched = len(sleep_result.data)
                        total_records += records_fetched
                        offset += batch_size
                        
                        # Check if we have more records to fetch
                        has_more = records_fetched == batch_size
                    else:
                        has_more = False
                
                return total_records
            
            # Execute the fetch operations and log the results
            hr_count = fetch_heart_rate_data()
            steps_count = fetch_step_data()
            activity_count = fetch_activity_data()
            sleep_count = fetch_sleep_data()
            
            logger.info(f"Retrieved health data for user {user_id}: heart rate: {hr_count}, steps: {steps_count}, activity: {activity_count}, sleep: {sleep_count}")
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error getting health data for analysis: {str(e)}", exc_info=True)
            return {
                "heartRate": [],
                "steps": [],
                "activeEnergy": [],
                "sleep": [],
                "workout": [],
                "distance": [],
                "basalEnergy": [],
                "flightsClimbed": [],
                "userInfo": {
                    "userId": user_id,
                    "age": 33,
                    "genderBinary": 1
                }
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
                "activeEnergy": True,
                "sleep": True,
                "workout": True
            }
            
            data_coverage = {
                "heartRate": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "steps": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
                "activeEnergy": {"count": 0, "days_covered": 0, "earliest_date": None, "latest_date": None},
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
                missing_data["activeEnergy"] = False
                data_coverage["activeEnergy"]["count"] = len(activity_result.data)
                dates = sorted(list(set([row['date'] for row in activity_result.data])))
                data_coverage["activeEnergy"]["days_covered"] = len(dates)
                data_coverage["activeEnergy"]["earliest_date"] = dates[0] if dates else None
                data_coverage["activeEnergy"]["latest_date"] = dates[-1] if dates else None
                
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
                    "activeEnergy": True,
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
        optional_data_types = ["activeEnergy", "workout"]
        
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
                
                # Add specific condition for workout and activeEnergy
                condition = None
                if data_type == 'workout':
                    condition = self.service_client.table(table) \
                        .select(date_field) \
                        .eq('person_id', user_id) \
                        .gt('very_active_minutes', 0) \
                        .gte(date_field, (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'))
                elif data_type == 'activeEnergy':
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