import os
import logging
from typing import Dict, List, Any, Optional
from supabase import create_client
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
        self.supabase_key = os.environ.get("SUPABASE_ANON_KEY", "")
        
        if not self.supabase_url or not self.supabase_key:
            logger.error("Supabase URL or key not found in environment variables")
            raise ValueError("Supabase URL and key must be set in environment variables")
        
        try:
            # Create Supabase client
            self.client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Error initializing Supabase client: {str(e)}")
            self.client = None
            raise
    
    # Authentication methods
    def register_user(self, email: str, password: str) -> Dict[str, Any]:
        """Register a new user with Supabase Auth"""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info(f"Successfully registered user: {email}")
                
                # Create a profile entry in the profiles table
                if response.user.id:
                    try:
                        self.client.table('profiles').insert({
                            'id': response.user.id,
                            'email': email,
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.now().isoformat()
                        }).execute()
                        logger.info(f"Created profile for user: {email}")
                    except Exception as e:
                        logger.error(f"Error creating profile for new user: {str(e)}")
                
                return {
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email
                    },
                    "session": {
                        "access_token": response.session.access_token,
                        "refresh_token": response.session.refresh_token,
                        "expires_at": response.session.expires_at
                    }
                }
            else:
                error_msg = "Failed to register user: No user or session in response"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error registering user: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Log in an existing user with Supabase Auth"""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info(f"Successfully logged in user: {email}")
                return {
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email
                    },
                    "session": {
                        "access_token": response.session.access_token,
                        "refresh_token": response.session.refresh_token,
                        "expires_at": response.session.expires_at
                    }
                }
            else:
                error_msg = "Failed to log in: No user or session in response"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Error logging in user: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token from Supabase Auth"""
        try:
            response = self.client.auth.get_user(token)
            
            if response and response.user:
                logger.info(f"Successfully validated token for user: {response.user.id}")
                return {
                    "valid": True,
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email
                    }
                }
            else:
                return {"valid": False, "error": "Invalid token"}
        except Exception as e:
            logger.error(f"Error validating token: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def store_health_data(self, user_id: str, health_data: Dict[str, Any]) -> bool:
        """Store health data in Supabase"""
        try:
            # Store heart rate data
            if 'heartRate' in health_data and health_data['heartRate']:
                for entry in health_data['heartRate']:
                    if 'startDate' in entry and 'value' in entry:
                        date = datetime.fromisoformat(entry['startDate'].replace('Z', '+00:00')).date()
                        self.client.table('fitbit_heart_rate_level').insert({
                            'person_id': user_id,
                            'date': str(date),
                            'avg_rate': float(entry['value'])
                        }).execute()
            
            # Store steps data
            if 'steps' in health_data and health_data['steps']:
                # Group steps by date
                daily_steps = {}
                for entry in health_data['steps']:
                    if 'endDate' in entry and 'value' in entry:
                        date = datetime.fromisoformat(entry['endDate'].replace('Z', '+00:00')).date()
                        date_str = str(date)
                        if date_str not in daily_steps:
                            daily_steps[date_str] = 0
                        daily_steps[date_str] += int(entry['value'])
                
                # Insert daily step totals
                for date, sum_steps in daily_steps.items():
                    self.client.table('fitbit_intraday_steps').insert({
                        'person_id': user_id,
                        'date': date,
                        'sum_steps': sum_steps
                    }).execute()
            
            # Store sleep data
            if 'sleep' in health_data and health_data['sleep']:
                # Process sleep data by date
                sleep_by_date = {}
                for entry in health_data['sleep']:
                    if 'startDate' in entry and 'value' in entry:
                        date = datetime.fromisoformat(entry['startDate'].replace('Z', '+00:00')).date()
                        date_str = str(date)
                        if date_str not in sleep_by_date:
                            sleep_by_date[date_str] = {
                                'minute_asleep': 0,
                                'entries': []
                            }
                        
                        duration = int(entry['value'])
                        sleep_by_date[date_str]['minute_asleep'] += duration
                        sleep_by_date[date_str]['entries'].append({
                            'start_time': entry.get('startDate'),
                            'end_time': entry.get('endDate'),
                            'duration': duration,
                            'level': 'asleep'
                        })
                
                # Insert sleep summary for each date
                for date, sleep_data in sleep_by_date.items():
                    # Insert daily summary
                    self.client.table('fitbit_sleep_daily_summary').insert({
                        'person_id': user_id,
                        'sleep_date': date,
                        'is_main_sleep': True,
                        'minute_asleep': sleep_data['minute_asleep'],
                        'minute_in_bed': sleep_data['minute_asleep'] + 10,
                        'minute_awake': 10
                    }).execute()
            
            # Store activity data (from workout)
            if 'workout' in health_data and health_data['workout']:
                # Group workouts by date
                activity_by_date = {}
                for entry in health_data['workout']:
                    if 'startDate' in entry:
                        date = datetime.fromisoformat(entry['startDate'].replace('Z', '+00:00')).date()
                        date_str = str(date)
                        if date_str not in activity_by_date:
                            activity_by_date[date_str] = {
                                'very_active_minutes': 0,
                                'calories_out': 0
                            }
                        
                        # Add workout duration to very active minutes
                        if 'duration' in entry:
                            duration_seconds = float(entry['duration'])
                            duration_minutes = int(duration_seconds / 60)
                            activity_by_date[date_str]['very_active_minutes'] += duration_minutes
                        
                        # Add calories
                        if 'totalEnergyBurned' in entry:
                            activity_by_date[date_str]['calories_out'] += int(float(entry['totalEnergyBurned']))
                
                # Insert activity data for each date
                for date, activity_data in activity_by_date.items():
                    # Calculate other activity estimates
                    fairly_active = int(activity_data['very_active_minutes'] * 0.5)
                    lightly_active = int(activity_data['very_active_minutes'] * 2)
                    sedentary = 1440 - activity_data['very_active_minutes'] - fairly_active - lightly_active
                    
                    self.client.table('fitbit_activity').insert({
                        'person_id': user_id,
                        'date': date,
                        'very_active_minutes': activity_data['very_active_minutes'],
                        'fairly_active_minutes': fairly_active,
                        'lightly_active_minutes': lightly_active,
                        'sedentary_minutes': max(0, sedentary),
                        'calories_out': activity_data['calories_out'],
                        'activity_calories': int(activity_data['calories_out'] * 0.7),
                        'calories_bmr': int(activity_data['calories_out'] * 0.3)
                    }).execute()
            
            logger.info(f"Successfully stored health data for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing health data: {str(e)}", exc_info=True)
            return False
    
    def store_analysis_result(self, result: Dict[str, Any]) -> bool:
        """Store mental health analysis result in Supabase"""
        try:
            # Store result in the mental_health_analysis table
            logger.info(f"Storing analysis result for user {result.get('user_id')}")
            
            # Create the data payload
            analysis_data = {
                'user_id': result.get('user_id'),
                'prediction': result.get('prediction', 0),
                'risk_level': result.get('risk_level', 'Unknown'),
                'risk_score': result.get('risk_score', 0.0),
                'contributing_factors': result.get('contributing_factors', {}),
                'analysis_date': result.get('analysis_date', datetime.now().isoformat()),
                'created_at': datetime.now().isoformat()
            }
            
            # Insert the analysis result
            response = self.client.table('mental_health_analysis').insert(analysis_data).execute()
            
            if hasattr(response, 'data') and response.data:
                logger.info(f"Successfully stored analysis result for user {result.get('user_id')}")
                return True
            else:
                logger.error(f"Failed to store analysis result: {getattr(response, 'error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error storing analysis result: {str(e)}")
            return False
    
    def get_latest_analysis(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest analysis result for a user"""
        try:
            result = self.client.table('analysis_results') \
                .select('*') \
                .eq('person_id', user_id) \
                .order('analysis_date', desc=True) \
                .limit(1) \
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                logger.info(f"No analysis result found for user {user_id}")
                return None
                
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
                
                result = self.client.table(table) \
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
    
    def get_health_data_for_analysis(self, user_id: str, days: int = 60) -> Dict[str, Any]:
        """Get health data from Supabase for analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            logger.info(f"Fetching {days} days of health data for user {user_id}")
            
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
                    "personId": user_id,
                    "age": 33,
                    "genderBinary": 1
                }
            }
            
            # Fetch heart rate data
            heart_rate_result = self.client.table('fitbit_heart_rate_level') \
                .select('*') \
                .eq('person_id', user_id) \
                .gte('date', start_date_str) \
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
            
            # Fetch step data
            steps_result = self.client.table('fitbit_intraday_steps') \
                .select('*') \
                .eq('person_id', user_id) \
                .gte('date', start_date_str) \
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
            
            # Fetch activity data
            activity_result = self.client.table('fitbit_activity') \
                .select('*') \
                .eq('person_id', user_id) \
                .gte('date', start_date_str) \
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
            
            # Fetch sleep data
            sleep_result = self.client.table('fitbit_sleep_daily_summary') \
                .select('*') \
                .eq('person_id', user_id) \
                .gte('sleep_date', start_date_str) \
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
            
            logger.info(f"Retrieved health data for user {user_id}")
            
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
                    "personId": user_id,
                    "age": 33,
                    "genderBinary": 1
                }
            } 