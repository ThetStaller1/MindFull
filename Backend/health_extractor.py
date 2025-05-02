#!/usr/bin/env python
"""
HealthKit Data Extractor

This module processes data received from the iOS app's HealthKit integration,
organizing it into structured formats suitable for further processing.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class HealthKitExtractor:
    """
    Extracts and organizes HealthKit data received from the iOS app.
    
    This class is responsible for taking the raw HealthKit data sent from
    the iOS app and organizing it into pandas DataFrames for each data type.
    """
    
    def __init__(self):
        """Initialize the HealthKit extractor"""
        logger.info("HealthKit extractor initialized")
        
        # Define mappings for HealthKit types to our internal types
        self.healthkit_types = {
            # Heart rate data
            'heart_rate': 'HKQuantityTypeIdentifierHeartRate',
            
            # Step count data
            'steps': 'HKQuantityTypeIdentifierStepCount',
            
            # Energy data
            'active_energy': 'HKQuantityTypeIdentifierActiveEnergyBurned',
            'basal_energy': 'HKQuantityTypeIdentifierBasalEnergyBurned',
            
            # Sleep data
            'sleep_analysis': 'HKCategoryTypeIdentifierSleepAnalysis',
            
            # Workout data
            'workouts': 'HKWorkoutTypeIdentifier'
        }
    
    def process_healthkit_data(self, health_data, person_id=None, age=None, gender_binary=None):
        """
        Process HealthKit data into Fitbit-compatible format for storage in Supabase.
        
        Args:
            health_data: Raw HealthKit data from the iOS app
            person_id: User ID string
            age: User age (integer)
            gender_binary: Gender as binary (0=male, 1=female)
            
        Returns:
            pandas DataFrame with features for the mental health model
        """
        logger.info("Processing HealthKit data into Fitbit format")
        
        # Convert to dictionary if it's a Pydantic model
        data_dict = health_data
        if hasattr(health_data, 'dict'):
            data_dict = health_data.dict()
        
        # Extract all HealthKit data first
        extracted_data = self.extract_all_data(data_dict)
        
        # Convert to Fitbit format for Supabase storage
        fitbit_data = {}
        
        # Heart rate level data
        if 'heart_rate' in extracted_data and not extracted_data['heart_rate'].empty:
            heart_rate_df = extracted_data['heart_rate']
            if 'date' not in heart_rate_df.columns:
                heart_rate_df['date'] = pd.to_datetime(heart_rate_df['timestamp']).dt.date if 'timestamp' in heart_rate_df.columns else pd.to_datetime(heart_rate_df['startDate']).dt.date
            
            # Group by date to calculate daily heart rate metrics
            heart_rate_daily = heart_rate_df.groupby('date').agg({
                'value': ['mean', 'min', 'max']
            }).reset_index()
            
            # Flatten multi-index columns
            heart_rate_daily.columns = ['date', 'avg_rate', 'min_rate', 'max_rate']
            
            # Add person_id and estimate resting rate
            heart_rate_daily['person_id'] = person_id
            heart_rate_daily['resting_rate'] = heart_rate_daily['min_rate'] * 0.9  # Rough estimate
            
            # Convert date to string
            heart_rate_daily['date'] = heart_rate_daily['date'].astype(str)
            
            # Store in fitbit_data
            fitbit_data['fitbit_heart_rate_level'] = heart_rate_daily
        
        # Steps data
        if 'steps' in extracted_data and not extracted_data['steps'].empty:
            steps_df = extracted_data['steps']
            if 'date' not in steps_df.columns:
                steps_df['date'] = pd.to_datetime(steps_df['timestamp']).dt.date if 'timestamp' in steps_df.columns else pd.to_datetime(steps_df['startDate']).dt.date
            
            # Group by date to calculate daily step totals
            steps_daily = steps_df.groupby('date').agg({
                'value': 'sum'
            }).reset_index()
            
            # Rename columns
            steps_daily.columns = ['date', 'sum_steps']
            
            # Add person_id
            steps_daily['person_id'] = person_id
            
            # Convert date to string
            steps_daily['date'] = steps_daily['date'].astype(str)
            
            # Store in fitbit_data
            fitbit_data['fitbit_intraday_steps'] = steps_daily
        
        # Activity data from active energy and workouts
        activity_records = []
        
        # Process active energy
        if 'active_energy' in extracted_data and not extracted_data['active_energy'].empty:
            active_energy_df = extracted_data['active_energy']
            if 'date' not in active_energy_df.columns:
                active_energy_df['date'] = pd.to_datetime(active_energy_df['timestamp']).dt.date if 'timestamp' in active_energy_df.columns else pd.to_datetime(active_energy_df['startDate']).dt.date
            
            # Group by date
            energy_daily = active_energy_df.groupby('date').agg({
                'value': 'sum'
            }).reset_index()
            
            # Calculate active minutes based on energy
            for _, row in energy_daily.iterrows():
                date = row['date']
                activity_calories = row['value']
                
                # Default activity minutes distribution
                very_active_minutes = 30
                fairly_active_minutes = 30
                lightly_active_minutes = 60
                
                # Calculate sedentary minutes
                sedentary_minutes = 24 * 60 - (very_active_minutes + fairly_active_minutes + lightly_active_minutes)
                
                activity_records.append({
                    'person_id': person_id or '1001',
                    'date': str(date),
                    'activity_calories': int(activity_calories),
                    'very_active_minutes': very_active_minutes,
                    'fairly_active_minutes': fairly_active_minutes,
                    'lightly_active_minutes': lightly_active_minutes,
                    'sedentary_minutes': sedentary_minutes
                })
        
        # Update with workout data if available
        if 'workouts' in extracted_data and not extracted_data['workouts'].empty:
            workout_df = extracted_data['workouts']
            if 'date' not in workout_df.columns:
                workout_df['date'] = pd.to_datetime(workout_df['startDate']).dt.date
            
            # Process workouts to calculate active minutes
            for _, row in workout_df.iterrows():
                date = row['date']
                
                # Find existing record for this date or create new one
                existing_record = next((r for r in activity_records if r['date'] == str(date)), None)
                
                if existing_record:
                    # Update existing record with workout data
                    existing_record['very_active_minutes'] += int(row.get('duration', 0) / 60)  # Convert seconds to minutes
                else:
                    # Create new record for this date
                    sedentary_minutes = 24 * 60 - int(row.get('duration', 0) / 60)
                    activity_records.append({
                        'person_id': person_id,
                        'date': str(date),
                        'activity_calories': int(row.get('totalEnergyBurned', 300)),
                        'very_active_minutes': int(row.get('duration', 0) / 60),
                        'fairly_active_minutes': 30,
                        'lightly_active_minutes': 60,
                        'sedentary_minutes': sedentary_minutes
                    })
        
        # Create activity DataFrame
        if activity_records:
            activity_df = pd.DataFrame(activity_records)
            fitbit_data['fitbit_activity'] = activity_df
        
        # Sleep data
        if 'sleep_analysis' in extracted_data and not extracted_data['sleep_analysis'].empty:
            sleep_df = extracted_data['sleep_analysis']
            
            # Calculate sleep time in minutes - prefer explicit duration field if available
            sleep_df['sleep_date'] = pd.to_datetime(sleep_df['startDate']).dt.date
            
            # Check if explicit duration field exists
            if 'duration' in sleep_df.columns:
                sleep_df['minute_asleep'] = sleep_df['duration']
                logger.info("Using explicit duration field for sleep summary calculations")
            else:
                # Fall back to calculating from time difference
                sleep_df['minute_asleep'] = (pd.to_datetime(sleep_df['endDate']) - pd.to_datetime(sleep_df['startDate'])).dt.total_seconds() / 60
                logger.info("Calculated sleep duration from timestamps for summary (fallback method)")
            
            # Group by date
            sleep_daily = sleep_df.groupby('sleep_date').agg({
                'minute_asleep': 'sum',
                'startDate': 'min',
                'endDate': 'max'
            }).reset_index()
            
            # Add calculated fields
            sleep_daily['minute_awake'] = 30  # Default estimate
            sleep_daily['minute_in_bed'] = sleep_daily['minute_asleep'] + sleep_daily['minute_awake']
            sleep_daily['person_id'] = person_id
            
            # Clean up and format
            sleep_daily = sleep_daily[['person_id', 'sleep_date', 'minute_asleep', 'minute_awake', 'minute_in_bed']]
            sleep_daily['sleep_date'] = sleep_daily['sleep_date'].astype(str)
            
            fitbit_data['fitbit_sleep_daily_summary'] = sleep_daily
            
            # Create sleep level data if possible
            if 'value' in sleep_df.columns:
                sleep_level_records = []
                
                for _, row in sleep_df.iterrows():
                    # Get duration from either the explicit duration field or the calculated minute_asleep
                    duration = row.get('duration', row['minute_asleep'])
                    
                    sleep_level_records.append({
                        'person_id': person_id or '1001',
                        'sleep_date': row['sleep_date'].strftime('%Y-%m-%d'),
                        'level': 'light',  # Default level
                        'start_time': pd.to_datetime(row['startDate']).strftime('%H:%M:%S'),
                        'duration': int(duration)
                    })
                
                if sleep_level_records:
                    sleep_level_df = pd.DataFrame(sleep_level_records)
                    fitbit_data['fitbit_sleep_level'] = sleep_level_df
        
        logger.info(f"Processed {len(fitbit_data)} Fitbit-compatible data tables")
        for table, df in fitbit_data.items():
            logger.info(f"  - {table}: {len(df)} records")
        
        # Extract features from fitbit_data for model prediction
        # Create a features DataFrame with demographic info
        features_df = pd.DataFrame({
            'age': [age if age is not None else 0],
            'gender_binary': [gender_binary if gender_binary is not None else 0],
            'person_id': [person_id if person_id is not None else '1001']
        })
        
        # Extract heart rate features if available
        if 'fitbit_heart_rate_level' in fitbit_data:
            hr_df = fitbit_data['fitbit_heart_rate_level']
            features_df['heart_rate_avg'] = hr_df['avg_rate'].mean() if not hr_df.empty else 0
            features_df['heart_rate_min'] = hr_df['min_rate'].min() if not hr_df.empty else 0
            features_df['heart_rate_max'] = hr_df['max_rate'].max() if not hr_df.empty else 0
            features_df['heart_rate_std'] = hr_df['avg_rate'].std() if not hr_df.empty and len(hr_df) > 1 else 0
        
        # Extract step features if available
        if 'fitbit_intraday_steps' in fitbit_data:
            steps_df = fitbit_data['fitbit_intraday_steps']
            features_df['steps_daily_mean'] = steps_df['sum_steps'].mean() if not steps_df.empty else 0
            features_df['steps_daily_std'] = steps_df['sum_steps'].std() if not steps_df.empty and len(steps_df) > 1 else 0
            features_df['steps_daily_max'] = steps_df['sum_steps'].max() if not steps_df.empty else 0
        
        # Extract activity features if available
        if 'fitbit_activity' in fitbit_data:
            activity_df = fitbit_data['fitbit_activity']
            features_df['activity_very_active_minutes_mean'] = activity_df['very_active_minutes'].mean() if not activity_df.empty else 0
            features_df['activity_very_active_minutes_std'] = activity_df['very_active_minutes'].std() if not activity_df.empty and len(activity_df) > 1 else 0
            features_df['activity_very_active_minutes_max'] = activity_df['very_active_minutes'].max() if not activity_df.empty else 0
            features_df['activity_calories_mean'] = activity_df['activity_calories'].mean() if not activity_df.empty else 0
            features_df['activity_calories_std'] = activity_df['activity_calories'].std() if not activity_df.empty and len(activity_df) > 1 else 0
        
        # Extract sleep features if available
        if 'fitbit_sleep_daily_summary' in fitbit_data:
            sleep_df = fitbit_data['fitbit_sleep_daily_summary']
            features_df['sleep_minute_asleep_mean'] = sleep_df['minute_asleep'].mean() if not sleep_df.empty else 0
            features_df['sleep_minute_asleep_std'] = sleep_df['minute_asleep'].std() if not sleep_df.empty and len(sleep_df) > 1 else 0
            features_df['sleep_minute_awake_mean'] = sleep_df['minute_awake'].mean() if not sleep_df.empty else 0
            features_df['sleep_minute_awake_std'] = sleep_df['minute_awake'].std() if not sleep_df.empty and len(sleep_df) > 1 else 0
            
            # Add placeholder values for deep, light, and REM sleep since we don't have actual values
            features_df['sleep_minute_deep_mean'] = features_df['sleep_minute_asleep_mean'] * 0.2 if not sleep_df.empty else 0
            features_df['sleep_minute_deep_std'] = features_df['sleep_minute_asleep_std'] * 0.2 if not sleep_df.empty and len(sleep_df) > 1 else 0
            features_df['sleep_minute_light_mean'] = features_df['sleep_minute_asleep_mean'] * 0.5 if not sleep_df.empty else 0
            features_df['sleep_minute_light_std'] = features_df['sleep_minute_asleep_std'] * 0.5 if not sleep_df.empty and len(sleep_df) > 1 else 0
            features_df['sleep_minute_rem_mean'] = features_df['sleep_minute_asleep_mean'] * 0.3 if not sleep_df.empty else 0
            features_df['sleep_minute_rem_std'] = features_df['sleep_minute_asleep_std'] * 0.3 if not sleep_df.empty and len(sleep_df) > 1 else 0
        
        logger.info(f"Created features DataFrame with {features_df.shape[1]} columns")
        return features_df
    
    def extract_all_data(self, health_data: Any) -> Dict[str, pd.DataFrame]:
        """
        Extract and organize all health data into DataFrames by type.
        
        Args:
            health_data: The health data received from the iOS app
                         (can be a Pydantic model or dictionary)
        
        Returns:
            Dictionary with data type keys and pandas DataFrame values
        """
        # Convert to dictionary if it's a Pydantic model
        data_dict = health_data
        if hasattr(health_data, 'dict'):
            data_dict = health_data.dict()
        
        extracted_data = {}
        
        # Extract heart rate data
        if data_dict.get('heartRate'):
            heart_rate_df = self._extract_heart_rate_data(data_dict['heartRate'])
            if not heart_rate_df.empty:
                extracted_data['heart_rate'] = heart_rate_df
        
        # Extract step data
        if data_dict.get('steps'):
            steps_df = self._extract_step_data(data_dict['steps'])
            if not steps_df.empty:
                extracted_data['steps'] = steps_df
        
        # Extract active energy data
        if data_dict.get('activeEnergy'):
            active_energy_df = self._extract_energy_data(data_dict['activeEnergy'], energy_type='active')
            if not active_energy_df.empty:
                extracted_data['active_energy'] = active_energy_df
        
        # Extract basal energy data
        if data_dict.get('basalEnergy'):
            basal_energy_df = self._extract_energy_data(data_dict['basalEnergy'], energy_type='basal')
            if not basal_energy_df.empty:
                extracted_data['basal_energy'] = basal_energy_df
        
        # Extract sleep data
        if data_dict.get('sleep'):
            sleep_df = self._extract_sleep_data(data_dict['sleep'])
            if not sleep_df.empty:
                extracted_data['sleep_analysis'] = sleep_df
        
        # Extract workout data
        if data_dict.get('workout'):
            workout_df = self._extract_workout_data(data_dict['workout'])
            if not workout_df.empty:
                extracted_data['workouts'] = workout_df
        
        # Log extraction summary
        for data_type, df in extracted_data.items():
            logger.info(f"Extracted {len(df)} {data_type} records")
        
        return extracted_data
    
    def _extract_heart_rate_data(self, heart_rate_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract and process heart rate data"""
        if not heart_rate_data:
            return pd.DataFrame()
        
        # Create DataFrame from records
        df = pd.DataFrame(heart_rate_data)
        
        # Convert date strings to datetime objects
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Ensure value is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Extract date component for date-based aggregation
        if 'startDate' in df.columns:
            df['date'] = df['startDate'].dt.date
        
        return df
    
    def _extract_step_data(self, step_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract and process step count data"""
        if not step_data:
            return pd.DataFrame()
        
        # Create DataFrame from records
        df = pd.DataFrame(step_data)
        
        # Convert date strings to datetime objects
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Ensure value is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Extract date component for date-based aggregation
        if 'startDate' in df.columns:
            df['date'] = df['startDate'].dt.date
        
        return df
    
    def _extract_energy_data(self, energy_data: List[Dict[str, Any]], energy_type: str) -> pd.DataFrame:
        """Extract and process energy data (active or basal)"""
        if not energy_data:
            return pd.DataFrame()
        
        # Create DataFrame from records
        df = pd.DataFrame(energy_data)
        
        # Convert date strings to datetime objects
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Ensure value is numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Extract date component for date-based aggregation
        if 'startDate' in df.columns:
            df['date'] = df['startDate'].dt.date
        
        # Add energy type column for identification
        df['energy_type'] = energy_type
        
        return df
    
    def _extract_sleep_data(self, sleep_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract and process sleep analysis data with improved session handling"""
        if not sleep_data:
            return pd.DataFrame()
        
        # Create DataFrame from records
        df = pd.DataFrame(sleep_data)
        
        # Convert date strings to datetime objects
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Calculate duration in minutes - prefer explicit duration field if available
        if 'duration' in df.columns:
            # Use the explicit duration field sent from iOS app
            df['duration_minutes'] = df['duration']
            logger.info("Using explicit duration field from iOS app for sleep data")
        elif 'startDate' in df.columns and 'endDate' in df.columns:
            # Fall back to calculating from time difference
            df['duration_minutes'] = (df['endDate'] - df['startDate']).dt.total_seconds() / 60
            logger.info("Calculated sleep duration from timestamps (fallback method)")
        
        # Extract sleep date (the date when sleep started)
        if 'startDate' in df.columns:
            df['sleep_date'] = df['startDate'].dt.date
        
        # Preserve session ID if available
        if 'sessionID' in df.columns:
            logger.info("Found sleep session IDs in the data")
        else:
            # If no session ID, we'll identify sessions in the mapper
            logger.info("No sleep session IDs found, will identify sessions by time proximity in mapper")
        
        # Convert sleep stage value if present
        if 'value' in df.columns:
            # Map HealthKit sleep stage values to standardized values
            # For HKCategoryValueSleepAnalysis
            sleep_stage_map = {
                '0': 'inBed',
                '1': 'asleep',
                '2': 'awake',
                '3': 'deep',
                '4': 'rem',
                '5': 'core'
            }
            
            # Try to map the values or keep original if not in map
            df['sleep_stage'] = df['value'].astype(str).map(
                lambda x: sleep_stage_map.get(x, x)
            )
        
        # If sleep_stage field is directly available (from iOS app), use it
        if 'sleep_stage' in df.columns:
            logger.info("Sleep stage information available directly")
        
        return df
    
    def _extract_workout_data(self, workout_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract and process workout data"""
        if not workout_data:
            return pd.DataFrame()
        
        # Create DataFrame from records
        df = pd.DataFrame(workout_data)
        
        # Convert date strings to datetime objects
        if 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
        
        # Extract duration
        if 'duration' in df.columns:
            df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        elif 'startDate' in df.columns and 'endDate' in df.columns:
            df['duration'] = (df['endDate'] - df['startDate']).dt.total_seconds()
        
        # Extract workout date
        if 'startDate' in df.columns:
            df['workout_date'] = df['startDate'].dt.date
        
        # Convert energy and distance fields to numeric
        for field in ['totalEnergyBurned', 'totalDistance']:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def _calculate_seconds_from_midnight(self, timestamp: datetime) -> int:
        """Calculate seconds from midnight (start of day) for a timestamp.
        
        Args:
            timestamp: The datetime to calculate from
            
        Returns:
            Integer number of seconds from midnight
        """
        midnight = datetime.combine(timestamp.date(), datetime.min.time())
        return int((timestamp - midnight).total_seconds()) 