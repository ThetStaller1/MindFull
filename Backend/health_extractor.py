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
        Process HealthKit data into features for the mental health model, exactly matching
        the original training data format.
        
        Args:
            health_data: Raw HealthKit data from the iOS app
            person_id: User ID string
            age: User age (integer)
            gender_binary: Gender as binary (0=male, 1=female)
            
        Returns:
            pandas DataFrame with features for the mental health model
        """
        logger.info("Processing HealthKit data into model-ready features")
        
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
            steps_daily.columns = ['date', 'steps']
            
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
                    'sedentary_minutes': sedentary_minutes,
                    'calories_out': int(activity_calories) + 1500  # Base calories + activity
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
                        'sedentary_minutes': sedentary_minutes,
                        'calories_out': int(row.get('totalEnergyBurned', 300)) + 1500  # Base calories + activity
                    })
        
        # Create activity DataFrame
        if activity_records:
            activity_df = pd.DataFrame(activity_records)
            # Calculate activity ratio matching the original model
            activity_df['total_active_minutes'] = (activity_df['very_active_minutes'] + 
                                                activity_df['fairly_active_minutes'] + 
                                                activity_df['lightly_active_minutes'])
            activity_df['activity_ratio'] = activity_df['total_active_minutes'] / (activity_df['sedentary_minutes'] + 1)
            fitbit_data['fitbit_activity'] = activity_df
        
        # Sleep data - Exact match to the original model processing
        sleep_records = []
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
            
            # Add sleep stage information
            if 'sleep_stage' in sleep_df.columns:
                # Calculate minutes per sleep stage from the detailed data
                sleep_stage_minutes = {}
                
                for _, row in sleep_df.iterrows():
                    date = row['sleep_date']
                    stage = row.get('sleep_stage', 'unknown').lower()
                    duration = row['minute_asleep']
                    
                    # Skip invalid durations
                    if duration <= 0:
                        continue
                    
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str not in sleep_stage_minutes:
                        sleep_stage_minutes[date_str] = {
                            'minute_deep': 0,
                            'minute_light': 0,
                            'minute_rem': 0,
                            'minute_awake': 0,
                            'minute_asleep': 0
                        }
                    
                    # Map the sleep stage and accumulate minutes
                    if 'deep' in stage:
                        sleep_stage_minutes[date_str]['minute_deep'] += duration
                    elif 'light' in stage or 'core' in stage:
                        sleep_stage_minutes[date_str]['minute_light'] += duration
                    elif 'rem' in stage:
                        sleep_stage_minutes[date_str]['minute_rem'] += duration
                    elif 'awake' in stage or 'wake' in stage:
                        sleep_stage_minutes[date_str]['minute_awake'] += duration
                    
                    # Accumulate total sleep time
                    if stage != 'awake' and stage != 'wake':
                        sleep_stage_minutes[date_str]['minute_asleep'] += duration
                
                # Create summary records
                for date_str, minutes in sleep_stage_minutes.items():
                    sleep_records.append({
                        'person_id': person_id,
                        'sleep_date': date_str,
                        'minute_asleep': minutes['minute_asleep'],
                        'minute_deep': minutes['minute_deep'],
                        'minute_light': minutes['minute_light'],
                        'minute_rem': minutes['minute_rem'],
                        'minute_awake': minutes['minute_awake'],
                        'minute_in_bed': minutes['minute_asleep'] + minutes['minute_awake']
                    })
            else:
                # Group by date for basic summary if no stage data
                sleep_daily = sleep_df.groupby('sleep_date').agg({
                    'minute_asleep': 'sum',
                    'startDate': 'min',
                    'endDate': 'max'
                }).reset_index()
                
                # Add calculated fields
                sleep_daily['minute_awake'] = 30  # Default estimate
                sleep_daily['minute_in_bed'] = sleep_daily['minute_asleep'] + sleep_daily['minute_awake']
                sleep_daily['person_id'] = person_id
                
                # Add sleep stage estimates based on typical percentages
                sleep_daily['minute_deep'] = sleep_daily['minute_asleep'] * 0.2
                sleep_daily['minute_light'] = sleep_daily['minute_asleep'] * 0.5
                sleep_daily['minute_rem'] = sleep_daily['minute_asleep'] * 0.25
                
                # Convert to records
                for _, row in sleep_daily.iterrows():
                    sleep_records.append({
                        'person_id': row['person_id'],
                        'sleep_date': row['sleep_date'].strftime('%Y-%m-%d'),
                        'minute_asleep': row['minute_asleep'],
                        'minute_deep': row['minute_deep'],
                        'minute_light': row['minute_light'],
                        'minute_rem': row['minute_rem'],
                        'minute_awake': row['minute_awake'],
                        'minute_in_bed': row['minute_in_bed']
                    })
        
        # Create sleep DataFrame and calculate sleep regularity features
        if sleep_records:
            sleep_df = pd.DataFrame(sleep_records)
            sleep_df['sleep_date'] = pd.to_datetime(sleep_df['sleep_date'])
            sleep_df = sleep_df.sort_values(['person_id', 'sleep_date'])
            
            # Add sleep regularity metrics to match original model
            sleep_df['sleep_start'] = sleep_df['sleep_date']
            sleep_df['next_day_start'] = sleep_df.groupby('person_id')['sleep_start'].shift(-1)
            sleep_df['time_diff'] = (sleep_df['next_day_start'] - sleep_df['sleep_start']).dt.total_seconds() / 3600
            
            # Add weekend vs weekday comparison
            sleep_df['is_weekend'] = sleep_df['sleep_date'].dt.dayofweek.isin([5, 6])
            
            # Convert back to strings for Supabase
            sleep_df['sleep_date'] = sleep_df['sleep_date'].dt.strftime('%Y-%m-%d')
            
            fitbit_data['fitbit_sleep_daily_summary'] = sleep_df
        
        logger.info(f"Processed {len(fitbit_data)} Fitbit-compatible data tables")
        for table, df in fitbit_data.items():
            logger.info(f"  - {table}: {len(df)} records")
        
        # ---------------------------------------------------------------
        # CREATE FEATURES FOR MODEL USING EXACT SAME APPROACH AS ORIGINAL
        # ---------------------------------------------------------------
        features_df = pd.DataFrame({
            'age': [age if age is not None else 0],
            'gender_binary': [gender_binary if gender_binary is not None else 0],
            'person_id': [person_id if person_id is not None else '1001']
        })
        
        # 1. SLEEP FEATURES - Match original model
        if 'fitbit_sleep_daily_summary' in fitbit_data and not fitbit_data['fitbit_sleep_daily_summary'].empty:
            sleep_df = fitbit_data['fitbit_sleep_daily_summary']
            sleep_df['sleep_date'] = pd.to_datetime(sleep_df['sleep_date'])
            sleep_df = sleep_df.sort_values(['person_id', 'sleep_date'])
            
            # Calculate basic sleep metrics
            sleep_metrics = {}
            
            # Basic sleep metrics
            for col in ['minute_asleep', 'minute_deep', 'minute_light', 'minute_rem', 'minute_awake']:
                if col in sleep_df.columns:
                    metrics = sleep_df.groupby('person_id')[col].agg(['mean', 'std', 'min', 'max']).reset_index()
                    # Only keep first row since we have a single person_id
                    for stat in ['mean', 'std', 'min', 'max']:
                        if stat in metrics.columns:
                            col_name = f'sleep_{col}_{stat}'
                            sleep_metrics[col_name] = metrics.iloc[0][stat]
            
            # Sleep regularity
            if 'time_diff' in sleep_df.columns:
                regularity = sleep_df.groupby('person_id')['time_diff'].agg(['std', 'mean']).reset_index()
                if not regularity.empty:
                    sleep_metrics['sleep_time_diff_std'] = regularity.iloc[0]['std']
                    sleep_metrics['sleep_time_diff_mean'] = regularity.iloc[0]['mean']
            
            # Weekend vs weekday differences (social jetlag)
            if 'is_weekend' in sleep_df.columns:
                weekend_stats = sleep_df[sleep_df['is_weekend']].groupby('person_id')['minute_asleep'].mean().reset_index()
                weekday_stats = sleep_df[~sleep_df['is_weekend']].groupby('person_id')['minute_asleep'].mean().reset_index()
                
                if not weekend_stats.empty and not weekday_stats.empty:
                    weekend_avg = weekend_stats.iloc[0]['minute_asleep'] if len(weekend_stats) > 0 else 0
                    weekday_avg = weekday_stats.iloc[0]['minute_asleep'] if len(weekday_stats) > 0 else 0
                    sleep_metrics['sleep_social_jetlag'] = abs(weekend_avg - weekday_avg)
            
            # Add sleep metrics to features
            for key, value in sleep_metrics.items():
                if pd.notna(value):  # Only add if not NaN
                    features_df[key] = value
                else:
                    features_df[key] = 0
        
        # 2. ACTIVITY FEATURES - Match original model
        if 'fitbit_activity' in fitbit_data and not fitbit_data['fitbit_activity'].empty:
            activity_df = fitbit_data['fitbit_activity']
            
            # Calculate activity metrics
            activity_metrics = {}
            
            # Calculate all required metrics
            for col in ['very_active_minutes', 'fairly_active_minutes', 'lightly_active_minutes', 
                        'sedentary_minutes', 'activity_ratio', 'steps', 'calories_out']:
                if col in activity_df.columns:
                    stats = ['mean', 'std']
                    if col in ['very_active_minutes', 'steps', 'calories_out']:
                        stats.append('max')
                    
                    metrics = activity_df.groupby('person_id')[col].agg(stats).reset_index()
                    # Only keep first row since we have a single person_id
                    for stat in stats:
                        col_name = f'activity_{col}_{stat}'
                        if len(metrics) > 0:
                            activity_metrics[col_name] = metrics.iloc[0][stat]
            
            # Add activity metrics to features
            for key, value in activity_metrics.items():
                if pd.notna(value):  # Only add if not NaN
                    features_df[key] = value
                else:
                    features_df[key] = 0
        
        # 3. HEART RATE FEATURES - Match original model
        if 'fitbit_heart_rate_level' in fitbit_data and not fitbit_data['fitbit_heart_rate_level'].empty:
            hr_df = fitbit_data['fitbit_heart_rate_level']
            
            # Calculate heart rate metrics
            hr_metrics = {}
            
            # Compute all required heart rate stats
            stats = ['mean', 'std', 'min', 'max']
            if 'avg_rate' in hr_df.columns:
                metrics = hr_df.groupby('person_id')['avg_rate'].agg(stats).reset_index()
                if not metrics.empty:
                    for stat in stats:
                        hr_metrics[f'hr_avg_rate_{stat}'] = metrics.iloc[0][stat]
                
                # Add skew if we have enough data points
                if len(hr_df) > 2:
                    try:
                        from scipy.stats import skew
                        skew_val = skew(hr_df['avg_rate'])
                        hr_metrics['hr_avg_rate_skew'] = skew_val
                    except:
                        hr_metrics['hr_avg_rate_skew'] = 0
                else:
                    hr_metrics['hr_avg_rate_skew'] = 0
            
            # Add heart rate metrics to features
            for key, value in hr_metrics.items():
                if pd.notna(value):  # Only add if not NaN
                    features_df[key] = value
                else:
                    features_df[key] = 0
        
        # Fill any missing values with means or zeros to match original model
        features_df = features_df.fillna(features_df.mean())
        for col in features_df.columns:
            if pd.isna(features_df[col]).any():
                features_df[col] = 0
        
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
            # Calculate from time difference
            df['duration_minutes'] = (df['endDate'] - df['startDate']).dt.total_seconds() / 60
            logger.info("Calculated sleep duration from timestamps")
        
        # Extract sleep date (the date when sleep started)
        if 'startDate' in df.columns:
            df['sleep_date'] = df['startDate'].dt.date
        
        # Preserve session ID if available
        if 'sessionID' in df.columns:
            logger.info("Found sleep session IDs in the data")
        else:
            # Sort by start time
            df = df.sort_values('startDate')
            
            # Identify sleep sessions (gap of more than 30 minutes indicates a new session)
            df['time_diff'] = df['startDate'].diff().dt.total_seconds() / 60
            df['new_session'] = (df['time_diff'] > 30) | (df['time_diff'].isna())
            df['sessionID'] = df['new_session'].cumsum()
            logger.info("Generated session IDs based on time proximity for sleep data")
        
        # Convert sleep stage value if present
        if 'value' in df.columns:
            # Map HealthKit sleep stage values to standardized values
            sleep_stage_map = {
                # Numeric values
                '0': 'in_bed',
                '1': 'asleep',  # Unspecified
                '2': 'awake',
                '3': 'deep',
                '4': 'rem',
                '5': 'core',
                
                # String values directly from HealthKit
                'HKCategoryValueSleepAnalysisInBed': 'in_bed',
                'HKCategoryValueSleepAnalysisAsleepUnspecified': 'asleep',
                'HKCategoryValueSleepAnalysisAsleepCore': 'core',
                'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
                'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
                'HKCategoryValueSleepAnalysisAwake': 'awake'
            }
            
            # Try to map the values or keep original if not in map
            df['sleep_stage'] = df['value'].astype(str).map(
                lambda x: sleep_stage_map.get(x, 'unknown')
            )
        
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