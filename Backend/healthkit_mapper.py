#!/usr/bin/env python
"""
HealthKit to Fitbit Data Format Mapper

This module converts extracted HealthKit data from iOS to Fitbit format
for compatibility with the existing mental health model and Supabase storage.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class HealthKitToFitbitMapper:
    """
    Converts HealthKit data extracted from iOS app to Fitbit format 
    for compatibility with the existing mental health model.
    """
    
    def __init__(self):
        """Initialize the HealthKit to Fitbit mapper"""
        logger.info("HealthKit to Fitbit mapper initialized")
        
        # Define heart rate zone boundaries
        self.heart_rate_zones = {
            "Out of Range": {"min": 30, "max": 99},
            "Fat Burn": {"min": 99, "max": 139},
            "Cardio": {"min": 139, "max": 169},
            "Peak": {"min": 169, "max": 220}
        }
        
        # Sleep stage mapping from HealthKit to Fitbit
        self.sleep_stage_mapping = {
            "HKCategoryValueSleepAnalysisInBed": "restless",
            "HKCategoryValueSleepAnalysisAsleepUnspecified": "light",
            "HKCategoryValueSleepAnalysisAsleepCore": "light",
            "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
            "HKCategoryValueSleepAnalysisAsleepREM": "rem",
            "HKCategoryValueSleepAnalysisAwake": "wake"
        }
    
    def convert_to_fitbit_format(self, extracted_data: Dict[str, pd.DataFrame], person_id: str) -> Dict[str, pd.DataFrame]:
        """
        Convert extracted HealthKit data to Fitbit format for Supabase storage.
        
        Args:
            extracted_data: Dictionary with DataFrames for each health data type
            person_id: User ID to assign to the data
            
        Returns:
            Dictionary of DataFrames in Fitbit format ready for Supabase
        """
        fitbit_data = {}
        
        # Convert heart rate data
        if 'heart_rate' in extracted_data and not extracted_data['heart_rate'].empty:
            # Create both heart rate level and heart rate summary
            fitbit_data['fitbit_heart_rate_level'] = self._convert_heart_rate_data(
                extracted_data['heart_rate'], person_id
            )
            
            # Add heart rate summary table
            fitbit_data['fitbit_heart_rate_summary'] = self._convert_heart_rate_summary(
                extracted_data['heart_rate'], person_id
            )
        
        # Convert steps data
        if 'steps' in extracted_data and not extracted_data['steps'].empty:
            fitbit_data['fitbit_intraday_steps'] = self._convert_steps_data(
                extracted_data['steps'], person_id
            )
        
        # Convert active energy data
        if 'active_energy' in extracted_data and not extracted_data['active_energy'].empty:
            activity_df = self._convert_active_energy_to_activity(
                extracted_data['active_energy'],
                extracted_data.get('workout', pd.DataFrame()),
                person_id
            )
            
            if not activity_df.empty:
                fitbit_data['fitbit_activity'] = activity_df
        
        # Convert sleep data
        if 'sleep' in extracted_data and not extracted_data['sleep'].empty:
            sleep_daily_summary, sleep_level = self._convert_sleep_data(
                extracted_data['sleep'], person_id
            )
            
            if not sleep_daily_summary.empty:
                fitbit_data['fitbit_sleep_daily_summary'] = sleep_daily_summary
            
            if not sleep_level.empty:
                fitbit_data['fitbit_sleep_level'] = sleep_level
        
        # Log conversion summary
        for data_type, df in fitbit_data.items():
            logger.info(f"Converted to {data_type}: {len(df)} records")
        
        return fitbit_data
    
    def _convert_heart_rate_summary(self, heart_rate_df: pd.DataFrame, person_id: str) -> pd.DataFrame:
        """
        Convert HealthKit heart rate data to Fitbit heart rate summary format with zone information.
        
        Args:
            heart_rate_df: DataFrame with heart rate data from HealthKit
            person_id: User ID
            
        Returns:
            DataFrame in Fitbit heart_rate_summary format
        """
        if heart_rate_df.empty:
            return pd.DataFrame(columns=[
                'person_id', 'date', 'zone_name', 'min_heart_rate', 
                'max_heart_rate', 'minute_in_zone', 'calorie_count'
            ])
        
        # Group heart rate data by day
        try:
            if 'date' not in heart_rate_df.columns:
                heart_rate_df['date'] = pd.to_datetime(heart_rate_df['startDate']).dt.date
        except Exception as e:
            logger.error(f"Error converting dates: {e}")
            # If error in conversion, try a direct assignment
            heart_rate_df['date'] = [d.date() if isinstance(d, datetime) else datetime.now().date() 
                                for d in heart_rate_df['startDate']]
        
        # Calculate daily heart rate zones
        result_data = []
        
        for date, group in heart_rate_df.groupby('date'):
            try:
                # Calculate min and max heart rates for the day
                min_hr = group['value'].min()
                max_hr = group['value'].max()
                
                # Calculate time spent in each heart rate zone
                for zone_name, zone_range in self.heart_rate_zones.items():
                    zone_data = group[(group['value'] >= zone_range['min']) & 
                                    (group['value'] <= zone_range['max'])]
                    
                    if not zone_data.empty:
                        # Calculate total minutes in zone (assuming data points are 1 minute apart)
                        minutes_in_zone = len(zone_data)
                        
                        # Estimate calorie count (simple estimate - can be improved)
                        avg_hr = zone_data['value'].mean()
                        calorie_count = int(minutes_in_zone * avg_hr * 0.1)
                        
                        zone_min_hr = zone_data['value'].min()
                        zone_max_hr = zone_data['value'].max()
                        
                        result_data.append({
                            'person_id': person_id,
                            'date': str(date),
                            'zone_name': zone_name,
                            'min_heart_rate': int(zone_min_hr),
                            'max_heart_rate': int(zone_max_hr),
                            'minute_in_zone': int(minutes_in_zone),
                            'calorie_count': int(calorie_count)
                        })
            except Exception as e:
                logger.error(f"Error processing heart rate zone for date {date}: {e}")
                continue
        
        # Return empty dataframe if no valid data
        if not result_data:
            return pd.DataFrame(columns=[
                'person_id', 'date', 'zone_name', 'min_heart_rate', 
                'max_heart_rate', 'minute_in_zone', 'calorie_count'
            ])
            
        return pd.DataFrame(result_data)
    
    def _convert_heart_rate_data(self, heart_rate_df: pd.DataFrame, person_id: str) -> pd.DataFrame:
        """
        Convert HealthKit heart rate data to Fitbit format.
        
        Args:
            heart_rate_df: DataFrame with heart rate data from HealthKit
            person_id: User ID
            
        Returns:
            DataFrame in Fitbit heart_rate_level format
        """
        if heart_rate_df.empty:
            return pd.DataFrame()
        
        # Group by date and calculate average heart rate
        if 'date' not in heart_rate_df.columns and 'startDate' in heart_rate_df.columns:
            heart_rate_df['date'] = heart_rate_df['startDate'].dt.date
        
        # Calculate daily heart rate statistics
        daily_hr = heart_rate_df.groupby('date').agg({
            'value': ['mean', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten the multi-level columns
        daily_hr.columns = ['date', 'avg_rate', 'min_rate', 'max_rate', 'sample_count']
        
        # Create Fitbit format DataFrame with exact column names from Supabase
        fitbit_hr = pd.DataFrame({
            'person_id': person_id,
            'date': daily_hr['date'].astype(str),
            'avg_rate': daily_hr['avg_rate'].round(1)
            # We're only keeping the required fields that match the Supabase schema
            # 'fitbit_heart_rate_level' in Supabase only has person_id, date, avg_rate
        })
        
        return fitbit_hr
    
    def _convert_steps_data(self, steps_df: pd.DataFrame, person_id: str) -> pd.DataFrame:
        """
        Convert HealthKit steps data to Fitbit format.
        
        Args:
            steps_df: DataFrame with steps data from HealthKit
            person_id: User ID
            
        Returns:
            DataFrame in Fitbit intraday_steps format
        """
        if steps_df.empty:
            return pd.DataFrame()
        
        # Group by date to get daily totals
        if 'date' not in steps_df.columns and 'startDate' in steps_df.columns:
            steps_df['date'] = steps_df['startDate'].dt.date
        
        # Sum steps by date
        daily_steps = steps_df.groupby('date')['value'].sum().reset_index()
        
        # Create Fitbit format DataFrame matching Supabase schema
        fitbit_steps = pd.DataFrame({
            'person_id': person_id,
            'date': daily_steps['date'].astype(str),
            'sum_steps': daily_steps['value'].astype(int)
        })
        
        return fitbit_steps
    
    def _convert_active_energy_to_activity(self, 
                                        active_energy_df: pd.DataFrame, 
                                        workout_df: pd.DataFrame, 
                                        person_id: str) -> pd.DataFrame:
        """
        Convert HealthKit active energy and workout data to Fitbit activity format.
        
        Args:
            active_energy_df: DataFrame with active energy data from HealthKit
            workout_df: DataFrame with workout data from HealthKit
            person_id: User ID
            
        Returns:
            DataFrame in Fitbit activity format
        """
        if active_energy_df.empty:
            return pd.DataFrame()
        
        # Group by date to get daily totals for active energy
        if 'date' not in active_energy_df.columns and 'startDate' in active_energy_df.columns:
            active_energy_df['date'] = active_energy_df['startDate'].dt.date
        
        # Sum active energy by date
        daily_energy = active_energy_df.groupby('date')['value'].sum().reset_index()
        
        # Calculate active minutes from workout data if available
        very_active_minutes = {}
        fairly_active_minutes = {}
        lightly_active_minutes = {}
        
        if not workout_df.empty:
            # Group workouts by date
            if 'workout_date' not in workout_df.columns and 'startDate' in workout_df.columns:
                workout_df['workout_date'] = workout_df['startDate'].dt.date
            
            # Calculate minutes by intensity based on workout type and duration
            for date, group in workout_df.groupby('workout_date'):
                # For simplicity, we'll consider all workout time as very active
                # Convert duration from seconds to minutes
                total_duration_minutes = group['duration'].sum() / 60
                
                # Store minutes by date
                very_active_minutes[date] = total_duration_minutes
                fairly_active_minutes[date] = 0  # Placeholder
                lightly_active_minutes[date] = 0  # Placeholder
        
        # Create activity DataFrame with default values - matching Supabase schema
        activity_records = []
        
        for _, row in daily_energy.iterrows():
            date = row['date']
            activity_calories = row['value']
            
            # Get active minutes for this date or use defaults
            very_active = very_active_minutes.get(date, 0)
            fairly_active = fairly_active_minutes.get(date, 0)
            
            # Estimate lightly active minutes based on calories if no workout data
            if date not in lightly_active_minutes:
                # Rough estimation: 1 kcal ≈ 0.1 lightly active minute
                lightly_active = activity_calories * 0.1
            else:
                lightly_active = lightly_active_minutes[date]
            
            # Calculate sedentary minutes (total day minutes - active minutes)
            total_day_minutes = 24 * 60  # minutes in a day
            sedentary_minutes = total_day_minutes - (very_active + fairly_active + lightly_active)
            
            # Ensure we don't have negative sedentary minutes
            sedentary_minutes = max(0, sedentary_minutes)
            
            # Create record with columns matching Supabase schema
            activity_records.append({
                'person_id': person_id,
                'date': str(date),
                'activity_calories': int(activity_calories),
                'calories_bmr': 1500,  # Default basal metabolic rate
                'calories_out': int(activity_calories + 1500),  # BMR + activity calories
                'floors': 0,  # Default value
                'elevation': 0,  # Default value
                'fairly_active_minutes': int(fairly_active),
                'lightly_active_minutes': int(lightly_active),
                'marginal_calories': 0,  # Default value
                'sedentary_minutes': int(sedentary_minutes),
                'steps': 0,  # Will be updated from steps data
                'very_active_minutes': int(very_active)
            })
        
        # Create DataFrame
        activity_df = pd.DataFrame(activity_records)
        
        return activity_df
    
    def _convert_sleep_data(self, sleep_df: pd.DataFrame, person_id: str) -> tuple:
        """
        Convert HealthKit sleep data to Fitbit sleep formats with improved stage mapping and session handling.
        
        Args:
            sleep_df: DataFrame with sleep data from HealthKit
            person_id: User ID
            
        Returns:
            Tuple of (sleep_daily_summary_df, sleep_level_df)
        """
        if sleep_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Convert dates to datetime objects if needed
        try:
            if not pd.api.types.is_datetime64_any_dtype(sleep_df['startDate']):
                sleep_df['startDate'] = pd.to_datetime(sleep_df['startDate'])
            if not pd.api.types.is_datetime64_any_dtype(sleep_df['endDate']):
                sleep_df['endDate'] = pd.to_datetime(sleep_df['endDate'])
        except Exception as e:
            logger.error(f"Error converting sleep dates: {e}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Get sleep session date (the date when sleep started)
        sleep_df['sleep_date'] = sleep_df['startDate'].dt.date
        
        # Calculate duration in minutes
        if 'duration_minutes' not in sleep_df.columns:
            sleep_df['duration_minutes'] = (
                (sleep_df['endDate'] - sleep_df['startDate']).dt.total_seconds() / 60
            ).astype(int)
            logger.info("Calculated sleep duration from timestamps in mapper")
        
        # Enhanced sleep stage mapping
        apple_to_fitbit_stage = {
            # Mapped from the standardized stages in extractor
            'in_bed': 'restless',
            'asleep': 'light',
            'awake': 'wake',
            'deep': 'deep',
            'rem': 'rem',
            'core': 'light',
            'unknown': 'light'
        }
        
        # Map sleep stages
        if 'sleep_stage' in sleep_df.columns:
            sleep_df['level'] = sleep_df['sleep_stage'].map(
                lambda x: apple_to_fitbit_stage.get(x, 'light')
            )
        elif 'value' in sleep_df.columns:
            # Fallback direct mapping from HealthKit values
            direct_stage_map = {
                # Numeric values
                '0': 'restless',
                '1': 'light',
                '2': 'wake',
                '3': 'deep',
                '4': 'rem',
                '5': 'light',
                
                # String values
                'HKCategoryValueSleepAnalysisInBed': 'restless',
                'HKCategoryValueSleepAnalysisAsleepUnspecified': 'light',
                'HKCategoryValueSleepAnalysisAsleepCore': 'light',
                'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
                'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
                'HKCategoryValueSleepAnalysisAwake': 'wake'
            }
            sleep_df['level'] = sleep_df['value'].astype(str).map(
                lambda x: direct_stage_map.get(x, 'light')
            )
        else:
            logger.warning("No sleep stage information found in sleep data")
            sleep_df['level'] = 'light'  # Default value
        
        # Ensure we have session IDs
        if 'sessionID' not in sleep_df.columns:
            # Sort by start time
            sleep_df = sleep_df.sort_values('startDate')
            
            # Identify sleep sessions (gap of more than 30 minutes indicates a new session)
            sleep_df['time_diff'] = sleep_df['startDate'].diff().dt.total_seconds() / 60
            sleep_df['new_session'] = (sleep_df['time_diff'] > 30) | (sleep_df['time_diff'].isna())
            sleep_df['sessionID'] = sleep_df['new_session'].cumsum()
            logger.info("Generated session IDs based on time proximity in mapper")
        
        # Create daily summary with accurate session information
        daily_sleep = []
        
        # Group by sleep date and session ID
        for (sleep_date, session_id), session_data in sleep_df.groupby(['sleep_date', 'sessionID']):
            # Calculate session metrics
            session_start = session_data['startDate'].min()
            session_end = session_data['endDate'].max()
            
            # Total time in bed equals the sum of all segment durations in this session
            in_bed_time = session_data['duration_minutes'].sum()
            
            # Calculate time spent in each sleep stage
            asleep_time = session_data[session_data['level'].isin(['light', 'deep', 'rem'])]['duration_minutes'].sum()
            deep_time = session_data[session_data['level'] == 'deep']['duration_minutes'].sum()
            light_time = session_data[session_data['level'] == 'light']['duration_minutes'].sum()
            rem_time = session_data[session_data['level'] == 'rem']['duration_minutes'].sum()
            awake_time = session_data[session_data['level'] == 'wake']['duration_minutes'].sum()
            restless_time = session_data[session_data['level'] == 'restless']['duration_minutes'].sum()
            
            # Calculate time after wakeup
            # In Fitbit format, this is typically the time spent awake at the end of a sleep session
            after_wakeup_time = 0
            
            # Sort segments by start time
            sorted_segments = session_data.sort_values('startDate')
            
            # Find wake segments at the end of the session
            if not sorted_segments.empty and 'wake' in sorted_segments['level'].values:
                # Identify continuous wake segments at the end
                reversed_segments = sorted_segments.iloc[::-1]  # Reverse to start from the end
                continuous_wake = 0
                for _, row in reversed_segments.iterrows():
                    if row['level'] == 'wake':
                        continuous_wake += row['duration_minutes']
                    else:
                        break  # Stop once we hit a non-wake segment
                
                after_wakeup_time = continuous_wake
            
            # Determine if this is the main sleep for the day
            # Typically the longest sleep session is considered main sleep
            is_main_sleep = 1  # We'll update this later after checking all sessions for the day
            
            # Create daily summary record for this session
            daily_sleep.append({
                'person_id': person_id,
                'sleep_date': str(sleep_date),
                'is_main_sleep': is_main_sleep,
                'minute_in_bed': int(in_bed_time),
                'minute_asleep': int(asleep_time),
                'minute_after_wakeup': int(after_wakeup_time),
                'minute_awake': int(awake_time),
                'minute_restless': int(restless_time),
                'minute_deep': int(deep_time),
                'minute_light': int(light_time),
                'minute_rem': int(rem_time),
                'minute_wake': int(awake_time),
                'start_time': session_start,
                'end_time': session_end,
                'session_id': session_id
            })
        
        # Convert to DataFrame
        daily_sleep_df = pd.DataFrame(daily_sleep)
        
        if not daily_sleep_df.empty:
            # Mark the longest sleep session for each day as the main sleep
            for sleep_date, day_group in daily_sleep_df.groupby('sleep_date'):
                if len(day_group) > 1:
                    # Find the longest session (by minute_in_bed)
                    main_sleep_idx = day_group['minute_in_bed'].idxmax()
                    # Update is_main_sleep
                    daily_sleep_df.loc[daily_sleep_df.index != main_sleep_idx, 'is_main_sleep'] = 0
            
            # Remove temporary columns
            daily_sleep_df = daily_sleep_df.drop(['start_time', 'end_time', 'session_id'], axis=1, errors='ignore')
        
        # Create sleep level dataframe with precise timestamps and durations
        sleep_levels = []
        
        for _, row in sleep_df.iterrows():
            session_id = row['sessionID']
            sleep_date = row['sleep_date']
            
            # Get the is_main_sleep value for this session
            is_main_sleep = 1  # Default
            if not daily_sleep_df.empty:
                # Find the corresponding session in daily_sleep
                session_entries = daily_sleep_df[
                    (daily_sleep_df['sleep_date'] == str(sleep_date)) &
                    (daily_sleep_df['session_id'] == session_id)
                ]
                if not session_entries.empty:
                    is_main_sleep = session_entries.iloc[0]['is_main_sleep']
            
            # Calculate seconds from start of day for better sorting
            seconds_from_start_of_day = (row['startDate'] - 
                                         pd.Timestamp.combine(row['sleep_date'], pd.Timestamp.min.time())).total_seconds()
            
            sleep_levels.append({
                'person_id': person_id,
                'sleep_date': str(sleep_date),
                'is_main_sleep': is_main_sleep,
                'level': row['level'],
                'date': row['startDate'].strftime('%Y-%m-%d %H:%M:%S'),
                'duration_in_min': int(row['duration_minutes']),
                'seconds_from_start_of_day': int(seconds_from_start_of_day)
            })
        
        sleep_daily_summary_df = daily_sleep_df
        sleep_level_df = pd.DataFrame(sleep_levels)
        
        logger.info(f"Converted {len(sleep_df)} sleep records into {len(sleep_daily_summary_df)} daily summaries and {len(sleep_level_df)} sleep level entries")
        
        return sleep_daily_summary_df, sleep_level_df
