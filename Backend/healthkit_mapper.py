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
            fitbit_data['fitbit_heart_rate_level'] = self._convert_heart_rate_data(
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
        Convert HealthKit sleep data to Fitbit sleep formats.
        
        Args:
            sleep_df: DataFrame with sleep data from HealthKit
            person_id: User ID
            
        Returns:
            Tuple of (sleep_daily_summary_df, sleep_level_df)
        """
        if sleep_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Process sleep data to get daily summaries
        # Ensure sleep_date is available
        if 'sleep_date' not in sleep_df.columns and 'startDate' in sleep_df.columns:
            sleep_df['sleep_date'] = sleep_df['startDate'].dt.date
        
        # Calculate summary by date
        sleep_summary = sleep_df.groupby('sleep_date').agg({
            'duration_minutes': 'sum',
            'startDate': 'min',
            'endDate': 'max',
        }).reset_index()
        
        # Create daily summary records
        summary_records = []
        level_records = []
        
        for _, row in sleep_summary.iterrows():
            date = row['sleep_date']
            start_time = row['startDate']
            end_time = row['endDate']
            total_minutes = row['duration_minutes']
            
            # Filter sleep_df for this date to get sleep levels
            date_sleep = sleep_df[sleep_df['sleep_date'] == date]
            
            # Extract sleep stages if available
            minute_asleep = 0
            minute_deep = 0
            minute_light = 0
            minute_rem = 0
            minute_wake = 0
            minute_after_wakeup = 0
            minute_awake = 0
            minute_restless = 0
            
            if 'sleep_stage' in date_sleep.columns:
                for _, stage_row in date_sleep.iterrows():
                    stage = stage_row.get('sleep_stage', 'unknown')
                    duration = stage_row.get('duration_minutes', 0)
                    
                    if stage in ['asleep', 'deep', 'light', 'rem']:
                        minute_asleep += duration
                        
                        if stage == 'deep':
                            minute_deep += duration
                        elif stage == 'light' or stage == 'core':
                            minute_light += duration
                        elif stage == 'rem':
                            minute_rem += duration
                    elif stage == 'awake':
                        minute_awake += duration
                    elif stage == 'inBed':
                        # In bed but not asleep is considered restless
                        minute_restless += duration
            else:
                # If no sleep stage data, assume all duration is asleep
                minute_asleep = total_minutes
            
            # Calculate minute_in_bed as the total of all sleep states
            minute_in_bed = minute_asleep + minute_awake + minute_restless
            
            # Create daily summary record - using exact column names from Supabase
            summary_records.append({
                'person_id': person_id,
                'sleep_date': str(date),
                'is_main_sleep': True,  # Default to main sleep
                'minute_in_bed': int(minute_in_bed),
                'minute_asleep': int(minute_asleep),
                'minute_after_wakeup': int(minute_after_wakeup),
                'minute_awake': int(minute_awake),
                'minute_restless': int(minute_restless),
                'minute_deep': int(minute_deep),
                'minute_light': int(minute_light),
                'minute_rem': int(minute_rem),
                'minute_wake': int(minute_wake)
            })
            
            # Create sleep level records - using exact column names from Supabase
            if 'sleep_stage' in date_sleep.columns:
                # Create a unique sleep ID for this night
                sleep_id = f"{person_id}-{date}-{hash(str(start_time))}"[:36]
                level_id = 1
                
                # Track sleep stages for this date
                for _, stage_row in date_sleep.iterrows():
                    stage = stage_row.get('sleep_stage', 'unknown')
                    start = stage_row.get('startDate')
                    end = stage_row.get('endDate')
                    duration = stage_row.get('duration_minutes', 0)
                    
                    # Map sleep stage to Fitbit format
                    if stage == 'deep':
                        level_value = 'deep'
                    elif stage == 'rem':
                        level_value = 'rem'
                    elif stage in ['light', 'core']:
                        level_value = 'light'
                    elif stage == 'awake':
                        level_value = 'wake'
                    elif stage == 'asleep':
                        level_value = 'asleep'
                    elif stage == 'inBed':
                        level_value = 'restless'
                    else:
                        level_value = 'unknown'
                    
                    # Create sleep level record for this segment with exact column names from Supabase
                    level_records.append({
                        'person_id': person_id,
                        'sleep_date': str(date),
                        'sleep_id': sleep_id,
                        'is_main_sleep': True,
                        'level_id': level_id,
                        'level': level_value,
                        'start_time': start,
                        'end_time': end,
                        'duration': int(duration)
                    })
                    level_id += 1
        
        # Create DataFrames
        daily_summary_df = pd.DataFrame(summary_records)
        sleep_level_df = pd.DataFrame(level_records)
        
        return daily_summary_df, sleep_level_df
