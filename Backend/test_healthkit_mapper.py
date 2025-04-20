"""
HealthKit to Fitbit Mapper (Test Version)

This module provides functionality to convert Apple HealthKit data to the All of Us Fitbit dataset format.
The All of Us Fitbit dataset is used for mental health detection algorithms.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Any


class HealthKitToFitbitMapper:
    """
    Class to handle the conversion from Apple HealthKit data to Fitbit format
    compatible with the All of Us dataset structure.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the mapper with output directory for the transformed files.
        
        Args:
            output_dir (str): Directory where the converted Fitbit-format files will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Define heart rate zone boundaries (can be customized)
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
        
        # Activity type mapping from HealthKit to Fitbit
        self.workout_type_mapping = {
            "HKWorkoutActivityTypeWalking": "Walk",
            "HKWorkoutActivityTypeRunning": "Run",
            "HKWorkoutActivityTypeCycling": "Bike",
            "HKWorkoutActivityTypeHiking": "Hike",
            "HKWorkoutActivityTypeElliptical": "Elliptical",
            "HKWorkoutActivityTypeTraditionalStrengthTraining": "Weights",
            "HKWorkoutActivityTypeHighIntensityIntervalTraining": "Interval Workout",
            "HKWorkoutActivityTypeMixedCardio": "Aerobic Workout",
            # Default for any other workout type
            "default": "Workout"
        }

    def map_heart_rate_summary(self, healthkit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map HealthKit heart rate data to Fitbit heart rate summary format.
        """
        if healthkit_data.empty:
            return pd.DataFrame(columns=[
                'person_id', 'date', 'zone_name', 'min_heart_rate', 
                'max_heart_rate', 'minute_in_zone', 'calorie_count'
            ])
            
        # Group heart rate data by day
        try:
            healthkit_data['date'] = pd.to_datetime(healthkit_data['startDate']).dt.date
        except Exception as e:
            print(f"Error converting dates: {e}")
            # If error in conversion, try a direct assignment
            healthkit_data['date'] = [d.date() if isinstance(d, datetime) else datetime.now().date() 
                                    for d in healthkit_data['startDate']]
        
        # Calculate daily heart rate zones
        result_data = []
        
        for date, group in healthkit_data.groupby('date'):
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
                            'person_id': group['person_id'].iloc[0] if 'person_id' in group.columns else '',
                            'date': date,
                            'zone_name': zone_name,
                            'min_heart_rate': zone_min_hr,
                            'max_heart_rate': zone_max_hr,
                            'minute_in_zone': minutes_in_zone,
                            'calorie_count': calorie_count
                        })
            except Exception as e:
                print(f"Error processing heart rate zone for date {date}: {e}")
                continue
        
        # Return empty dataframe if no valid data
        if not result_data:
            return pd.DataFrame(columns=[
                'person_id', 'date', 'zone_name', 'min_heart_rate', 
                'max_heart_rate', 'minute_in_zone', 'calorie_count'
            ])
            
        return pd.DataFrame(result_data)

    def map_heart_rate_level(self, healthkit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map HealthKit heart rate data to Fitbit heart rate level format.
        """
        if healthkit_data.empty:
            return pd.DataFrame(columns=['person_id', 'date', 'avg_rate'])
        
        # Convert startDate to date
        try:
            healthkit_data['date'] = pd.to_datetime(healthkit_data['startDate']).dt.date
        except Exception as e:
            print(f"Error converting dates: {e}")
            # If error in conversion, try a direct assignment
            healthkit_data['date'] = [d.date() if isinstance(d, datetime) else datetime.now().date() 
                                     for d in healthkit_data['startDate']]
        
        try:
            # Calculate average heart rate per day
            result_df = (healthkit_data.groupby(['person_id', 'date'] if 'person_id' in healthkit_data.columns else ['date'])
                        .agg({'value': 'mean'})
                        .reset_index()
                        .rename(columns={'value': 'avg_rate'}))
            
            # Add person_id column if not present
            if 'person_id' not in result_df.columns:
                result_df['person_id'] = ''
            
            # Convert avg_rate to integer
            result_df['avg_rate'] = result_df['avg_rate'].astype(int)
            
            return result_df
        except Exception as e:
            print(f"Error calculating heart rate level: {e}")
            return pd.DataFrame(columns=['person_id', 'date', 'avg_rate'])

    def map_intraday_steps(self, healthkit_data: pd.DataFrame) -> pd.DataFrame:
        """
        Map HealthKit step count data to Fitbit intraday steps format.
        """
        if healthkit_data.empty:
            return pd.DataFrame(columns=['person_id', 'date', 'sum_steps'])
        
        try:
            # Convert dates to datetime
            healthkit_data['startDate'] = pd.to_datetime(healthkit_data['startDate'])
            
            # Extract date and sum steps for each day
            result_df = (healthkit_data.groupby(['person_id', healthkit_data['startDate'].dt.date] 
                                              if 'person_id' in healthkit_data.columns 
                                              else [healthkit_data['startDate'].dt.date])
                        .agg({'value': 'sum'})
                        .reset_index()
                        .rename(columns={'startDate': 'date', 'value': 'sum_steps'}))
            
            # Add person_id column if not present
            if 'person_id' not in result_df.columns:
                result_df['person_id'] = ''
                
            return result_df
        except Exception as e:
            print(f"Error processing step data: {e}")
            return pd.DataFrame(columns=['person_id', 'date', 'sum_steps'])

    def map_activity(self, 
                    healthkit_workouts: pd.DataFrame, 
                    healthkit_steps: pd.DataFrame,
                    healthkit_flights: Optional[pd.DataFrame] = None,
                    healthkit_calories: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Map HealthKit workout, step, and other activity data to Fitbit activity format.
        """
        # Initialize an empty result DataFrame with all required columns
        result_columns = [
            'person_id', 'date', 'activity_calories', 'calories_bmr', 
            'calories_out', 'elevation', 'fairly_active_minutes', 
            'floors', 'lightly_active_minutes', 'marginal_calories', 
            'sedentary_minutes', 'steps', 'very_active_minutes'
        ]
        
        if healthkit_workouts.empty and healthkit_steps.empty:
            return pd.DataFrame(columns=result_columns)
        
        result_data = []
        
        try:
            # Process steps data by day if available
            if not healthkit_steps.empty:
                try:
                    healthkit_steps['date'] = pd.to_datetime(healthkit_steps['startDate']).dt.date
                    
                    # Group by person_id if it exists, otherwise just by date
                    groupby_cols = ['person_id', 'date'] if 'person_id' in healthkit_steps.columns else ['date']
                    
                    daily_steps = (healthkit_steps.groupby(groupby_cols)
                                .agg({'value': 'sum'})
                                .reset_index()
                                .rename(columns={'value': 'steps'}))
                    
                    if 'person_id' not in daily_steps.columns:
                        daily_steps['person_id'] = ''
                    
                    for _, row in daily_steps.iterrows():
                        activity_dict = {col: 0 for col in result_columns}
                        activity_dict['person_id'] = row['person_id']
                        activity_dict['date'] = row['date']
                        activity_dict['steps'] = row['steps']
                        
                        # Default sedentary minutes (16 hours)
                        activity_dict['sedentary_minutes'] = 16 * 60
                        
                        result_data.append(activity_dict)
                except Exception as e:
                    print(f"Error processing steps for activity: {e}")
            
            # Process workouts data if available
            if not healthkit_workouts.empty:
                try:
                    healthkit_workouts['date'] = pd.to_datetime(healthkit_workouts['startDate']).dt.date
                    
                    # Calculate duration in minutes
                    healthkit_workouts['duration_minutes'] = (
                        (pd.to_datetime(healthkit_workouts['endDate']) - 
                        pd.to_datetime(healthkit_workouts['startDate'])).dt.total_seconds() / 60
                    )
                    
                    # Map workout types to activity intensity
                    def map_workout_to_intensity(workout_type):
                        if not workout_type or pd.isna(workout_type):
                            return 'lightly_active_minutes'
                            
                        high_intensity = ['HKWorkoutActivityTypeRunning', 'HKWorkoutActivityTypeHighIntensityIntervalTraining']
                        medium_intensity = ['HKWorkoutActivityTypeCycling', 'HKWorkoutActivityTypeElliptical', 
                                           'HKWorkoutActivityTypeTraditionalStrengthTraining']
                        low_intensity = ['HKWorkoutActivityTypeWalking', 'HKWorkoutActivityTypeYoga', 
                                        'HKWorkoutActivityTypeFlexibility']
                        
                        if workout_type in high_intensity:
                            return 'very_active_minutes'
                        elif workout_type in medium_intensity:
                            return 'fairly_active_minutes'
                        else:
                            return 'lightly_active_minutes'
                    
                    healthkit_workouts['intensity_category'] = healthkit_workouts['workoutActivityType'].apply(map_workout_to_intensity)
                    
                    # Group workouts by day and intensity category
                    groupby_cols = ['person_id', 'date', 'intensity_category'] if 'person_id' in healthkit_workouts.columns else ['date', 'intensity_category']
                    
                    workout_minutes = (healthkit_workouts.groupby(groupby_cols)
                                    .agg({'duration_minutes': 'sum'})
                                    .reset_index())
                    
                    if 'person_id' not in workout_minutes.columns:
                        workout_minutes['person_id'] = ''
                    
                    # Process calories if available
                    if healthkit_calories is not None and not healthkit_calories.empty:
                        try:
                            healthkit_calories['date'] = pd.to_datetime(healthkit_calories['startDate']).dt.date
                            
                            groupby_cols = ['person_id', 'date'] if 'person_id' in healthkit_calories.columns else ['date']
                            
                            daily_calories = (healthkit_calories.groupby(groupby_cols)
                                           .agg({'value': 'sum'})
                                           .reset_index()
                                           .rename(columns={'value': 'calories_out'}))
                            
                            if 'person_id' not in daily_calories.columns:
                                daily_calories['person_id'] = ''
                            
                            # Estimate BMR as 65% of total calories
                            daily_calories['calories_bmr'] = daily_calories['calories_out'] * 0.65
                            daily_calories['activity_calories'] = daily_calories['calories_out'] - daily_calories['calories_bmr']
                        except Exception as e:
                            print(f"Error processing calories data: {e}")
                            daily_calories = None
                    else:
                        daily_calories = None
                    
                    # Process elevation/floors if available
                    if healthkit_flights is not None and not healthkit_flights.empty:
                        try:
                            healthkit_flights['date'] = pd.to_datetime(healthkit_flights['startDate']).dt.date
                            
                            groupby_cols = ['person_id', 'date'] if 'person_id' in healthkit_flights.columns else ['date']
                            
                            daily_flights = (healthkit_flights.groupby(groupby_cols)
                                          .agg({'value': 'sum'})
                                          .reset_index()
                                          .rename(columns={'value': 'floors'}))
                            
                            if 'person_id' not in daily_flights.columns:
                                daily_flights['person_id'] = ''
                            
                            # Convert floors to elevation (1 floor ≈ 3 meters)
                            daily_flights['elevation'] = daily_flights['floors'] * 3
                        except Exception as e:
                            print(f"Error processing flights data: {e}")
                            daily_flights = None
                    else:
                        daily_flights = None
                    
                    # Update the result_data with workout information
                    for _, row in workout_minutes.iterrows():
                        # Find or create day entry
                        day_entry = next((item for item in result_data 
                                        if item['person_id'] == row['person_id'] and 
                                        item['date'] == row['date']), None)
                        
                        if day_entry is None:
                            day_entry = {col: 0 for col in result_columns}
                            day_entry['person_id'] = row['person_id']
                            day_entry['date'] = row['date']
                            day_entry['sedentary_minutes'] = 16 * 60  # Default sedentary minutes
                            result_data.append(day_entry)
                        
                        # Update the appropriate activity minutes category
                        day_entry[row['intensity_category']] += row['duration_minutes']
                        
                        # Adjust sedentary minutes accordingly
                        day_entry['sedentary_minutes'] -= row['duration_minutes']
                    
                    # Update calories and elevation data if available
                    if daily_calories is not None:
                        for _, row in daily_calories.iterrows():
                            day_entry = next((item for item in result_data 
                                            if item['person_id'] == row['person_id'] and 
                                            item['date'] == row['date']), None)
                            
                            if day_entry is not None:
                                day_entry['calories_out'] = row['calories_out']
                                day_entry['calories_bmr'] = row['calories_bmr']
                                day_entry['activity_calories'] = row['activity_calories']
                                day_entry['marginal_calories'] = int(day_entry['activity_calories'] * 0.1)
                    
                    if daily_flights is not None:
                        for _, row in daily_flights.iterrows():
                            day_entry = next((item for item in result_data 
                                            if item['person_id'] == row['person_id'] and 
                                            item['date'] == row['date']), None)
                            
                            if day_entry is not None:
                                day_entry['floors'] = row['floors']
                                day_entry['elevation'] = row['elevation']
                except Exception as e:
                    print(f"Error processing workout data: {e}")
            
            # Ensure all numeric columns are filled with appropriate values
            result_df = pd.DataFrame(result_data)
            numeric_columns = ['activity_calories', 'calories_bmr', 'calories_out', 
                              'elevation', 'fairly_active_minutes', 'floors', 
                              'lightly_active_minutes', 'marginal_calories', 
                              'sedentary_minutes', 'steps', 'very_active_minutes']
            
            for col in numeric_columns:
                if col in result_df.columns:
                    result_df[col] = result_df[col].fillna(0).astype(int)
                else:
                    result_df[col] = 0
            
            return result_df
        
        except Exception as e:
            print(f"Error in map_activity: {e}")
            return pd.DataFrame(columns=result_columns)

    def map_sleep_daily_summary(self, healthkit_sleep: pd.DataFrame) -> pd.DataFrame:
        """
        Map HealthKit sleep data to Fitbit sleep daily summary format.
        """
        if healthkit_sleep.empty:
            return pd.DataFrame(columns=[
                'person_id', 'sleep_date', 'is_main_sleep', 'minute_in_bed', 
                'minute_asleep', 'minute_after_wakeup', 'minute_awake', 
                'minute_restless', 'minute_deep', 'minute_light', 'minute_rem', 'minute_wake'
            ])
        
        try:
            # Convert dates to datetime
            healthkit_sleep['startDate'] = pd.to_datetime(healthkit_sleep['startDate'])
            healthkit_sleep['endDate'] = pd.to_datetime(healthkit_sleep['endDate'])
            
            # Calculate duration in minutes
            healthkit_sleep['duration_minutes'] = (
                (healthkit_sleep['endDate'] - healthkit_sleep['startDate']).dt.total_seconds() / 60
            )
            
            # Map sleep values to Fitbit categories
            def map_sleep_value(value):
                mapping = {
                    'HKCategoryValueSleepAnalysisInBed': 'in_bed',
                    'HKCategoryValueSleepAnalysisAsleepUnspecified': 'asleep',
                    'HKCategoryValueSleepAnalysisAsleepCore': 'light',
                    'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
                    'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
                    'HKCategoryValueSleepAnalysisAwake': 'awake'
                }
                return mapping.get(value, 'unknown')
            
            healthkit_sleep['sleep_type'] = healthkit_sleep['value'].apply(map_sleep_value)
            
            # Extract sleep date (using start date)
            healthkit_sleep['sleep_date'] = healthkit_sleep['startDate'].dt.date
            
            # Group by person and date to create daily summaries
            daily_sleep = []
            
            # Use person_id if available
            groupby_cols = ['person_id', 'sleep_date'] if 'person_id' in healthkit_sleep.columns else ['sleep_date']
            
            for group_key, group in healthkit_sleep.groupby(groupby_cols):
                try:
                    if isinstance(group_key, tuple):
                        person_id, sleep_date = group_key
                    else:
                        sleep_date = group_key
                        person_id = ''
                    
                    # Calculate total time in bed
                    in_bed_time = group[group['sleep_type'] == 'in_bed']['duration_minutes'].sum()
                    
                    # Calculate time in each sleep stage
                    asleep_time = group[group['sleep_type'] == 'asleep']['duration_minutes'].sum()
                    deep_time = group[group['sleep_type'] == 'deep']['duration_minutes'].sum()
                    light_time = group[group['sleep_type'] == 'light']['duration_minutes'].sum()
                    rem_time = group[group['sleep_type'] == 'rem']['duration_minutes'].sum()
                    awake_time = group[group['sleep_type'] == 'awake']['duration_minutes'].sum()
                    
                    # Estimate restless time (can't directly map from HealthKit)
                    # Assuming restless is a subset of awake time
                    restless_time = awake_time * 0.3
                    wake_time = awake_time * 0.7
                    
                    # Calculate time after wakeup (estimate as last 10 minutes of sleep session)
                    after_wakeup_time = min(10, in_bed_time * 0.05)
                    
                    daily_sleep.append({
                        'person_id': person_id,
                        'sleep_date': sleep_date,
                        'is_main_sleep': 1,  # Assume main sleep for now
                        'minute_in_bed': int(in_bed_time),
                        'minute_asleep': int(asleep_time + deep_time + light_time + rem_time),
                        'minute_after_wakeup': int(after_wakeup_time),
                        'minute_awake': int(awake_time),
                        'minute_restless': int(restless_time),
                        'minute_deep': int(deep_time),
                        'minute_light': int(light_time + asleep_time),  # Combine light and unspecified asleep
                        'minute_rem': int(rem_time),
                        'minute_wake': int(wake_time)
                    })
                except Exception as e:
                    print(f"Error processing sleep summary for date {sleep_date}: {e}")
                    continue
            
            return pd.DataFrame(daily_sleep)
        except Exception as e:
            print(f"Error in map_sleep_daily_summary: {e}")
            return pd.DataFrame(columns=[
                'person_id', 'sleep_date', 'is_main_sleep', 'minute_in_bed', 
                'minute_asleep', 'minute_after_wakeup', 'minute_awake', 
                'minute_restless', 'minute_deep', 'minute_light', 'minute_rem', 'minute_wake'
            ])

    def map_sleep_level(self, healthkit_sleep: pd.DataFrame) -> pd.DataFrame:
        """
        Map HealthKit sleep data to Fitbit sleep level format.
        """
        if healthkit_sleep.empty:
            return pd.DataFrame(columns=[
                'person_id', 'sleep_date', 'is_main_sleep', 'level', 'date', 'duration_in_min'
            ])
        
        try:
            # Convert dates to datetime
            healthkit_sleep['startDate'] = pd.to_datetime(healthkit_sleep['startDate'])
            healthkit_sleep['endDate'] = pd.to_datetime(healthkit_sleep['endDate'])
            
            # Calculate duration in minutes
            healthkit_sleep['duration_in_min'] = (
                (healthkit_sleep['endDate'] - healthkit_sleep['startDate']).dt.total_seconds() / 60
            ).astype(int)
            
            # Extract sleep date
            healthkit_sleep['sleep_date'] = healthkit_sleep['startDate'].dt.date
            
            # Map HealthKit sleep stages to Fitbit levels
            stage_mapping = {
                'HKCategoryValueSleepAnalysisInBed': 'restless',
                'HKCategoryValueSleepAnalysisAsleepUnspecified': 'light',
                'HKCategoryValueSleepAnalysisAsleepCore': 'light',
                'HKCategoryValueSleepAnalysisAsleepDeep': 'deep',
                'HKCategoryValueSleepAnalysisAsleepREM': 'rem',
                'HKCategoryValueSleepAnalysisAwake': 'wake'
            }
            
            healthkit_sleep['level'] = healthkit_sleep['value'].map(stage_mapping)
            
            # For each sleep entry, create a proper Fitbit-format row
            sleep_levels = []
            
            for _, row in healthkit_sleep.iterrows():
                try:
                    sleep_levels.append({
                        'person_id': row['person_id'] if 'person_id' in row else '',
                        'sleep_date': row['sleep_date'],
                        'is_main_sleep': 1,  # Assume main sleep for now
                        'level': row['level'],
                        'date': row['startDate'].strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_in_min': row['duration_in_min']
                    })
                except Exception as e:
                    print(f"Error processing sleep level entry: {e}")
                    continue
            
            return pd.DataFrame(sleep_levels)
        except Exception as e:
            print(f"Error in map_sleep_level: {e}")
            return pd.DataFrame(columns=[
                'person_id', 'sleep_date', 'is_main_sleep', 'level', 'date', 'duration_in_min'
            ])

    def save_to_csv(self, df: pd.DataFrame, dataset_type: str, group_id: str) -> str:
        """
        Save the converted dataframe to a CSV file in the Fitbit format.
        """
        filename = f"dataset_{group_id}_{dataset_type}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        # Make sure the DataFrame isn't empty
        if df.empty:
            print(f"Warning: Empty DataFrame for {dataset_type}, creating empty file")
            df = pd.DataFrame(columns=['person_id'])  # Minimal empty dataframe
        
        df.to_csv(filepath, index=False)
        return filepath

    def process_healthkit_export(self, 
                                healthkit_export: Dict[str, pd.DataFrame], 
                                person_id: str,
                                group_id: str) -> Dict[str, str]:
        """
        Process the entire HealthKit export and convert it to Fitbit format.
        """
        # Add person_id to all dataframes
        for key in healthkit_export:
            healthkit_export[key]['person_id'] = person_id
        
        result_files = {}
        
        try:
            # Process heart rate data
            if 'heart_rate' in healthkit_export and not healthkit_export['heart_rate'].empty:
                print("Processing heart rate data...")
                hr_summary = self.map_heart_rate_summary(healthkit_export['heart_rate'])
                hr_level = self.map_heart_rate_level(healthkit_export['heart_rate'])
                
                hr_summary_path = self.save_to_csv(hr_summary, 'fitbit_heart_rate_summary', group_id)
                hr_level_path = self.save_to_csv(hr_level, 'fitbit_heart_rate_level', group_id)
                
                result_files['heart_rate_summary'] = hr_summary_path
                result_files['heart_rate_level'] = hr_level_path
            
            # Process steps data
            if 'steps' in healthkit_export and not healthkit_export['steps'].empty:
                print("Processing step data...")
                intraday_steps = self.map_intraday_steps(healthkit_export['steps'])
                intraday_steps_path = self.save_to_csv(intraday_steps, 'fitbit_intraday_steps', group_id)
                result_files['intraday_steps'] = intraday_steps_path
            
            # Process activity data
            print("Processing activity data...")
            workout_data = healthkit_export.get('workouts', pd.DataFrame())
            steps_data = healthkit_export.get('steps', pd.DataFrame())
            flights_data = healthkit_export.get('flights_climbed', pd.DataFrame())
            calories_data = healthkit_export.get('active_energy', pd.DataFrame())
            
            activity_data = self.map_activity(workout_data, steps_data, flights_data, calories_data)
            activity_path = self.save_to_csv(activity_data, 'fitbit_activity', group_id)
            result_files['activity'] = activity_path
            
            # Process sleep data
            if 'sleep_analysis' in healthkit_export and not healthkit_export['sleep_analysis'].empty:
                print("Processing sleep data...")
                sleep_daily = self.map_sleep_daily_summary(healthkit_export['sleep_analysis'])
                sleep_level = self.map_sleep_level(healthkit_export['sleep_analysis'])
                
                sleep_daily_path = self.save_to_csv(sleep_daily, 'fitbit_sleep_daily_summary', group_id)
                sleep_level_path = self.save_to_csv(sleep_level, 'fitbit_sleep_level', group_id)
                
                result_files['sleep_daily_summary'] = sleep_daily_path
                result_files['sleep_level'] = sleep_level_path
            
            return result_files
        
        except Exception as e:
            print(f"Error in process_healthkit_export: {e}")
            return result_files 