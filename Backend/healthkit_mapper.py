#!/usr/bin/env python
"""
HealthKit to Fitbit Data Format Mapper

This module converts extracted Apple HealthKit data to Fitbit format for 
compatibility with the mental health assessment algorithm that was trained 
on the "All of Us" NIH wearable dataset. 

Mapping Strategy:
----------------
The module implements a comprehensive mapping between Apple HealthKit and Fitbit data formats:

1. Direct Mappings:
   - person_id: User identifier from the system
   - date: Date of activity formatted as YYYY-MM-DD
   - activity_calories: Mapped from HealthKit's activeEnergyBurned
   - calories_bmr: Mapped from HealthKit's basalEnergyBurned
   - calories_out: Calculated as sum of activeEnergyBurned + basalEnergyBurned
   - steps: Mapped from HealthKit's stepCount
   - floors: Mapped from HealthKit's flightsClimbed

2. Derived Mappings:
   - elevation: Converted from flightsClimbed using standard conversion (3m per floor)
   - activity minutes categories:
     * very_active_minutes: Minutes in "Cardio" or "Peak" heart rate zones or high-intensity workouts
     * fairly_active_minutes: Minutes in "Fat Burn" heart rate zone or moderate workouts
     * lightly_active_minutes: Minutes with activity but below moderate intensity
     * sedentary_minutes: Total daily minutes (1440) minus all active minutes
   - marginal_calories: Estimated as 9% of total calories (calories_out)

3. Validation Checks:
   - Active + BMR calories should equal total calories
   - Total minutes across all activity levels should equal 1440 (24 hours)
   - Activity minutes are scaled appropriately when exceeding constraints

The mapping logic incorporates both heart rate data and workout data to determine 
activity intensity levels, providing a more accurate representation of activity
compared to using only step counts or calories.
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
        
        This method orchestrates the conversion of all HealthKit data types to their 
        corresponding Fitbit format for processing by the mental health assessment algorithm.
        It handles heart rate, steps, active energy, basal energy, flights climbed, 
        and sleep data, applying the mapping logic defined in each specialized converter method.
        
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
        
        # Convert active energy data to activity data (comprehensive mapping)
        if 'active_energy' in extracted_data and not extracted_data['active_energy'].empty:
            # Pass all relevant data types to the activity converter for comprehensive mapping
            activity_df = self._convert_active_energy_to_activity(
                active_energy_df=extracted_data['active_energy'],
                workout_df=extracted_data.get('workout', pd.DataFrame()),
                person_id=person_id,
                basal_energy_df=extracted_data.get('basal_energy'),
                steps_df=extracted_data.get('steps'),
                flights_df=extracted_data.get('flights_climbed'),
                heart_rate_df=extracted_data.get('heart_rate')
            )
            
            if not activity_df.empty:
                fitbit_data['fitbit_activity'] = activity_df
                logger.info(f"Generated activity data with {len(activity_df)} records")
        else:
            logger.warning("No active energy data available - cannot generate activity records")
        
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
                                 person_id: str,
                                 basal_energy_df: Optional[pd.DataFrame] = None,
                                 steps_df: Optional[pd.DataFrame] = None,
                                 flights_df: Optional[pd.DataFrame] = None,
                                 heart_rate_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Convert HealthKit data to Fitbit activity format following the mapping plan.
        
        This function implements comprehensive mapping from HealthKit data to Fitbit activity format
        for mental health assessment algorithm compatibility. It maps the following fields:
        
        Direct mappings:
        - person_id: User identifier
        - date: Date of activity (YYYY-MM-DD)
        - activity_calories: From activeEnergyBurned
        - calories_bmr: From basalEnergyBurned
        - calories_out: Sum of activity_calories and calories_bmr
        - steps: From stepCount
        - floors: From flightsClimbed
        
        Derived mappings:
        - elevation: Calculated from flightsClimbed (3 meters per floor)
        - very_active_minutes: Minutes in high heart rate zones or intense workouts
        - fairly_active_minutes: Minutes in moderate heart rate zones or moderate workouts
        - lightly_active_minutes: Minutes with activity below moderate threshold
        - sedentary_minutes: Total daily minutes (1440) minus all active minutes
        - marginal_calories: Estimated as percentage of calories_out
        
        Args:
            active_energy_df: DataFrame with active energy data from HealthKit
            workout_df: DataFrame with workout data from HealthKit
            person_id: User ID
            basal_energy_df: Optional DataFrame with basal energy data
            steps_df: Optional DataFrame with steps data
            flights_df: Optional DataFrame with flights climbed data
            heart_rate_df: Optional DataFrame with heart rate data
            
        Returns:
            DataFrame in Fitbit activity format with complete data mapping
        """
        if active_energy_df.empty:
            logger.warning("No active energy data available for activity calculation")
            # Return an empty DataFrame with the expected columns instead of an empty list
            return pd.DataFrame(columns=[
                'person_id', 'date', 'activity_calories', 'calories_bmr', 'calories_out',
                'floors', 'elevation', 'fairly_active_minutes', 'lightly_active_minutes',
                'marginal_calories', 'sedentary_minutes', 'steps', 'very_active_minutes'
            ])
        
        # Group by date to get daily totals for active energy
        if 'date' not in active_energy_df.columns and 'startDate' in active_energy_df.columns:
            active_energy_df['date'] = pd.to_datetime(active_energy_df['startDate']).dt.date
        
        # Sum active energy by date
        daily_energy = active_energy_df.groupby('date')['value'].sum().reset_index()
        daily_energy = daily_energy.rename(columns={'value': 'activity_calories'})
        
        # Process basal energy data (calories_bmr)
        if basal_energy_df is not None and not basal_energy_df.empty:
            if 'date' not in basal_energy_df.columns and 'startDate' in basal_energy_df.columns:
                basal_energy_df['date'] = pd.to_datetime(basal_energy_df['startDate']).dt.date
            
            # Sum basal energy by date
            daily_basal = basal_energy_df.groupby('date')['value'].sum().reset_index()
            daily_basal = daily_basal.rename(columns={'value': 'calories_bmr'})
            
            # Merge with active energy data
            daily_energy = pd.merge(daily_energy, daily_basal, on='date', how='outer')
        else:
            # Use a reasonable default BMR if no data available
            # BMR varies by age, gender, weight but ~1500-1800 is reasonable for adults
            daily_energy['calories_bmr'] = 1600
            logger.info("Using default BMR of 1600 calories (no basal energy data available)")
        
        # Calculate total calories (activity + BMR)
        daily_energy['calories_out'] = daily_energy['activity_calories'] + daily_energy['calories_bmr']
        
        # Process steps data
        if steps_df is not None and not steps_df.empty:
            if 'date' not in steps_df.columns and 'startDate' in steps_df.columns:
                steps_df['date'] = pd.to_datetime(steps_df['startDate']).dt.date
            
            # Sum steps by date
            daily_steps = steps_df.groupby('date')['value'].sum().reset_index()
            daily_steps = daily_steps.rename(columns={'value': 'steps'})
            
            # Merge with energy data
            daily_energy = pd.merge(daily_energy, daily_steps, on='date', how='outer')
        else:
            daily_energy['steps'] = 0
            logger.info("No steps data available")
        
        # Process floors/elevation data
        if flights_df is not None and not flights_df.empty:
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
            logger.info("No floors/elevation data available")
        
        # Calculate activity minutes based on heart rate data and workout data
        activity_minutes = self._calculate_activity_minutes(heart_rate_df, workout_df)
        
        # Merge activity minutes with daily energy data
        if not activity_minutes.empty:
            daily_energy = pd.merge(daily_energy, activity_minutes, on='date', how='outer')
        else:
            # Provide default activity minutes based on activity calories if no HR data
            daily_energy['very_active_minutes'] = daily_energy['activity_calories'].apply(
                lambda cals: min(int(cals * 0.01), 120)  # ~1% of activity calories, max 2 hours
            )
            daily_energy['fairly_active_minutes'] = daily_energy['activity_calories'].apply(
                lambda cals: min(int(cals * 0.02), 240)  # ~2% of activity calories, max 4 hours
            )
            daily_energy['lightly_active_minutes'] = daily_energy['activity_calories'].apply(
                lambda cals: min(int(cals * 0.03), 360)  # ~3% of activity calories, max 6 hours
            )
            logger.info("Using estimated activity minutes based on calories (no heart rate/workout data)")
        
        # Calculate sedentary minutes
        total_day_minutes = 24 * 60  # 1440 minutes in a day
        daily_energy['sedentary_minutes'] = total_day_minutes - (
            daily_energy['very_active_minutes'] + 
            daily_energy['fairly_active_minutes'] + 
            daily_energy['lightly_active_minutes']
        )
        # Ensure sedentary minutes are not negative
        daily_energy['sedentary_minutes'] = daily_energy['sedentary_minutes'].apply(lambda x: max(0, x))
        
        # Calculate marginal calories (typically ~8-10% of total calories)
        daily_energy['marginal_calories'] = (daily_energy['calories_out'] * 0.09).astype(int)
        
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
        
        # Format the final dataframe
        activity_df = pd.DataFrame({
            'person_id': person_id,
            'date': daily_energy['date'].astype(str),
            'activity_calories': daily_energy['activity_calories'].astype(int),
            'calories_bmr': daily_energy['calories_bmr'].astype(int),
            'calories_out': daily_energy['calories_out'].astype(int),
            'floors': daily_energy['floors'].astype(int),
            'elevation': daily_energy['elevation'].astype(int),
            'fairly_active_minutes': daily_energy['fairly_active_minutes'].astype(int),
            'lightly_active_minutes': daily_energy['lightly_active_minutes'].astype(int),
            'marginal_calories': daily_energy['marginal_calories'].astype(int),
            'sedentary_minutes': daily_energy['sedentary_minutes'].astype(int),
            'steps': daily_energy['steps'].astype(int),
            'very_active_minutes': daily_energy['very_active_minutes'].astype(int)
        })
        
        # Validate the generated data
        activity_df = self._validate_activity_data(activity_df)
        logger.info(f"Generated {len(activity_df)} daily activity records")
        
        return activity_df

    def _calculate_activity_minutes(self, heart_rate_df: Optional[pd.DataFrame] = None, 
                                  workout_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate activity minutes from heart rate and workout data.
        
        Activity level categorization:
        - very_active_minutes: "Cardio" or "Peak" heart rate zones or high-intensity workouts
        - fairly_active_minutes: "Fat burn" heart rate zone or moderate-intensity workouts
        - lightly_active_minutes: Minutes with activity but below moderate intensity
        
        Args:
            heart_rate_df: Optional DataFrame with heart rate data
            workout_df: Optional DataFrame with workout data
            
        Returns:
            DataFrame with date and activity minutes columns
        """
        result_data = []
        
        # Calculate activity minutes from heart rate data
        if heart_rate_df is not None and not heart_rate_df.empty:
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
        if workout_df is not None and not workout_df.empty:
            # Ensure date column exists
            if 'workout_date' not in workout_df.columns and 'startDate' in workout_df.columns:
                workout_df['workout_date'] = pd.to_datetime(workout_df['startDate']).dt.date
            
            # Define workout intensity mapping
            workout_intensity = {
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
            
            # Group workouts by date
            for date, group in workout_df.groupby('workout_date'):
                very_active_mins = 0
                fairly_active_mins = 0
                lightly_active_mins = 0
                
                # Process each workout
                for _, workout in group.iterrows():
                    # Get duration in minutes
                    duration_mins = workout['duration'] / 60 if 'duration' in workout else 0
                    
                    # Determine intensity based on workout type
                    workout_type = workout.get('workoutActivityType', 'default')
                    intensity = workout_intensity.get(workout_type, workout_intensity['default'])
                    
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
        2. Active energy burned should correlate with activity minutes and steps
        3. Ensure calories_out = activity_calories + calories_bmr
        
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
        validated_df['marginal_calories'] = (validated_df['calories_out'] * 0.09).astype(int)
        
        return validated_df
    
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
