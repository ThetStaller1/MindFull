"""
HealthKit Data Extractor (Test Version)

This module provides functionality to extract and organize health data from Apple HealthKit.
It's designed to work with data exported from the iOS HealthKit API, structuring it 
in a way that can be easily processed by the HealthKit to Fitbit mapper.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional


class HealthKitExtractor:
    """
    Class to handle the extraction and organization of Apple HealthKit data.
    """
    
    def __init__(self):
        """
        Initialize the HealthKit extractor with mappings between HealthKit identifiers
        and the data types needed for Fitbit conversion.
        """
        # HealthKit type identifiers
        self.healthkit_types = {
            # Heart rate data
            'heart_rate': 'HKQuantityTypeIdentifierHeartRate',
            
            # Step count data
            'steps': 'HKQuantityTypeIdentifierStepCount',
            
            # Energy data
            'active_energy': 'HKQuantityTypeIdentifierActiveEnergyBurned',
            'basal_energy': 'HKQuantityTypeIdentifierBasalEnergyBurned',
            
            # Distance and elevation data
            'flights_climbed': 'HKQuantityTypeIdentifierFlightsClimbed',
            'distance_walking_running': 'HKQuantityTypeIdentifierDistanceWalkingRunning',
            
            # Sleep data
            'sleep_analysis': 'HKCategoryTypeIdentifierSleepAnalysis',
            
            # Workout data
            'workouts': 'HKWorkoutTypeIdentifier'
        }
        
        # Expected columns for each data type
        self.expected_columns = {
            'heart_rate': ['startDate', 'endDate', 'value', 'device'],
            'steps': ['startDate', 'endDate', 'value', 'device'],
            'active_energy': ['startDate', 'endDate', 'value', 'device'],
            'basal_energy': ['startDate', 'endDate', 'value', 'device'],
            'flights_climbed': ['startDate', 'endDate', 'value', 'device'],
            'distance_walking_running': ['startDate', 'endDate', 'value', 'device'],
            'sleep_analysis': ['startDate', 'endDate', 'value', 'device'],
            'workouts': ['startDate', 'endDate', 'duration', 'workoutActivityType', 'totalDistance', 'totalEnergyBurned', 'device']
        }
    
    def convert_healthkit_datetime(self, date_str: str) -> datetime:
        """
        Convert HealthKit datetime string to Python datetime object.
        
        Args:
            date_str: Datetime string from HealthKit
            
        Returns:
            Converted datetime object
        """
        # Handle empty or None values
        if not date_str:
            return datetime.now()
            
        # HealthKit timestamps might be in different formats, handle accordingly
        try:
            # ISO 8601 format with Z for UTC
            if 'Z' in date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            # ISO 8601 format with timezone offset
            elif '+' in date_str or '-' in date_str and 'T' in date_str:
                return datetime.fromisoformat(date_str)
            # Apple's default format (2023-04-01 08:30:45 -0400)
            else:
                # Try different formats
                try:
                    # Format with timezone
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")
                except ValueError:
                    # Format without timezone
                    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Fallback for other formats or errors
            print(f"Warning: Could not parse date '{date_str}', using current time")
            return datetime.now()
    
    def extract_heart_rate(self, healthkit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract heart rate data from HealthKit.
        
        Args:
            healthkit_data: List of heart rate data points from HealthKit
            
        Returns:
            DataFrame with heart rate data
        """
        heart_rate_records = []
        type_id = self.healthkit_types['heart_rate']
        
        for record in healthkit_data:
            if record.get('type') == type_id:
                try:
                    value = record.get('value', '0')
                    # Handle string values
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            value = 0
                    
                    heart_rate_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': value,
                        'device': record.get('device', '')
                    })
                except (ValueError, KeyError) as e:
                    # Skip records with invalid data
                    print(f"Skipping heart rate record due to error: {e}")
                    continue
        
        if heart_rate_records:
            return pd.DataFrame(heart_rate_records)
        else:
            return pd.DataFrame(columns=self.expected_columns['heart_rate'])
    
    def extract_steps(self, healthkit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract step count data from HealthKit.
        
        Args:
            healthkit_data: List of step count data points from HealthKit
            
        Returns:
            DataFrame with step count data
        """
        step_records = []
        type_id = self.healthkit_types['steps']
        
        for record in healthkit_data:
            if record.get('type') == type_id:
                try:
                    value = record.get('value', '0')
                    # Handle string values
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            value = 0
                            
                    step_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': value,
                        'device': record.get('device', '')
                    })
                except (ValueError, KeyError) as e:
                    # Skip records with invalid data
                    print(f"Skipping steps record due to error: {e}")
                    continue
        
        if step_records:
            return pd.DataFrame(step_records)
        else:
            return pd.DataFrame(columns=self.expected_columns['steps'])
    
    def extract_energy(self, healthkit_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Extract energy (calories) data from HealthKit.
        
        Args:
            healthkit_data: List of energy data points from HealthKit
            
        Returns:
            Dictionary with active and basal energy DataFrames
        """
        active_energy_records = []
        basal_energy_records = []
        
        active_type_id = self.healthkit_types['active_energy']
        basal_type_id = self.healthkit_types['basal_energy']
        
        for record in healthkit_data:
            try:
                value = record.get('value', '0')
                # Handle string values
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = 0
                        
                if record.get('type') == active_type_id:
                    active_energy_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': value,
                        'device': record.get('device', '')
                    })
                elif record.get('type') == basal_type_id:
                    basal_energy_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': value,
                        'device': record.get('device', '')
                    })
            except (ValueError, KeyError) as e:
                # Skip records with invalid data
                print(f"Skipping energy record due to error: {e}")
                continue
        
        result = {}
        
        if active_energy_records:
            result['active_energy'] = pd.DataFrame(active_energy_records)
        else:
            result['active_energy'] = pd.DataFrame(columns=self.expected_columns['active_energy'])
        
        if basal_energy_records:
            result['basal_energy'] = pd.DataFrame(basal_energy_records)
        else:
            result['basal_energy'] = pd.DataFrame(columns=self.expected_columns['basal_energy'])
        
        return result
    
    def extract_flights_climbed(self, healthkit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract flights climbed data from HealthKit.
        
        Args:
            healthkit_data: List of flights climbed data points from HealthKit
            
        Returns:
            DataFrame with flights climbed data
        """
        flights_records = []
        type_id = self.healthkit_types['flights_climbed']
        
        for record in healthkit_data:
            if record.get('type') == type_id:
                try:
                    value = record.get('value', '0')
                    # Handle string values
                    if isinstance(value, str):
                        try:
                            value = float(value)
                        except ValueError:
                            value = 0
                            
                    flights_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': value,
                        'device': record.get('device', '')
                    })
                except (ValueError, KeyError) as e:
                    # Skip records with invalid data
                    print(f"Skipping flights record due to error: {e}")
                    continue
        
        if flights_records:
            return pd.DataFrame(flights_records)
        else:
            return pd.DataFrame(columns=self.expected_columns['flights_climbed'])
    
    def extract_sleep(self, healthkit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract sleep analysis data from HealthKit.
        
        Args:
            healthkit_data: List of sleep analysis data points from HealthKit
            
        Returns:
            DataFrame with sleep analysis data
        """
        sleep_records = []
        type_id = self.healthkit_types['sleep_analysis']
        
        for record in healthkit_data:
            if record.get('type') == type_id:
                try:
                    sleep_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'value': record.get('value', ''),
                        'device': record.get('device', '')
                    })
                except (ValueError, KeyError) as e:
                    # Skip records with invalid data
                    print(f"Skipping sleep record due to error: {e}")
                    continue
        
        if sleep_records:
            return pd.DataFrame(sleep_records)
        else:
            return pd.DataFrame(columns=self.expected_columns['sleep_analysis'])
    
    def extract_workouts(self, healthkit_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract workout data from HealthKit.
        
        Args:
            healthkit_data: List of workout data points from HealthKit
            
        Returns:
            DataFrame with workout data
        """
        workout_records = []
        type_id = self.healthkit_types['workouts']
        
        for record in healthkit_data:
            if record.get('type') == type_id or record.get('workoutActivityType') is not None:
                try:
                    # Convert string values to floats where needed
                    duration = record.get('duration', '0')
                    if isinstance(duration, str):
                        try:
                            duration = float(duration)
                        except ValueError:
                            duration = 0
                            
                    total_distance = record.get('totalDistance', '0')
                    if isinstance(total_distance, str):
                        try:
                            total_distance = float(total_distance)
                        except ValueError:
                            total_distance = 0
                            
                    total_energy = record.get('totalEnergyBurned', '0')
                    if isinstance(total_energy, str):
                        try:
                            total_energy = float(total_energy)
                        except ValueError:
                            total_energy = 0
                    
                    workout_records.append({
                        'startDate': self.convert_healthkit_datetime(record.get('startDate', '')),
                        'endDate': self.convert_healthkit_datetime(record.get('endDate', '')),
                        'duration': duration,
                        'workoutActivityType': record.get('workoutActivityType', ''),
                        'totalDistance': total_distance,
                        'totalEnergyBurned': total_energy,
                        'device': record.get('device', '')
                    })
                except (ValueError, KeyError) as e:
                    # Skip records with invalid data
                    print(f"Skipping workout record due to error: {e}")
                    continue
        
        if workout_records:
            return pd.DataFrame(workout_records)
        else:
            return pd.DataFrame(columns=self.expected_columns['workouts'])
    
    def extract_all_data(self, healthkit_data: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Extract all relevant data from HealthKit export.
        
        Args:
            healthkit_data: List of all health data points from HealthKit
            
        Returns:
            Dictionary with DataFrames for each data type
        """
        result = {}
        
        # Extract heart rate data
        print("Extracting heart rate data...")
        result['heart_rate'] = self.extract_heart_rate(healthkit_data)
        
        # Extract steps data
        print("Extracting step data...")
        result['steps'] = self.extract_steps(healthkit_data)
        
        # Extract energy data
        print("Extracting energy data...")
        energy_data = self.extract_energy(healthkit_data)
        result['active_energy'] = energy_data['active_energy']
        result['basal_energy'] = energy_data['basal_energy']
        
        # Extract flights climbed data
        print("Extracting flights climbed data...")
        result['flights_climbed'] = self.extract_flights_climbed(healthkit_data)
        
        # Extract sleep data
        print("Extracting sleep data...")
        result['sleep_analysis'] = self.extract_sleep(healthkit_data)
        
        # Extract workout data
        print("Extracting workout data...")
        result['workouts'] = self.extract_workouts(healthkit_data)
        
        return result 