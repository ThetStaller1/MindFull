#!/usr/bin/env python
"""
Test Apple Health XML Export Converter

This script processes data directly from the XML export file that Apple Health
provides when you use the "Export All Health Data" option.
"""

import os
import sys
import xml.etree.ElementTree as ET
import json
import pandas as pd
from datetime import datetime
import argparse

from test_healthkit_extractor import HealthKitExtractor
from test_healthkit_mapper import HealthKitToFitbitMapper

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert Apple Health exported XML data to Fitbit format'
    )
    
    parser.add_argument(
        '--input', 
        required=True, 
        help='Path to the Apple Health export.xml file'
    )
    
    parser.add_argument(
        '--output', 
        required=True, 
        help='Directory where the converted Fitbit format files will be saved'
    )
    
    parser.add_argument(
        '--person_id', 
        default="1001",
        help='Person ID to use in the converted data (default: 1001)'
    )
    
    parser.add_argument(
        '--group_id',
        default="59116210", 
        help='Group ID to use in the filename (59116210 for control or 82793569 for subject)'
    )
    
    return parser.parse_args()

def convert_xml_to_healthkit_format(xml_path):
    """
    Convert Apple Health XML export to a format compatible with our HealthKit extractor.
    
    Args:
        xml_path: Path to the export.xml file
    
    Returns:
        List of dictionaries with health data in JSON format
    """
    print(f"Parsing Apple Health XML export from {xml_path}...")
    
    # Parse the XML file - this may take some time for large exports
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    
    # Map Apple Health record types to HealthKit types
    type_mapping = {
        "HKQuantityTypeIdentifierHeartRate": "HKQuantityTypeIdentifierHeartRate",
        "HKQuantityTypeIdentifierStepCount": "HKQuantityTypeIdentifierStepCount",
        "HKQuantityTypeIdentifierActiveEnergyBurned": "HKQuantityTypeIdentifierActiveEnergyBurned",
        "HKQuantityTypeIdentifierBasalEnergyBurned": "HKQuantityTypeIdentifierBasalEnergyBurned",
        "HKQuantityTypeIdentifierFlightsClimbed": "HKQuantityTypeIdentifierFlightsClimbed",
        "HKCategoryTypeIdentifierSleepAnalysis": "HKCategoryTypeIdentifierSleepAnalysis",
        "HKWorkoutTypeIdentifier": "HKWorkoutTypeIdentifier"
    }
    
    result = []
    
    # Process Record elements (main health data points)
    for record in root.findall('.//Record'):
        record_type = record.get('type')
        if record_type in type_mapping:
            record_data = {
                "type": record_type,
                "startDate": record.get('startDate'),
                "endDate": record.get('endDate'),
                "value": record.get('value'),
                "device": record.get('device', '')
            }
            result.append(record_data)
    
    # Process Workout elements
    for workout in root.findall('.//Workout'):
        workout_data = {
            "type": "HKWorkoutTypeIdentifier",
            "startDate": workout.get('startDate'),
            "endDate": workout.get('endDate'),
            "workoutActivityType": workout.get('workoutActivityType'),
            "duration": workout.get('duration', '0'),
            "totalDistance": workout.get('totalDistance', '0'),
            "totalEnergyBurned": workout.get('totalEnergyBurned', '0'),
            "device": workout.get('device', '')
        }
        result.append(workout_data)
    
    print(f"Extracted {len(result)} health data records from XML export")
    return result

def main():
    """Main function to test the conversion with Apple Health XML export."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Convert XML data to format compatible with our extractor
    healthkit_data = convert_xml_to_healthkit_format(args.input)
    
    # Save intermediate JSON (for debugging purposes)
    json_path = os.path.join(args.output, 'converted_healthkit_data.json')
    print(f"Saving intermediate JSON to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(healthkit_data, f, indent=2)
    
    # Extract and organize HealthKit data
    print("Extracting and organizing health data...")
    extractor = HealthKitExtractor()
    extracted_data = extractor.extract_all_data(healthkit_data)
    
    # Print summary of extracted data
    print("\nExtracted Data Summary:")
    for data_type, data_frame in extracted_data.items():
        print(f"  - {data_type}: {len(data_frame)} records")
    
    # Convert to Fitbit format
    print(f"Converting data to Fitbit format (person_id: {args.person_id}, group_id: {args.group_id})...")
    mapper = HealthKitToFitbitMapper(args.output)
    result_files = mapper.process_healthkit_export(extracted_data, args.person_id, args.group_id)
    
    # Print summary of converted files
    print("\nConversion complete. Generated files:")
    for data_type, file_path in result_files.items():
        print(f"  - {data_type}: {file_path}")

if __name__ == "__main__":
    main() 