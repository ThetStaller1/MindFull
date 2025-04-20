# MindWatch iOS App

MindWatch is an iOS app that analyzes your HealthKit data to provide mental health insights using a machine learning model.

## Features

- Collects and displays various health metrics from Apple HealthKit
- Sends health data to a backend server for mental health analysis
- Displays a mental health risk assessment with contributing factors
- Provides recommendations based on the analysis results

## Setup

1. Open the project in Xcode 15 or later
2. Make sure you have the required development provisioning profiles
3. Configure the app to use HealthKit capabilities
4. Build and run on an iOS device or simulator

## HealthKit Data Usage

The app requests and uses the following HealthKit data types:

- Heart rate
- Step count
- Active energy burned
- Distance
- Sleep analysis
- Basal energy burned
- Flights climbed
- Workout data

## Backend Integration

The app communicates with a FastAPI backend to analyze the health data. To configure the backend URL:

1. For development, the app connects to `http://localhost:8000` by default
2. For production, update the `APIService.swift` file with your production server URL

## Project Structure

- `model/` - Core data models and services
  - `HealthData.swift` - SwiftData model for health metrics
  - `HealthKitManager.swift` - Manages HealthKit data access
  - `HealthViewModel.swift` - View model for health data
  - `APIService.swift` - Handles communication with the backend
- `MentalHealthView.swift` - View for displaying mental health analysis
- `HealthDataView.swift` - View for displaying health metrics

## Privacy

The app transmits health data to the backend server for analysis. All data is transmitted securely and no personal health information is stored on the server beyond the duration of the analysis.

## Requirements

- iOS 17.0+
- Xcode 15.0+
- HealthKit compatible device 