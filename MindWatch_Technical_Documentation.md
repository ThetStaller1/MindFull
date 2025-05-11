# MindWatch Technical Documentation

## Project Overview

MindWatch is a comprehensive mental health monitoring application that leverages Apple Watch and HealthKit data to analyze a user's mental health status. The system consists of three main components:

1. **iOS Client Application**: Built with Swift and SwiftUI, interfaces with Apple HealthKit and Apple Watch
2. **FastAPI Backend**: Python-based server providing API endpoints, data processing, and mental health analysis
3. **Supabase Database**: Cloud-based PostgreSQL database for secure data storage and authentication

## System Architecture

### Component Overview

#### iOS Application (Swift/SwiftUI)
- Collects and processes HealthKit data from Apple Watch
- Authenticates users through backend API (not directly with Supabase)
- Presents analysis results and educational resources
- Uses MVVM architecture pattern

#### Backend Server (Python/FastAPI)
- Provides RESTful API endpoints for authentication, data upload, and analysis
- Interfaces with Supabase for database operations
- Contains the mental health prediction model (XGBoost)
- Processes and transforms health data for analysis

#### Database (Supabase/PostgreSQL)
- Stores user authentication data
- Maintains health data records with timestamp indexing
- Stores analysis results and user profiles
- Provides secure authentication mechanisms

### Data Flow

#### Authentication Flow
1. User registers/logs in through iOS app
2. iOS app sends credentials to FastAPI backend
3. Backend authenticates with Supabase and returns token to iOS app
4. iOS app stores token securely and uses for subsequent requests

#### Data Collection Flow
1. User grants HealthKit permissions in iOS app
2. App retrieves last sync timestamp from backend
3. App collects new HealthKit data since last sync (or 60 days if no previous sync)
4. Data is sent to backend via API
5. Backend transforms and stores data in Supabase

#### Analysis Flow
1. User initiates mental health screening in iOS app
2. Backend retrieves 60 days of health data from Supabase
3. Data is processed through mental health algorithm (XGBoost model)
4. Analysis results are stored in Supabase
5. Results are returned to iOS app for display

## Technical Implementation

### iOS Application

#### Technologies Used
- **Language**: Swift 5.x
- **Framework**: SwiftUI, Combine
- **Data Access**: HealthKit framework
- **Networking**: URLSession for API requests
- **Architecture Pattern**: MVVM (Model-View-ViewModel)

#### Key Components

##### Authentication Module
- `AuthView.swift`: Login and registration interface
- `AuthViewModel.swift`: Authentication business logic and state management

##### Health Data Module
- `HealthViewModel.swift`: Core health data collection and processing
- `HealthDataView.swift`: Interface for viewing and syncing health data
- `HealthDataPoint.swift`: Model representing individual health data points

##### Analysis Module
- `AnalysisView.swift`: Displays mental health analysis results
- `AnalysisResult.swift`: Model representing analysis output

##### Core Infrastructure
- `APIService.swift`: Handles all API communication with backend
- `Models.swift`: Core data models used throughout the application
- `MainTabView.swift`: Main navigation interface

#### Health Data Collection

The app collects the following health metrics from HealthKit:
- Heart rate (resting, average, min, max)
- Step count
- Active energy burned
- Basal energy burned
- Sleep analysis (duration and stages)
- Workouts (duration, type, energy burned)
- Flights climbed

### Backend Server

#### Technologies Used
- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Database Access**: Supabase Client
- **Machine Learning**: XGBoost
- **Data Processing**: Pandas, NumPy
- **Authentication**: OAuth2, JWT

#### Key Components

##### API Layer (`app.py`)
- REST API endpoints for authentication, data upload, analysis
- Dependency injection for Supabase client and predictor
- Authentication middleware for token validation
- Comprehensive error handling and logging

##### Health Data Processing (`health_extractor.py`)
- Standardizes HealthKit data format from iOS
- Transforms raw health data into model-ready features
- Handles edge cases and data normalization
- Maps Apple HealthKit types to internal data model

##### Mental Health Prediction (`app.py` - MentalHealthPredictor class)
- Loads trained XGBoost model from file
- Processes health data for prediction
- Calculates contributing factors to mental health score
- Provides data quality assessment

##### Database Interface (`supabase_client.py`)
- Encapsulates all Supabase interactions
- Handles authentication operations
- Manages health data storage and retrieval
- Processes analysis results storage

#### Machine Learning Model

- **Algorithm**: XGBoost Classifier
- **Features**: Processed from health data including:
  - Sleep patterns (duration, consistency, stages)
  - Physical activity metrics (steps, exercise time)
  - Heart rate patterns and variability
  - Daily activity patterns
  - Energy expenditure

- **Output**: 
  - Binary classification (0-1)
  - Risk level (Low, Moderate, High)
  - Contributing factors
  - Confidence score

### Database Schema

#### Authentication Tables
- User authentication records
- Session tokens and refresh tokens

#### Health Data Tables
- Heart rate measurements with timestamps
- Step count data with timestamps
- Sleep analysis records with stages
- Activity and energy expenditure records
- Workout records with metadata

#### Analysis Tables
- Mental health analysis results
- Timestamps of analyses
- Contributing factors
- Risk scores and levels

## API Endpoints

### Authentication Endpoints
- `POST /register`: Register new user
- `POST /login`: Authenticate user
- `POST /validate-token`: Validate authentication token
- `POST /logout`: End user session

### Health Data Endpoints
- `POST /analyze`: Upload and analyze health data
- `GET /latest-data-timestamp/{user_id}`: Get timestamp of latest data
- `GET /data-quality/{user_id}`: Check completeness of health data
- `GET /check-missing-data/{user_id}`: Identify missing data types

### Analysis Endpoints
- `GET /latest-analysis/{user_id}`: Get most recent analysis
- `GET /analysis-history/{user_id}`: Get history of analyses
- `POST /check-analysis`: Check if new analysis is needed

### User Profile Endpoints
- `POST /update-profile`: Update user demographic data
- `GET /data-collection-guidance`: Get guidance on data collection

## Security Considerations

- All database operations route through backend API, never directly from iOS app
- Authentication tokens stored securely on iOS device
- User data encrypted in transit with HTTPS
- Authorization required for all sensitive endpoints
- Input validation on all API endpoints
- Limited data retention policies

## Development Environment

### iOS Development
- Xcode 14+
- Swift 5.x
- iOS 15+ deployment target
- Apple Developer account for HealthKit capabilities

### Backend Development
- Python 3.9+
- FastAPI framework
- Supabase client libraries
- Docker containers for deployment
- Environment variables for configuration

## Deployment Considerations

### iOS App Deployment
- App Store review considerations for HealthKit usage
- Privacy policy requirements for health data
- Beta testing through TestFlight

### Backend Deployment
- Server-side environment configuration
- Database connection settings
- Machine learning model deployment
- API security and rate limiting

## Data Processing Pipeline

1. **Data Collection**: iOS app collects HealthKit data
2. **Data Transmission**: Secure transmission to backend API
3. **Data Transformation**: Backend transforms raw data to standardized format
4. **Feature Engineering**: Creation of model-ready features
5. **Model Prediction**: XGBoost model predicts mental health status
6. **Result Storage**: Analysis results stored in database
7. **Result Presentation**: Results returned to iOS app for display

## Error Handling and Monitoring

- Comprehensive error handling at all levels
- Detailed logging of system operations
- Rate limiting and request validation
- Fallback mechanisms for data processing
- Data quality assessment before analysis

## Performance Considerations

- Efficient data retrieval based on timestamps
- Batch processing of health data
- Periodic data synchronization
- Backend caching for frequent operations
- Response time monitoring and optimization

## Conclusion

MindWatch represents a sophisticated integration of mobile health technology, cloud services, and machine learning to deliver mental health insights from wearable device data. The architecture ensures security, efficiency, and reliability while maintaining a user-friendly interface for accessing complex health analytics. 