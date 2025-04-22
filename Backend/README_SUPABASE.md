# MindWatch Backend with Supabase Integration

This document provides instructions for setting up and using the MindWatch backend with Supabase integration.

## Overview

The MindWatch backend has been updated to store health data and analysis results in Supabase, allowing for:

- User authentication
- Persistent storage of health data
- Scheduled analysis (every 7 days)
- Retrieval of historical analysis results

The iOS app communicates with the backend API, which in turn interacts with Supabase, maintaining a clean separation of concerns.

## Setup Instructions

### 1. Environment Setup

1. Create a `.env` file in the `Backend` directory (copy from `.env.example`):
   ```
   # MindWatch API Environment Variables
   
   # Supabase credentials
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_ANON_KEY=your-supabase-anon-key
   
   # API settings
   API_HOST=0.0.0.0
   API_PORT=8000
   DEBUG=True
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Database Setup

The Supabase project "Mindwatch" has been set up with the following tables:

1. `profiles` - User profile information
2. `person_dataset` - User demographic data
3. `fitbit_heart_rate_level` - Heart rate data
4. `fitbit_heart_rate_summary` - Heart rate zone summaries
5. `fitbit_intraday_steps` - Step count data
6. `fitbit_activity` - Activity data
7. `fitbit_sleep_daily_summary` - Sleep summary data
8. `fitbit_sleep_level` - Detailed sleep stage data
9. `analysis_results` - Mental health analysis results

All tables have Row Level Security (RLS) enabled to ensure data privacy.

## API Endpoints

### Authentication

All endpoints require an authentication token, which should be provided in the request header:

```
Authorization: Bearer <token>
```

### Endpoints

1. **Health Check**
   - `GET /health`
   - Returns API status

2. **Analyze Health Data**
   - `POST /analyze`
   - Processes health data, stores in Supabase, runs the analysis model, and stores results
   - Request body: HealthKit data in required format

3. **Check Analysis Status**
   - `POST /check-analysis`
   - Checks if a new analysis should be run (7 days since last analysis)
   - Request body: `{ "userId": "user-id", "forceRun": false }`

4. **Get Latest Analysis**
   - `GET /latest-analysis/{user_id}`
   - Returns the most recent analysis result for a user

## Data Flow

1. iOS app authenticates with Supabase (through the backend)
2. iOS app collects HealthKit data and sends to backend
3. Backend stores data in Supabase tables
4. Backend runs the analysis model
5. Backend stores analysis results in Supabase
6. iOS app requests latest analysis from backend
7. Backend retrieves analysis from Supabase and returns to the app

## Scheduled Analysis

The system will automatically check if a new analysis is needed:

1. When the user logs in
2. When explicitly requested via the `/check-analysis` endpoint
3. Analysis is performed at most once every 7 days, unless manually triggered

## Troubleshooting

- **Missing Environment Variables**: Ensure the `.env` file is properly set up with Supabase credentials
- **Authentication Errors**: Verify that the authentication token is being correctly passed and validated
- **Database Connection Issues**: Check Supabase dashboard for connectivity issues

## Development Notes

- The backend uses the `supabase-py` library to interact with Supabase
- Environment variables are loaded using `python-dotenv`
- Row Level Security ensures that users can only access their own data
- The mental health prediction model is unchanged, maintaining compatibility with the original dataset format

## Security Considerations

- Authentication tokens should be securely stored on the client device
- All API requests are authenticated
- Database access is restricted through Row Level Security
- Sensitive health data is only processed on the backend
- No direct communication between iOS app and Supabase 