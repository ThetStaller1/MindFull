# MindFull API - FastAPI Backend

This backend serves the MindWatch iOS app by processing HealthKit data and predicting mental health risk using a pre-trained XGBoost model.

## Setup

1. Make sure you have Python 3.8+ installed
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Use the included run script:

```bash
./run.sh
```

Or run directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

- `GET /health` - Check if the API is running
- `POST /analyze` - Analyze HealthKit data and return mental health risk assessment

## Data Format

The API expects HealthKit data in the following format:

```json
{
  "heartRate": [
    {
      "type": "HKQuantityTypeIdentifierHeartRate",
      "startDate": "2023-01-01T12:00:00Z",
      "endDate": "2023-01-01T12:00:05Z",
      "value": 72.0,
      "unit": "count/min"
    }
  ],
  "steps": [...],
  "activeEnergy": [...],
  "sleep": [...],
  "workout": [...],
  "distance": [...],
  "basalEnergy": [...],
  "flightsClimbed": [...],
  "userInfo": {
    "personId": "1001",
    "age": 33,
    "genderBinary": 1
  }
}
```

## Model

The backend uses a pre-trained XGBoost model to analyze health data patterns and predict mental health risk. The model files are located in the `model/` directory:

- `mental_health_model.xgb` - The trained XGBoost model
- `scaler.save` - Feature scaler/names
- `feature_importance.csv` - Feature importance data

## Testing iOS Data Format

The `test_ios_data_format.py` script allows you to capture raw data that would be sent by the iOS app and save it as CSV files in the Fitbit format expected by the mental health model. This helps verify that the data transformation logic is working correctly.

### Usage

```bash
python test_ios_data_format.py --input sample_healthkit_data.json --output converted_data
```

Parameters:
- `--input`: Path to a JSON file containing HealthKit data in the format expected by the API
- `--output`: Directory where the CSV files will be saved
- `--person_id` (optional): Person ID to use in the output files (default: "1001")
- `--group_id` (optional): Group ID to use in filenames (default: "59116210" for control group)

### How it Works

1. The script takes raw HealthKit data in JSON format (same format the iOS app sends to the API)
2. It transforms this data into CSV files that match the Fitbit format expected by the model
3. These CSV files include:
   - Heart rate data
   - Steps data
   - Activity summary
   - Sleep summary
4. The raw JSON is also saved for reference

### Workflow

1. The iOS app collects HealthKit data and sends it to the `/analyze` endpoint in JSON format
2. The `test_ios_data_format.py` script captures this data and saves it as CSV files
3. These CSV files can be compared with those produced by `test_xml_converter.py`
4. The mental health model (`test_mental_health_model.py`) can then process these CSV files

### Example

```bash
# First, run the test to save the sample data as CSV files
python test_ios_data_format.py --input sample_healthkit_data.json --output converted_data

# Then analyze this data using the mental health model
python test_mental_health_model.py --data_dir converted_data --model_dir model
```

This allows you to verify that:
1. The iOS app is sending data in the correct format
2. The data transformation logic is working as expected
3. The mental health model is producing accurate results

# MindFull Backend Updates

## Data Processing Enhancements

This update addresses issues with data processing and Supabase storage in the MindFull application. The primary focus is on ensuring that health data is properly stored in the required Supabase tables, specifically `fitbit_heart_rate_summary` and `fitbit_sleep_level` which were previously missing.

### Key Changes

1. **Enhanced Heart Rate Processing**
   - Added support for generating heart rate zone summaries
   - Implemented calculation of time spent in heart rate zones
   - Created proper mapping to the `fitbit_heart_rate_summary` table

2. **Enhanced Sleep Data Processing**
   - Improved sleep stage mapping from HealthKit to Fitbit format
   - Added support for generating detailed sleep level records
   - Fixed sleep duration calculations for more accurate reporting
   - Ensured proper mapping to the `fitbit_sleep_level` table

3. **Supabase Client Updates**
   - Updated health data storage to include all required tables
   - Implemented proper data transformation for Supabase compatibility
   - Added batch processing for efficient data upload

### Technical Implementation

The implementation was based on the conversion logic found in the test files (`test_xml_converter.py`, `test_healthkit_mapper.py`, and `test_healthkit_extractor.py`), but adapted to work with the direct HealthKit data received from the iOS app.

#### Modified Files:

1. **Backend/healthkit_mapper.py**
   - Added `_convert_heart_rate_summary()` method to generate heart rate zone statistics
   - Updated sleep data processing to include detailed sleep level information
   - Implemented proper stage mapping from HealthKit to Fitbit sleep stages

2. **Backend/supabase_client.py**
   - Enhanced `upload_health_data()` to support all required Supabase tables
   - Added heart rate summary processing including zone calculations
   - Added proper sleep level processing and storage

### Data Flow

The updated data flow is as follows:

1. iOS app collects HealthKit data (heart rate, steps, sleep, etc.)
2. Data is sent to the backend API
3. Backend processes the data using `healthkit_extractor.py` and `healthkit_mapper.py`
4. Transformed data is stored in all required Supabase tables
5. Stored data is used for mental health analysis

### Respiratory Rate Data

Per requirements, respiratory rate data is not processed. The iOS app may still collect this data, but it is ignored during the processing and storage phases. 