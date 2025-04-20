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