# Test HealthKit to Fitbit Converter for XML Export

This test code helps convert your Apple Health export (XML format) to the All of Us Fitbit dataset format used for mental health detection.

## Files

1. `test_xml_converter.py` - Main script for processing XML exports
2. `test_healthkit_extractor.py` - Extracts and organizes health data from the XML
3. `test_healthkit_mapper.py` - Maps the health data to Fitbit format

## How to Use

1. Export your health data from the Apple Health app:
   - Open the Health app on your iPhone
   - Tap your profile icon at the top-right
   - Scroll down and tap "Export All Health Data"
   - This creates a zip file which you need to transfer to your computer

2. Unzip the export (it contains an `export.xml` file)

3. Run the test script:
   ```bash
   python test_xml_converter.py --input /path/to/apple_health_export/export.xml --output /path/for/output --person_id 1001 --group_id 59116210
   ```

   Parameters:
   - `--input`: Path to the export.xml file
   - `--output`: Directory where the converted files will be saved
   - `--person_id`: Person ID to use in the data (default: 1001)
   - `--group_id`: Group ID for filenames (59116210 for control, 82793569 for subject)

4. The script will:
   - Parse the XML file (may take some time for large exports)
   - Extract relevant health data
   - Convert to Fitbit format
   - Save CSV files in the output directory

## Handling XML Errors

Apple Health exports can have XML format issues. If you encounter XML parsing errors, you may need to fix the XML structure first. Common issues include:
- Duplicate startDate attributes
- Malformed DTD 
- Missing element declarations

## Output Files

The converter creates these files in Fitbit format:
- `dataset_[group_id]_fitbit_heart_rate_summary.csv`
- `dataset_[group_id]_fitbit_heart_rate_level.csv`
- `dataset_[group_id]_fitbit_intraday_steps.csv`
- `dataset_[group_id]_fitbit_activity.csv`
- `dataset_[group_id]_fitbit_sleep_daily_summary.csv`
- `dataset_[group_id]_fitbit_sleep_level.csv` 