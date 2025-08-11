#!/usr/bin/env python3
"""
Test script for Data Analyst Agent - Sales Dataset Analysis
Tests CSV analysis, statistical calculations, and data visualization based on the project template.
"""

import requests
import json
import time
import base64
import io
import tempfile
import os
import csv
from PIL import Image
from typing import Dict, Any, Optional
import numpy as np

class SalesDatasetTester:
    def __init__(self, api_url="http://localhost:8000/api/"):
        self.api_url = api_url
        self.test_question = """
Analyze `sample-weather.csv`.

Return a JSON object with keys:
- `average_temp_c`: number
- `max_precip_date`: string
- `min_temp_c`: number
- `temp_precip_correlation`: number
- `average_precip_mm`: number
- `temp_line_chart`: base64 PNG string under 100kB
- `precip_histogram`: base64 PNG string under 100kB

Answer:
1. What is the average temperature in Celsius?
2. On which date was precipitation highest?
3. What is the minimum temperature recorded?
4. What is the correlation between temperature and precipitation?
5. What is the average precipitation in millimeters?
6. Plot temperature over time as a line chart with a red line. Encode as base64 PNG.
7. Plot precipitation as a histogram with orange bars. Encode as base64 PNG.
"""

        # Expected answers based on the carefully crafted CSV data
        self.expected_answers = {
            "total_sales": 1140.0,
            "top_region": "West",
            "day_sales_correlation": 0.2228124549277306,
            "median_sales": 140.0,
            "total_sales_tax": 114.0
        }

    def _create_sample_csv_data(self) -> list:
        """
        Creates a precisely calculated sample sales dataset that will produce
        the exact expected answers for all validation checks.
        """
        # This dataset is engineered to meet all constraints simultaneously.
        csv_data = [
            ['order_id', 'date', 'region', 'sales'],
            ['1', '2023-01-20', 'North', '80'],
            ['2', '2023-01-01', 'North', '90'],
            ['3', '2023-01-22', 'East',  '100'],
            ['4', '2023-01-05', 'South', '120'],
            ['5', '2023-01-15', 'West',  '140'],  # This is the median value
            ['6', '2023-01-25', 'East',  '150'],
            ['7', '2023-01-10', 'South', '150'],
            ['8', '2023-01-28', 'West',  '150'],
            ['9', '2023-01-12', 'West',  '160']
        ]
        # REGIONAL TOTALS: North=170, South=270, East=250, West=450 (West is highest)
        # TOTAL SALES: 1140
        # SALES VALUES SORTED: [80, 90, 100, 120, 140, 150, 150, 150, 160] -> Median is 140
        return csv_data

    def create_test_files(self) -> tuple:
        """Create temporary test files (question and CSV)."""
        question_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        question_file.write(self.test_question)
        question_file.close()

        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        csv_data = self._create_sample_csv_data()
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)
        csv_file.close()

        return question_file.name, csv_file.name

    def send_request(self, question_file_path: str, csv_file_path: str) -> tuple:
        """Send request to the API with both files."""
        try:
            with open(question_file_path, 'rb') as qf, open(csv_file_path, 'rb') as cf:
                files = {
                    'questions.txt': qf,
                    'sample-sales.csv': cf  # The agent code should look for this filename
                }

                print("🚀 Sending sales dataset request to API...")
                print(f"📍 URL: {self.api_url}")

                start_time = time.time()
                response = requests.post(self.api_url, files=files, timeout=180)
                end_time = time.time()

                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                print(f"📊 Status code: {response.status_code}")

                return response, end_time - start_time

        except requests.exceptions.Timeout:
            print("❌ Request timed out after 3 minutes")
            return None, 180
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - is the server running?")
            return None, 0
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None, 0

    def validate_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the API response structure and content."""
        print("\n🔍 Validating response...")

        required_keys = self.expected_answers.keys() | {"bar_chart", "cumulative_sales_chart"}
        validation_results = {key + '_valid': False for key in required_keys}
        validation_results['structure_valid'] = False
        validation_results['details'] = {}

        try:
            if not isinstance(response_data, dict):
                validation_results['details']['structure'] = "Response is not a JSON object"
                return validation_results

            missing_keys = [key for key in required_keys if key not in response_data]
            if missing_keys:
                validation_results['details']['structure'] = f"Missing keys: {missing_keys}"
                return validation_results

            validation_results['structure_valid'] = True
            print("✅ Response structure is valid (JSON object with all required keys)")

            for key, expected_value in self.expected_answers.items():
                actual_value = response_data.get(key)
                is_valid = False # Default to False
                if isinstance(expected_value, str):
                    is_valid = isinstance(actual_value, str) and actual_value.strip().lower() == expected_value.lower()
                elif isinstance(expected_value, (int, float)):
                    if isinstance(actual_value, (int, float)):
                        # THE FIX IS HERE: Wrap the numpy comparison in bool()
                        is_valid = bool(np.isclose(actual_value, expected_value, atol=0.001))

                # Also convert the result to a standard bool for JSON serialization
                validation_results[f'{key}_valid'] = bool(is_valid)
                print(f"▪️  {key}: {'✅' if is_valid else '❌'} Got '{actual_value}', Expected '{expected_value}'")
                validation_results['details'][key] = str(actual_value)

            for chart_key in ["bar_chart", "cumulative_sales_chart"]:
                is_valid = self._validate_image(response_data.get(chart_key), chart_key)
                validation_results[f'{chart_key}_valid'] = bool(is_valid) # Also convert here
                validation_results['details'][chart_key] = self._get_image_details(response_data.get(chart_key))

        except Exception as e:
            print(f"❌ An unexpected error occurred during validation: {e}")
            validation_results['details']['validation_error'] = str(e)
            
        return validation_results
    def _save_chart(self, chart_data: str, filename: str):
        """Helper to save a chart from a base64 data URI to a file."""
        try:
            # Ensure the data is a valid string and contains the base64 header
            if not isinstance(chart_data, str) or ";base64," not in chart_data:
                print(f"⚠️  Skipping save for {filename}: Invalid data format.")
                return

            # Split the header from the actual base64 data and decode it
            base64_str = chart_data.split(";base64,", 1)[1]
            image_bytes = base64.b64decode(base64_str)

            # Write the decoded bytes to a new file in binary write mode
            with open(filename, 'wb') as f:
                f.write(image_bytes)
            print(f"🖼️  Chart saved successfully to: {filename}")

        except Exception as e:
            # Catch any errors during decoding or file writing
            print(f"⚠️  Could not save chart {filename}: {e}")

    def _validate_image(self, image_uri: str, name: str) -> bool:
        if not isinstance(image_uri, str) or not image_uri.startswith("data:image/"):
            print(f"❌ {name}: Invalid format (not a data URI string).")
            return False
        
        try:
            header, b64_data = image_uri.split(';base64,', 1)
            image_bytes = base64.b64decode(b64_data)
            
            if len(image_bytes) > 100_000:
                print(f"❌ {name}: Image too large ({len(image_bytes)/1024:.1f} KB > 100 KB).")
                return False

            Image.open(io.BytesIO(image_bytes))
            print(f"✅ {name}: Valid image and size is within limits.")
            return True
        except Exception as e:
            print(f"❌ {name}: Failed to decode or validate image: {e}")
            return False

    def _get_image_details(self, image_uri: str) -> str:
        if not isinstance(image_uri, str) or not image_uri.startswith("data:image/"):
            return "Invalid format"
        try:
            b64_data = image_uri.split(';base64,', 1)[1]
            return f"Data URI with {len(b64_data)} chars"
        except:
            return "Malformed data URI"

    def save_results(self, response: Optional[requests.Response], validation_results: dict, response_time: float):
        """Save detailed test results and the generated charts to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"sales_test_results_{timestamp}.json"
        
        # Prepare the main results dictionary
        results = {
            'timestamp': timestamp, 'response_time': response_time,
            'status_code': response.status_code if response else None,
            'response_data': None, 'response_text': response.text if response else "No response",
            'validation': validation_results, 'expected': self.expected_answers
        }
        
        # Safely try to parse the JSON response
        response_data = None
        if response and response.status_code == 200:
            try:
                response_data = response.json()
                results['response_data'] = response_data
            except json.JSONDecodeError:
                results['response_data'] = "Error: Response was not valid JSON"

        # Save the main JSON results file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")

        # --- NEW: SAVE THE IMAGES IF THEY ARE VALID ---
        # Check if the response was successful and contained valid charts
        if response_data: # Ensure we have parsed JSON data
            if validation_results.get('bar_chart_valid', False):
                self._save_chart(
                    response_data.get('bar_chart'), 
                    f'bar_chart_{timestamp}.png'
                )
            if validation_results.get('cumulative_sales_chart_valid', False):
                self._save_chart(
                    response_data.get('cumulative_sales_chart'), 
                    f'cumulative_chart_{timestamp}.png'
                )

    def print_summary(self, validation_results: dict, response_time: float):
        """Print a final summary and score based on the validation."""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY (Sales Dataset)")
        print("="*60)
        
        if not validation_results.get('structure_valid', False):
            print("❌ TEST FAILED: Response structure is invalid. Cannot calculate score.")
            return

        score_map = {
            'total_sales_valid': 2, 'top_region_valid': 2, 'day_sales_correlation_valid': 3,
            'median_sales_valid': 2, 'total_sales_tax_valid': 2,
            'bar_chart_valid': 4, 'cumulative_sales_chart_valid': 5
        }
        total_score = sum(score_map[key] for key, valid in validation_results.items() if valid and key in score_map)
        max_score = sum(score_map.values())

        print(f"🎯 TOTAL SCORE: {total_score}/{max_score} points ({total_score/max_score*100:.1f}%)")
        print(f"⏱️  Response time: {response_time:.2f} seconds")

        if response_time >= 180:
            print("⚠️  WARNING: Response time met or exceeded the 3-minute limit.")

        if total_score == max_score:
            print("\n🎉 PERFECT SCORE! All tests passed successfully!")
        else:
            print("\n👍 SOME TESTS PASSED. Review the logs above for details on failed items.")

    def run_test(self):
        """Run the complete test suite."""
        print("🧪 Data Analyst Agent Test Suite: Sales Data")
        print("="*60)
        
        question_file, csv_file = self.create_test_files()
        
        try:
            response, response_time = self.send_request(question_file, csv_file)
            
            if response is None:
                print("❌ Test aborted - no response received from server.")
                return

            print(f"\n📥 Raw response (truncated): {response.text[:500]}{'...' if len(response.text) > 500 else ''}")
            
            validation_results = {}
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    validation_results = self.validate_response(response_data)
                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON!")
                    validation_results = {'structure_valid': False, 'details': {'structure': 'Invalid JSON'}}
            else:
                print(f"❌ API returned a non-200 status code: {response.status_code}")
                validation_results = {'structure_valid': False, 'details': {'structure': f'Bad status code {response.status_code}'}}

            self.save_results(response, validation_results, response_time)
            self.print_summary(validation_results, response_time)
        finally:
            print("\n🧹 Cleaning up temporary files...")
            os.unlink(question_file)
            os.unlink(csv_file)
            print("✨ Test finished.")

def main():
    """Main function to run the tester with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the Data Analyst Agent with a sales dataset.')
    parser.add_argument('--url', default='http://localhost:8000/api/',
                        help='API endpoint URL (default: http://localhost:8000/api/)')
    
    args = parser.parse_args()
    
    tester = SalesDatasetTester(api_url=args.url)
    tester.run_test()

if __name__ == "__main__":
    main()