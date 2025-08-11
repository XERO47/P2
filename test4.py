#!/usr/bin/env python3
"""
Test script for Data Analyst Agent - Weather Dataset Analysis
Tests CSV analysis, statistical calculations, and data visualization.
This script is designed to replicate the validation logic from the provided promptfoo.yaml.
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

class WeatherDatasetTester:
    def __init__(self, api_url="http://localhost:8000/api/"):
        self.api_url = api_url
        self.test_question = """Analyze `sample-weather.csv`.

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
7. Plot precipitation as a histogram with orange bars. Encode as base64 PNG."""

        # These expected answers are calculated from the sample data
        # and match the values in the promptfoo.yaml assertions.
        self.expected_answers = {
            "average_temp_c": 5.1,
            "max_precip_date": "2024-01-06",
            "min_temp_c": 2,
            "temp_precip_correlation": -0.0413519224, # Note: a slight negative correlation
            "average_precip_mm": 0.9
        }

    def _create_sample_csv_data(self) -> str:
        """Returns the weather data as a CSV formatted string."""
        return """date,temperature_c,precip_mm
2024-01-01,5,0
2024-01-02,7,1
2024-01-03,6,0
2024-01-04,8,2
2024-01-05,6,0
2024-01-06,4,5
2024-01-07,3,0
2024-01-08,2,0
2024-01-09,4,1
2024-01-10,6,0
"""

    def create_test_files(self) -> tuple:
        """Create temporary test files (question and CSV)."""
        question_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, newline='', encoding='utf-8')
        question_file.write(self.test_question)
        question_file.close()

        csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8')
        csv_file.write(self._create_sample_csv_data())
        csv_file.close()

        return question_file.name, csv_file.name

    def send_request(self, question_file_path: str, csv_file_path: str) -> tuple:
        """Send request to the API with both files."""
        try:
            with open(question_file_path, 'rb') as qf, open(csv_file_path, 'rb') as cf:
                files = {
                    'questions.txt': qf,
                    'sample-weather.csv': cf # Agent should look for this filename
                }

                print("🚀 Sending weather dataset request to API...")
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
        """Validate the API response structure and content based on promptfoo.yaml."""
        print("\n🔍 Validating response...")

        required_keys = self.expected_answers.keys() | {"temp_line_chart", "precip_histogram"}
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
                is_valid = False
                if key == "max_precip_date":
                    is_valid = isinstance(actual_value, str) and actual_value.strip() == expected_value
                elif isinstance(expected_value, (int, float)):
                    is_valid = isinstance(actual_value, (int, float)) and np.isclose(actual_value, expected_value, atol=0.001)

                validation_results[f'{key}_valid'] = bool(is_valid) # Convert numpy.bool_ to bool
                print(f"▪️  {key}: {'✅' if is_valid else '❌'} Got '{actual_value}', Expected '{expected_value}'")
                validation_results['details'][key] = str(actual_value)
            
            # This script performs structural validation on the images.
            # The full visual validation (e.g., line color) is done by the llm-rubric in promptfoo.
            validation_results['temp_line_chart_valid'] = self._validate_image(response_data.get("temp_line_chart"), "Temperature Line Chart")
            validation_results['details']['temp_line_chart'] = self._get_image_details(response_data.get("temp_line_chart"))
            
            validation_results['precip_histogram_valid'] = self._validate_image(response_data.get("precip_histogram"), "Precipitation Histogram")
            validation_results['details']['precip_histogram'] = self._get_image_details(response_data.get("precip_histogram"))

        except Exception as e:
            print(f"❌ An unexpected error occurred during validation: {e}")
            validation_results['details']['validation_error'] = str(e)
            
        return validation_results

    def _validate_image(self, image_uri: str, name: str) -> bool:
        """Validates that a string is a valid, reasonably sized, base64 encoded image URI."""
        if not isinstance(image_uri, str) or not image_uri.startswith("data:image/"):
            print(f"❌ {name}: Invalid format (not a data URI string).")
            return False
        try:
            b64_data = image_uri.split(';base64,', 1)[1]
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
        """Returns a string with details about the image for reporting."""
        if not isinstance(image_uri, str) or not image_uri.startswith("data:image/"): return "Invalid format"
        try: return f"Data URI with {len(image_uri.split(';base64,', 1)[1])} chars"
        except: return "Malformed data URI"

    def _save_chart(self, chart_data: str, filename: str):
        """Saves a chart from a base64 data URI to a file."""
        try:
            if chart_data and ";base64," in chart_data:
                image_bytes = base64.b64decode(chart_data.split(";base64,", 1)[1])
                with open(filename, 'wb') as f: f.write(image_bytes)
                print(f"🖼️  Chart saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Could not save chart {filename}: {e}")

    def save_results(self, response: Optional[requests.Response], validation_results: dict, response_time: float):
        """Saves detailed test results and charts to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"weather_test_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp, 'response_time': response_time,
            'status_code': response.status_code if response else None,
            'response_data': None, 'response_text': response.text if response else "No response",
            'validation': validation_results, 'expected': self.expected_answers
        }
        response_data = None
        if response and response.status_code == 200:
            try:
                response_data = response.json()
                results['response_data'] = response_data
            except json.JSONDecodeError: results['response_data'] = "Error: Not valid JSON"
        
        with open(results_file, 'w') as f: json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")

        if response_data:
            self._save_chart(response_data.get('temp_line_chart'), f'temp_line_chart_{timestamp}.png')
            self._save_chart(response_data.get('precip_histogram'), f'precip_histogram_{timestamp}.png')

    def print_summary(self, validation_results: dict, response_time: float):
        """Prints a final summary and score based on the promptfoo weights."""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY (Weather Dataset)")
        print("="*60)
        
        if not validation_results.get('structure_valid', False):
            print("❌ TEST FAILED: Response structure is invalid. Cannot calculate score.")
            return

        score_map = {
            'average_temp_c_valid': 3, 'max_precip_date_valid': 3, 'min_temp_c_valid': 3,
            'temp_precip_correlation_valid': 3, 'average_precip_mm_valid': 3,
            'temp_line_chart_valid': 8, 'precip_histogram_valid': 8
        }
        total_score = sum(score_map[key] for key, valid in validation_results.items() if valid and key in score_map)
        max_score = sum(score_map.values())

        print(f"🎯 TOTAL SCORE: {total_score}/{max_score} points ({total_score/max_score*100:.1f}%)")
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        
        if total_score == max_score: print("\n🎉 PERFECT SCORE! All tests passed.")
        else: print("\n⚠️  SOME TESTS FAILED. Review logs for details.")

    def run_test(self):
        """Runs the complete test suite."""
        print("="*60)
        print("🧪 Starting Data Analyst Agent Test: Weather Dataset Analysis")
        print("="*60)
        
        question_file, csv_file = self.create_test_files()
        try:
            response, response_time = self.send_request(question_file, csv_file)
            if response is None:
                print("❌ Test aborted - no response from server.")
                return
            
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
    """Main function to run the tester."""
    import argparse
    parser = argparse.ArgumentParser(description='Test the Data Analyst Agent with a weather dataset.')
    parser.add_argument('--url', default='http://localhost:8000/api/',
                        help='API endpoint URL (e.g., http://localhost:8000/api/)')
    args = parser.parse_args()
    
    tester = WeatherDatasetTester(api_url=args.url)
    tester.run_test()

if __name__ == "__main__":
    main()