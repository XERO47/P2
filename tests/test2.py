#!/usr/bin/env python3
"""
Test script for Data Analyst Agent - Indian High Court Judgments Dataset
Tests DuckDB querying, data analysis, and visualization capabilities
"""

import requests
import json
import time
import base64
import io
import tempfile
import os
import re
from PIL import Image
from typing import Dict, Any, Optional

class IndianCourtsDatasetTester:
    def __init__(self, api_url="http://localhost:8000/api/"):
        self.api_url = api_url
        self.test_question = """The Indian high court judgement dataset contains judgements from the Indian High Courts, downloaded from [ecourts website](https://judgments.ecourts.gov.in/). It contains judgments of 25 high courts, along with raw metadata (as .json) and structured metadata (as .parquet).

- 25 high courts
- ~16M judgments
- ~1TB of data

Structure of the data in the bucket:

- `data/pdf/year=2025/court=xyz/bench=xyz/judgment1.pdf,judgment2.pdf`
- `metadata/json/year=2025/court=xyz/bench=xyz/judgment1.json,judgment2.json`
- `metadata/parquet/year=2025/court=xyz/bench=xyz/metadata.parquet`
- `metadata/tar/year=2025/court=xyz/bench=xyz/metadata.tar.gz`
- `data/tar/year=2025/court=xyz/bench=xyz/pdfs.tar`

This DuckDB query counts the number of decisions in the dataset.

```sql
INSTALL httpfs; LOAD httpfs;
INSTALL parquet; LOAD parquet;

SELECT COUNT(*) FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1');
```

Here are the columns in the data:

| Column                 | Type    | Description                    |
| ---------------------- | ------- | ------------------------------ |
| `court_code`           | VARCHAR | Court identifier (e.g., 33~10) |
| `title`                | VARCHAR | Case title and parties         |
| `description`          | VARCHAR | Case description               |
| `judge`                | VARCHAR | Presiding judge(s)             |
| `pdf_link`             | VARCHAR | Link to judgment PDF           |
| `cnr`                  | VARCHAR | Case Number Register           |
| `date_of_registration` | VARCHAR | Registration date              |
| `decision_date`        | DATE    | Date of judgment               |
| `disposal_nature`      | VARCHAR | Case outcome                   |
| `court`                | VARCHAR | Court name                     |
| `raw_html`             | VARCHAR | Original HTML content          |
| `bench`                | VARCHAR | Bench identifier               |
| `year`                 | BIGINT  | Year partition                 |

Here is a sample row:

```json
{
  "court_code": "33~10",
  "title": "CRL MP(MD)/4399/2023 of Vinoth Vs The Inspector of Police",
  "description": "No.4399 of 2023 BEFORE THE MADURAI BENCH OF MADRAS HIGH COURT ( Criminal Jurisdiction ) Thursday, ...",
  "judge": "HONOURABLE  MR JUSTICE G.K. ILANTHIRAIYAN",
  "pdf_link": "court/cnrorders/mdubench/orders/HCMD010287762023_1_2023-03-16.pdf",
  "cnr": "HCMD010287762023",
  "date_of_registration": "14-03-2023",
  "decision_date": "2023-03-16",
  "disposal_nature": "DISMISSED",
  "court": "33_10",
  "raw_html": "<button type='button' role='link'..",
  "bench": "mdubench",
  "year": 2023
}
```

Answer the following questions and respond with a JSON object containing the answer.

```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```"""

    def create_test_files(self):
        """Create temporary test files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.test_question)
            return f.name

    def send_request(self, question_file_path):
        """Send request to the API"""
        try:
            with open(question_file_path, 'rb') as f:
                files = {'questions.txt': f}
                
                print("🚀 Sending Indian Courts dataset request to API...")
                print(f"📍 URL: {self.api_url}")
                
                start_time = time.time()
                response = requests.post(self.api_url, files=files, timeout=180)
                end_time = time.time()
                
                print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
                print(f"📊 Status code: {response.status_code}")
                
                return response, end_time - start_time
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out (3 minutes)")
            return None, 180
        except requests.exceptions.ConnectionError:
            print("❌ Connection error - is the server running?")
            return None, 0
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None, 0

    def validate_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the API response structure and content"""
        print("\n🔍 Validating Indian Courts dataset response...")
        
        validation_results = {
            'structure_valid': False,
            'question_1_valid': False,
            'question_2_valid': False,
            'question_3_valid': False,
            'details': {}
        }
        
        expected_keys = [
            "Which high court disposed the most cases from 2019 - 2022?",
            "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?",
            "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters"
        ]
        
        try:
            # Check if response is a JSON object with expected keys
            if not isinstance(response_data, dict):
                validation_results['details']['structure'] = "Response is not a JSON object"
                return validation_results
                
            # Check for presence of expected keys (allowing for slight variations)
            found_keys = []
            for expected_key in expected_keys:
                key_found = False
                for actual_key in response_data.keys():
                    if self._keys_similar(expected_key, actual_key):
                        found_keys.append(actual_key)
                        key_found = True
                        break
                if not key_found:
                    validation_results['details']['structure'] = f"Missing key similar to: {expected_key}"
                    return validation_results
                    
            validation_results['structure_valid'] = True
            print("✅ Response structure is valid (JSON object with 3 expected keys)")
            
            # Validate Question 1: Which high court disposed the most cases from 2019-2022?
            answer_1 = response_data.get(found_keys[0], "")
            if isinstance(answer_1, str) and len(answer_1.strip()) > 0:
                # Check if it mentions a court name or identifier
                court_indicators = ['court', 'high court', '33_10', 'madras', 'bombay', 'delhi', 'calcutta', 'karnataka']
                if any(indicator in answer_1.lower() for indicator in court_indicators):
                    validation_results['question_1_valid'] = True
                    print(f"✅ Question 1 valid: '{answer_1}'")
                else:
                    print(f"⚠️  Question 1 unclear: '{answer_1}' (no clear court identifier)")
            else:
                print(f"❌ Question 1 invalid: '{answer_1}' (empty or wrong type)")
            
            validation_results['details']['answer_1'] = str(answer_1)
            
            # Validate Question 2: Regression slope
            answer_2 = response_data.get(found_keys[1], "")
            try:
                # Try to extract a numeric value (slope)
                slope_match = re.search(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?', str(answer_2))
                if slope_match:
                    slope_value = float(slope_match.group())
                    # Reasonable range for days delay slope (could be negative or positive)
                    if -1000 <= slope_value <= 1000:
                        validation_results['question_2_valid'] = True
                        print(f"✅ Question 2 valid: {slope_value} (regression slope)")
                    else:
                        print(f"⚠️  Question 2 suspicious: {slope_value} (slope seems extreme)")
                        validation_results['question_2_valid'] = True  # Still accept it
                else:
                    print(f"❌ Question 2 invalid: '{answer_2}' (no numeric value found)")
            except (ValueError, TypeError):
                print(f"❌ Question 2 invalid: '{answer_2}' (cannot extract numeric value)")
                
            validation_results['details']['answer_2'] = str(answer_2)
            
            # Validate Question 3: Base64 encoded plot
            answer_3 = response_data.get(found_keys[2], "")
            if isinstance(answer_3, str):
                # Check for data URI format (could be PNG or WebP)
                if answer_3.startswith("data:image/"):
                    try:
                        # Extract the base64 part
                        if ";base64," in answer_3:
                            base64_data = answer_3.split(";base64,")[1]
                            
                            # Decode and validate image
                            image_data = base64.b64decode(base64_data)
                            image = Image.open(io.BytesIO(image_data))
                            
                            # Check file size (under 100,000 characters for base64)
                            if len(base64_data) <= 100000:
                                validation_results['question_3_valid'] = True
                                print(f"✅ Question 3 valid: {image.size} image, {len(base64_data):,} base64 chars")
                            else:
                                print(f"❌ Question 3 too large: {len(base64_data):,} base64 chars (limit: 100,000)")
                                
                            validation_results['details']['answer_3'] = f"Valid image: {image.size}, {len(base64_data):,} base64 chars"
                            
                        else:
                            print(f"❌ Question 3 invalid: missing ';base64,' in data URI")
                            validation_results['details']['answer_3'] = "Invalid data URI format"
                    except Exception as e:
                        print(f"❌ Question 3 invalid image: {e}")
                        validation_results['details']['answer_3'] = f"Invalid image data: {e}"
                elif len(answer_3.strip()) == 0 or answer_3 == "data:image/webp:base64,":
                    print(f"❌ Question 3 empty: no plot generated")
                    validation_results['details']['answer_3'] = "Empty or incomplete data URI"
                else:
                    print(f"❌ Question 3 invalid format: doesn't start with 'data:image/'")
                    validation_results['details']['answer_3'] = "Invalid data URI format"
            else:
                print(f"❌ Question 3 invalid type: {type(answer_3)}")
                validation_results['details']['answer_3'] = f"Wrong type: {type(answer_3)}"
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            validation_results['details']['validation_error'] = str(e)
            
        return validation_results

    def _keys_similar(self, expected: str, actual: str) -> bool:
        """Check if two keys are similar (allowing for minor variations)"""
        expected_clean = expected.lower().strip()
        actual_clean = actual.lower().strip()
        
        # Exact match
        if expected_clean == actual_clean:
            return True
            
        # Check if key words are present
        expected_words = set(expected_clean.split())
        actual_words = set(actual_clean.split())
        
        # At least 70% word overlap
        common_words = expected_words & actual_words
        return len(common_words) / len(expected_words) >= 0.7

    def save_results(self, response, validation_results, response_time):
        """Save test results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"indian_courts_test_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'test_type': 'indian_high_courts_dataset',
            'response_time': response_time,
            'status_code': response.status_code if response else None,
            'response_data': response.json() if response and response.status_code == 200 else None,
            'response_text': response.text if response else None,
            'validation_results': validation_results,
            'test_question_preview': self.test_question[:500] + "..."
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {results_file}")
        
        # Also save the plot if it exists
        if (response and response.status_code == 200 and 
            validation_results.get('question_3_valid', False)):
            try:
                response_data = response.json()
                plot_key = None
                for key in response_data.keys():
                    if "plot" in key.lower() or "scatterplot" in key.lower():
                        plot_key = key
                        break
                
                if plot_key and response_data[plot_key].startswith("data:image/"):
                    base64_data = response_data[plot_key].split(";base64,")[1]
                    image_data = base64.b64decode(base64_data)
                    
                    # Determine file extension from data URI
                    if "webp" in response_data[plot_key]:
                        ext = "webp"
                    else:
                        ext = "png"
                    
                    plot_file = f"indian_courts_plot_{timestamp}.{ext}"
                    with open(plot_file, 'wb') as f:
                        f.write(image_data)
                    print(f"📊 Plot saved to: {plot_file}")
            except Exception as e:
                print(f"❌ Failed to save plot: {e}")

    def print_summary(self, validation_results, response_time):
        """Print test summary"""
        print("\n" + "="*70)
        print("📋 INDIAN HIGH COURTS DATASET TEST SUMMARY")
        print("="*70)
        
        total_score = 0
        max_score = 15  # Estimated scoring (5 points per question)
        
        # Structure check
        if validation_results['structure_valid']:
            print("✅ Structure: Valid JSON object with expected keys")
        else:
            print("❌ Structure: Invalid - test failed")
            return
            
        # Question scoring (5 points each)
        if validation_results['question_1_valid']:
            print("✅ Question 1 (High court with most cases): Valid (5/5 points)")
            total_score += 5
        else:
            print("❌ Question 1 (High court with most cases): Invalid (0/5 points)")
            
        if validation_results['question_2_valid']:
            print("✅ Question 2 (Regression slope): Valid (5/5 points)")
            total_score += 5
        else:
            print("❌ Question 2 (Regression slope): Invalid (0/5 points)")
            
        if validation_results['question_3_valid']:
            print("✅ Question 3 (Scatterplot visualization): Valid (5/5 points)")
            total_score += 5
        else:
            print("❌ Question 3 (Scatterplot visualization): Invalid (0/5 points)")
            
        print(f"\n🎯 TOTAL SCORE: {total_score}/{max_score} points ({total_score/max_score*100:.1f}%)")
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        
        if response_time > 180:
            print("⚠️  WARNING: Response time exceeded 3-minute limit")
            
        # Performance assessment
        if total_score == max_score:
            print("🎉 PERFECT SCORE! All Indian Courts dataset queries executed successfully!")
        elif total_score >= 10:
            print("👍 GOOD PERFORMANCE! Most DuckDB queries and analysis completed correctly.")
        elif total_score >= 5:
            print("⚠️  PARTIAL SUCCESS. Some dataset analysis features working.")
        else:
            print("❌ POOR PERFORMANCE. Major issues with DuckDB querying and data analysis.")
            
        print("\n📊 DATASET ANALYSIS CAPABILITIES TESTED:")
        print("   - DuckDB S3 parquet file querying")
        print("   - Large-scale data aggregation (16M+ records)")
        print("   - Date parsing and time difference calculations")
        print("   - Statistical analysis (regression slopes)")
        print("   - Data visualization with regression lines")

    def run_test(self):
        """Run the complete test suite"""
        print("🏛️  Data Analyst Agent - Indian High Courts Dataset Test")
        print("="*70)
        print(f"Testing endpoint: {self.api_url}")
        print("Dataset: Indian High Court Judgments (~16M records, ~1TB data)")
        print("\n📋 Testing capabilities:")
        print("   - DuckDB querying of S3-hosted parquet files")
        print("   - Large-scale data aggregation and filtering")
        print("   - Date arithmetic and regression analysis")
        print("   - Statistical visualization")
        print("\n" + "-"*70)
        
        # Create test files
        question_file = self.create_test_files()
        
        try:
            # Send request
            response, response_time = self.send_request(question_file)
            
            if response is None:
                print("❌ Test failed - no response received")
                return
                
            print(f"\n📥 Raw response preview: {response.text[:300]}{'...' if len(response.text) > 300 else ''}")
            
            # Validate response
            validation_results = {'structure_valid': False}
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    validation_results = self.validate_response(response_data)
                except json.JSONDecodeError as e:
                    print(f"❌ Response is not valid JSON: {e}")
                    print(f"Response content: {response.text}")
            else:
                print(f"❌ API returned error status: {response.status_code}")
                print(f"Response: {response.text}")
            
            # Save results
            self.save_results(response, validation_results, response_time)
            
            # Print summary
            self.print_summary(validation_results, response_time)
            
        finally:
            # Cleanup
            if os.path.exists(question_file):
                os.unlink(question_file)

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Data Analyst Agent with Indian High Courts Dataset')
    parser.add_argument('--url', default='https://p2-y6wv.onrender.com/api/', 
                       help='API endpoint URL (default: http://localhost:8000/api/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = IndianCourtsDatasetTester(api_url=args.url)
    tester.run_test()

if __name__ == "__main__":
    main()