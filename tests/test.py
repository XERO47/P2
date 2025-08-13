#!/usr/bin/env python3
"""
Test script for Data Analyst Agent
Tests the Wikipedia highest grossing films analysis task
"""

import requests
import json
import time
import base64
import io
import tempfile
import os
from PIL import Image
import re

class DataAnalystAgentTester:
    def __init__(self, api_url="http://localhost:5000/api/"):
        self.api_url = api_url
        self.test_question = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, "data:image/png;base64,iVBORw0KG..." under 100,000 bytes."""

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
                
                print("🚀 Sending request to API...")
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

    def validate_response(self, response_data):
        """Validate the API response structure and content"""
        print("\n🔍 Validating response...")
        
        validation_results = {
            'structure_valid': False,
            'answer_1_valid': False,
            'answer_2_valid': False,
            'answer_3_valid': False,
            'answer_4_valid': False,
            'details': {}
        }
        
        try:
            # Check if response is a JSON array with 4 elements
            if not isinstance(response_data, list):
                validation_results['details']['structure'] = "Response is not a JSON array"
                return validation_results
                
            if len(response_data) != 4:
                validation_results['details']['structure'] = f"Array has {len(response_data)} elements, expected 4"
                return validation_results
                
            validation_results['structure_valid'] = True
            print("✅ Response structure is valid (4-element array)")
            
            # Validate Answer 1: Number of $2bn movies before 2000
            answer_1 = response_data[0]
            if isinstance(answer_1, (int, str)):
                try:
                    num_value = int(answer_1)
                    if 0 <= num_value <= 10:  # Reasonable range
                        validation_results['answer_1_valid'] = True
                        print(f"✅ Answer 1 valid: {answer_1} movies")
                    else:
                        print(f"⚠️  Answer 1 suspicious: {answer_1} (seems too high/low)")
                except ValueError:
                    print(f"❌ Answer 1 invalid: '{answer_1}' is not a number")
            else:
                print(f"❌ Answer 1 invalid type: {type(answer_1)}")
            
            validation_results['details']['answer_1'] = str(answer_1)
            
            # Validate Answer 2: Earliest film over $1.5bn
            answer_2 = response_data[1]
            if isinstance(answer_2, str) and len(answer_2) > 0:
                # Check if it contains a movie title
                if any(keyword in answer_2.lower() for keyword in ['titanic', 'avatar', 'star wars', 'avengers']):
                    validation_results['answer_2_valid'] = True
                    print(f"✅ Answer 2 valid: '{answer_2}'")
                else:
                    print(f"⚠️  Answer 2 unclear: '{answer_2}'")
            else:
                print(f"❌ Answer 2 invalid: '{answer_2}'")
                
            validation_results['details']['answer_2'] = str(answer_2)
            
            # Validate Answer 3: Correlation coefficient
            answer_3 = response_data[2]
            try:
                corr_value = float(answer_3)
                if -1 <= corr_value <= 1:
                    validation_results['answer_3_valid'] = True
                    print(f"✅ Answer 3 valid: {corr_value} (correlation coefficient)")
                else:
                    print(f"❌ Answer 3 invalid: {corr_value} (not in range [-1, 1])")
            except (ValueError, TypeError):
                print(f"❌ Answer 3 invalid: '{answer_3}' is not a number")
                
            validation_results['details']['answer_3'] = str(answer_3)
            
            # Validate Answer 4: Base64 encoded image
            answer_4 = response_data[3]
            if isinstance(answer_4, str) and answer_4.startswith("data:image/png;base64,"):
                base64_data = answer_4.split(",")[1]
                try:
                    # Decode base64 and check if it's a valid image
                    image_data = base64.b64decode(base64_data)
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Check file size (should be under 100KB)
                    size_kb = len(image_data) / 1024
                    if size_kb <= 100:
                        validation_results['answer_4_valid'] = True
                        print(f"✅ Answer 4 valid: {image.size} image, {size_kb:.1f} KB")
                    else:
                        print(f"❌ Answer 4 too large: {size_kb:.1f} KB (limit: 100 KB)")
                        
                    validation_results['details']['answer_4'] = f"Valid image: {image.size}, {size_kb:.1f} KB"
                    
                except Exception as e:
                    print(f"❌ Answer 4 invalid image: {e}")
                    validation_results['details']['answer_4'] = f"Invalid image data: {e}"
            else:
                print(f"❌ Answer 4 invalid format: doesn't start with 'data:image/png;base64,'")
                validation_results['details']['answer_4'] = "Invalid data URI format"
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            validation_results['details']['validation_error'] = str(e)
            
        return validation_results

    def save_results(self, response, validation_results, response_time):
        """Save test results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'response_time': response_time,
            'status_code': response.status_code if response else None,
            'response_data': response.json() if response and response.status_code == 200 else None,
            'response_text': response.text if response else None,
            'validation_results': validation_results,
            'test_question': self.test_question
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to: {results_file}")
        
        # Also save the image if it exists
        if (response and response.status_code == 200 and 
            validation_results.get('answer_4_valid', False)):
            try:
                response_data = response.json()
                if len(response_data) >= 4:
                    image_data_uri = response_data[3]
                    base64_data = image_data_uri.split(",")[1]
                    image_data = base64.b64decode(base64_data)
                    
                    image_file = f"test_plot_{timestamp}.png"
                    with open(image_file, 'wb') as f:
                        f.write(image_data)
                    print(f"🖼️  Plot saved to: {image_file}")
            except Exception as e:
                print(f"❌ Failed to save plot: {e}")

    def print_summary(self, validation_results, response_time):
        """Print test summary"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY")
        print("="*60)
        
        total_score = 0
        max_score = 20  # Based on the evaluation rubric
        
        # Structure check (no points but required)
        if validation_results['structure_valid']:
            print("✅ Structure: Valid 4-element JSON array")
        else:
            print("❌ Structure: Invalid - test failed")
            return
            
        # Answer scoring (4 points each for answers 1-3, 8 points for answer 4)
        if validation_results['answer_1_valid']:
            print("✅ Answer 1: Valid (4/4 points)")
            total_score += 4
        else:
            print("❌ Answer 1: Invalid (0/4 points)")
            
        if validation_results['answer_2_valid']:
            print("✅ Answer 2: Valid (4/4 points)")
            total_score += 4
        else:
            print("❌ Answer 2: Invalid (0/4 points)")
            
        if validation_results['answer_3_valid']:
            print("✅ Answer 3: Valid (4/4 points)")
            total_score += 4
        else:
            print("❌ Answer 3: Invalid (0/4 points)")
            
        if validation_results['answer_4_valid']:
            print("✅ Answer 4: Valid visualization (8/8 points)")
            total_score += 8
        else:
            print("❌ Answer 4: Invalid visualization (0/8 points)")
            
        print(f"\n🎯 TOTAL SCORE: {total_score}/{max_score} points ({total_score/max_score*100:.1f}%)")
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        
        if response_time > 180:
            print("⚠️  WARNING: Response time exceeded 3-minute limit")
            
        if total_score == max_score:
            print("🎉 PERFECT SCORE! All tests passed!")
        elif total_score >= 16:
            print("👍 GOOD PERFORMANCE! Most tests passed.")
        elif total_score >= 8:
            print("⚠️  NEEDS IMPROVEMENT. Some tests failed.")
        else:
            print("❌ POOR PERFORMANCE. Major issues detected.")

    def run_test(self):
        """Run the complete test suite"""
        print("🧪 Data Analyst Agent Test Suite")
        print("="*60)
        print(f"Testing endpoint: {self.api_url}")
        print("\n📝 Test Question:")
        print(self.test_question)
        print("\n" + "-"*60)
        
        # Create test files
        question_file = self.create_test_files()
        
        try:
            # Send request
            response, response_time = self.send_request(question_file)
            
            if response is None:
                print("❌ Test failed - no response received")
                return
                
            print(f"\n📥 Raw response: {response.text[:500]}{'...' if len(response.text) > 500 else ''}")
            
            # Validate response
            validation_results = {'structure_valid': False}
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    validation_results = self.validate_response(response_data)
                except json.JSONDecodeError:
                    print("❌ Response is not valid JSON")
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
    
    parser = argparse.ArgumentParser(description='Test the Data Analyst Agent')
    parser.add_argument('--url', default='https://p2-y6wv.onrender.com/api/', 
                       help='API endpoint URL (default: https://p2-y6wv.onrender.com/api/)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = DataAnalystAgentTester(api_url=args.url)
    tester.run_test()

if __name__ == "__main__":
    main()