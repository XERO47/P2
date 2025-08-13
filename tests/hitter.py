import requests
import time
from datetime import datetime

def check_api(url, timeout=10):
    """
    Continuously check an API endpoint until it returns 200 OK
    
    Args:
        url (str): The API endpoint to check
        timeout (int): Request timeout in seconds (default: 10)
    """
    attempt = 0
    
    print(f"Starting API health check for: {url}")
    print("Press Ctrl+C to stop\n")
    
    while True:
        attempt += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            print(f"[{timestamp}] Attempt #{attempt}: Checking API...", end=" ")
            
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 200:
                print(f"✅ SUCCESS! Got 200 OK")
                print(f"Response time: {response.elapsed.total_seconds():.2f}s")
                print(f"Total attempts: {attempt}")
                break
            else:
                print(f"❌ Got {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {str(e)}")
        
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Stopped by user after {attempt} attempts")
            return
        
        print("Waiting 30 seconds before next attempt...\n")
        time.sleep(30)

if __name__ == "__main__":
    # Replace with your API endpoint
    api_url = "https://p2-y6wv.onrender.com/health/"  # Example URL that returns 503
    
    # Uncomment the line below and replace with your actual API URL
    # api_url = "https://your-api-endpoint.com/health"
    
    check_api(api_url)