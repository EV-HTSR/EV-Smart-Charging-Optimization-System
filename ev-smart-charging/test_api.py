import requests
import json

def test_api_endpoints():
    base_url = "http://localhost:8000"
    endpoints = [
        "/",
        "/health", 
        "/stations",
        "/sessions?limit=5",
        "/analytics/summary",
        "/analytics/hourly-usage?days=7"
    ]
    
    print("🧪 Testing API Endpoints...")
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            print(f"✅ {endpoint}: {response.status_code}")
            if response.status_code != 200:
                print(f"   ❌ Error: {response.text}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

if __name__ == "__main__":
    test_api_endpoints()
    