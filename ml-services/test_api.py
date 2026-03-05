"""
Test SEVAS API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

print("="*70)
print("🧪 TESTING SEVAS API")
print("="*70)

# Test 1: Health Check
print("\n📋 Test 1: Health Check")
print("-"*70)
response = requests.get(f"{BASE_URL}/api/health")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 2: Service Info
print("\n📋 Test 2: Service Info")
print("-"*70)
response = requests.get(f"{BASE_URL}/api/info")
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Test 3: Available Models
print("\n📋 Test 3: Available Models")
print("-"*70)
response = requests.get(f"{BASE_URL}/api/models")
print(f"Status: {response.status_code}")
print(f"Models: {json.dumps(response.json(), indent=2)}")

# Test 4: Analyze Image
print("\n📋 Test 4: Analyze Image")
print("-"*70)

image_path = 'uploads/test_image.jpg'

try:
    with open(image_path, 'rb') as f:
        files = {'file': f}
        data = {'detection_type': 'general'}
        
        response = requests.post(
            f"{BASE_URL}/api/analyze",
            files=files,
            data=data
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis Result:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.text}")
            
except FileNotFoundError:
    print(f"❌ File not found: {image_path}")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n" + "="*70)
print("✅ API TESTING COMPLETE")
print("="*70)