#!/usr/bin/env python3
"""
Test script for WAN 2.2 Koyeb deployment
Verifies API functionality without requiring full model loading
"""

import requests
import json
import time
import sys

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""
    
    print("🧪 Testing WAN 2.2 Koyeb API...")
    print(f"Base URL: {base_url}")
    print("-" * 50)
    
    try:
        # Test root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Service: {data.get('service')}")
        print(f"   Koyeb Mode: {data.get('environment', {}).get('koyeb_deployment')}")
        print(f"   CPU Mode: {data.get('environment', {}).get('cpu_mode')}")
        print()
        
        # Test health endpoint
        print("2. Testing health endpoint...")
        response = requests.get(f"{base_url}/healthz")
        print(f"   Status: {response.status_code}")
        health = response.json()
        print(f"   OK: {health.get('ok')}")
        print(f"   CPU %: {health.get('resources', {}).get('cpu_percent')}")
        print(f"   Memory: {health.get('resources', {}).get('memory_available_gb')}GB available")
        print(f"   Pipeline Loaded: {health.get('pipeline_loaded')}")
        print()
        
        # Test generation endpoint (will likely fail without models)
        print("3. Testing generate endpoint...")
        gen_data = {
            "prompt": "A cat playing with a ball",
            "width": 512,
            "height": 288,
            "num_frames": 17,
            "steps": 10
        }
        response = requests.post(f"{base_url}/generate", json=gen_data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"   Job ID: {job_id}")
            
            # Check job status
            print("4. Checking job status...")
            for i in range(5):
                time.sleep(2)
                status_response = requests.get(f"{base_url}/jobs/{job_id}")
                status_data = status_response.json()
                status = status_data.get('status')
                print(f"   Attempt {i+1}: {status}")
                
                if status == "done":
                    print("   ✅ Job completed successfully!")
                    break
                elif status == "error":
                    error = status_data.get('error', 'Unknown error')
                    print(f"   ❌ Job failed: {error[:100]}...")
                    break
            else:
                print("   ⏱️ Job still running after 10 seconds")
        else:
            print(f"   ❌ Generate failed: {response.text}")
        
        print("\n✅ API test completed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {base_url}")
        print("Make sure the server is running with:")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(base_url)