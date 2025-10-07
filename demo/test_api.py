"""
Test script for the STL Classification Web API
Tests all endpoints to ensure they're working correctly
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
API_BASE = f"{BASE_URL}/api"

def test_ping():
    """Test the ping endpoint."""
    print("Testing /api/ping...")
    try:
        response = requests.get(f"{API_BASE}/ping")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_models():
    """Test the getModels endpoint."""
    print("\nTesting /api/getModels...")
    try:
        # Test without filter
        response = requests.get(f"{API_BASE}/getModels")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total models: {data.get('total', 0)}")
        
        if data.get('models'):
            print("Sample model:")
            print(json.dumps(data['models'][0], indent=2))
        
        # Test with positive filter
        response = requests.get(f"{API_BASE}/getModels?type=positive")
        pos_data = response.json()
        print(f"Positive models: {pos_data.get('total', 0)}")
        
        # Test with negative filter
        response = requests.get(f"{API_BASE}/getModels?type=negative")
        neg_data = response.json()
        print(f"Negative models: {neg_data.get('total', 0)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_inference():
    """Test the doInference endpoint."""
    print("\nTesting /api/doInference...")
    try:
        # First get a model to test with
        models_response = requests.get(f"{API_BASE}/getModels")
        models_data = models_response.json()
        
        if not models_data.get('models'):
            print("No models available for testing")
            return False
        
        # Use the first model for testing
        test_model = models_data['models'][0]
        model_url = test_model['url']
        
        print(f"Testing inference with: {test_model['name']}")
        print(f"Model URL: {model_url}")
        print(f"Expected type: {test_model['type']}")
        
        # Perform inference
        inference_data = {"modelUrl": model_url}
        response = requests.post(f"{API_BASE}/doInference", 
                               json=inference_data,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                prediction = result['result']['prediction']
                metadata = result['result']['metadata']
                weights = result['result']['weights']
                
                print(f"Prediction: {prediction}")
                print(f"Confidence: {prediction['confidence']:.3f}")
                print(f"Predicted class: {prediction['labels'][prediction['class']]}")
                print(f"Inference time: {metadata['inference_time']:.3f}s")
                print(f"Weight layers: {len(weights)}")
                
                return True
            else:
                print(f"Inference failed: {result.get('error')}")
                return False
        else:
            print(f"HTTP Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_file_serving():
    """Test file serving endpoints."""
    print("\nTesting file serving...")
    try:
        # Get models to test file serving
        models_response = requests.get(f"{API_BASE}/getModels")
        models_data = models_response.json()
        
        if not models_data.get('models'):
            print("No models available for file serving test")
            return False
        
        # Test serving an STL file
        test_model = models_data['models'][0]
        file_url = f"{BASE_URL}{test_model['url']}"
        
        print(f"Testing file serving: {file_url}")
        
        response = requests.head(file_url)  # Use HEAD to avoid downloading large file
        print(f"File serve status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Content type: {response.headers.get('content-type', 'unknown')}")
            print(f"Content length: {response.headers.get('content-length', 'unknown')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("STL Classification API Test Suite")
    print("=" * 50)
    
    tests = [
        ("Ping", test_ping),
        ("Get Models", test_get_models),
        ("File Serving", test_file_serving),
        ("Inference", test_inference)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")
        time.sleep(1)  # Brief pause between tests
    
    print(f"\n{'='*50}")
    print("Test Results Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    print("Make sure the Flask server is running on localhost:5000")
    print("You can start it with: python demo/server.py")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")