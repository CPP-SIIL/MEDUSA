"""
Simple API test for the STL classification server
"""
import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_ping():
    """Test the ping endpoint"""
    print("Testing /api/ping endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/api/ping')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_get_models():
    """Test the getModels endpoint"""
    print("\nTesting /api/getModels endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/api/getModels')
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Number of models: {len(data['models'])}")
        if len(data['models']) > 0:
            print(f"First model: {data['models'][0]['name']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_inference():
    """Test the doInference endpoint with a sample model"""
    print("\nTesting /api/doInference endpoint...")
    try:
        # First get available models
        models_response = requests.get(f'{BASE_URL}/api/getModels')
        models = models_response.json()['models']
        
        if len(models) == 0:
            print("No models available for testing")
            return False
        
        # Use the first model for testing
        test_model = models[0]['name']
        print(f"Testing with model: {test_model}")
        
        # Send inference request
        inference_data = {
            'modelName': test_model
        }
        
        response = requests.post(
            f'{BASE_URL}/api/doInference',
            json=inference_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Processing time: {result['processing_time']:.4f}s")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("STL Classification API Test")
    print("=" * 40)
    
    # Test all endpoints
    ping_ok = test_ping()
    models_ok = test_get_models()
    inference_ok = test_inference()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Ping: {'✅' if ping_ok else '❌'}")
    print(f"Get Models: {'✅' if models_ok else '❌'}")
    print(f"Inference: {'✅' if inference_ok else '❌'}")
    
    if all([ping_ok, models_ok, inference_ok]):
        print("\n🎉 All tests passed! Server is ready for frontend integration.")
    else:
        print("\n⚠️ Some tests failed. Check server status.")

if __name__ == "__main__":
    main()