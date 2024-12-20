import requests
import json
import time

def test_health():
    """Test health check endpoint"""
    response = requests.get('http://localhost:5005/health')
    print("\n=== Health Check Test ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_prediction(data, test_name="Default"):
    """Test prediction endpoint with given data"""
    print(f"\n=== Prediction Test: {test_name} ===")
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:5005/predict', 
                           json=data,
                           headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()

def run_tests():
    """Run all tests"""
    # Wait for server to start
    time.sleep(2)
    
    # Test 1: Health Check
    test_health()
    
    # Test 2: Normal Operating Conditions
    normal_data = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40,
        'Tool wear [min]': 0
    }
    test_prediction(normal_data, "Normal Operating Conditions")
    
    # Test 3: High Temperature Conditions
    high_temp_data = {
        'Air temperature [K]': 310.1,
        'Process temperature [K]': 320.6,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40,
        'Tool wear [min]': 200
    }
    test_prediction(high_temp_data, "High Temperature")
    
    # Test 4: High Wear Conditions
    high_wear_data = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40,
        'Tool wear [min]': 250
    }
    test_prediction(high_wear_data, "High Tool Wear")
    
    # Test 5: Extreme Conditions
    extreme_data = {
        'Air temperature [K]': 315.1,
        'Process temperature [K]': 325.6,
        'Rotational speed [rpm]': 2000,
        'Torque [Nm]': 60,
        'Tool wear [min]': 250
    }
    test_prediction(extreme_data, "Extreme Conditions")
    
    # Test 6: Invalid Data (Missing Field)
    invalid_data = {
        'Air temperature [K]': 298.1,
        'Process temperature [K]': 308.6,
        'Rotational speed [rpm]': 1500,
        'Tool wear [min]': 0
        # Missing 'Torque [Nm]'
    }
    test_prediction(invalid_data, "Invalid Data (Missing Field)")

if __name__ == '__main__':
    run_tests() 