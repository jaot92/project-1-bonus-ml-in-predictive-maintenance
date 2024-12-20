import requests
import json
import time

BASE_URL = "https://predictive-maintenance-vgzh.onrender.com"

def test_health():
    """Test health check endpoint"""
    response = requests.get(f'{BASE_URL}/health')
    print("\n=== Health Check Test ===")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

def test_prediction(data, test_name="Default"):
    """Test prediction endpoint with given data"""
    print(f"\n=== Prediction Test: {test_name} ===")
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(f'{BASE_URL}/predict', 
                               json=data,
                               headers=headers,
                               timeout=30)  # Aumentamos el timeout por cold start
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except requests.exceptions.Timeout:
        print("La solicitud excedió el tiempo de espera (posible cold start)")
    except Exception as e:
        print(f"Error: {str(e)}")

def run_tests():
    """Run all tests"""
    print("Iniciando pruebas en producción...")
    print(f"URL base: {BASE_URL}")
    
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
    test_prediction(normal_data, "Condiciones Normales")
    
    # Test 3: High Risk Conditions
    high_risk_data = {
        'Air temperature [K]': 315.1,
        'Process temperature [K]': 325.6,
        'Rotational speed [rpm]': 2000,
        'Torque [Nm]': 60,
        'Tool wear [min]': 250
    }
    test_prediction(high_risk_data, "Condiciones de Alto Riesgo")

if __name__ == '__main__':
    run_tests() 