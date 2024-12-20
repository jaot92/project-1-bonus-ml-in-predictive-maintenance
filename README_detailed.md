# Predictive Maintenance with Machine Learning

## Project Overview
This project implements a machine learning solution for predictive maintenance using the AI4I 2020 Predictive Maintenance Dataset. The system predicts potential machine failures based on sensor data and operational parameters.

## Features
- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Machine Learning model training with Random Forest
- Hyperparameter tuning using Grid Search
- Web interface for real-time predictions
- REST API endpoints for integration

## Project Structure
```
project/
│
├── data/                    # Dataset directory
│   └── ai4i2020.csv        # Main dataset
│
├── models/                  # Saved models
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/              # Jupyter notebooks
│
├── src/                    # Source code
│   ├── app.py             # Flask application
│   ├── model_training.py  # Model training script
│   ├── exploratory_analysis.py  # EDA script
│   └── hyperparameter_tuning.py # Hyperparameter optimization
│
├── static/                 # Static files for web interface
│   ├── css/
│   └── js/
│
└── templates/              # HTML templates
    └── index.html         # Main web interface
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [project-directory]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python src/model_training.py
```

2. (Optional) Perform hyperparameter tuning:
```bash
python src/hyperparameter_tuning.py
```

3. Run the web application:
```bash
python src/app.py
```

4. Access the web interface at `http://localhost:5000`

## Model Performance
- Cross-validation F1 Score: 0.973
- Test Set Performance:
  - Precision (No Failure): 1.00
  - Recall (No Failure): 0.97
  - Precision (Failure): 0.52
  - Recall (Failure): 0.90

# API Documentation
   
Base URL: https://predictive-maintenance-vgzh.onrender.com
   
## Endpoints
   
### Health Check
GET /health
   
### Prediction
POST /predict
Content-Type: application/json

## API Endpoints

### Prediction Endpoint
- URL: `/predict`
- Method: `POST`
- Input Format:
```json
{
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1551,
    "Torque [Nm]": 42.8,
    "Tool wear [min]": 0
}
```
- Response Format:
```json
{
    "status": "success",
    "prediction": 0,
    "probability": {
        "no_failure": 0.95,
        "failure": 0.05
    }
}
```

### Health Check
- URL: `/health`
- Method: `GET`
- Response: `{"status": "healthy"}`

## Future Improvements
1. Implementation of additional machine learning algorithms
2. Real-time monitoring capabilities
3. Integration with external monitoring systems
4. Enhanced visualization of predictions and model explanations
5. Deployment to cloud platforms

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 

## Live Demo
The application is deployed and available at: [https://predictive-maintenance-vgzh.onrender.com/](https://predictive-maintenance-vgzh.onrender.com/)

### Example Usage with curl
```bash
# Health check
curl https://predictive-maintenance-vgzh.onrender.com/health

# Make a prediction
curl -X POST \
  https://predictive-maintenance-vgzh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Air temperature [K]": 298.1,
    "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 40,
    "Tool wear [min]": 0
  }'
```

### Valid Parameter Ranges
| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Air temperature | 295 | 304 | K |
| Process temperature | 305 | 313 | K |
| Rotational speed | 1300 | 2000 | rpm |
| Torque | 30 | 60 | Nm |
| Tool wear | 0 | 250 | min |