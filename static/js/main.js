// Initialize parameter chart
let parameterChart = null;

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const resultSection = document.getElementById('resultSection');
    const loadingSection = document.getElementById('loadingSection');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading state
        resultSection.style.display = 'none';
        loadingSection.style.display = 'block';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = parseFloat(value);
        });
        
        try {
            // Make prediction request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                displayResults(result, data);
            } else {
                showError(result.message);
            }
        } catch (error) {
            showError('An error occurred while making the prediction.');
        } finally {
            loadingSection.style.display = 'none';
            resultSection.style.display = 'block';
        }
    });
});

function displayResults(result, inputData) {
    // Update prediction alert
    const predictionAlert = document.getElementById('predictionAlert');
    const failureProb = result.probability.failure * 100;
    
    if (result.prediction === 1) {
        predictionAlert.className = 'alert alert-danger';
        predictionAlert.innerHTML = `<strong>Warning:</strong> High risk of machine failure detected! (${failureProb.toFixed(1)}% probability)`;
    } else if (failureProb > 20) {
        predictionAlert.className = 'alert alert-warning';
        predictionAlert.innerHTML = `<strong>Caution:</strong> Moderate risk of failure (${failureProb.toFixed(1)}% probability)`;
    } else {
        predictionAlert.className = 'alert alert-success';
        predictionAlert.innerHTML = `<strong>Good:</strong> Low risk of failure (${failureProb.toFixed(1)}% probability)`;
    }
    
    // Update probability bar
    const probBar = document.getElementById('failureProbability');
    probBar.style.width = `${failureProb}%`;
    probBar.textContent = `${failureProb.toFixed(1)}%`;
    
    // Calculate and display derived values
    const power = (inputData['Rotational speed [rpm]'] * inputData['Torque [Nm]']) / 9550;
    const tempDiff = inputData['Process temperature [K]'] - inputData['Air temperature [K]'];
    
    document.getElementById('powerValue').textContent = `${power.toFixed(2)} kW`;
    document.getElementById('tempDiffValue').textContent = `${tempDiff.toFixed(2)} K`;
    
    // Update parameter health chart
    updateParameterChart(inputData);
}

function updateParameterChart(data) {
    const normalRanges = {
        'Air temperature [K]': { min: 295, max: 304, current: data['Air temperature [K]'] },
        'Process temperature [K]': { min: 305, max: 313, current: data['Process temperature [K]'] },
        'Rotational speed [rpm]': { min: 1300, max: 2000, current: data['Rotational speed [rpm]'] },
        'Torque [Nm]': { min: 30, max: 60, current: data['Torque [Nm]'] },
        'Tool wear [min]': { min: 0, max: 200, current: data['Tool wear [min]'] }
    };
    
    // Calculate health scores (0-100)
    const healthScores = {};
    for (const [param, range] of Object.entries(normalRanges)) {
        const value = range.current;
        const min = range.min;
        const max = range.max;
        
        if (value < min) {
            healthScores[param] = (value / min) * 100;
        } else if (value > max) {
            healthScores[param] = Math.max(0, 100 - ((value - max) / max) * 100);
        } else {
            healthScores[param] = 100 - Math.abs(((value - (min + (max-min)/2)) / ((max-min)/2)) * 50);
        }
    }
    
    // Update or create chart
    const ctx = document.getElementById('parameterChart').getContext('2d');
    
    if (parameterChart) {
        parameterChart.destroy();
    }
    
    parameterChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(healthScores),
            datasets: [{
                label: 'Parameter Health',
                data: Object.values(healthScores),
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2,
                pointBackgroundColor: 'rgba(54, 162, 235, 1)'
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function showError(message) {
    const predictionAlert = document.getElementById('predictionAlert');
    predictionAlert.className = 'alert alert-danger';
    predictionAlert.textContent = message;
    document.getElementById('resultSection').style.display = 'block';
} 