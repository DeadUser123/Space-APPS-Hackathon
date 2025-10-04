// Prediction page JavaScript
let batchResults = null;

// Tab switching
function showTab(tabName) {
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    const buttons = document.querySelectorAll('.tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
}

// Load example data
function loadExample() {
    // Add slight randomization to create more realistic examples
    // Base values from a confirmed exoplanet (hot Jupiter-like)
    const randomVariation = () => 0.9 + (Math.random() * 0.2); // 90% to 110% of base value
    
    document.getElementById('orbital_period').value = (3.52 * randomVariation()).toFixed(3);
    document.getElementById('transit_duration').value = (2.8 * randomVariation()).toFixed(2);
    document.getElementById('transit_depth').value = Math.round(1200 * randomVariation());
    document.getElementById('planet_radius').value = (1.2 * randomVariation()).toFixed(2);
    document.getElementById('semi_major_axis').value = (0.045 * randomVariation()).toFixed(4);
    document.getElementById('insolation_flux').value = Math.round(350 * randomVariation());
    document.getElementById('equilibrium_temp').value = Math.round(1450 * randomVariation());
    document.getElementById('stellar_teff').value = Math.round(5800 * randomVariation());
    document.getElementById('stellar_radius').value = (1.1 * randomVariation()).toFixed(2);
    document.getElementById('stellar_logg').value = (4.4 * randomVariation()).toFixed(2);
}

// Single prediction form handler
document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value) || 0;
    });
    
    try {
        // Show loading state
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = 'Predicting...';
        submitBtn.disabled = true;
        
        // Make API call
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayResults(result);
        
        // Reset button
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        
    } catch (error) {
        alert('Error making prediction: ' + error.message);
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
});

// Display single prediction results
function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const resultContent = document.getElementById('result-content');
    
    const isExoplanet = result.prediction === 'Exoplanet';
    const confidencePercent = (result.confidence * 100).toFixed(2);
    const exoplanetProb = (result.probability_exoplanet * 100).toFixed(2);
    const falsePositiveProb = (result.probability_false_positive * 100).toFixed(2);
    
    resultContent.innerHTML = `
        <div class="result-prediction ${isExoplanet ? 'exoplanet' : 'false-positive'}">
            ${isExoplanet ? '🪐' : '❌'} ${result.prediction}
        </div>
        <div class="result-details">
            <div class="result-item">
                <h4>Confidence</h4>
                <p>${confidencePercent}%</p>
            </div>
            <div class="result-item">
                <h4>Exoplanet Probability</h4>
                <p style="color: ${isExoplanet ? '#10b981' : '#64748b'}">
                    ${exoplanetProb}%
                </p>
            </div>
            <div class="result-item">
                <h4>False Positive Probability</h4>
                <p style="color: ${!isExoplanet ? '#ef4444' : '#64748b'}">
                    ${falsePositiveProb}%
                </p>
            </div>
        </div>
        <div style="margin-top: 1.5rem; padding: 1rem; background: var(--light-bg); border-radius: 5px;">
            <h4 style="margin-bottom: 0.5rem;">Interpretation:</h4>
            <p>${getInterpretation(result)}</p>
        </div>
    `;
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function getInterpretation(result) {
    if (result.prediction === 'Exoplanet') {
        if (result.confidence > 0.9) {
            return 'High confidence exoplanet detection! This candidate shows strong characteristics of a confirmed exoplanet.';
        } else if (result.confidence > 0.7) {
            return 'Good confidence exoplanet detection. This candidate is likely an exoplanet but may require additional verification.';
        } else {
            return 'Moderate confidence exoplanet detection. Further observation recommended to confirm.';
        }
    } else {
        if (result.confidence > 0.9) {
            return 'High confidence that this is a false positive. The signal is likely caused by stellar variability or instrumental artifacts.';
        } else if (result.confidence > 0.7) {
            return 'Good confidence false positive classification. This signal likely does not represent a real exoplanet.';
        } else {
            return 'Moderate confidence false positive. Additional analysis may be needed.';
        }
    }
}

// File upload handling
const fileInput = document.getElementById('csv-file');
if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'Choose CSV file...';
        document.getElementById('file-name').textContent = fileName;
    });
}

// Batch prediction form handler
document.getElementById('batch-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files[0]) {
        alert('Please select a CSV file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
        const submitBtn = e.target.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        submitBtn.textContent = 'Processing...';
        submitBtn.disabled = true;
        
        const response = await fetch('/api/predict_batch', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Batch prediction failed');
        }
        
        const result = await response.json();
        batchResults = result;
        
        displayBatchResults(result);
        
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        
    } catch (error) {
        alert('Error processing file: ' + error.message);
        submitBtn.textContent = 'Process File';
        submitBtn.disabled = false;
    }
});

// Display batch results
function displayBatchResults(result) {
    const resultsDiv = document.getElementById('batch-results');
    const statsDiv = document.getElementById('batch-stats');
    const tableBody = document.getElementById('batch-table-body');
    
    // Calculate statistics
    const exoplanets = result.predictions.filter(p => p.prediction === 'Exoplanet').length;
    const falsePositives = result.predictions.filter(p => p.prediction === 'False Positive').length;
    const avgConfidence = (result.predictions.reduce((sum, p) => sum + p.confidence, 0) / result.total * 100).toFixed(2);
    
    // Display stats
    statsDiv.innerHTML = `
        <div class="stat-card">
            <h3>${result.total}</h3>
            <p>Total Candidates</p>
        </div>
        <div class="stat-card">
            <h3 style="color: var(--success-color)">${exoplanets}</h3>
            <p>Predicted Exoplanets</p>
        </div>
        <div class="stat-card">
            <h3 style="color: var(--danger-color)">${falsePositives}</h3>
            <p>False Positives</p>
        </div>
        <div class="stat-card">
            <h3>${avgConfidence}%</h3>
            <p>Avg Confidence</p>
        </div>
    `;
    
    // Populate table
    tableBody.innerHTML = '';
    result.predictions.forEach(pred => {
        const row = document.createElement('tr');
        const isExoplanet = pred.prediction === 'Exoplanet';
        row.innerHTML = `
            <td>${pred.row + 1}</td>
            <td style="color: ${isExoplanet ? 'var(--success-color)' : 'var(--danger-color)'}; font-weight: bold;">
                ${pred.prediction}
            </td>
            <td>${(pred.confidence * 100).toFixed(2)}%</td>
            <td>${(pred.probability_exoplanet * 100).toFixed(2)}%</td>
        `;
        tableBody.appendChild(row);
    });
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Download results as CSV
function downloadResults() {
    if (!batchResults) {
        alert('No results to download');
        return;
    }
    
    // Create CSV content
    let csv = 'Row,Prediction,Confidence,Exoplanet_Probability,False_Positive_Probability\n';
    batchResults.predictions.forEach(pred => {
        csv += `${pred.row + 1},${pred.prediction},${pred.confidence},${pred.probability_exoplanet},${pred.probability_false_positive}\n`;
    });
    
    // Create download link
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `exoplanet_predictions_${new Date().getTime()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}
