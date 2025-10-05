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
    
    document.getElementById('orbital_period').value = (Math.random() * 129995.7784).toFixed(3);
    document.getElementById('transit_duration').value = (Math.random() * 138.54).toFixed(2);
    document.getElementById('transit_depth').value = Math.round(Math.random() * 1541400);
    document.getElementById('planet_radius').value = (Math.random() * 200346).toFixed(2);
    document.getElementById('semi_major_axis').value = (Math.random() * 45).toFixed(4);
    document.getElementById('insolation_flux').value = Math.round(Math.random() * 10947554.55);
    document.getElementById('equilibrium_temp').value = Math.round(Math.random() * 14667);
    document.getElementById('stellar_teff').value = Math.round(Math.random() * 50000);
    document.getElementById('stellar_radius').value = (Math.random() * 230).toFixed(2);
    document.getElementById('stellar_logg').value = (Math.random() * 6).toFixed(2);
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
        const file = e.target.files[0];
        const baseLabel = document.getElementById('file-name');
        const selName = document.getElementById('selected-file-name');
        const label = document.getElementById('file-label');
        if (file) {
            baseLabel.textContent = 'File selected';
            selName.textContent = file.name;
            selName.style.display = 'block';
            label.classList.add('has-file');
        } else {
            baseLabel.textContent = 'Click or drag a CSV file here';
            selName.textContent = '';
            selName.style.display = 'none';
            label.classList.remove('has-file');
        }
    });
}

// Utility: build URL (supports being served behind a path on Render)
function apiUrl(path) {
    return path; // adjust here if a prefix is needed later
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
        
        const response = await fetch(apiUrl('/api/predict_batch'), {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let msg = 'Batch prediction failed';
            try {
                const errData = await response.json();
                if (errData.error) msg = errData.error;
            } catch {}
            throw new Error(msg);
        }
        
        const result = await response.json();
        batchResults = result;
        
        displayBatchResults(result);
        
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
        
    } catch (error) {
        const status = document.getElementById('batch-status');
        if (status) {
            status.style.color = 'var(--danger-color)';
            status.textContent = error.message;
        } else {
            alert('Error processing file: ' + error.message);
        }
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
        <div class="batch-stat-card">
            <p>Total Candidates</p>
            <h3>${result.total}</h3>
        </div>
        <div class="batch-stat-card">
            <p>Predicted Exoplanets</p>
            <h3 style="color: var(--success-color)">${exoplanets}</h3>
        </div>
        <div class="batch-stat-card">
            <p>False Positives</p>
            <h3 style="color: var(--danger-color)">${falsePositives}</h3>
        </div>
        <div class="batch-stat-card">
            <p>Avg Confidence</p>
            <h3>${avgConfidence}%</h3>
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

// Drag & drop support
const dropZone = document.getElementById('file-drop-zone');
const fileLabel = document.getElementById('file-label');
if (dropZone && fileLabel) {
    ;['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, (e)=>{
        e.preventDefault(); e.stopPropagation(); fileLabel.classList.add('dragover');
    }));
    ;['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, (e)=>{
        e.preventDefault(); e.stopPropagation(); fileLabel.classList.remove('dragover');
    }));
    dropZone.addEventListener('drop', (e)=>{
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const f = e.dataTransfer.files[0];
            if (!f.name.endsWith('.csv')) { alert('Please drop a .csv file'); return; }
            document.getElementById('csv-file').files = e.dataTransfer.files;
            document.getElementById('file-name').textContent = 'File selected';
            const selName = document.getElementById('selected-file-name');
            selName.textContent = f.name;
            selName.style.display = 'block';
            document.getElementById('file-label').classList.add('has-file');
        }
    });
}

// Sample CSV download
const sampleBtn = document.getElementById('download-sample');
if (sampleBtn) {
    sampleBtn.addEventListener('click', () => {
        const header = 'orbital_period,transit_duration,transit_depth,planet_radius,semi_major_axis,insolation_flux,equilibrium_temp,stellar_teff,stellar_radius,stellar_logg\n';
        const sampleRows = [
            '3.52,2.5,1200,1.2,0.045,350,1450,5800,1.1,4.4',
            '12.7,5.1,800,2.4,0.09,290,900,5600,0.95,4.3'
        ];
        const blob = new Blob([header + sampleRows.join('\n')], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'sample_exoplanet_batch.csv';
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
}

// Reset batch results
const resetBtn = document.getElementById('reset-batch');
if (resetBtn) {
    resetBtn.addEventListener('click', () => {
        document.getElementById('batch-results').style.display = 'none';
        document.getElementById('batch-table-body').innerHTML = '';
        batchResults = null;
        document.getElementById('batch-status').textContent = '';
        document.getElementById('csv-file').value = '';
        document.getElementById('file-name').textContent = 'Click or drag a CSV file here';
        const sel = document.getElementById('selected-file-name');
        sel.textContent=''; sel.style.display='none';
        document.getElementById('file-label').classList.remove('has-file');
        window.scrollTo({ top: document.getElementById('batch-tab').offsetTop, behavior: 'smooth'});
    });
}
