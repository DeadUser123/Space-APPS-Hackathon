// Metrics page JavaScript

// Load metrics on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadMetrics();
});

async function loadMetrics() {
    try {
        // Load model info
        const infoResponse = await fetch('/api/model_info');
        const modelInfo = await infoResponse.json();
        
        // Display model info
        displayModelInfo(modelInfo);
        
        // Load visualizations
        const vizResponse = await fetch('/api/visualizations');
        const visualizations = await vizResponse.json();
        
        // Display visualizations
        displayVisualizations(visualizations);
        
        // Hide loading, show content
        document.getElementById('loading').style.display = 'none';
        document.getElementById('metrics-content').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading metrics:', error);
        document.getElementById('loading').innerHTML = `
            <p style="color: var(--danger);">
                Error loading metrics. Make sure the model is trained and the server is running.
            </p>
        `;
    }
}

function displayModelInfo(info) {
    const modelInfoDiv = document.getElementById('model-info');
    
    if (!info.model_loaded) {
        modelInfoDiv.innerHTML = '<p style="color: var(--danger-color);">Model not loaded. Please train the model first.</p>';
        return;
    }
    
    modelInfoDiv.innerHTML = `
        <div class="info-item">
            <strong>Training Epoch</strong>
            <span>${info.epoch || 'N/A'}</span>
        </div>
        <div class="info-item">
            <strong>Accuracy</strong>
            <span>${(info.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="info-item">
            <strong>Precision</strong>
            <span>${(info.precision * 100).toFixed(2)}%</span>
        </div>
        <div class="info-item">
            <strong>Recall</strong>
            <span>${(info.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="info-item">
            <strong>F1-Score</strong>
            <span>${(info.f1_score * 100).toFixed(2)}%</span>
        </div>
        <div class="info-item">
            <strong>ROC-AUC</strong>
            <span>${(info.roc_auc * 100).toFixed(2)}%</span>
        </div>
        <div class="info-item">
            <strong>Input Features</strong>
            <span>${info.num_features}</span>
        </div>
        <div class="info-item">
            <strong>Best Accuracy</strong>
            <span>${(info.best_acc * 100).toFixed(2)}%</span>
        </div>
    `;
}

function displayVisualizations(viz) {
    // Display metrics plot
    if (viz.metrics) {
        document.getElementById('metrics-plot').src = 'data:image/png;base64,' + viz.metrics;
    }
    
    // Display confusion matrix
    if (viz.confusion_matrix) {
        document.getElementById('confusion-matrix').src = 'data:image/png;base64,' + viz.confusion_matrix;
    }
    
    // Display ROC curve
    if (viz.roc_curve) {
        document.getElementById('roc-curve').src = 'data:image/png;base64,' + viz.roc_curve;
    }

    if (viz.feature_importance) {
        document.getElementById('feature-importance').src = 'data:image/png;base64,' + viz.feature_importance;
    }

    if (viz.dataset_distribution) {
        document.getElementById('dataset-distribution').src = 'data:image/png;base64,' + viz.dataset_distribution;
    }

    if (viz.confidence_distribution) {
        document.getElementById('confidence-distribution').src = 'data:image/png;base64,' + viz.confidence_distribution;
    }
}

// Refresh metrics
async function refreshMetrics() {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('metrics-content').style.display = 'none';
    await loadMetrics();
}
