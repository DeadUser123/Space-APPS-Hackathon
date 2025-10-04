// Metrics page JavaScript (static visualization mode)

document.addEventListener('DOMContentLoaded', async () => {
    await loadModelInfo();
    // Content is already visible; images are static in HTML.
});

async function loadModelInfo() {
    try {
        const resp = await fetch('/api/model_info');
        const info = await resp.json();
        displayModelInfo(info);
    } catch (e) {
        console.error('Failed to load model info', e);
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

// No visualization JS needed; images are static.
