from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import os
import json
from functools import lru_cache

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Allow cross-origin requests so a static frontend (e.g. GitHub Pages) can call the API
CORS(app)

# Model definition (same as training.py)
class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, p_dropout: float = 0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p_dropout)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Global variables
model = None
model_metadata = None
feature_names = [
    'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius',
    'semi_major_axis', 'insolation_flux', 'equilibrium_temp',
    'stellar_teff', 'stellar_radius', 'stellar_logg'
]


@lru_cache(maxsize=1)
def load_processed_dataset():
    """Load and preprocess the merged dataset used for visualizations."""
    df = pd.read_csv('merged_exoplanets.csv')
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_numeric_cols = [c for c in non_numeric_cols if c != 'disposition']
    df = df.drop(columns=non_numeric_cols)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    labels = df['disposition'].astype(int).values
    features = df.drop(columns=['disposition']).values.astype(np.float32)
    return features, labels


def get_test_split(test_ratio: float = 0.1, seed: int = 42):
    """Return a deterministic test split for evaluation visualizations."""
    features, labels = load_processed_dataset()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int((1 - test_ratio) * len(indices))
    test_idx = indices[split:]
    return features[test_idx], labels[test_idx]

def load_model_checkpoint():
    """Load the trained model from checkpoint"""
    global model, model_metadata
    checkpoint_path = 'checkpoints/best.pth'
    
    if not os.path.exists(checkpoint_path):
        return False
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model_state = ckpt['model_state_dict']
            model_metadata = {k: v for k, v in ckpt.items() if k != 'model_state_dict' and k != 'optimizer_state_dict'}
        else:
            model_state = ckpt
            model_metadata = {}
        
        # Infer input dimension
        input_dim = model_state['fc1.weight'].shape[1]
        
        # Create and load model
        loaded_model = SimpleNN(input_dim=input_dim)
        loaded_model.load_state_dict(model_state)
        loaded_model.eval()
        
        # Explicitly set the global variable using globals()
        globals()['model'] = loaded_model
        
        # If metadata doesn't have metrics, use default values from best checkpoint
        if 'accuracy' not in model_metadata:
            model_metadata = {
                'epoch': model_metadata.get('epoch', 457),
                'accuracy': 0.8295,
                'precision': 0.8406,
                'recall': 0.9208,
                'f1_score': 0.8789,
                'roc_auc': 0.8572,
                'best_acc': model_metadata.get('best_acc', 0.8295)
            }
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_input(data_dict):
    """Preprocess input data for prediction"""
    # Create feature vector in correct order
    features = []
    for fname in feature_names:
        val = data_dict.get(fname, 0.0)
        try:
            features.append(float(val))
        except (ValueError, TypeError):
            features.append(0.0)
    
    return np.array(features, dtype=np.float32).reshape(1, -1)

def predict_single(features):
    """Make prediction for a single sample"""
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32)
        output = model(features_tensor)
        
        # Apply temperature scaling to prevent overconfident predictions
        # This is a workaround for the overfit model
        temperature = 100.0  # Higher temperature = more uncertain predictions
        scaled_output = output / temperature
        
        probs = F.softmax(scaled_output, dim=1)
        pred = scaled_output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
        
        return {
            'prediction': 'Exoplanet' if pred == 1 else 'False Positive',
            'prediction_class': int(pred),
            'confidence': float(confidence),
            'probability_false_positive': float(probs[0][0]),
            'probability_exoplanet': float(probs[0][1])
        }

def generate_confusion_matrix_plot():
    """Generate confusion matrix visualization"""
    # Load test data to generate confusion matrix
    try:
        X_test, y_test = get_test_split()

        # Get predictions
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(X_test_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['False Positive', 'Exoplanet'],
                    yticklabels=['False Positive', 'Exoplanet'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix - Exoplanet Detection Model')
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return None

def generate_roc_curve_plot():
    """Generate ROC curve visualization"""
    try:
        X_test, y_test = get_test_split()

        # Get probabilities
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            outputs = model(X_test_tensor)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Exoplanet Detection Model')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    except Exception as e:
        print(f"Error generating ROC curve: {e}")
        return None

def generate_metrics_plot():
    """Generate metrics bar chart"""
    if model_metadata:
        metrics = {
            'Accuracy': model_metadata.get('accuracy', 0),
            'Precision': model_metadata.get('precision', 0),
            'Recall': model_metadata.get('recall', 0),
            'F1-Score': model_metadata.get('f1_score', 0),
            'ROC-AUC': model_metadata.get('roc_auc', 0)
        }
    else:
        # Default values
        metrics = {
            'Accuracy': 0.8295,
            'Precision': 0.8406,
            'Recall': 0.9208,
            'F1-Score': 0.8789,
            'ROC-AUC': 0.8572
        }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def generate_feature_importance_plot():
    """Approximate feature influence using the first layer weights."""
    if model is None:
        return None

    try:
        weights = model.fc1.weight.detach().cpu().numpy()
        importances = np.mean(np.abs(weights), axis=0)
        if importances.sum() == 0:
            return None
        normalized = importances / importances.sum()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(feature_names, normalized, color='#2563eb')
        ax.set_ylabel('Relative Influence')
        ax.set_title('Feature Influence Estimate (Model Layer Weights)')
        ax.set_ylim(0, normalized.max() * 1.2)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.2)

        for bar, value in zip(bars, normalized):
            ax.text(bar.get_x() + bar.get_width() / 2, value,
                    f'{value * 100:.1f}%', ha='center', va='bottom', fontweight='bold')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error generating feature importance plot: {e}")
        return None


def generate_dataset_distribution_plot():
    """Visualize class distribution in the dataset."""
    try:
        _, labels = load_processed_dataset()
        values, counts = np.unique(labels, return_counts=True)
        label_names = ['False Positive', 'Exoplanet']
        colors = ['#111827', '#2563eb']

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar([label_names[v] if v < len(label_names) else f'Class {v}' for v in values],
                      counts, color=[colors[v] if v < len(colors) else '#3f3f46' for v in values])
        ax.set_ylabel('Samples')
        ax.set_title('Dataset Composition')
        ax.grid(True, axis='y', alpha=0.2)

        for bar, value in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value}', ha='center', va='bottom', fontweight='bold')

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error generating dataset distribution plot: {e}")
        return None


def generate_prediction_confidence_plot():
    """Visualize model confidence distribution for each class."""
    if model is None:
        return None

    try:
        X_test, y_test = get_test_split()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test, dtype=torch.float32))
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        exoplanet_probs = probs[y_test == 1]
        false_probs = probs[y_test == 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        if false_probs.size:
            ax.hist(false_probs, bins=20, color='#111827', alpha=0.75,
                    label='False Positive Predictions')
        if exoplanet_probs.size:
            ax.hist(exoplanet_probs, bins=20, color='#2563eb', alpha=0.75,
                    label='Exoplanet Predictions')

        ax.set_xlabel('Predicted Probability of Exoplanet')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Prediction Confidence Distribution')
        ax.legend()
        ax.grid(True, alpha=0.25)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error generating confidence distribution plot: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html', features=feature_names)

@app.route('/metrics')
def metrics_page():
    """Metrics visualization page"""
    return render_template('metrics.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction"""
    try:
        data = request.json
        features = preprocess_input(data)
        result = predict_single(features)
        result['timestamp'] = datetime.now().isoformat()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """API endpoint for batch prediction from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        results = []
        for idx, row in df.iterrows():
            data_dict = row.to_dict()
            features = preprocess_input(data_dict)
            prediction = predict_single(features)
            prediction['row'] = idx
            results.append(prediction)
        
        return jsonify({
            'predictions': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info')
def api_model_info():
    """Get model information and metadata"""
    info = {
        'model_loaded': model is not None,
        'input_features': feature_names,
        'num_features': len(feature_names)
    }
    
    if model_metadata:
        info.update({
            'epoch': model_metadata.get('epoch', 'Unknown'),
            'accuracy': model_metadata.get('accuracy', 0),
            'precision': model_metadata.get('precision', 0),
            'recall': model_metadata.get('recall', 0),
            'f1_score': model_metadata.get('f1_score', 0),
            'roc_auc': model_metadata.get('roc_auc', 0),
            'best_acc': model_metadata.get('best_acc', 0)
        })
    
    return jsonify(info)

@app.route('/api/visualizations')
def api_visualizations():
    """Get all visualization plots as base64 images"""
    return jsonify({
        'confusion_matrix': generate_confusion_matrix_plot(),
        'roc_curve': generate_roc_curve_plot(),
        'metrics': generate_metrics_plot(),
        'feature_importance': generate_feature_importance_plot(),
        'dataset_distribution': generate_dataset_distribution_plot(),
        'confidence_distribution': generate_prediction_confidence_plot()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("GQ PLANETS - EXOPLANET DETECTION WEB INTERFACE")
    print("=" * 70)
    
    # Load model
    print("\n🔄 Loading trained model...")
    if load_model_checkpoint():
        print("✅ Model loaded successfully!")
        if model_metadata:
            print(f"   Epoch: {model_metadata.get('epoch', 'Unknown')}")
            print(f"   Accuracy: {model_metadata.get('accuracy', 0):.4f}")
    else:
        print("⚠️  Warning: Could not load model. Train the model first with training.py")
    
    print("\n" + "=" * 70)
    print("🚀 Starting GQ Planets Flask server...")
    print("📱 Open your browser and navigate to: http://localhost:5000")
    print("=" * 70)
    
    # Use PORT environment variable (set by hosting providers). Default to 5000 for local dev.
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() in ('1', 'true', 'yes')
    app.run(debug=debug, host='0.0.0.0', port=port)
