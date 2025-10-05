import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import errno

DATASET_PATH = 'merged_exoplanets.csv'


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, p_dropout: float = 0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p_dropout)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def load_dataset():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    df = pd.read_csv(DATASET_PATH)

    if 'disposition' not in df.columns:
        raise ValueError("CSV must contain a 'disposition' column with labels 0/1")

    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_numeric_cols = [c for c in non_numeric_cols if c != 'disposition']
    df = df.drop(columns=non_numeric_cols)

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    labels = df['disposition'].astype(int).values
    features = df.drop(columns=['disposition']).values.astype(np.float32)

    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_test, y_test = features[test_idx], labels[test_idx]

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    test_ds = TensorDataset(X_test_t, y_test_t)
    
    return {
        "test": DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0),
        "test_size": len(test_ds),
    }


def load_model_from_checkpoint(checkpoint_path: str, input_dim: int = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
        metadata = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    else:
        model_state = ckpt
        metadata = {}

    if input_dim is None:
        if 'fc1.weight' in model_state:
            input_dim = model_state['fc1.weight'].shape[1]
        else:
            raise ValueError('Could not infer input_dim')

    model = SimpleNN(input_dim=input_dim)
    model.load_state_dict(model_state)
    model.eval()
    return model, metadata


def evaluate_model(model, test_loader, test_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sigmoid for probabilities
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(pred.cpu().numpy().flatten().tolist())
            all_targets.extend(target.cpu().numpy().flatten().tolist())

    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    acc = correct / test_size if test_size > 0 else 0.0

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.0
    
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'correct': correct,
        'total': test_size
    }


def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_test_arrays():
    """Return X_test, y_test using the same preprocessing/split as load_dataset()."""
    seed = 42
    np.random.seed(seed)
    df = pd.read_csv(DATASET_PATH)
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    non_numeric_cols = [c for c in non_numeric_cols if c != 'disposition']
    df = df.drop(columns=non_numeric_cols)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())
    labels = df['disposition'].astype(int).values
    features = df.drop(columns=['disposition']).values.astype(np.float32)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    test_idx = idx[split:]
    X_test, y_test = features[test_idx], labels[test_idx]
    return X_test, y_test


def plot_roc_curve(y_true, probs, out_path):
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', color='#2563eb')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Exoplanet Detection Model')
        ax.legend(loc='lower right')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote ROC curve to {out_path}")
    except Exception as e:
        print(f"Failed to write ROC curve: {e}")


def plot_confusion_matrix(cm, out_path):
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['False Positive', 'Exoplanet'],
                    yticklabels=['False Positive', 'Exoplanet'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix - Exoplanet Detection Model')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote confusion matrix to {out_path}")
    except Exception as e:
        print(f"Failed to write confusion matrix: {e}")


def plot_metrics_bar(metrics, out_path):
    try:
        names = list(metrics.keys())
        values = [metrics[k] for k in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Metrics')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.3f}', ha='center')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote metrics bar chart to {out_path}")
    except Exception as e:
        print(f"Failed to write metrics bar chart: {e}")


def plot_feature_influence(model, feature_names, out_path):
    try:
        weights = model.fc1.weight.detach().cpu().numpy()
        importances = np.mean(np.abs(weights), axis=0)
        if importances.sum() == 0:
            print("Feature importances all zero; skipping")
            return
        normalized = importances / importances.sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(feature_names, normalized, color='#2563eb')
        ax.set_title('Feature Influence (mean abs weight)')
        plt.xticks(rotation=45, ha='right')
        for bar, v in zip(bars, normalized):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.001, f'{v:.2f}', ha='center')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote feature influence to {out_path}")
    except Exception as e:
        print(f"Failed to write feature influence: {e}")


def plot_dataset_composition(out_path):
    try:
        df = pd.read_csv(DATASET_PATH)
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        non_numeric_cols = [c for c in non_numeric_cols if c != 'disposition']
        df = df.drop(columns=non_numeric_cols)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(df.mean())
        labels = df['disposition'].astype(int).values
        values, counts = np.unique(labels, return_counts=True)
        # Map to readable names
        label_names = ['False Positive', 'Exoplanet']
        names = [label_names[v] if v < len(label_names) else f'Class {v}' for v in values]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, counts, color=['#111827', '#2563eb'])
        ax.set_ylabel('Samples')
        ax.set_title('Dataset Composition')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{cnt}', ha='center', va='bottom')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Wrote dataset composition to {out_path}")
    except Exception as e:
        print(f"Failed to write dataset composition: {e}")


if __name__ == '__main__':
    print("=" * 70)
    print("EXOPLANET DETECTION MODEL - PERFORMANCE EVALUATION")
    print("=" * 70)
    
    checkpoint_path = os.path.join('checkpoints', 'best.pth')
    
    try:
        # Load model
        print("\n📦 Loading best model checkpoint...")
        model, meta = load_model_from_checkpoint(checkpoint_path, input_dim=None)
        print(f"✅ Model loaded successfully!")
        print(f"   Input dimension: {model.fc1.in_features}")
        
        if meta:
            print(f"\n📊 Checkpoint metadata:")
            for key, value in meta.items():
                if key != 'optimizer_state_dict':
                    print(f"   {key}: {value}")
        
        # Load test data
        print("\n📊 Loading test dataset...")
        data = load_dataset()
        test_loader = data['test']
        test_size = data['test_size']
        print(f"✅ Test set size: {test_size} samples")
        
        # Evaluate
        print("\n🔍 Evaluating model performance...")
        results = evaluate_model(model, test_loader, test_size)
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 70)
        print(f"Accuracy:   {results['accuracy']:.4f} ({results['correct']}/{results['total']} = {100 * results['accuracy']:.2f}%)")
        print(f"Precision:  {results['precision']:.4f}")
        print(f"Recall:     {results['recall']:.4f}")
        print(f"F1-Score:   {results['f1_score']:.4f}")
        print(f"ROC-AUC:    {results['roc_auc']:.4f}")
        
        cm = results['confusion_matrix']
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print(f"                    Predicted")
        print(f"                 False     True")
        print(f"Actual False  {cm[0][0]:7d}  {cm[0][1]:7d}")
        print(f"       True   {cm[1][0]:7d}  {cm[1][1]:7d}")
        
        # Calculate additional stats
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print("\n" + "=" * 70)
        print("DETAILED STATISTICS")
        print("=" * 70)
        print(f"True Positives:   {tp:7d} (Correctly identified exoplanets)")
        print(f"True Negatives:   {tn:7d} (Correctly rejected false positives)")
        print(f"False Positives:  {fp:7d} (Incorrectly identified as exoplanets)")
        print(f"False Negatives:  {fn:7d} (Missed exoplanets)")
        print(f"Specificity:      {specificity:.4f} (True negative rate)")
        print("=" * 70)

        # Persist evaluation metrics into checkpoint metadata and metadata.json fallback (so website metrics automatically update)
        try:
            eval_meta = {
                'eval_accuracy': float(results['accuracy']),
                'eval_precision': float(results['precision']),
                'eval_recall': float(results['recall']),
                'eval_f1_score': float(results['f1_score']),
                'eval_roc_auc': float(results['roc_auc']),
                'eval_date_utc': datetime.now(datetime.UTC).isoformat() + 'Z',
                'eval_test_size': int(results.get('total', test_size))
            }

            plain_meta = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'roc_auc': float(results['roc_auc'])
            }

            ckpt_dir = os.path.dirname(checkpoint_path) or '.'
            os.makedirs(ckpt_dir, exist_ok=True)

            if os.path.exists(checkpoint_path):
                backup = checkpoint_path + f'.bak.{datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")}' # create backup
                try:
                    shutil.copy2(checkpoint_path, backup)
                except Exception:
                    backup = None

                ckpt = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                    # merge evaluation metrics into checkpoint dict (plain keys + prefixed keys)
                    ckpt.update(plain_meta)
                    ckpt.update(eval_meta)
                    torch.save(ckpt, checkpoint_path)
                    print(f"Saved evaluation metrics into checkpoint: {checkpoint_path}")
                    if backup:
                        print(f"Backup of original checkpoint saved to: {backup}")
                else:
                    # raw state_dict: write metadata.json next to checkpoint
                    meta_path = os.path.join(ckpt_dir, 'metadata.json')
                    merged = {**plain_meta, **eval_meta}
                    with open(meta_path, 'w') as f:
                        json.dump(merged, f, indent=2)
                    print(f"Checkpoint is a raw state_dict; wrote metadata to {meta_path}")
            else:
                meta_path = os.path.join(ckpt_dir, 'metadata.json')
                merged = {**plain_meta, **eval_meta}
                with open(meta_path, 'w') as f:
                    json.dump(merged, f, indent=2)
                print(f"No checkpoint found; wrote metadata to {meta_path}")
        except Exception as e:
            print(f"Warning: failed to save metadata: {e}")

        # generate visualization pngs
        try:
            out_dir = os.path.join('static', 'images', 'metrics')
            _ensure_dir(out_dir)

            # get test arrays and predictions
            X_test, y_test = get_test_arrays()
            with torch.no_grad():
                outputs = model(torch.tensor(X_test, dtype=torch.float32))
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            # ROC
            plot_roc_curve(y_test, probs, os.path.join(out_dir, 'roc_curve.png'))

            # Confusion matrix
            preds = (probs > 0.5).astype(int)
            cm = confusion_matrix(y_test, preds)
            plot_confusion_matrix(cm, os.path.join(out_dir, 'confusion_matrix.png'))

            # Metrics bar
            metrics = {
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'ROC-AUC': results['roc_auc']
            }
            plot_metrics_bar(metrics, os.path.join(out_dir, 'metrics.png'))

            # Feature influence (use actual feature names matching app.py)
            feature_names = [
                'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius',
                'semi_major_axis', 'insolation_flux', 'equilibrium_temp',
                'stellar_teff', 'stellar_radius', 'stellar_logg'
            ]
            # If model has different input dim, fall back to generated names
            if len(feature_names) != model.fc1.in_features:
                feature_names = [f'feat_{i}' for i in range(model.fc1.in_features)]
            plot_feature_influence(model, feature_names, os.path.join(out_dir, 'feature_importance.png'))
            # Dataset composition
            plot_dataset_composition(os.path.join(out_dir, 'dataset_distribution.png'))

        except Exception as e:
            print(f"Warning: failed to generate/save visualizations: {e}")

    except FileNotFoundError:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Run training.py to create a model first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
