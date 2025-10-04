import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import os

DATASET_PATH = 'merged_exoplanets.csv'

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
            
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    correct = sum([1 for p, t in zip(all_preds, all_targets) if p == t])
    acc = correct / test_size if test_size > 0 else 0.0

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
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
        
    except FileNotFoundError:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        print("   Run training.py to create a model first.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
