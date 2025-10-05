import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

DATASET_PATH = 'merged_exoplanets.csv'

# MODEL TAKES IN FOLLOWING: orbital_period, transit_duration, transit_depth, planet_radius, insolation_flux, equilibrium_temp, stellar_teff, stellar_logg, stellar_radius, semi_major_axis

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
        x = self.dropout(x)          # apply dropout before final layer
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

    df = df.apply(pd.to_numeric, errors='coerce').fillna(df.mean())

    labels = df['disposition'].astype(int).values
    features = df.drop(columns=['disposition']).values.astype(np.float32)

    feature_info = {
        'orbital_period': (0.1638211, 129995.7784),
        'transit_duration': (0.0, 138.54),
        'transit_depth': (0.0, 1541400.0),
        'planet_radius': (0.08, 200346.0),
        'insolation_flux': (0.0, 10947554.55),
        'equilibrium_temp': (25.0, 14667.0),
        'stellar_teff': (2550.0, 50000.0),
        'stellar_logg': (0.047, 5.96065),
        'stellar_radius': (0.109, 229.908),
        'semi_major_axis': (0.0013702438396151, 44.9892)
    }
    # feature_names = list(feature_info.keys())

    # synthetic examples of unrealistics
    # num_synthetic = int(0.1 * len(features))
    # synthetic_X = np.zeros((num_synthetic, features.shape[1]), dtype=float)
    # synthetic_y = np.zeros(num_synthetic, dtype=int)  # always False Positive

    # for i in range(num_synthetic):
    #     num_extreme = np.random.randint(1, features.shape[1]+1)  # number of extreme columns
    #     extreme_cols = np.random.choice(features.shape[1], num_extreme, replace=False)
    #     for j in range(features.shape[1]):
    #         min_val, max_val = feature_info[feature_names[j]]
    #         if j in extreme_cols:
    #             # make extreme (e.g., 10× to 100× max)
    #             synthetic_X[i, j] = max_val * np.random.uniform(10, 100)
    #         else:
    #             # normal realistic value
    #             synthetic_X[i, j] = np.random.uniform(min_val, max_val)

    # Train/test split
    idx = np.arange(len(features))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    print(f"Total rows: {len(features)}, train: {len(train_ds)}, test: {len(test_ds)}")

    return {
        "train": DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0),
        "test": DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0),
        "input_dim": features.shape[1],
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }

    
def train(epoch = 0):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / loaders_info['train_size']
    print(f"Epoch {epoch}: Train loss: {avg_loss:.6f}")
    return avg_loss


def save_checkpoint(state: dict, is_best: bool, checkpoint_dir: str, filename: str = 'checkpoint.pth'):
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth')
        torch.save(state, best_path)

def test():
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Compute loss
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0)

            # Sigmoid for probabilities
            probs = torch.sigmoid(output)
            pred = (probs > 0.5).long()

            all_probs.extend(probs.cpu().numpy().flatten().tolist())
            all_preds.extend(pred.cpu().numpy().flatten().tolist())
            all_targets.extend(target.cpu().numpy().flatten().tolist())

            # Count correct predictions
            correct += (pred.cpu() == target.cpu().long()).sum().item()

    test_loss = test_loss / loaders_info['test_size']
    acc = correct / loaders_info['test_size'] if loaders_info['test_size'] > 0 else 0.0

    # Calculate advanced metrics
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
        roc_auc = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{loaders_info['test_size']} ({100. * acc:.2f}%)")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              False  True")
    print(f"Actual False  {cm[0][0]:5d} {cm[0][1]:5d}")
    print(f"       True   {cm[1][0]:5d} {cm[1][1]:5d}")
    print("-" * 50)
    
    return acc, test_loss, precision, recall, f1, roc_auc

if __name__ == "__main__":
    data = load_dataset()

    # split returned dict into loaders and metadata
    loaders = {
        'train': data['train'],
        'test': data['test']
    }
    loaders_info = {
        'train_size': data['train_size'],
        'test_size': data['test_size']
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model with correct input dimension
    input_dim = data['input_dim']
    print(input_dim)
    model = SimpleNN(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # outputs are raw logits; use CrossEntropyLoss which expects class indices as targets
    loss_fn = nn.BCEWithLogitsLoss()
    # checkpointing setup
    checkpoint_dir = 'checkpoints'
    best_acc = 0.8 # default 80%
    
    # Track metrics over time
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }

    for epoch in range(1, 1000):
        train_loss = train(epoch)
        acc, test_loss, precision, recall, f1, roc_auc = test()
        
        # Record metrics
        metrics_history['epoch'].append(epoch)
        metrics_history['train_loss'].append(train_loss)
        metrics_history['test_loss'].append(test_loss)
        metrics_history['accuracy'].append(acc)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1_score'].append(f1)
        metrics_history['roc_auc'].append(roc_auc)

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
        }

        # save best model
        if acc > best_acc:
            best_acc = acc
            state['best_acc'] = best_acc
            save_checkpoint(state, is_best=True, checkpoint_dir=checkpoint_dir, filename=f'checkpoint_epoch_{epoch}.pth')
            print(f"✨ New best model! Accuracy: {100. * best_acc:.2f}%")
    
    # save metrics to CSV
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.to_csv('training_metrics.csv', index=False)
    print(f"\n✅ Training complete! Metrics saved to training_metrics.csv")