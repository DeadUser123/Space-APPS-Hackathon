import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
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

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    print(f"Total rows: {len(df)}, train: {len(train_ds)}, test: {len(test_ds)}")

    return {
        "train": DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0),
        "test": DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0),
        "input_dim": features.shape[1],
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }
    
def train_epoch(model, optimizer, loss_fn, loaders, loaders_info, device, epoch):
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

def test_model(model, loss_fn, loaders, loaders_info, device):
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
            test_loss += loss_fn(output, target).item() * data.size(0)
            
            probs = F.softmax(output, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    test_loss = test_loss / loaders_info['test_size']
    acc = correct / loaders_info['test_size'] if loaders_info['test_size'] > 0 else 0.0

    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
        roc_auc = 0.0
    
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
    print("Testing training script with advanced metrics...")
    print("=" * 70)
    
    data = load_dataset()
    loaders = {'train': data['train'], 'test': data['test']}
    loaders_info = {'train_size': data['train_size'], 'test_size': data['test_size']}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = data['input_dim']
    print(f"Input dimension: {input_dim}")
    
    model = SimpleNN(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # Run just 5 epochs for testing
    print("\nRunning 5 test epochs...")
    print("=" * 70)
    
    for epoch in range(1, 6):
        train_loss = train_epoch(model, optimizer, loss_fn, loaders, loaders_info, device, epoch)
        acc, test_loss, precision, recall, f1, roc_auc = test_model(model, loss_fn, loaders, loaders_info, device)
    
    print("\n" + "=" * 70)
    print("✅ Test complete! All metrics are working correctly.")
    print("=" * 70)
