import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

DATASET_PATH = 'merged_exoplanets.csv'

# MODEL TAKES IN FOLLOWING: orbital_period, transit_duration, transit_depth, planet_radius, insolation_flux, equilibrium_temp, stellar_teff, stellar_logg, stellar_radius, semi_major_axis

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
        x = self.dropout(x)          # apply dropout before final layer
        x = self.fc3(x)
        return x
    
def load_dataset():
    seed = 42 # prevent model from being retrained on smth else
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Read CSV
    df = pd.read_csv(DATASET_PATH)

    # Expectation: 'disposition' column is the label (0 or 1). Drop non-numeric columns like 'source'.
    if 'disposition' not in df.columns:
        raise ValueError("CSV must contain a 'disposition' column with labels 0/1")

    # Drop text/categorical columns that are not numeric
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    # keep 'disposition' even if it's object-y; convert explicitly below
    non_numeric_cols = [c for c in non_numeric_cols if c != 'disposition']
    df = df.drop(columns=non_numeric_cols)

    # Convert all columns to numeric, coerce errors and fill missing values with column mean
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.mean())

    # Separate features and labels
    labels = df['disposition'].astype(int).values
    features = df.drop(columns=['disposition']).values.astype(np.float32)

    # Train/test split (changed to 90/10)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # Convert to tensors and create TensorDataset
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    from torch.utils.data import TensorDataset
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    
    # show sizes for transparency
    print(f"Total rows: {len(df)}, train: {len(train_ds)}, test: {len(test_ds)}")

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

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # store for potential metrics
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

    test_loss = test_loss / loaders_info['test_size']
    acc = correct / loaders_info['test_size'] if loaders_info['test_size'] > 0 else 0.0

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{loaders_info['test_size']} ({100. * acc:.2f}%)")
    return acc, test_loss

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
    loss_fn = nn.CrossEntropyLoss()
    # checkpointing setup
    checkpoint_dir = 'checkpoints'
    best_acc = 0.0
    save_every = 10

    for epoch in range(1, 1000):
        train(epoch)
        acc, test_loss = test()

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }

        # save periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(state, is_best=False, checkpoint_dir=checkpoint_dir, filename=f'checkpoint_epoch_{epoch}.pth')

        # save best model
        if acc > best_acc:
            best_acc = acc
            state['best_acc'] = best_acc
            save_checkpoint(state, is_best=True, checkpoint_dir=checkpoint_dir, filename=f'checkpoint_epoch_{epoch}.pth')