import torch
from torch import nn
import torch.nn.functional as F
import os

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

def load_model_from_checkpoint(checkpoint_path: str, input_dim: int) -> nn.Module:
    """Load a model from a checkpoint.

    The training script saves a dict with keys like 'model_state_dict'. This function
    accepts either that dict or a raw state_dict. If `input_dim` is None, the function
    will try to infer the input dimension from the saved weights (fc1.weight).
    Returns (model, metadata_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # load to CPU by default (safe)
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # determine whether ckpt is a dict containing model_state_dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model_state = ckpt['model_state_dict']
        metadata = {k: v for k, v in ckpt.items() if k != 'model_state_dict'}
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        # looks like a raw state_dict
        model_state = ckpt
        metadata = {}
    else:
        # fallback: try to treat ckpt itself as a state dict
        try:
            model_state = dict(ckpt)
            metadata = {}
        except Exception:
            raise RuntimeError("Unrecognized checkpoint format")

    # infer input_dim if not provided
    inferred_input_dim = None
    if 'fc1.weight' in model_state:
        inferred_input_dim = model_state['fc1.weight'].shape[1]
    else:
        # find the first 2D tensor (weight) and use its second dim
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                inferred_input_dim = v.shape[1]
                break

    if input_dim is None:
        if inferred_input_dim is None:
            raise ValueError('Could not infer input_dim from checkpoint; provide input_dim explicitly')
        input_dim = int(inferred_input_dim)

    model = SimpleNN(input_dim=input_dim)
    model.load_state_dict(model_state)
    model.eval()  # set to evaluation mode
    return model, metadata


if __name__ == '__main__':
    # example usage: load the best checkpoint saved by training
    ckpt_path = os.path.join('checkpoints', 'best.pth')
    try:
        model, meta = load_model_from_checkpoint(ckpt_path, input_dim=None)
        print(f"Loaded model from {ckpt_path}; inferred input_dim={model.fc1.in_features}")
        if meta:
            print("Checkpoint metadata keys:", list(meta.keys()))
    except FileNotFoundError:
        print(f"No checkpoint found at {ckpt_path}. Run training.py to create checkpoints first.")