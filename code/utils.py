import json
import os
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_checkpoint(state: Dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    torch.save(state, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt

def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    return float(accuracy_score(y_true, y_pred))

def summarize_classification(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict:
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    return {'confusion_matrix': cm, 'classification_report': report}

def plot_history(history: Dict[str, List[float]], save_path: str) -> None:
    ensure_dir(str(Path(save_path).parent))

    plt.figure(figsize=(8, 5))
    plt.plot(history.get('train_loss', []), label='train_loss')
    plt.plot(history.get('val_loss', []), label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(save_path).with_name('loss_curve.png')), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.get('train_acc', []), label='train_acc')
    plt.plot(history.get('val_acc', []), label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(Path(save_path).with_name('accuracy_curve.png')), dpi=200)
    plt.close()

class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True

        improved = score > self.best_score if self.mode == 'max' else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
