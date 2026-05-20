import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision_compat import ensure_torchvision_compat
ensure_torchvision_compat()
from torchvision import datasets, transforms
from tqdm import tqdm

from models import FocalLoss, build_model
from utils import compute_accuracy, ensure_dir, get_device, save_json, seed_everything

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def parse_args():
    parser = argparse.ArgumentParser(description='Class imbalance experiments on CIFAR-10')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output results/cifar10_imbalance')
    parser.add_argument('--minority_fraction', type=float, default=0.2, help='Fraction kept for selected minority classes')
    parser.add_argument('--minority_classes', type=int, nargs='+', default=[0, 1, 8], help='Class indices made minority')
    return parser.parse_args()

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return total_loss / len(loader.dataset), compute_accuracy(labels_all, preds_all)

def train(model, loader, val_loader, criterion, optimizer, device, epochs):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        preds_all, labels_all = [], []
        total_loss = 0.0
        for images, labels in tqdm(loader, desc=f'Train epoch {epoch + 1}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            preds_all.extend(preds.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

        train_loss = total_loss / len(loader.dataset)
        train_acc = compute_accuracy(labels_all, preds_all)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        best_val_acc = max(best_val_acc, val_acc)
        print(f'epoch={epoch + 1}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')
    return best_val_acc

def create_imbalanced_subset(dataset, minority_classes, minority_fraction, seed=42):
    rng = torch.Generator().manual_seed(seed)
    targets = torch.tensor(dataset.targets)
    selected_indices = []
    stats = {}

    for cls in range(10):
        cls_indices = torch.where(targets == cls)[0]
        keep = int(len(cls_indices) * minority_fraction) if cls in minority_classes else len(cls_indices)
        keep = max(keep, 1)
        perm = cls_indices[torch.randperm(len(cls_indices), generator=rng)[:keep]]
        selected_indices.extend(perm.tolist())
        stats[int(cls)] = int(keep)

    return Subset(dataset, selected_indices), stats

def get_subset_targets(subset):
    return [subset.dataset.targets[i] for i in subset.indices]

def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)
    device = get_device()

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    base_train = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_transform)

    imbalanced_train, class_counts = create_imbalanced_subset(base_train, args.minority_classes, args.minority_fraction, args.seed)
    subset_targets = torch.tensor(get_subset_targets(imbalanced_train))
    counts = torch.bincount(subset_targets, minlength=10).float()
    class_weights = counts.sum() / (10 * counts.clamp(min=1.0))
    sample_weights = [1.0 / counts[label].item() for label in subset_targets.tolist()]

    regular_loader = DataLoader(imbalanced_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    weighted_loader = DataLoader(imbalanced_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    results = {'class_counts': class_counts}

    # Baseline cross-entropy
    model = build_model('resnet18', num_classes=10, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    results['baseline_ce'] = {'best_val_acc': train(model, regular_loader, val_loader, ce_loss, optimizer, device, args.epochs)}

    # Class-weighted loss
    model = build_model('resnet18', num_classes=10, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    results['class_weighted_ce'] = {'best_val_acc': train(model, regular_loader, val_loader, weighted_ce, optimizer, device, args.epochs)}

    # Weighted sampler
    model = build_model('resnet18', num_classes=10, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    results['weighted_sampler'] = {'best_val_acc': train(model, weighted_loader, val_loader, ce_loss, optimizer, device, args.epochs)}

    # Focal loss extension
    model = build_model('resnet18', num_classes=10, pretrained=True).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    focal_loss = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    results['focal_loss'] = {'best_val_acc': train(model, regular_loader, val_loader, focal_loss, optimizer, device, args.epochs)}

    results['subset_distribution'] = dict(Counter(get_subset_targets(imbalanced_train)))
    save_json(results, str(Path(args.output_dir) / 'imbalance_results.json'))
    print(results)

if __name__ == '__main__':
    main()
