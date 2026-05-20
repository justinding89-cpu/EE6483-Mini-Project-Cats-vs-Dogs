import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision_compat import ensure_torchvision_compat
ensure_torchvision_compat()
from torchvision import datasets, transforms
from tqdm import tqdm

from models import build_model
from utils import compute_accuracy, ensure_dir, get_device, plot_history, save_checkpoint, save_json, seed_everything


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'mobilenet_v3_small', 'smallcnn'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='output results/cifar10')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc='Eval', leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

    return total_loss / len(loader.dataset), compute_accuracy(labels_all, preds_all)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc='Train', leave=False):
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

    return total_loss / len(loader.dataset), compute_accuracy(labels_all, preds_all)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(args.image_size, padding=12),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train_aug = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=train_transform)
    full_train_eval = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=eval_transform)

    val_len = int(len(full_train_aug) * args.val_ratio)
    train_len = len(full_train_aug) - val_len
    indices = torch.randperm(len(full_train_aug), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    train_set = Subset(full_train_aug, train_indices)
    val_set = Subset(full_train_eval, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_name, num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f'train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'class_names': test_set.classes,
                'args': vars(args),
            }, str(Path(args.output_dir) / 'best_cifar10_model.pth'))

    save_json(history, str(Path(args.output_dir) / 'history.json'))
    plot_history(history, str(Path(args.output_dir) / 'history.json'))
    print(f'Best validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
