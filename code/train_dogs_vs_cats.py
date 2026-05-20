import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from dataset import get_class_weights_from_imagefolder, get_imagefolder_datasets, get_sample_weights_from_imagefolder
from models import FocalLoss, build_model
from utils import EarlyStopping, compute_accuracy, ensure_dir, get_device, load_checkpoint, plot_history, save_checkpoint, save_json, seed_everything, summarize_classification

def parse_args():
    parser = argparse.ArgumentParser(description='Train Dogs vs Cats classifier')
    parser.add_argument('--data_root', type=str, required=True, help='Dataset root containing train/, val/, test/')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'mobilenet_v3_small', 'smallcnn'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output results/dogs_vs_cats')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'step', 'none'])
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--use_weighted_sampler', action='store_true')
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--use_focal_loss', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=4)
    parser.add_argument('--resume', type=str, default='')
    return parser.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = compute_accuracy(labels_all, preds_all)
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    preds_all, labels_all = [], []

    for images, labels in tqdm(loader, desc='Val', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = compute_accuracy(labels_all, preds_all)
    summary = summarize_classification(labels_all, preds_all, class_names)
    return epoch_loss, epoch_acc, summary

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    train_dir = str(Path(args.data_root) / 'train')
    val_dir = str(Path(args.data_root) / 'val')

    train_dataset, val_dataset = get_imagefolder_datasets(train_dir, val_dir, args.image_size)

    if args.use_weighted_sampler:
        sample_weights = get_sample_weights_from_imagefolder(train_dataset)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_name, num_classes=2, pretrained=True, freeze_backbone=args.freeze_backbone).to(device)

    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights_from_imagefolder(train_dataset).to(device)

    if args.use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.gamma, patience=2)
    elif args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=args.early_stop_patience, mode='max')

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer=optimizer, map_location=device)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        history = ckpt.get('history', history)

    class_names = train_dataset.classes
    print(f'Using device: {device}')
    print(f'Classes: {class_names}')

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_summary = evaluate(model, val_loader, criterion, device, class_names)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

        improved = early_stopping.step(val_acc)
        if improved and val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'class_names': class_names,
                'args': vars(args),
            }, str(Path(args.output_dir) / 'best_model.pth'))
            save_json(val_summary, str(Path(args.output_dir) / 'best_val_metrics.json'))

        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()

        if early_stopping.should_stop:
            print('Early stopping triggered.')
            break
    plot_history(history, str(Path(args.output_dir) / 'history.json'))
    save_json(history, str(Path(args.output_dir) / 'history.json'))
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'Artifacts saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
