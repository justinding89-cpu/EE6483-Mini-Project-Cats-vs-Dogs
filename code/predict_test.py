import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestImageDataset, build_eval_transform
from models import build_model
from utils import get_device, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Dogs vs Cats test set and export submission.csv')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to dataset/test directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'mobilenet_v3_small', 'smallcnn'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_csv', type=str, default='submission.csv')
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()

    dataset = TestImageDataset(args.test_dir, transform=build_eval_transform(args.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model_name, num_classes=2, pretrained=False)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model = model.to(device)
    model.eval()

    ids, labels = [], []
    for images, image_ids in tqdm(loader, desc='Predict'):
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        normalized_ids = []
        for x in image_ids:
            if hasattr(x, 'item'):
                x = x.item()
            sx = str(x)
            normalized_ids.append(int(sx) if sx.isdigit() else sx)
        ids.extend(normalized_ids)
        labels.extend(preds)

    df = pd.DataFrame({'id': ids, 'label': labels})
    if pd.api.types.is_numeric_dtype(df['id']):
        df = df.sort_values('id').reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f'Saved submission file to {args.output_csv}')
    print(df.head())

if __name__ == '__main__':
    main()
