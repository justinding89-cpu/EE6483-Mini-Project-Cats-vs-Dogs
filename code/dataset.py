from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision_compat import ensure_torchvision_compat
ensure_torchvision_compat()
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # transforms.Resize((image_size, image_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

class TestImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            [p for p in self.root_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}],
            key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
        return image, image_id

def get_imagefolder_datasets(train_dir: str, val_dir: str, image_size: int = 224):
    train_dataset = datasets.ImageFolder(train_dir, transform=build_train_transform(image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_eval_transform(image_size))
    return train_dataset, val_dataset

def get_class_weights_from_imagefolder(dataset: datasets.ImageFolder) -> torch.Tensor:
    targets = [label for _, label in dataset.samples]
    num_classes = len(dataset.classes)
    counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp(min=1.0))
    return weights

def get_sample_weights_from_imagefolder(dataset: datasets.ImageFolder) -> List[float]:
    targets = [label for _, label in dataset.samples]
    counts = torch.bincount(torch.tensor(targets)).float()
    class_weights = 1.0 / counts.clamp(min=1.0)
    sample_weights = [class_weights[t].item() for t in targets]
    return sample_weights
