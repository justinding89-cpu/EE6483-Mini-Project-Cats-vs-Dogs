# Dogs vs. Cats Classification with Transfer Learning + CIFAR-10 Imbalance Study

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red" />
  <img src="https://img.shields.io/badge/Task-Image%20Classification-green" />
  <img src="https://img.shields.io/badge/Backbone-ResNet%2FMobileNet-orange" />
</p>

## 📌 Project Overview

This repository contains a reproducible deep-learning pipeline for **Dogs vs. Cats image classification** and an extended study on **CIFAR-10 multi-class classification under class imbalance**.

The main task is to classify whether an input image contains a **dog** or a **cat**. The project compares multiple CNN-based models, including **ResNet18**, **ResNet34**, **MobileNetV3-Small**, and a custom **SmallCNN** trained from scratch. It also explores data augmentation, transfer learning, learning-rate scheduling, early stopping, and imbalance-aware training strategies.

> 中文简介：本项目基于 PyTorch 实现猫狗二分类任务，并进一步扩展到 CIFAR-10 多分类和类别不平衡实验。项目重点展示了迁移学习、数据增强、模型对比和不平衡数据处理方法。

\---

## ✨ Highlights

* 🐶🐱 **Dogs vs. Cats binary image classification**
* 🔥 **Transfer learning** with ImageNet-pretrained ResNet and MobileNet backbones
* 🧠 Custom **SmallCNN baseline** for comparison
* 📈 Training / validation loss and accuracy curve visualization
* ⚖️ CIFAR-10 class imbalance experiments
* 🧪 Comparison of imbalance remedies:

  * Class-weighted cross entropy
  * Weighted random sampling
  * Focal loss
* 📄 Automatic `submission.csv` generation for test prediction
* 🔁 Reproducible training with fixed random seed

\---

## 🏆 Experimental Results

### Dogs vs. Cats Validation Performance

|Model|Parameters|Best Validation Accuracy|Observation|
|-|-:|-:|-|
|ResNet34|21.29M|**99.2%**|Best overall performance|
|ResNet18|11.18M|**99.0%**|Best balance between accuracy and efficiency|
|ResNet18 without augmentation|11.18M|98.8%|Strong but less stable|
|MobileNetV3-Small|1.52M|98.3%|Lightweight and efficient|
|SmallCNN|0.42M|87.9%|Simple baseline trained from scratch|

### CIFAR-10 Extension

A fine-tuned ResNet18 was adapted from binary classification to 10-class classification.

|Task|Model|Best Validation Accuracy|
|-|-|-:|
|CIFAR-10 multi-class classification|ResNet18|**96.8%**|

### CIFAR-10 Class Imbalance Study

Classes `0`, `1`, and `8` were reduced to 20% of their original training samples to simulate an imbalanced setting.

|Method|Best Validation Accuracy|Delta vs. Baseline|
|-|-:|-:|
|Baseline Cross Entropy|93.92%|0.00%|
|Class-weighted Cross Entropy|93.76%|-0.16%|
|Weighted Random Sampler|**94.04%**|+0.12%|
|Focal Loss|93.67%|-0.25%|

\---

## 📁 Repository Structure

```text
.
├── code/
│   ├── dataset.py                  # Dataset loading and image transforms
│   ├── models.py                   # ResNet, MobileNetV3, SmallCNN, FocalLoss
│   ├── train\_dogs\_vs\_cats.py        # Main training script for Dogs vs. Cats
│   ├── predict\_test.py              # Generate prediction CSV for test images
│   ├── train\_cifar10.py             # CIFAR-10 multi-class training script
│   ├── imbalance\_experiment.py      # CIFAR-10 imbalance experiments
│   ├── utils.py                     # Metrics, checkpointing, plotting, reproducibility
│   ├── torchvision\_compat.py        # Torchvision compatibility helper
│   ├── requirements.txt             # Python dependencies
│   └── output results/              # Training curves and saved results
├── report/
│   └── Report.pdf                   # Project report
├── task/
│   └── EE6483-Project2.pdf          # Project task description
├── submission.csv                   # Test prediction result
└── README.md
```

> Recommended cleanup before publishing: do not upload `.idea/`, `\_\_pycache\_\_/`, local datasets, or large `.pth` model files directly to GitHub.

\---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR\_USERNAME/dogs-vs-cats-transfer-learning.git
cd dogs-vs-cats-transfer-learning/code
```

Create a virtual environment:

```bash
python -m venv .venv

# Windows
.venv\\Scripts\\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

\---

## 📦 Dataset Preparation

The Dogs vs. Cats dataset should be organized as follows:

```text
dataset/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

Label definition used for submission:

```text
0 = cat
1 = dog
```

\---

## 🚀 Training

### Train ResNet18 on Dogs vs. Cats

```bash
python train\_dogs\_vs\_cats.py \\
  --data\_root /path/to/dataset \\
  --model\_name resnet18 \\
  --epochs 12 \\
  --batch\_size 32 \\
  --lr 1e-4 \\
  --output\_dir outputs/dogs\_vs\_cats\_resnet18
```

### Train ResNet34

```bash
python train\_dogs\_vs\_cats.py \\
  --data\_root /path/to/dataset \\
  --model\_name resnet34 \\
  --epochs 12 \\
  --batch\_size 32 \\
  --lr 1e-4 \\
  --output\_dir outputs/dogs\_vs\_cats\_resnet34
```

### Train MobileNetV3-Small

```bash
python train\_dogs\_vs\_cats.py \\
  --data\_root /path/to/dataset \\
  --model\_name mobilenet\_v3\_small \\
  --epochs 12 \\
  --batch\_size 32 \\
  --lr 1e-4 \\
  --output\_dir outputs/dogs\_vs\_cats\_mobilenetv3small
```

### Train SmallCNN

```bash
python train\_dogs\_vs\_cats.py \\
  --data\_root /path/to/dataset \\
  --model\_name smallcnn \\
  --epochs 15 \\
  --batch\_size 32 \\
  --lr 1e-4 \\
  --output\_dir outputs/dogs\_vs\_cats\_smallcnn
```

\---

## 📤 Generate Test Submission

```bash
python predict\_test.py \\
  --test\_dir /path/to/dataset/test \\
  --checkpoint outputs/dogs\_vs\_cats\_resnet18/best\_model.pth \\
  --model\_name resnet18 \\
  --output\_csv outputs/submission.csv
```

The generated CSV file follows the format:

```csv
id,label
1,0
2,1
3,1
...
```

\---

## 🔟 CIFAR-10 Multi-class Classification

```bash
python train\_cifar10.py \\
  --data\_root ./data \\
  --model\_name resnet18 \\
  --epochs 15 \\
  --batch\_size 64 \\
  --lr 1e-4 \\
  --output\_dir outputs/cifar10\_resnet18
```

\---

## ⚖️ CIFAR-10 Class Imbalance Experiment

```bash
python imbalance\_experiment.py \\
  --data\_root ./data \\
  --epochs 8 \\
  --minority\_fraction 0.2 \\
  --minority\_classes 0 1 8
```

This experiment compares baseline cross entropy, class-weighted loss, weighted sampling, and focal loss.

\---

## 📊 Training Curves

Example training curves are saved under the output directory after training:

```text
outputs/.../loss\_curve.png
outputs/.../accuracy\_curve.png
```

If you keep the original project output folder, example paths include:

```text
code/output results/dogs\_vs\_cats\_resnet18/loss\_curve.png
code/output results/dogs\_vs\_cats\_resnet18/accuracy\_curve.png
code/output results/dogs\_vs\_cats\_resnet34/loss\_curve.png
code/output results/dogs\_vs\_cats\_resnet34/accuracy\_curve.png
```

\---

## 🧠 Method Summary

The project treats Dogs vs. Cats as a supervised closed-set binary classification problem. Since the dataset is medium-sized and visually structured, transfer learning is an effective strategy. ImageNet-pretrained CNN backbones provide strong generic visual features, while the final classification layer is replaced for the target two-class task.

The main training pipeline includes:

1. Image resizing to `224 × 224`
2. Random horizontal flipping
3. Random rotation
4. Light color jitter
5. ImageNet normalization
6. Cross-entropy loss
7. Adam optimizer
8. Learning-rate scheduling
9. Early stopping
10. Checkpoint saving based on best validation accuracy

For CIFAR-10, the classifier head is changed to 10 output classes and CIFAR-specific normalization is used.

\---

## 📌 Key Takeaways

* Transfer learning significantly outperforms training a small CNN from scratch.
* ResNet34 achieves the highest validation accuracy, while ResNet18 provides the best accuracy-efficiency trade-off.
* Data augmentation improves training stability and generalization.
* MobileNetV3-Small is a strong choice when model size and inference efficiency matter.
* For moderate class imbalance, weighted sampling can be more effective than simply modifying the loss function.

\---

## ⭐ How to Make This Repository More Attractive

To help more people discover and star this project:

1. Use a clear repository name, for example:

```text
   dogs-vs-cats-transfer-learning
   pytorch-dogs-cats-classification
   image-classification-resnet-cifar10
   ```

2. Add GitHub topics:

```text
   pytorch, deep-learning, computer-vision, image-classification, resnet, mobilenet, transfer-learning, cifar10, class-imbalance, dogs-vs-cats
   ```

3. Add result images to the README, such as accuracy curves and loss curves.
4. Keep the README in English for a wider audience, but include a short Chinese description if needed.
5. Upload large model weights to **GitHub Releases**, **Google Drive**, **OneDrive**, or **Hugging Face**, then link them in the README instead of committing them directly.
6. Pin this repository on your GitHub profile.
7. Add a short demo section showing example predictions.
8. Add a clean `requirements.txt`, `.gitignore`, and `LICENSE` file.

\---

## ⚠️ Notes on Large Files

GitHub blocks files larger than 100 MB in normal Git commits. This project contains large `.pth` model checkpoints, so you should not upload them directly with normal Git.

Recommended options:

* Ignore model weights with `.gitignore`
* Use Git LFS
* Upload weights to GitHub Releases or cloud storage
* Keep only training curves, metrics JSON files, report, and source code in the repository

\---

## 📄 License

This project is released for academic and educational purposes. You may add an MIT License if you want others to reuse the code more easily.

\---

## 🙋‍♂️ Author

**Justin Ding**

Research interests: Machine Learning, Deep Learning, Computer Vision, Large Language Models, and AI Agents.

If you find this project useful, please consider giving it a ⭐!

