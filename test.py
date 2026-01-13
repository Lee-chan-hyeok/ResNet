from model import BasicBlock, Bottleneck, ResNet
from torchinfo import summary
from dataset import ResNetDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# For f1-score
from sklearn.metrics import f1_score

import numpy as np
import torch
import torch.nn as nn
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# def F1Score(y_true, y_pred):
#     TP = np.sum(y_pred == y_true)
#     FP = np.sum(y_pred != y_true)
#     FN = np.sum(y_pred != y_true)

#     precision = TP / (TP + FP + 1e-5)
#     recall = TP / (TP + FN + 1e-5)

#     f1_score = float(2.0 * precision * recall) / (precision + recall + 1e-5)

#     return f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=149, num_channels=3).to(device)
# summary(model, (1, 3, 224, 224))
model_path = torch.load("checkpoint/v3/77epoch_best.pth", map_location=device)
model.load_state_dict(model_path)

# --------- config ---------
data_path = "data/test_data"
batch_size = 64

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)   # 이거 빼야되나 말아야되나...
])
# --------------------------------

test_dataset = ResNetDataset(data_path=data_path, task="test", transform=test_transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

best_acc = 0.0

checkpoint_save_path = "checkpoint/v3"

os.makedirs(checkpoint_save_path, exist_ok=True)

correct = 0
total = 0
test_loss = 0.0

# for f1-score
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader, desc="Test loader"):
        image, label = data
        image, label = image.to(device), label.to(device)

        out = model(image)
        loss = loss_fn(out, label)

        test_loss += loss.item() * label.shape[0]

        pred = out.argmax(dim=1)
        correct += (pred == label).sum().item()  # 맞힌 데이터 개수
        
        total += label.shape[0]  # label 전체 개수

        # For f1-score
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    test_loss /= total
    test_acc = correct / total
    top1_error = 1 - test_acc

    # Calc f1-score
    f1 = f1_score(all_labels, all_preds, average='macro')
    # custom_f1 = F1Score(all_labels, all_preds)

    print(f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, F1-score: {f1:.4f}")
    # print(f"Custom F1-score: {custom_f1:.4f}")

    #print(f"test_loss: {test_loss:.2f}, test_acc: {test_acc:.2f}")
    print(f"### Top-1 error : {top1_error:.2f}")
