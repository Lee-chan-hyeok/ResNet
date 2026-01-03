from model import BasicBlock, Bottleneck, ResNet
from torchinfo import summary
from dataset import ResNetDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import os

from utils import save_loss_accuracy_graph, save_lr_graph


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device: {device}")

# ResNet101 (Bottleneck이랑 block 개수 설정)
model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=149, num_channels=3).to(device)
summary(model, (1, 3, 224, 224))

# --------- config ---------
data_path = "data/train_data"
batch_size = 64
warmup_epoch=5
total_epochs = 100
lr = 1e-3

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# --------------------------------

train_dataset = ResNetDataset(data_path=data_path, task="train", transform=train_transform)
valid_dataset = ResNetDataset(data_path=data_path, task="val", transform=val_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4
)

# Warmup scheduler를 위한 설정
warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,   # 초기 lr의 0.1배의 값부터 시작함.
    total_iters=warmup_epoch
)

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=total_epochs-warmup_epoch,
    eta_min=1e-5)

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, cosine],
    milestones=[warmup_epoch]   # warup_epoch끝나고 스케줄러 다음껄로 바꿈
)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

os.makedirs("checkpoint", exist_ok=True)

# early stop, best model save
early_stopping = 20
patience = 0
best_acc = 0.0

# lr 관련
lr_list = []

# loss, acc관련
train_loss_list, valid_loss_list = [], []
train_acc_list, valid_acc_list = [], []
graph_save_path = "checkpoint/v3"

for epoch in range(total_epochs):

    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for data in tqdm(train_loader, desc="train loader"):
        image, label = data
        image, label = image.to(device), label.to(device)

        out = model(image)
        loss = loss_fn(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # accuracy
        pred = out.argmax(dim=1)
        train_acc += (label == pred).sum().item()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    print(f"train_loss: {train_loss}, train_acc: {train_acc}")


    valid_loss = 0.0
    valid_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, desc="validation loader"):
            image, label = data
            image, label = image.to(device), label.to(device)

            out = model(image)
            loss = loss_fn(out, label)

            valid_loss += loss.item()
            # accuracy
            pred = out.argmax(dim=1)
            valid_acc += (label == pred).sum().item()

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)
        print(f"valid_loss: {valid_loss}, valid_acc: {valid_acc}")

    print(f"Complete {epoch+1} epoch !!!")

    # lr 관련
    current_lr = optimizer.param_groups[0]['lr']
    lr_list.append(current_lr)
    lr_scheduler.step()

    # loss, acc 관련
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

    save_loss_accuracy_graph(
        train_loss_list,
        valid_loss_list,
        train_acc_list,
        valid_acc_list,
        graph_save_path
    )
    
    save_lr_graph(
        lr_list,
        graph_save_path
    )

    # early stopping & save best model (accuracy 기준)
    if valid_acc > best_acc:
        best_acc = valid_acc
        patience = 0
        torch.save(model.state_dict(), f"checkpoint/{epoch+1}epoch_best.pth")
        print(f"### save best model ###")
    else:
        patience += 1
        if patience > early_stopping:
            print(f"### Early stop! ###")
            break