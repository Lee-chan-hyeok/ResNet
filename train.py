from model import BasicBlock, ResNet34
from torchinfo import summary
from dataset import ResNetDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ResNet(BasicBlock, [3, 4, 6, 3])
model = ResNet34(num_classes=1000)
model.to(device)
summary(model, (1, 3, 224, 224))

# --------- config 설정값 ---------
data_path = "data/images"
batch_size = 32
total_epoch = 10
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# --------------------------------

full_dataset = ResNetDataset(data_path=data_path, transform=transform)

train_size = int(len(full_dataset) * 0.8)
valid_size = len(full_dataset) - train_size

# prepare dataset
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

os.makedirs("checkpoint", exist_ok=True)

# early stop, best model save
early_stopping = 5
patience = 0
best_acc = 0.0

for epoch in range(total_epoch):

    train_loss = 0.0
    train_acc = 0.0
    model.train()
    for data in tqdm(train_loader, desc="train loader"):
        image, label = data
        image.to(device), label.to(device)

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
            image.to(device), label.to(device)

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

    # early stopping & save best model
    if valid_acc > best_acc:
        patience = 0
        best_acc = valid_acc
        torch.save(model.state_dict(), "checkpoint/best.pth")
        print(f"### save best model ###")
    else:
        patience += 1
        if patience > early_stopping:
            print(f"### Early stop! ###")
            break