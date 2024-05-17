# -*- coding: utf-8 -*-
"""
Created on Wed May 8 14:37:08 2024
采用CNN对railway fault数据集分类
@author: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os, sys,copy
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_DIR)

import torchvision
print(torch.__version__)
print(torchvision.__version__)


class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128,kernel_size=3, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.fc1 = nn.Linear(in_features=64*54*54, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=2)

    def forward(self, x):
        #x [3, 224, 224]
        x = F.relu(self.conv1(x)) #[128, 222, 222]
        x = F.max_pool2d(x, 2, 2) #[128, 111, 111]
        x = F.relu(self.conv2(x)) #[64, 109, 109]
        x = F.max_pool2d(x, 2, 2) #[64, 54, 54]

        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, epoch):
    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        pred = model(data)
        loss = nn.CrossEntropyLoss(pred, label)
        loss.backward()
        optimizer.step()
    print("loss:", loss.item())

def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad(): 
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print("acc:", 100 * correct / len(test_loader.dataset))
            

#预处理数据
#t图片大小为[3,224,224]
batch_size = 100


transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 根据需要调整尺寸
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
    # 可以添加其他转换...
])
train_dataset = datasets.ImageFolder('train', transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

print(f"length : {len(train_dataset)}")

valid_dataset = datasets.ImageFolder('train', transform=transform)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = datasets.ImageFolder('test', transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

for images, labels in train_dataloader:
    # 处理你的图片批次...
    pass




lr = 1e-2
momentum = 0.5
model = myNet()


epochs = 1
isTrain = False

if(isTrain):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        train(model, train_dataloader, optimizer, epoch)
        test(model, test_dataloader)

torch.save(model.state_dict(),"railway_fault_cnn0.pt")
#torch.save(model, "railway_cnn1.pt")

isTest = True

if(isTest):
    model.load_state_dict(torch.load("railway_fault_cnn0.pt"))
    test(model, test_dataloader)

    