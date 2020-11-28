import os
import torch
import numpy as np
import torchvision
from torch import nn
from utils1 import decode_segmap
from PIL import Image
from Model import SegNet
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

class MyDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.datas = self.getData(data_dir)
        self.labels = self.getLabel(label_dir)
        self.transform = transform

    def __getitem__(self, index):
        imgPath = self.datas[index]
        labelPath = self.labels[index]
        imgTensor = self.transform(Image.open(self.data_dir + imgPath).convert('RGB'))
        labelTensor = Image.open(self.label_dir + labelPath).convert('L')
        labelTensor = torchvision.transforms.Resize(size=(512, 512))(labelTensor)
        labelTensor = np.array(labelTensor)
        labelTensor = torch.tensor(labelTensor)
        return imgTensor, labelTensor

    def __len__(self):
        return len(self.datas)

    def getData(self, data_dir):
        files = []
        for file in os.listdir(data_dir):
            files.append(file)
        return files

    def getLabel(self, label_dir):
        labels = []
        for label in os.listdir(label_dir):
            labels.append(label)
        return labels

model = SegNet().cuda()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(512, 512)),
    torchvision.transforms.ToTensor(),
])

epoch = 2

epochList = []

lossfun = nn.CrossEntropyLoss().cuda()

writer = SummaryWriter(comment='SegNet')

mydataset = MyDataset('/root/PycharmProjects/Pytorch_S/camvid/images/', '/root/PycharmProjects/Pytorch_S/camvid/labels/', transform=transform)

trainLoader = DataLoader(mydataset, batch_size=4, num_workers=2, shuffle=True, drop_last=True)

optimter = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

scheduler = lr_scheduler.MultiStepLR(optimter, milestones=[6, 15, 22, 30, 36, 41, 47], gamma=0.5)

for i in range(epoch):
    cnt = 0
    epoch_loss = 0
    for image, label in trainLoader:
        image = image.cuda()
        label = label.cuda()
        label = torch.squeeze(label)                       # label: 4 * 512 * 512
        output = model(image)                              # shape: 4 * 32 * 512 * 512
        loss = lossfun(output.float(), label.long())
        optimter.zero_grad()
        loss.backward()
        optimter.step()
        cnt += 1
        epoch_loss += loss.item()
        if cnt % 50 == 0:
            output = torch.argmax(output, dim=1).cpu().data.numpy()   # 4 * 512 * 512
            for j in range(4):
                img = decode_segmap(output[j, ...], 32)         # 512 * 512 * 3
                img = torch.tensor(img.transpose((2, 0, 1)))    # 3 * 512 * 512
                writer.add_image(tag=f'epoch-{i}-test-{cnt}-{j}', img_tensor=img)
                writer.add_image(tag=f'epoch-{i}-test-{cnt}-{j}-label', img_tensor=torch.unsqueeze(label[j, ...] * 5, dim=0))

        print(f'epoch: {i}, cnt: {cnt}, loss: {loss.item()}, lr: {scheduler.get_last_lr()}')
    scheduler.step()
    epochList.append(epoch_loss / cnt)

for index, epoch in enumerate(epochList):
    print(f'epoch: {index}, loss: {epoch}')