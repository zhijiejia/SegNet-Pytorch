from torch import nn
from torch.nn import functional as F

class SegNet(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(2, 2)

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.unpooling1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
    
        self.unpooling2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.block7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.unpooling3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.block8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.unpooling4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.block9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.unpooling5 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)

        self.block10 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.output = nn.Conv2d(64, num_classes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):

        x_size = x.shape[2:]

        x1 = self.block1(x)
        x2, index1 = self.pooling(x1)

        x3 = self.block2(x2)
        x4, index2 = self.pooling(x3)

        x5 = self.block3(x4)
        x6, index3 = self.pooling(x5)

        x7 = self.block4(x6)
        x8, index4 = self.pooling(x7)

        x9 = self.block5(x8)

        x10, index5 = self.pooling(x9)

        x11 = self.unpooling1(x10)
        x11 = F.interpolate(x11, scale_factor=2, mode='bilinear', align_corners=True)
        x12 = self.block6(x11)

        x13 = self.unpooling2(x12)
        x13 = F.interpolate(x13, scale_factor=2, mode='bilinear', align_corners=True)
        x14 = self.block7(x13)

        x15 = self.unpooling3(x14)
        x15 = F.interpolate(x15, scale_factor=2, mode='bilinear', align_corners=True)
        x16 = self.block8(x15)

        x17 = self.unpooling4(x16)
        x17 = F.interpolate(x17, scale_factor=2, mode='bilinear', align_corners=True)
        x18 = self.block9(x17)

        x19 = self.unpooling5(x18)
        x19 = F.interpolate(x19, scale_factor=2, mode='bilinear', align_corners=True)
        x20 = self.block10(x19)

        x21 = self.output(x20)
        x21 = F.interpolate(x21, size=x_size, mode='bilinear', align_corners=True)

        return x21
