import torch
import torch.nn as nn
from dataset import Heatmap2D_Dataset
from torchinfo import summary


class Heatmap2D_Encoder(nn.Module):
    # input: (bs, 3, 256, 256)
    # output: (bs, 128, 28, 28)
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3
            ),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.mp = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.conv3(x)
        x = self.mp(x)
        x = self.conv4(x)
        return x


class Heatmap2D_Decoder(nn.Module):
    # input: (bs, 128, 28, 28)
    # output: (bs, 4, 64, 64)
    def __init__(self):
        super().__init__()

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=2,
                stride=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=1
            ),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=4,
                kernel_size=1
            )
        )


    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.outconv(x)
        return x



class Heatmap2D_Model(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.encoder = Heatmap2D_Encoder()
        self.decoder = Heatmap2D_Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




if __name__ == "__main__":
    model = Heatmap2D_Model()
    summary(model=model, input_size=(8, 3, 256, 256))




