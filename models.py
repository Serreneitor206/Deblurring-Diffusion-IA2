import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        # Decoder con skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)  
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)  
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)  
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)  
        d1 = self.dec1(d1)

        out = self.final(d1)
        return self.sigmoid(out)


class KernelGenerator(nn.Module):
    def __init__(self, kernel_size, latent_dim=200, hidden_dim=2000, n_layers=3, mode='diffusion'):
        super().__init__()
        self.kernel_size = kernel_size
        self.mode = mode
        if mode == 'standard':
            self.fc_in = nn.Linear(latent_dim, hidden_dim)
        else:
            self.fc_in = nn.Linear(kernel_size*kernel_size, hidden_dim)

        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dim, kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        x = self.fc_in(z)
        x = self.hidden(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x.view(-1, 1, self.kernel_size, self.kernel_size)