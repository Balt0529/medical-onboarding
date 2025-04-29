import torch
# from torch import Module
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv3d(3, 3, 3, padding=1)
        self.conv3 = nn.Conv3d(3, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(10 * 14 * 10 * 64, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm3d1 = nn.BatchNorm3d(3)
        self.batchnorm3d12 = nn.BatchNorm3d(3)
        self.batchnorm3d2 = nn.BatchNorm3d(32)
        self.batchnorm3d3 = nn.BatchNorm3d(64)
        self.batchnorm1 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.batchnorm3d1(self.conv1(x)))
        x = F.relu(self.batchnorm3d12(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3d2(self.conv3(x)))
        x = F.relu(self.batchnorm3d3(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 10 * 14 * 10 * 64)
        x = self.dropout(x)
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


class CAE3D(nn.Module):
    def __init__(self):
        super(CAE3D, self).__init__()
        
        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),            
            
            nn.Conv3d(3, 3, kernel_size=3, padding=1),          
            nn.ReLU(),
            nn.Conv3d(3, 32, kernel_size=3, padding=1),         
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),            

            nn.Conv3d(32, 64, kernel_size=3, padding=1),        
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),        
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),             
        )

        # Flatten + FC
        self.fc1 = nn.Linear(64 * 10 * 14 * 10, 512)              
        self.fc2 = nn.Linear(512, 256)
        
        # デコーダ用全結合
        self.defc1 = nn.Linear(256, 512)
        self.defc2 = nn.Linear(512, 64 * 10 * 14 * 10)
        
        # デコーダConv
        self.decoder_conv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(3, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # エンコード処理
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        encoded = self.fc2(self.fc1(x))  # 出力256次元

        # デコード処理
        x = self.defc2(self.defc1(encoded))
        x = x.view(x.size(0), 64, 10, 14, 10)
        x = self.decoder_conv(x)
        return x, encoded


class VAE3D(nn.Module):
    def __init__(self):
        super(VAE3D, self).__init__()

        # エンコーダー
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        # Flatten + 潜在変数パラメータ μ, logσ²
        self.fc1 = nn.Linear(64 * 10 * 14 * 10, 512)
        self.fc_mu = nn.Linear(512, 256)
        self.fc_logvar = nn.Linear(512, 256)

        # デコーダー用全結合
        self.defc1 = nn.Linear(256, 512)
        self.defc2 = nn.Linear(512, 64 * 10 * 14 * 10)

        # デコーダー
        self.decoder_conv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(32, 3, kernel_size=3, padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),

            nn.Conv3d(3, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # エンコード
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        h = self.fc1(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 再パラメータ化
        z = self.reparameterize(mu, logvar)

        # デコード
        x = self.defc1(z)
        x = self.defc2(x)
        x = x.view(x.size(0), 64, 10, 14, 10)
        x = self.decoder_conv(x)

        return x, mu, logvar
