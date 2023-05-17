import torch.nn as nn
import torch
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        )

    def forward(self, x):
        return x + self.res(x)


class TraBlock(nn.Module):
    def __init__(self, dim):
        super(TraBlock, self).__init__()

        self.tra1 = nn.Sequential(ResBlock(dim))
        self.tra2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=2, bias=True),
            ResBlock(dim * 2)
        )
        self.fution = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x1 = self.tra1(x)
        x2 = F.interpolate(self.tra2(x), scale_factor=2, mode='bilinear')
        x = torch.cat((x1, x2), dim=1)
        x = self.fution(x)
        return x


class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        self.grad = nn.ModuleList([
            nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True),
            ResBlock(dim)
        ])

    def forward(self, x, y, Phi, PhiT):
        x_pixel = self.grad[0](x)
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32, bias=None)
        delta = y - Phix
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0, bias=None))
        x_delta = self.grad[1](x_pixel)
        return self.grad[2](x_delta)


class AttBlock(nn.Module):
    def __init__(self, dim):
        super(AttBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = x.reshape(B, C, H//32, 32, W//32, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 32, 32)
        x1 = self.conv2(x).reshape(-1, C, 32*32)
        x2 = self.conv3(x).reshape(-1, C, 32*32).transpose(1, 2)
        att = (x2 @ x1) * self.scale
        att = att.softmax(dim=1)
        x = (x.reshape(-1, C, 32*32) @ att).reshape(-1, C, 32, 32)
        x = self.conv4(x)
        x = x.reshape(B, H//32, W//32, C, 32, 32).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
  
        return x


class OPNet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(OPNet, self).__init__()

        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        self.LayerNo = LayerNo
        self.TraBlocks = nn.ModuleList([TraBlock(self.base) for i in range(LayerNo)])
        self.GradBlocks = nn.ModuleList([GradBlock(self.base) for i in range(LayerNo)])
        self.AttBlocks = nn.ModuleList([AttBlock(self.base) for i in range(LayerNo)])

    def forward(self, x):
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x = F.conv2d(y, PhiT, padding=0, bias=None)
        x = nn.PixelShuffle(32)(x)

        x = self.conv1(x)

        for i in range(self.LayerNo):
            x = self.GradBlocks[i](x, y, Phi, PhiT) + x
            x = self.TraBlocks[i](x) + x
            x = self.AttBlocks[i](x) + x

        x = self.conv2(x)

        phi_cons = torch.mm(self.Phi, self.Phi.t()).squeeze().squeeze()

        return x, phi_cons
