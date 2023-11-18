import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, classes=3):
        super(UNet, self).__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        self.double_conv_downs = nn.ModuleList(
            [self._double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2)

        self.upsample = nn.Upsample(scale_factor=2)

        self.up_convs = nn.ModuleList(
            [nn.Conv2d(in_channels=layer, out_channels=layer // 2, kernel_size=3, padding=1) for layer in self.layers[::-1][:-2]])

        self.double_conv_ups = nn.ModuleList(
            [self._double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]])
        
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

    def _double_conv(self, in_f, out_f, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_f),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_f, out_channels=out_f, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            # nn.BatchNorm2d(out_f)
        )

        
    def forward(self, x):

        down_outputs = []
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                down_outputs.append(x)
                x = nn.MaxPool2d(kernel_size=2)(x)

        down_outputs = down_outputs[::-1]

        for skip_con, up_conv, double_conv_up in zip(down_outputs, self.up_convs, self.double_conv_ups):
            x = self.upsample(x)
            x = up_conv(x)
            if x.shape != skip_con.shape:
                x = tf.resize(x, skip_con.shape[2:])
            x = torch.cat((x, skip_con), dim=1)
            x = double_conv_up(x)
            
        x = self.final_conv(x)
        return x