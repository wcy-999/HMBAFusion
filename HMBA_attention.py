import torch
from torch import nn

class channel_attention(nn.Module):

    def __init__(self, in_channel, ratio=4):

        super(channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        b, c, h, w = inputs.shape

        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)

        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        x = x.view([b, c, 1, 1])
        outputs = inputs * x

        return outputs

class spatial_attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)

        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        x = self.conv(x)
        x = self.sigmoid(x)
        outputs = inputs * x

        return outputs


class cbam(nn.Module):

    def __init__(self, in_channel, ratio=4, kernel_size=7):

        super(cbam, self).__init__()

        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    def forward(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)

        return x

