import torch
import torch.nn as nn

class WindowAttention2(nn.Module):
    def __init__(self, in_channels, window_size=3):
        super(WindowAttention2, self).__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.query_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.max_pool = nn.MaxPool2d(window_size, stride=1,
                                     padding=self.window_size // 2)

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Compute queries, keys, and values
        queries = self.query_conv(x)
        keys = self.key_conv(x)
        values = self.value_conv(x)

        # Reshape queries and keys to (batch_size, channels, num_windows)
        queries = queries.view(batch_size, channels, -1)

        keys = self.max_pool(keys).view(batch_size, -1, channels).transpose(1, 2)

        # Compute query-key logits and apply softmax along window dimension
        qk_logits = torch.matmul(queries, keys.transpose(1, 2))
        qk_probs = self.softmax(qk_logits)

        # Compute weighted sum of values using query-key probabilities
        values = self.max_pool(values).view(batch_size, channels, -1)
        out = torch.matmul(qk_probs.permute(0, 2, 1), values)

        # Reshape output to (batch_size, channels, height, width)
        out = out.view(batch_size, self.out_channels, height, width)

        # Apply final convolutional layer to output
        out = self.out_conv(out)

        return out