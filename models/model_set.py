import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.l1 = nn.Linear(embedding_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.l1(x.type(torch.float32))
        x = F.relu(x)
        x = self.mlp(x)
        opt = torch.sigmoid(x)
        return opt

class LSTMcls(nn.Module):

    def __init__(self, hidden_dim, layers = 1):
        super(LSTMcls, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(1, hidden_dim,layers,bidirectional=True,batch_first=True)
        self.mlp = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        x = torch.unsqueeze(x,-1)
        lstm_out, _ = self.lstm(x)
        last_state = lstm_out[:,-1,:]
        output = self.mlp(last_state)
        output = torch.sigmoid(output)
        return output


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        if stride == 1:
            self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
        else:
            self.projector = nn.Conv1d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, stride=1):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                stride=stride,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, hidden_dims, depth=10, output_dims=16, target=1):
        super(CNN, self).__init__()

        self.emb = nn.Linear(1, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.mlp = nn.Linear(output_dims, target)
        self.proj = nn.Sequential(
            nn.Linear(output_dims, int(output_dims / 2)),
            nn.ReLU(),
            nn.Linear(int(output_dims / 2), 1),
        )
        self.usedMLP = self.mlp

    def replaceMLP(self, target=1):
        self.usedMLP = self.proj

    def forward(self, x):
        x = torch.unsqueeze(x, -1)
        #         print(x.shape)
        x = self.emb(x)
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # B x Ch x T
        x = torch.mean(x, 1)
        #         print(x.shape)
        x = self.usedMLP(x)
        opt = torch.sigmoid(x)
        return opt


class paddingCNN(nn.Module):
    def __init__(self, hidden_dims, depth=10, output_dims=16, target=1, mask=False):
        super(paddingCNN, self).__init__()

        self.emb = nn.Linear(1, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.mask = mask
        self.mlp = nn.Linear(output_dims, target)
        self.proj = nn.Sequential(
            nn.Linear(output_dims, int(output_dims / 2)),
            nn.ReLU(),
            nn.Linear(int(output_dims / 2), 1),
        )
        self.usedMLP = self.mlp

    def replaceMLP(self, target=1):
        self.usedMLP = self.proj

    def forward(self, xm):
        x, mask = xm
        if self.mask:
            x = x * mask  # remove the flipped on the padding

        x = torch.unsqueeze(x, -1)
        #         print('input x',x.shape) B* T * Ch
        #         print('input mask',mask.shape)  B * T
        x = self.emb(x)
        #         print('emb',x.shape) B* T * F
        x = x.transpose(1, 2)
        #         print('trans',x.shape) # B x Ch x T 4096, 128, 184
        x = self.feature_extractor(x)
        #         print('feature',x.shape) # B x Ch x T 4096, 128, 184
        x = x.transpose(1, 2)  # B x T x Ch 4096, 184, 128
        if self.mask:
            unsqz_mask = torch.unsqueeze(mask, -1)  # B*T*1
            x = x * unsqz_mask
            x = torch.sum(x, 1) / torch.sum(unsqz_mask, 1)
        else:
            x = torch.mean(x, 1)
        #         print(x.shape)
        x = self.usedMLP(x)
        opt = torch.sigmoid(x)
        return opt

