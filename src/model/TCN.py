from torch.nn.utils import weight_norm
import torch.nn as nn

class Chomp1d(nn.Module):
    """Ensures causal convolution - no information leakage from future to past
    i.e. output at time t in convolved only with elements from time t and earlier in 
    the previous layer

    Ref: https://dida.do/blog/temporal-convolutional-networks-for-sequence-modeling
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride, padding, dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride, padding, dilation))
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp, self.relu, self.dropout,
            self.conv2, self.chomp, self.relu, self.dropout,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
    

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            tb = TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size, dropout=dropout
            )
            layers.append(tb)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, output_size, kernel_size=2, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # Adjust dimensions to (batch_size, num_channels, sequence_length)
        x = x.transpose(1, 2)
        y = self.tcn(x)

        # Apply the linear layer to each time step
        y = self.linear(y.transpose(1, 2))
        return y
