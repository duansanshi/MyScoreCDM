import torch
from torch import nn
import sys
sys.path.append("../../")
from tcn import TemporalConvNet




def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer



class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2, emb_dropout=0.2):
        super(TCN, self).__init__()
        self.encoder = Conv1d_with_init(1, input_size, 1)
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.decoder = nn.Linear(input_size, output_size)
        self.decoder.weight = self.encoder.weight
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        # input has dimension (N, L_in), and emb has dimension (N, L_in, C_in)
        emb = self.drop(self.encoder(x))
        y = self.tcn(emb.transpose(1, 2))
        o = self.decoder(y.transpose(1, 2))
        return o.contiguous()



if __name__ == "__main__":
    tcn = TCN(24, 24, [24,24,24])
    a = torch.randn(13, 1, 24)
    b = tcn(a)
    print(b.shape)