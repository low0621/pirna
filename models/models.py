import torch
from torch import nn

class ppi_model(nn.Module):
    def __init__(self, mode='train'):
        super(ppi_model, self).__init__()
        self.mode = mode
        self.siteConv = ResidualBlock(4, 32, se=True)
        self.piRNAConv = ResidualBlock(4, 32, se=True)
        # self.respiRNA = nn.Sequential(ResidualBlock(16, 32))
        # self.resmRNA = nn.Sequential(ResidualBlock(16, 32))
        # self.pi_pooling = nn.AvgPool1d(7)
        # self.m_pooling = nn.AvgPool1d(6)
        self.encoder_layer_pi = nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0.1)
        self.transformer_encoder_pi = nn.TransformerEncoder(self.encoder_layer_pi, num_layers=1)
        self.encoder_layer_m = nn.TransformerEncoderLayer(d_model=32, nhead=4, dropout=0.1)
        self.transformer_encoder_m = nn.TransformerEncoder(self.encoder_layer_m, num_layers=1)
        self.flatten = nn.Flatten()
        self.dense_pi = nn.Linear(21, 31)
        self.conv = nn.Conv1d(64, 32, 3, padding=1)
        self.dense = nn.Sequential(nn.Linear(992, 128),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.Linear(64, 2)
                                   )
        self.sm = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pirna, mrna, label=None):
        pi_x = pirna.permute(0, 2, 1)
        m_x = mrna.permute(0, 2, 1)

        pi_x = self.piRNAConv(pi_x)
        # pi_x = self.respiRNA(pi_x)
        # pi_x = self.pi_pooling(pi_x)

        m_x = self.siteConv(m_x)
        # m_x = self.resmRNA(m_x)
        # m_x = self.m_pooling(m_x)

        pi_x = pi_x.permute(2, 0, 1)
        m_x = m_x.permute(2, 0, 1)

        trans_pi = self.transformer_encoder_pi(pi_x)
        trans_m = self.transformer_encoder_m(m_x)

        trans_pi = trans_pi.permute(1, 2, 0)
        trans_m = trans_m.permute(1, 2, 0)
        trans_pi = self.dense_pi(trans_pi)

        cat_x = torch.cat((trans_pi, trans_m), 1)
        conv_x = self.conv(cat_x)
        conv_x = conv_x.permute(0, 2, 1)
        flatt_x = self.flatten(conv_x)
        output = self.dense(flatt_x)#cat_x)

        predictions = self.sm(output)

        if self.mode == 'test':
            return predictions

        loss = self.loss_fn(output, label.long())

        return predictions, loss

def resConv(in_channels, out_channels):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, se=False, stride=1):
        super(ResidualBlock, self).__init__()
        self.use_se = se
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels
        self.conv1 = resConv(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resConv(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout()
        self.se = SELayer(out_channels, reduction=4)
        self.conv3 = resConv(in_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        if self.use_se:
            out = self.se(out)
        if self.in_channels != self.out_channels:
            residual = self.conv3(x)
            residual = self.bn3(residual)
        out += residual
        out = self.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avgpool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), x.size(1), 1)

        return x*y.expand_as(x)
