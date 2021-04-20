import torch.nn as nn
from torchvision.models import resnet18

resnet = resnet18(pretrained=True)

class CRNN(nn.Module):
    def __init__(self, num_chars, rnn_hidden_size=256, dropout=0.1):
        super(CRNN, self).__init__()
        self.num_chars = num_chars
        self.rnn_hidden_size = rnn_hidden_size
        self.dropout = dropout
        
        # Get resnet18 without the last 3 layers
        resnet_modules = list(resnet.children())[:-3]
        self.cnn_p1 = nn.Sequential(*resnet_modules)
        
        # Add custom layers
        self.cnn_p2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3,6),
                stride=1,
                padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Linear(1024, 256)
        
        # RNN
        self.rnn1 = nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.rnn2 = nn.GRU(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.linear2 = nn.Linear(self.rnn_hidden_size*2, num_chars)
        
    def forward(self, x):
        x = self.cnn_p1(x)
        x = self.cnn_p2(x)
        x = x.permute(0,3,1,2)

        batch_size = x.size(0)
        T = x.size(1)
        x = x.view(batch_size, T, -1)
        x = self.linear1(x)

        x, hidden = self.rnn1(x)
        feature_size = x.size(2)
        x = x[:, :, :feature_size//2] + x[:, :, feature_size//2:]

        x, hidden = self.rnn2(x)
        x = self.linear2(x)
        x = x.permute(1,0,2)

        return x