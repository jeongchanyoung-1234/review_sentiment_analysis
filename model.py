import torch
import torch.nn as nn


class RNNclassifier(nn.Module):
    def __init__(self, input_size, emb_dim, hidden_size, n_layers, n_classes, dropout):
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_classes = n_classes

        super().__init__()

        # x = (bs, length)
        self.emb = nn.Embedding(input_size, emb_dim)
        # x = (bs, length, dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        # x = (bs, length, hidden * 2)
        self.generator = nn.Linear(in_features=hidden_size * 2,
                                   out_features=n_classes)
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x = (bs, length)
        x1 = self.emb(x)
        # x = (bs, length, dim)
        x2, _ = self.rnn(x1)
        # x = (bs, length, hidden * 2)
        y = self.activation(self.generator(x2[:, -1]))

        return y

class CNNclassifier(nn.Module):
    def __init__(self,
                 input_size,
                 emb_dim,
                 window_sizes,
                 n_filters,
                 use_padding,
                 use_batchnorm,
                 dropout,
                 n_classes):
        # emb
        self.input_size = input_size
        self.emb_dim = emb_dim
        # cnn module
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        self.use_padding = use_padding
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        # generator
        self.n_classes = n_classes
        super().__init__()

        # x = (bs, length)
        self.emb = nn.Embedding(input_size, emb_dim)
        # x = (bs, length, emb)
        self.cnnmodule = nn.ModuleList()

        for window_size in window_sizes:
            self.cnnmodule.append(nn.Sequential(
                    nn.Conv2d(in_channels=1,
                              out_channels=n_filters,
                              kernel_size=(window_size, emb_dim)),
                              # padding=(1, 1) if use_padding else 0),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filters) if use_batchnorm else nn.Dropout(dropout),
                )
            )
        # x = (bs, n_filters, length -  window_size + 1, 1)
        self.generator = nn.Linear(in_features=n_filters * len(window_sizes),
                                   out_features=n_classes)
        self.activation = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        # x = (bs, length)
        x = self.emb(x)
        # x = (bs, length, emb)
        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            pad = x.new(x.size(0), min_length - x.size(1), self.emb_dim).zero_()
            x = torch.cat([x, pad], dim=1)

        x = x.unsqueeze(dim=1)
        # x = (bs, 1, length, emb)

        cnn_outs = []
        for block in self.cnnmodule:
            cnn_out = block(x)
            # x = (bs, n_filters, length - window_size + 1, 1)
            cnn_out = nn.functional.max_pool1d(input=cnn_out.squeeze(dim=-1),
                                               kernel_size=cnn_out.size(-2)).squeeze(dim=-1)
            # x = (bs, n_filters)
            cnn_outs += [cnn_out]

        x = torch.cat(cnn_outs, dim=-1)
        # x = (bs, sum(n_filters))
        x = self.generator(x)
        y = self.activation(x)

        return y