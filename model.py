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