import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import RNNclassifier
from trainer import Trainer
from data_loader import DataLoader


def define_argparse():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--file_path', type=str,
                   default='C:/Users/JCY/3-nlp_basics/05-text_classification/review.sorted.uniq.refined.tok.shuf.train.tsv')

    p.add_argument('--gpu_id', type=int, default= 0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--verbose', type=int, default=2)

    # rnn
    p.add_argument('--emb_dim', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=32)
    p.add_argument('--n_layers', type=int, default=3)

    config = p.parse_args()

    return config


def main(config) :
    # dataloader = DataLoader()
    # train_loaders, val_loaders = dataloader.get_loaders(config)

    dataloader = DataLoader()
    train_loader, valid_loader = dataloader.get_loaders(config)

    print('Train size : {} Valid size : {}'.format(
        len(train_loader.dataset),
        len(valid_loader.dataset),
    ))

    input_size = len(dataloader.text.vocab)
    n_classes = len(dataloader.label.vocab)

    model = RNNclassifier(input_size, config.emb_dim, config.hidden_size, config.n_layers, n_classes, .2)
    loss = nn.NLLLoss() # expect 1d tensor
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(config)
    trainer.train(model, optimizer, loss, train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparse()
    main(config)