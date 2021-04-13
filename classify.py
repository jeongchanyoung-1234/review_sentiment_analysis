import sys
import argparse

import torch
from torchtext.legacy import data

from model import RNNclassifier, CNNclassifier


def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--file_path', type=str, required=True)
    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--topk', type=int, default=1)

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    config = p.parse_args()

    return config


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:{}'.format(config.gpu_id)
    )

    rnn_dict = saved_data['rnn']
    cnn_dict = saved_data['cnn']
    train_config = saved_data['config']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    text_field = data.Field(batch_first=True)
    label_field = data.Field(sequential=False,
                             unk_token=None)

    text_field.vocab = vocab
    label_field.vocab = classes

    lines = []
    # for line in sys.stdin:
    #     if line.strip() != '':
    #         lines.append(line.strip().split(' ')[:train_config.max_length])
    with open(config.file_path, 'r', -1, 'utf-8') as f:
        linelist = f.readlines()
        for line in linelist:
            if line.strip() != '' :
                lines.append(line.strip().split(' ')[:train_config.max_length])

    with torch.no_grad():
        ensemble = []
        if rnn_dict != None and not config.drop_rnn:
            model = RNNclassifier(input_size=len(vocab),
                                  emb_dim=train_config.emb_dim,
                                  hidden_size=train_config.hidden_size,
                                  n_layers=train_config.n_layers,
                                  n_classes=len(classes),
                                  dropout=train_config.dropout)
            model.load_state_dict(rnn_dict)
            ensemble.append(model)

        if cnn_dict != None and not config.drop_cnn:
            model = CNNclassifier(input_size=len(vocab),
                                  emb_dim=train_config.emb_dim,
                                  window_sizes=train_config.window_sizes,
                                  n_filters=train_config.n_filters,
                                  use_padding=False,
                                  use_batchnorm=train_config.use_batchnorm,
                                  dropout=train_config.dropout,
                                  n_classes=len(classes))
            model.load_state_dict(cnn_dict)
            ensemble.append(model)

        y_hats = []
        for model in ensemble:
            model.eval()

            y_hat = []
            for i in range(0, len(lines), config.batch_size):
                x = text_field.numericalize(
                    text_field.pad(lines[i:i + config.batch_size]),
                    device = 'cpu' if config.gpu_id == -1 else 'cuda:{}'.format(config.gpu_id)
                )

                y_hat.append(model(x).cpu())
                # y_hat = (bs, class)
            y_hat = torch.cat(y_hat, dim=0)

            y_hats.append(y_hat)
        y_hats = torch.stack(y_hats, dim=0).exp()
        # y_hats = (n_models, bs, class)
        y_hats = torch.mean(y_hats, dim=0)
        # y_hats = (bs, class)

        probs, indices = torch.topk(y_hats, config.topk, dim=-1)

        with open('{}_prediction.txt'.format(config.model_fn[:-4]), 'w', -1, encoding='utf-8') as f:
            for i in range(len(lines)) :
                f.write('{}\t{}\n'.format(
                    ' '.join(classes.itos[indices[i][j]] for j in range(config.topk)),
                    ' '.join(lines[i])
                ))
        # for i in range(len(lines)):
        #     sys.stdout.write('{}\t{}\n'.format(
        #         ' '.join(classes.itos[indices[i][j]] for j in range(config.topk)),
        #         ' '.join(lines[i])
        #     ))


if __name__ == '__main__':
    config = define_argparse()
    main(config)

