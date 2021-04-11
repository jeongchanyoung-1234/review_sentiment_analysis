from torchtext.legacy import data

class DataLoader(object):
    def __init__(self,
                 max_vocab=9999,
                 min_freq=1,
                 init_token='<bos>',
                 eos_token='<eos>'):
        super().__init__()

        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.label = data.Field(sequential=False,
                                unk_token=None)
        self.text = data.Field(init_token=init_token,
                               eos_token=eos_token,
                               batch_first=True,
                               )

    def get_loaders(self, config):
        train, valid = data.TabularDataset(
            path=config.file_path,
            format='tsv',
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=config.train_ratio)

        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=config.batch_size,
            device= 'cuda:{}'.format(config.gpu_id) if config.gpu_id >= 0 else 'cpu',
            shuffle=True,
            sort_key = lambda x: len(x.text),
            sort_within_batch = True
        )

        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=self.max_vocab, min_freq=self.min_freq)

        return self.train_loader, self.valid_loader