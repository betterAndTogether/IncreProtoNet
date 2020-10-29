import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json



class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """
    def __init__(self, data_file, baserel2index_file, encoder):
        if not os.path.exists(data_file):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(data_file))
        self.classes = list(self.json_data.keys())
        self.class2index = json.load(open(baserel2index_file))
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
            item['h'][2][0],
            item['t'][2][0])
        return word, pos1, pos2, mask 

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):

        # 均匀地去采集数据,每个batch 采集一个relation
        data_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        data_label = []
        # shuffle classes index
        np.random.shuffle(self.classes)
        class_name = np.random.choice(self.classes, 1)[0]
        # for i, class_name in enumerate(self.classes):
        idx = np.random.choice(list(range(len(self.json_data[class_name]))), 1)[0]

        word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][idx])
        word = torch.tensor(word).long()
        pos1 = torch.tensor(pos1).long()
        pos2 = torch.tensor(pos2).long()
        mask = torch.tensor(mask).long()
        self.__additem__(data_set, word, pos1, pos2, mask)
        data_label += [self.class2index[class_name]]

        return data_set, data_label

    def __len__(self):
        return 999999999999999


def collate_fn(data):
    batch_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_label = []

    data_set, data_label = zip(*data)

    for i in range(len(data_set)):
        for k in data_set[i]:
            batch_set[k] += data_set[i][k]
        batch_label += data_label[i]

    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label


def get_loader(data_file, baserel2index_file, encoder, batch_size, num_workers=4,  collate_fn=collate_fn):

    dataset = FewRelDataset(data_file, baserel2index_file, encoder)

    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)

    return iter(data_loader)

