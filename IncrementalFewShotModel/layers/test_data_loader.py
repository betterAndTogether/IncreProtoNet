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
    def __init__(self, base_data_file, novel_data_file, baserel2index_file, encoder, baseN, novelN, K, Q, bi_pos_num, bi_neg_num):
        if not os.path.exists(base_data_file):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.base_json_data = json.load(open(base_data_file))
        self.novel_json_data = json.load(open(novel_data_file))
        self.base_classes = list(self.base_json_data.keys())
        self.novel_classes = list(self.novel_json_data.keys())

        self.baseClass2index = json.load(open(baserel2index_file))
        self.encoder = encoder
        self.baseN = baseN
        self.novelN = novelN
        self.K = K
        self.Q = Q
        self.bi_pos_num = bi_pos_num
        self.bi_neg_num = bi_neg_num

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

        novelSupport_set = {'word':[], 'pos1':[], 'pos2':[], 'mask':[]}
        base_label = []  # 不构建支持样本，但构建存在哪些baseRels
        query_set = {'word':[], 'pos1':[], 'pos2':[], 'mask':[]}
        query_label = []
        novel_query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        novel_query_label = []
        base_query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        base_query_label = []

        # 随机抽取baseN个关系作为baseRels
        baseClasses = random.sample(self.base_classes, self.baseN)
        # 随机抽取novelN个关系作为novelRels
        novelClasses = random.sample(self.novel_classes, self.novelN)

        # 可视化固定选取的关系：
        # baseClasses = ["P991", "P6", "P176"]
        # novelClasses = ["P921", "P2094"]

        """
            构造baseRels的query set 
        """
        for i, class_name in enumerate(baseClasses):
            indices = np.random.choice(list(range(len(self.base_json_data[class_name]))), self.Q, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.base_json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(query_set, word, pos1, pos2, mask)

            # 注意： base_label[0]的关系对应的label_index为0，以此类推
            base_label += [self.baseClass2index[class_name]]
            query_label += [i] * self.Q

        """
            构建novelRels Support Set
        """
        startLabelIndex = len(baseClasses)
        for i, class_name in enumerate(novelClasses):

            indices = np.random.choice(list(range(len(self.novel_json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.novel_json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(novelSupport_set, word, pos1, pos2, mask)
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask)
                count += 1

            query_label += [startLabelIndex+i] * self.Q

        """
            base bi classifier query  shape=[baseN, 2Q]
        """
        oveall_classes = []
        oveall_classes.extend(self.base_classes)
        oveall_classes.extend(self.novel_classes)

        for i, class_name in enumerate(baseClasses):
            indices = np.random.choice(list(range(len(self.base_json_data[class_name]))), self.bi_pos_num, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.base_json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(base_query_set, word, pos1, pos2, mask)
            base_query_label += [1] * self.bi_pos_num

            # 选择负类
            noise_classes = []
            for rel in oveall_classes:
                if rel != class_name:
                    noise_classes.append(rel)
            neg_classes = np.random.choice(noise_classes, self.bi_neg_num, replace=True)
            for rel in neg_classes:
                if rel in self.base_classes:
                    indice = np.random.choice(list(range(len(self.base_json_data[rel]))), 1, False)[0]
                    word, pos1, pos2, mask = self.__getraw__(self.base_json_data[rel][indice])
                else:
                    indice = np.random.choice(list(range(len(self.novel_json_data[rel]))), 1, False)[0]
                    word, pos1, pos2, mask = self.__getraw__(self.novel_json_data[rel][indice])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(base_query_set, word, pos1, pos2, mask)
            base_query_label += [0] * self.bi_neg_num

        """
            Novel bi-classifier query shape=[novelN, 2Q]
        """
        for i, class_name in enumerate(novelClasses):
            indices = np.random.choice(list(range(len(self.novel_json_data[class_name]))), self.bi_pos_num, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.novel_json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(novel_query_set, word, pos1, pos2, mask)
            novel_query_label += [1] * self.bi_pos_num

            # 选择负类
            noise_classes = []
            for rel in oveall_classes:
                if rel != class_name:
                    noise_classes.append(rel)
            neg_classes = np.random.choice(noise_classes, self.bi_neg_num, replace=True)
            for rel in neg_classes:
                if rel in self.base_classes:
                    indice = np.random.choice(list(range(len(self.base_json_data[rel]))), 1, False)[0]
                    word, pos1, pos2, mask = self.__getraw__(self.base_json_data[rel][indice])
                else:
                    indice = np.random.choice(list(range(len(self.novel_json_data[rel]))), 1, False)[0]
                    word, pos1, pos2, mask = self.__getraw__(self.novel_json_data[rel][indice])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(novel_query_set, word, pos1, pos2, mask)
            novel_query_label += [0] * self.bi_neg_num

        return novelSupport_set, query_set, query_label, base_label, novel_query_set, novel_query_label, base_query_set, base_query_label

    def __len__(self):
        return 999999999999999


def collate_fn(data):

    batch_novelSupportSet = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_novel_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_base_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_queryLabel = []
    batch_baseLabel = []
    batch_novelQueryLabel = []
    batch_baseQueryLabel = []

    novelSupport_set, query_set, query_label, base_label,novel_query_set, novel_query_label, base_query_set, base_query_label = zip(*data)

    for i in range(len(novelSupport_set)):
        for k in novelSupport_set[i]:
            batch_novelSupportSet[k] += novelSupport_set[i][k]
        for k in query_set[i]:
            batch_query[k] += query_set[i][k]
        for k in novel_query_set[i]:
            batch_novel_query[k] += novel_query_set[i][k]
        for k in base_query_set[i]:
            batch_base_query[k] += base_query_set[i][k]

        batch_queryLabel += query_label[i]
        batch_baseLabel += base_label[i]
        batch_novelQueryLabel += novel_query_label[i]
        batch_baseQueryLabel += base_query_label[i]

    for k in batch_novelSupportSet:
        batch_novelSupportSet[k] = torch.stack(batch_novelSupportSet[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_novel_query:
        batch_novel_query[k] = torch.stack(batch_novel_query[k], 0)
    for k in batch_base_query:
        batch_base_query[k] = torch.stack(batch_base_query[k], 0)

    batch_baseLabel = torch.tensor(batch_baseLabel)
    batch_queryLabel = torch.tensor(batch_queryLabel)
    batch_novelQueryLabel = torch.tensor(batch_novelQueryLabel)
    batch_baseQueryLabel = torch.tensor(batch_baseQueryLabel)

    return batch_novelSupportSet, batch_query, batch_queryLabel, batch_baseLabel, batch_novel_query, \
           batch_novelQueryLabel, batch_base_query, batch_baseQueryLabel


def get_loader(base_data_file, novel_data_file, baserel2index_file, encoder, batch_size,  baseN, novelN, K, Q, bi_pos_num, bi_neg_num, num_workers=4,  collate_fn=collate_fn):

    dataset = FewRelDataset(base_data_file, novel_data_file, baserel2index_file, encoder, baseN, novelN, K, Q, bi_pos_num, bi_neg_num)

    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)

    return iter(data_loader)

