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
    def __init__(self, data_file, baserel2index_file, encoder, baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num):
        if not os.path.exists(data_file):
            print("[ERROR] Data file does not exist!")
            assert(0)
        self.json_data = json.load(open(data_file))
        self.classes = list(self.json_data.keys())
        self.baseClass2index = json.load(open(baserel2index_file))
        self.encoder = encoder
        self.baseN = baseN
        self.novelN = novelN
        self.K = K
        self.Q = Q
        self.triplet_num = triplet_num
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

        novelSupport_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        base_label = []  # 不构建支持样本，但构建存在哪些baseRels
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_label = []
        triplet_base_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        triplet_novel_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

        # 随机抽取baseN个关系作为baseRels
        baseClasses = random.sample(self.classes, self.baseN)
        novelOverallClasses = list(filter(lambda x: x not in baseClasses, self.classes))
        # 随机抽取novelN个关系作为novelRels
        novelClasses = random.sample(novelOverallClasses, self.novelN)

        # selectedClasses = []
        # selectedClasses.extend(baseClasses)
        # selectedClasses.extend(novelClasses)

        """
            构造baseRels的query set 
        """
        for i, class_name in enumerate(baseClasses):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.Q, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
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

            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)
            count = 0
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
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
            triplet_query_set
        """
        for i, class_name in enumerate(novelClasses):
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.triplet_num, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(triplet_novel_set, word, pos1, pos2, mask)

            # 选择负类
            novel_neg_classes = []
            for item in novelClasses:
                if item != class_name:
                    novel_neg_classes.append(item)

            neg_classes = []
            neg_classes_sub = np.random.choice(baseClasses, self.triplet_num//2, replace=True)
            neg_classes.extend(neg_classes_sub)
            neg_classes_sub = np.random.choice(novel_neg_classes, self.triplet_num-self.triplet_num//2, replace=True)
            neg_classes.extend(neg_classes_sub)
            for rel in neg_classes:
                indice = np.random.choice(list(range(len(self.json_data[rel]))), 1, False)[0]
                word, pos1, pos2, mask = self.__getraw__(self.json_data[rel][indice])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(triplet_base_set, word, pos1, pos2, mask)

        return novelSupport_set, query_set, query_label, base_label, triplet_base_set, triplet_novel_set

    def __len__(self):
        return 999999999999999


def collate_fn(data):

    batch_novelSupportSet = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_novel_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_queryLabel = []
    batch_base_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_baseLabel = []
    batch_novelQueryLabel= []
    batch_baseQueryLabel = []
    batch_triplet_base = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_triplet_novel = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

    novelSupport_set, query_set, query_label, base_label, triplet_base_set, triplet_novel_set = zip(*data)

    for i in range(len(novelSupport_set)):
        for k in novelSupport_set[i]:
            batch_novelSupportSet[k] += novelSupport_set[i][k]
        for k in query_set[i]:
            batch_query[k] += query_set[i][k]
        for k in triplet_base_set[i]:
            batch_triplet_base[k] += triplet_base_set[i][k]
        for k in triplet_novel_set[i]:
            batch_triplet_novel[k] += triplet_novel_set[i][k]

        batch_queryLabel += query_label[i]
        batch_baseLabel += base_label[i]

    for k in batch_novelSupportSet:
        batch_novelSupportSet[k] = torch.stack(batch_novelSupportSet[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    for k in batch_triplet_base:
        batch_triplet_base[k] = torch.stack(batch_triplet_base[k], 0)
    for k in batch_triplet_novel:
        batch_triplet_novel[k] = torch.stack(batch_triplet_novel[k], 0)

    batch_baseLabel = torch.tensor(batch_baseLabel)
    batch_queryLabel = torch.tensor(batch_queryLabel)

    return batch_novelSupportSet, batch_query, batch_queryLabel, batch_baseLabel, batch_triplet_base, batch_triplet_novel


def get_loader(data_file, baserel2index_file, encoder, batch_size,  baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num, num_workers=4,  collate_fn=collate_fn):

    dataset = FewRelDataset(data_file, baserel2index_file, encoder, baseN, novelN, K, Q, triplet_num, bi_pos_num, bi_neg_num)

    data_loader = data.DataLoader(dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn)

    return iter(data_loader)
