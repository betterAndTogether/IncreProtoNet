# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
from torch import nn
import torch
from BaseModel.layers import framework

"""
    这里编写自己的模型代码
"""


class DeepProto(framework.BaseREModel):

    def __init__(self, sentence_encoder, num_rels, hidden_size, pl_weight):
        framework.BaseREModel.__init__(self, sentence_encoder)

        self.num_rels = num_rels
        self.hidden_size = hidden_size
        # 初始化prototype
        self.prototypes = torch.nn.Parameter(torch.zeros(self.num_rels, self.hidden_size), requires_grad=True)
        self.cost = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss(reduce=True, size_average=True)
        self.pl_weight = pl_weight

    def forward(self, x, y):
        """
        :param x: {"word":[index..], "pos1":[], "pos2":[], "mask":[]}
        :param y: shape=[B*num_rels]
        :return:
        """
        # shape = [B, hidden_size]
        word_embed, sen_embed = self.sentence_encoder(x)
        # shape= [B, num_rels]
        dist = self.__distance__(sen_embed, self.prototypes)
        preds = torch.argmin(dist, 1)

        # exit()
        acc = self.accuracy(preds, y)
        softmax_loss = self.softmax_loss(dist, y)
        prototype_loss = self.pl_loss(sen_embed, y, self.prototypes)
        loss = softmax_loss + self.pl_weight * prototype_loss
        return preds, acc, loss

    def softmax_loss(self, dist, y):
        logits = -dist
        N = dist.size(-1)
        mean_loss = self.cost(logits.view(-1, N), y.view(-1))
        return mean_loss

    def pl_loss(self, sen_embed, label, prototypes):
        batch_num = torch.tensor(sen_embed.size(0)).float()
        batch_prototypes = prototypes[label]  # shape=[B, hidden_size]
        loss = self.mseloss(sen_embed, batch_prototypes)
        return loss

    def accuracy(self, pred, label):
        """
        :param pred:
        :param label:
        :return:
        """
        acc = torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))
        return acc

    def __distance__(self, sen_embed, prototypes):
        """
        计算每个sen_embed到所有的prototye的距离
        :param sen_embed: shape=[B, hidden_size]
        :param prototypes: shape=[num_rels, hidden_size]
        :return: shape= [B, num_rels]
        """
        f_2 = torch.sum(torch.pow(sen_embed, 2), 1, keepdim=True)  # shape= [B*num_rels, 1]
        c_2 = torch.sum(torch.pow(prototypes, 2), 1, keepdim=True) # shape= [num_rels, 1]
        prototypes = torch.transpose(prototypes, 1, 0)  # shape= [hidden_size, num_rels]
        # shape= [B*num_rels, num_rels]
        dist = f_2 - 2 * torch.matmul(sen_embed, prototypes) + torch.transpose(c_2, 1, 0)

        # num_rels = prototypes.size(0)
        # B = sen_embed.size(0)
        # sen_embed = sen_embed.unsqueeze(1).expand(-1, num_rels, -1)  # shape=[B, num_rels, hidden_size]
        # prototypes = prototypes.unsqueeze(0).expand(B, -1, -1)  # shape=[B, num_rels, hidden_size]
        # dist = torch.pow(sen_embed - prototypes, 2).sum(-1)
        # print(dist)
        # print(dist.shape)
        # exit()
        return dist









