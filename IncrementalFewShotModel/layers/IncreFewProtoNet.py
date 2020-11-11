# coding=utf-8
import sys
from os.path import normpath,join,dirname
sys.path.append(normpath(join(dirname(__file__), '..')))
from torch import nn
import torch
from IncrementalFewShotModel.layers import framework as framework


class IncreProto(framework.IncreFewShotREModel):

    def __init__(self, baseModel, Meta_CNN, topKEmbed, top_K):
        framework.IncreFewShotREModel.__init__(self, Meta_CNN)

        self.baseModel = baseModel
        hidden_size = self.baseModel.prototypes.size(-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_linear_layer = nn.Linear(hidden_size, 1)
        self.novel_linear_layer = nn.Linear(hidden_size, 1)

        # shape = [num_rels, topk, hidden_size]
        self.top_K = top_K
        self.topKEmbed = torch.nn.Parameter(torch.tensor(topKEmbed), requires_grad=False).float()

        self.cost = nn.CrossEntropyLoss()

    def forward(self, novelSupport_set, query_set, query_label, base_label, K, hidden_size, baseN, novelN, Q,
                triplet_base, triplet_novel, triplet_num,
                margin, biQ, triplet_loss_w, is_train):
        """
        :param novelSupport_set: shape=[B*novelN*K, max_length]
        :param query_set: shape=[B*(baseN+novelN)*Q, hidden_size]
        :param query_label: shape=[B*(baseN+novelN)*Q]
        :param base_label: shape=[B*baseN]
        :param K: k shot
        :param hidden_size:
        :param baseN:
        :param novelN:
        :param base_query: shape=[B*baseN*2*Q, hidden_size]
        :param base_query_label: shape =[B*baseN*2*Q]
        :param novel_query: shape=[B*novelN*2*Q, hidden_size]
        :param novel_query_label: shape =[B*novelN*2Q]
        :param is_train: True or False
        :return:
        """
        """
            sentence embedding 
        """
        with torch.no_grad():

            # shape = [B*novelN*K, sen_len, hidden_size]
            novel_support_w, _ = self.baseModel.sentence_encoder(novelSupport_set)
            # shape=[B*(baseN+novelN)*Q, hidden_size]
            both_query_w, both_query = self.baseModel.sentence_encoder(query_set)
            if is_train:
                # shape =[B*baseN*triplet_num, hidden_size]
                triplet_query_base_w, triplet_query_base = self.baseModel.sentence_encoder(triplet_base)
                # shape =[B*novelN*triplet_num, sen_len, hidden_size]
                triplet_query_novel_w, triplet_query_novel = self.baseModel.sentence_encoder(triplet_novel)

        """
            base bi_classifier 
        """
        # base_prototypes.shape=[B, baseN, hidden_size]
        base_prototypes = self.base_prototype_generator(base_label, baseN, hidden_size)
        """
           novel bi_classifier 
        """
        # novel_prototypes.shape=[B, novelN, hidden_size]
        novel_prototypes = self.novel_prototype_generator(novel_support_w, novelN, K, hidden_size)

        """
            多分类
        """
        """
            merge both query 和 meat_query 
        """
        # shape=[B*(baseN+novelN)*Q, hidden_size]
        merge_representation = self.merge_protoNet(both_query_w, both_query, novel_prototypes, base_prototypes, (baseN+novelN)*Q, hidden_size)

        # IncreProtoNet without prototype attention alignment
        # merge_representation = self.merge_protoNet_without_attention(both_query_w, both_query)

        # base model
        # merge_representation = self.merge_protoNet_without_metaCNN(both_query)

        """
            classifier 
        """
        # shape = [B, 1,  baseN+novelN, hidden_size]
        merge_prototypes = torch.cat((base_prototypes, novel_prototypes), 1).unsqueeze(1)
        B = merge_prototypes.size(0)
        # shape=[B, (baseN+novelN)*Q, 1, hidden_size]
        merge_representation = merge_representation.view(-1, (baseN+novelN)*Q, hidden_size).unsqueeze(2)
        # shape=[B, (baseN+novelN)*Q, baseN+novelN]
        logits = torch.pow(merge_representation - merge_prototypes, 2).sum(-1)
        both_preds = torch.argmin(logits, -1)  # 取最短距离
        both_loss = self.softmax_loss(-logits, query_label)

        base_acc, novel_acc, both_acc = self.accuracy(both_preds, query_label, baseN, novelN, B, Q)

        """
            novel_prototype_adapter  
        """
        if is_train:
            merge_base_query = self.merge_protoNet(triplet_query_base_w, triplet_query_base, novel_prototypes, base_prototypes, triplet_num*novelN, hidden_size)
            merge_novel_query = self.merge_protoNet(triplet_query_novel_w, triplet_query_novel, novel_prototypes, base_prototypes, triplet_num*novelN, hidden_size)
            triplet_loss = self.triplet_loss_calc(merge_base_query, merge_novel_query, novel_prototypes, triplet_num, margin, novelN, hidden_size)
            loss = both_loss + triplet_loss_w * triplet_loss
        else:
            loss = both_loss

        return base_acc, novel_acc, both_acc, both_preds, loss, logits

    def merge_protoNet_without_metaCNN(self, both_query):
        return both_query

    def merge_protoNet_without_attention(self, both_query_w, both_query):
        # shape=[B*(baseN+novelN)*Q, hidden_size]
        meta_both_query = self.MetaCNN_Encoder(both_query_w)
        merge_representation = 0.5 * both_query + 0.5 * meta_both_query

        return merge_representation

    def merge_protoNet(self, both_query_w, both_query, novel_prototypes, base_prototypes, totalQ, hidden_size):
        """
            merge both query 和 meat_query
        """
        # shape=[B*(baseN+novelN)*Q, hidden_size]
        meta_both_query = self.MetaCNN_Encoder(both_query_w)
        # shape=[B, (baseN+novelN)*Q, hidden_size], 即每个query都会有一个meta prototype
        meta_merge_prototypes = self.mergePrototypes(meta_both_query, novel_prototypes, totalQ, hidden_size)
        # shape=[B, (baseN+novelN)*Q, hidden_size]
        base_merge_prototypes = self.mergePrototypes(both_query, base_prototypes, totalQ, hidden_size)
        # shape=[B*(baseN+novelN)*Q, 1]
        meta_weight = torch.pow(meta_both_query - meta_merge_prototypes.view(-1, hidden_size), 2).sum(-1).unsqueeze(1)
        # shape=[B*(baseN+novelN)*Q, 1]
        base_weight = torch.pow(both_query - base_merge_prototypes.view(-1, hidden_size), 2).sum(-1).unsqueeze(1)
        # shape=[B*(baseN+novelN)*Q, 2, 1]
        merge_weight = torch.softmax(torch.cat((-base_weight, -meta_weight), 1), -1).unsqueeze(2)
        # shape=[B*(baseN+novelN)*Q, 2, hidden_size]
        merge_query = torch.cat((both_query.unsqueeze(1), meta_both_query.unsqueeze(1)), 1)
        # shape=[B*(baseN+novelN)*Q, hidden_size]
        merge_representation = torch.sum(merge_query * merge_weight, 1)

        return merge_representation

    def mergePrototypes(self, novel_both_query, novel_prototypes, totalQ, hidden_size):
        """
        :param novel_both_query: shape=[B*(baseN+novelN)*Q, hidden_size]
        :param novel_prototypes: shape=[B, novelN, hidden_size]
        :return:
        """
        # totalQ = (baseN+novelN)*Q
        # shape=[B, (baseN+novelN)*Q, 1, hidden_size]
        novel_both_query = novel_both_query.view(-1, totalQ, hidden_size).unsqueeze(2)
        # shape=[B, 1, novelN, hidden_size]
        novel_prototypes = novel_prototypes.unsqueeze(1)
        # shape = [B, (baseN+novelN)*Q, novelN]
        dists = torch.pow(novel_both_query - novel_prototypes, 2).sum(-1)
        # shape = [B, (baseN+novelN)*Q, novelN]
        weights = torch.softmax(-dists, -1)

        # merge prototypes
        # shape=[B, (baseN+novelN)*Q, novelN, hidden_size]
        novel_prototypes = novel_prototypes.expand(-1, totalQ, -1, -1)
        # shape = [B, (baseN+novelN)*Q, novelN, 1]
        weights = weights.unsqueeze(3)
        # shape=[B, (baseN+novelN)*Q, hidden_size]
        merge_prototypes = torch.sum(novel_prototypes * weights, 2)

        return merge_prototypes

    def triplet_loss_calc(self, triplet_query_base, triplet_query_novel, novel_prototypes,
                                triplet_num, margin, novelN, hidden_size):
        """

        :param triplet_query_base: shape =[B*novelN*triplet_num, hidden_size]
        :param triplet_query_novel_w: shape =[B*novelN*triplet_num, sen_len, hidden_size]
        :param novel_prototypes: shape=[B, novelN, hidden_size]
        :param margin:
        :return:
        """
        # 使用meta_cnn编码
        # triplet_query_novel = self.MetaCNN_Encoder(triplet_query_novel)

        # shape = [B, novelN, triplet_num, hidden_size]
        novel_prototypes = novel_prototypes.unsqueeze(2).expand(-1, -1, triplet_num, -1)
        # shape =[B, novelN, triplet_num, hidden_size]
        triplet_query_novel = triplet_query_novel.view(-1, novelN, triplet_num, hidden_size)
        # shape = [B, novelN, triplet_num]
        pos_dist = torch.pow(triplet_query_novel - novel_prototypes, 2).sum(-1)
        # shape =[B, novelN, triplet_num, hidden_size]
        triplet_query_base = triplet_query_base.view(-1, novelN, triplet_num, hidden_size)
        # shape =[B, novelN, triplet_num, hidden_size]
        neg_dist = torch.pow(triplet_query_base - novel_prototypes, 2).sum(-1)

        B = neg_dist.size(0)
        loss_value = (margin + pos_dist - neg_dist).view(-1).unsqueeze(1)
        zero_tensors = torch.zeros(B * novelN * triplet_num, 1).to(self.device)
        loss, _ = torch.max(torch.cat((loss_value, zero_tensors), -1), 1)
        loss = torch.mean(loss, -1)

        return loss

    def novel_prototype_generator(self, novel_support, novelN, K, hidden_size):
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novelN:
        :param K:
        :param hidden_size:
        :return:
        """
        """
            novel prototypes 
        """
        # shape = [B*novelN*K, hidden_size]
        novel_support = self.MetaCNN_Encoder(novel_support)
        # shape = [B*novelN, K, hidden_size]
        novel_support = novel_support.view(-1, K, hidden_size)
        # shape = [B, novelN, hidden_size]
        novel_prototypes = torch.mean(novel_support, 1).view(-1, novelN, hidden_size)

        return novel_prototypes

    def base_prototype_generator(self, base_label, baseN, hidden_size):

        # shape = [B*baseN, topK, hidden_size]
        base_topKEmbed = self.topKEmbed.view(-1, self.top_K, hidden_size)[base_label].to(self.device)
        # shape = [B, baseN, hidden_size]
        base_prototypes = torch.mean(base_topKEmbed, 1).view(-1, baseN, hidden_size)

        return base_prototypes

    def accuracy(self, pred, label, baseN, novelN, B, Q):
        """
        :param pred: shape=[B*(novelN+baseN)*Q]
        :param label: shape=[B*(novelN+baseN)*Q]
        :return:
        """
        # both accuracy
        both_acc = torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

        # base accuracy
        # shape = [B*baseN*Q]
        base_label = label.view(-1, (baseN+novelN)*Q)[:, :(baseN*Q)].reshape(-1)
        base_pred = pred.view(-1, (baseN+novelN)*Q)[:, :(baseN*Q)].reshape(-1)
        base_acc = torch.mean((base_pred == base_label).type(torch.FloatTensor))

        # novel accuracy
        # shape = [B*baseN*Q]
        novel_label = label.view(-1, (baseN + novelN) * Q)[:, (baseN * Q):].reshape(-1)
        novel_pred = pred.view(-1, (baseN + novelN) * Q)[:, (baseN * Q):].reshape(-1)
        novel_acc = torch.mean((novel_pred == novel_label).type(torch.FloatTensor))

        return base_acc, novel_acc, both_acc

    def softmax_loss(self, dist, y):
        logits = dist
        N = dist.size(-1)
        mean_loss = self.cost(logits.view(-1, N), y.view(-1))
        return mean_loss




