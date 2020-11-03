# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
from torch import nn
import torch
from IncrementalFewShotModel.layers import framework as framework

"""
    这里编写自己的模型代码
    该版本知识为了简单实现baseline, novelprototypes只是简单的使用求平均得到
"""


class IncreProto(framework.IncreFewShotREModel):

    def __init__(self, baseModel, Meta_CNN, topKEmbed, top_K):
        framework.IncreFewShotREModel.__init__(self, Meta_CNN)

        """
            冻结baseModel,训练的时候仍然需要过滤此类参数。
        """
        self.baseModel = baseModel
        """
            模型需要使得 
        """
        hidden_size = self.baseModel.prototypes.size(-1)
        base_num_rels = self.baseModel.prototypes.size(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_linear_layer = nn.Linear(hidden_size, 1)
        self.novel_linear_layer = nn.Linear(hidden_size, 1)

        # shape = [num_rels, topk, hidden_size]
        self.top_K = top_K
        self.topKEmbed = torch.nn.Parameter(torch.tensor(topKEmbed), requires_grad=False).float()

        self.base_belta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.base_delta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.base_belta.data.fill_(2.0)
        self.base_delta.data.fill_(1.0)

        self.novel_belta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.novel_delta = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.novel_belta.data.fill_(2.0)
        self.novel_delta.data.fill_(1.0)

        self.cost = nn.CrossEntropyLoss()
        self.mseloss = nn.MSELoss(reduce=True, size_average=True)

    def forward(self, novelSupport_set, query_set, query_label, base_label, K, hidden_size, baseN, novelN, Q,
                novel_query, novel_query_label, base_query, base_query_label, triplet_base, triplet_novel, triplet_num,
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
            # shape = [B*baseN*2*Q, hidden_size]
            _, base_query = self.baseModel.sentence_encoder(base_query)
            # shape = [B*novelN*2*Q, hidden_size]
            novel_query_w, novel_query = self.baseModel.sentence_encoder(novel_query)
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
        # base_radius.shape=[B*baseN], base_prototypes.shape=[B, baseN, hidden_size]
        base_radius, base_prototypes, base_loss, bi_base_acc = self.base_bi_classifier(base_query, base_query_label,
                                                                                    base_label, baseN, biQ, hidden_size)
        """
           novel bi_classifier 
        """
        # novel_radius.shape=[B*novelN], novel_prototypes.shape=[B, novelN, hidden_size]
        novel_radius, novel_prototypes, novel_loss, bi_novel_acc = self.novel_bi_classifier(novel_support_w, novel_query_w, novel_query, base_prototypes,
                                                                                         novel_query_label, novelN, K,
                                                                                         biQ, hidden_size)

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
            adapter_loss = self.novel_prototype_adapter(merge_base_query, merge_novel_query, novel_prototypes, triplet_num, margin, novelN, hidden_size)
            # adapter_loss = self.prototype_loss(triplet_query_novel_w, novel_prototypes, triplet_num, margin, novelN, hidden_size)
            # loss = base_loss + novel_loss + both_loss + 0.1 * adapter_loss
            # loss = novel_loss + both_loss #+ 0.1 * adapter_loss
            loss = both_loss + triplet_loss_w * adapter_loss
            # loss = both_loss
        else:
            # loss = base_loss + novel_loss + both_loss
            # loss = both_loss
            loss = novel_loss + both_loss

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

    def novel_prototype_adapter(self, triplet_query_base, triplet_query_novel, novel_prototypes,
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

    def prototype_loss(self, triplet_query_novel_w, novel_prototypes, triplet_num, margin, novelN, hidden_size):
        """
        :param triplet_query_novel_w: shape =[B*novelN*triplet_num, sen_len, hidden_size]
        :param novel_prototypes: shape=[B, novelN, hidden_size]
        :param triplet_num:
        :param margin:
        :param novelN:
        :param hidden_size:
        :return:
        """
        # 使用meta_cnn编码
        # shape = [B*novelN*tirplet_num, hidden_size]
        triplet_query_novel = self.MetaCNN_Encoder(triplet_query_novel_w)
        # shape=[B*novelN*triplet_num, hidden_size]
        novel_prototypes = novel_prototypes.unsqueeze(2).expand(-1, -1, triplet_num, -1).reshape(-1, hidden_size)
        loss = self.mseloss(triplet_query_novel, novel_prototypes)

        return loss

    def novel_bi_classifier(self, novel_support, novel_query_w, novel_query, base_prototypes, novel_query_label, novelN, K, biQ, hidden_size):
        """
        :param novel_support: shape = [B*novelN*K, sen_len, hidden_size]
        :param novel_query:
        :param novel_query_label:
        :param novelN:
        :param K:
        :param Q:
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

        """
            prototype attention 
            # shape=[B*(baseN+novelN)*Q, hidden_size]
            meta_both_query = self.MetaCNN_Encoder(both_query_w)
            # shape=[B, (baseN+novelN)*Q, hidden_size], 即每个query都会有一个meta prototype
            meta_merge_prototypes = self.mergePrototypes(meta_both_query, novel_prototypes, baseN, novelN, Q, hidden_size)
            # shape=[B, (baseN+novelN)*Q, hidden_size]
            base_merge_prototypes = self.mergePrototypes(both_query, base_prototypes, baseN, novelN, Q, hidden_size)
            # shape=[B*(baseN+novelN)*Q, 1]
            base_weight = torch.pow(meta_both_query - meta_merge_prototypes.view(-1, hidden_size), 2).sum(-1).unsqueeze(1)
            # shape=[B*(baseN+novelN)*Q, 1]
            meta_weight = torch.pow(both_query - base_merge_prototypes.view(-1, hidden_size), 2).sum(-1).unsqueeze(1)
            # shape=[B*(baseN+novelN)*Q, 2, 1]
            merge_weight = torch.softmax(torch.cat((-base_weight, -meta_weight), 1), -1).unsqueeze(2)
            # shape=[B*(baseN+novelN)*Q, 2, hidden_size]
            merge_query = torch.cat((both_query.unsqueeze(1), meta_both_query.unsqueeze(1)), 1)
            # shape=[B*(baseN+novelN)*Q, hidden_size]
            merge_representation = torch.sum(merge_query * merge_weight, 1)
        """
        # # shape=[B*novelN*biQ, hidden_size]
        # meta_novel_query = self.MetaCNN_Encoder(novel_query_w)
        # # shape=[B, novelN*biQ, hidden_size], 即每个query都会有一个meta prototype
        # meta_novel_prototypes = self.mergePrototypes(meta_novel_query, novel_prototypes, novelN*biQ, hidden_size)
        # # shape=[B, novelN*biQ, hidden_size]
        # base_novel_prototypes = self.mergePrototypes(novel_query, base_prototypes, novelN*biQ, hidden_size)
        # # shape=[B*(baseN+novelN)*Q, 1]
        # base_weight = torch.pow(meta_novel_query - meta_novel_prototypes.view(-1, hidden_size), 2).sum(-1).unsqueeze(1)
        # exit()


        """
            bi_classifier 
        """

        # shape = [B, novelN]
        novel_radius = self.novel_radius_measurement(novel_support, novel_prototypes, novelN, K, hidden_size)
        # shape = [B*novelN, biQ, hidden_size]
        novel_prototypes_re = novel_prototypes.view(-1, hidden_size).unsqueeze(1).expand(-1, biQ, -1)
        # shape = [B*novelN, biQ, hidden_size]
        novel_query = novel_query.view(-1, biQ, hidden_size)
        # shape = [B*novelN, biQ, 1]
        novel_dists = torch.pow(novel_query - novel_prototypes_re, 2).sum(-1).unsqueeze(2)
        # shape = [B*novelN, biQ, 1]
        novel_radius_re = novel_radius.unsqueeze(1).expand(-1, biQ).unsqueeze(2)
        # shape = [B*novelN*biQ, 2]
        novel_logits = torch.cat((novel_dists, novel_radius_re), -1).view(-1, 2)
        novel_preds = torch.argmax(novel_logits, -1)
        novel_acc = torch.mean((novel_preds.view(-1) == novel_query_label.view(-1)).type(torch.FloatTensor))
        novel_loss = self.softmax_loss(novel_logits, novel_query_label)

        return novel_radius, novel_prototypes, novel_loss, novel_acc

    def base_bi_classifier(self, base_query, base_query_label, base_label, baseN, biQ, hidden_size):

        # sum_Q = 2*Q

        # shape = [B*baseN, topK, hidden_size]
        base_topKEmbed = self.topKEmbed.view(-1, self.top_K, hidden_size)[base_label].to(self.device)
        # shape = [B, baseN, hidden_size]
        base_prototypes = torch.mean(base_topKEmbed, 1).view(-1, baseN, hidden_size)
        # shape = [B*baseN]
        base_radius = self.base_radius_measurement(base_topKEmbed, base_prototypes, baseN, self.top_K, hidden_size)

        # 计算base_query到base_prototype之间的距离
        # shape =[B*baseN, 2*Q, hidden_size]
        base_prototypes_re = base_prototypes.view(-1, hidden_size).unsqueeze(1).expand(-1, biQ, -1)
        # shape = [B*baseN, 2*Q, hidden_size]
        base_query = base_query.view(-1, biQ, hidden_size)
        # shape = [B*baseN, 2*Q, 1]
        base_dist = torch.pow(base_query - base_prototypes_re, 2).sum(-1).unsqueeze(2)
        # shape = [B*baseN, 2*Q, 1]
        base_radius_re = base_radius.unsqueeze(1).expand(-1, biQ).unsqueeze(2)
        # shape = [B*baseN*2*Q, 2]
        base_logits = torch.cat((base_dist, base_radius_re), -1).view(-1, 2)
        base_preds = torch.argmax(base_logits, -1)
        base_acc = torch.mean((base_preds.view(-1) == base_query_label.view(-1)).type(torch.FloatTensor))
        base_loss = self.softmax_loss(base_logits, base_query_label)

        return base_radius, base_prototypes, base_loss, base_acc

    def novel_radius_measurement(self, novel_support, novel_prototypes, novelN, K, hidden_size):
        """
        :param novel_support: shape = [B*novelN, K, hidden_size]
        :param novel_prototypes: shape = [B, novelN, hidden_size]
        :return:
        """
        # shape = [B*novelN, hidden_size]
        max_value, _ = novel_support.max(1)
        min_value, _ = novel_support.min(1)
        # shape = [B*novelN, hidden_size]
        variable = max_value - min_value
        alpha = self.novel_linear_layer(variable)
        # shape = [B, novelN]
        alpha = torch.tanh(alpha.squeeze(-1)).view(-1, novelN)

        """
            计算prototype到每个shot的距离 
        """
        # shape =[B*novelN, K, hidden_size]
        novel_prototypes = novel_prototypes.view(-1, hidden_size).unsqueeze(1).expand(-1, K, -1)
        # shape = [B*novelN, K]
        dist = torch.pow(novel_support - novel_prototypes, 2).sum(-1)
        # 方式一
        # shape = [B*baseN]
        mean_dist = torch.mean(dist, -1) * self.novel_belta
        wids = mean_dist.view(-1, novelN) * (self.novel_delta + alpha)
        wids = wids.view(-1)

        # 方式二
        # shape = [B*baseN]
        # max_dist, _ = torch.max(dist, -1)
        # wids = max_dist.view(-1, novelN) * (1.0 + alpha)
        # wids = wids.view(-1)

        # print("novel=========")
        # print(wids)
        # exit()

        return wids

    def base_radius_measurement(self, base_topKEmbed, base_prototypes, baseN, topK, hidden_size):
        """
        :param base_topKEmbed: shape = [B*baseN, topK, hidden_size]
        :param base_prototypes: shape = [B, baseN, hidden_size]
        :param baseN:
        :param topK:
        :param hidden_size:
        :return:
        """
        """
            捕捉每个关系的特征值浮动变化 
        """
        # shape=[B*baseN, hidden_size]
        max_value, _ = base_topKEmbed.max(1)
        min_value, _ = base_topKEmbed.min(1)
        # shape = [B*baseN, hidden_size]
        variable = max_value - min_value
        alpha = self.base_linear_layer(variable)
        alpha = torch.tanh(alpha.squeeze(-1)).view(-1, baseN)
        """
            计算prototype到每个shot的距离 
        """
        # shape =[B*baseN, topK, hidden_size]
        base_prototypes = base_prototypes.view(-1, hidden_size).unsqueeze(1).expand(-1, topK, -1)
        # shape = [B*baseN, topK]
        dist = torch.pow(base_topKEmbed - base_prototypes, 2).sum(-1)
        # shape = [B*baseN]
        mean_dist = torch.mean(dist, -1) * self.base_belta
        wids = mean_dist.view(-1, baseN) * (self.base_delta + alpha)
        wids = wids.view(-1)
        # 方式二
        # shape = [B*baseN]
        # max_dist, _ = torch.max(dist, -1)
        # wids = max_dist.view(-1, baseN) * (1.0 + alpha)
        # wids = wids.view(-1)
        return wids

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




