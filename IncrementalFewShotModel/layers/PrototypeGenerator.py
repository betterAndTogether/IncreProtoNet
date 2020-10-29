# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
import os
import numpy as np
import json
import torch
from  utils.path_util import from_project_root
from utils import json_util


def base_topk_selector(basemodel, sentence_encoder, init_prototypes, top_k, train_baseData_path, baserel2index_file, re_selected, is_bert, is_simQ):

    """
        获取所有每个relation所有的sentence 表示，并获取前top K 代表性的样本。并生成 prototypes
    """
    if (os.path.isfile(from_project_root("BaseModel/checkpoint/topKEmbed-bert-{}.txt").format(top_k)) or \
            os.path.isfile(from_project_root("BaseModel/checkpoint/topKEmbed-glove-{}.txt").format(top_k))) and not re_selected:
        if is_bert:
            prototypes = np.loadtxt(from_project_root("BaseModel/checkpoint/topKEmbed-bert-{}.txt".format(top_k)))
        else:
            prototypes = np.loadtxt(from_project_root("BaseModel/checkpoint/topKEmbed-glove-{}.txt".format(top_k)))
    else:
        base_json_data = json.load(open(train_baseData_path))
        baseClass2index = json.load(open(baserel2index_file))

        if is_simQ:
            # shape=[num_rels*top_k, hidden_size]
            prototypes = generate_prototype_simQ(base_json_data, baseClass2index, basemodel, sentence_encoder,
                                            init_prototypes, top_k)
        else:
            # shape=[num_rels*top_k, hidden_size]
            prototypes = generate_prototype(base_json_data, baseClass2index, basemodel, sentence_encoder, init_prototypes, top_k)
        if is_bert:
            np.savetxt(from_project_root("BaseModel/checkpoint/topKEmbed-bert-{}.txt".format(top_k)), np.array(prototypes))
        else:
            np.savetxt(from_project_root("BaseModel/checkpoint/topKEmbed-glove-{}.txt".format(top_k)),
                       np.array(prototypes))

    return prototypes


def generate_prototype(base_json_data, baseClass2index, basemodel, encoder, init_prototypes, top_k):
    """

    :param base_json_data:
    :param baseClass2index:
    :param basemodel:
    :param encoder:
    :param init_prototypes:
    :param topK:
    :return:
    """
    index2BaseClass = {index:rel for (rel, index) in baseClass2index.items()}
    relations = base_json_data.keys()
    topk_sen_embed = []

    for i in range(len(relations)):
        rel = index2BaseClass[i]
        prototype = init_prototypes[baseClass2index[rel]]
        sen_embeds = []
        dists = []
        for i, ins in enumerate(base_json_data[rel]):
            batch_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}

            word, pos1, pos2, mask = __getraw__(ins, encoder)
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            __additem__(batch_set, word, pos1, pos2, mask)
            for k in batch_set:
                batch_set[k] = torch.stack(batch_set[k], 0)

            word_embed, sentence_embed = basemodel.sentence_encoder(batch_set)
            sentence_embed = sentence_embed.cpu().data.numpy()
            dis = __dis__(sentence_embed, prototype)

            sen_embeds.extend(sentence_embed)
            dists.append(dis)

        #
        dists = np.array(dists)
        top_k_idx = dists.argsort()[::-1][-top_k:]
        strong_embeds = np.array(sen_embeds)[top_k_idx]  # shape=[10, hidden_size]
        topk_sen_embed.extend(strong_embeds)
        # print(topk_sen_embed)

    return topk_sen_embed


def generate_prototype_simQ(base_json_data, baseClass2index, basemodel, encoder, init_prototypes, top_k):
    """

    :param base_json_data:
    :param baseClass2index:
    :param basemodel:
    :param encoder:
    :param init_prototypes:
    :param topK:
    :return:
    """
    index2BaseClass = {index:rel for (rel, index) in baseClass2index.items()}
    relations = base_json_data.keys()
    topk_sen_embed = []

    for i in range(len(relations)):
        rel = index2BaseClass[i]
        prototype = init_prototypes[baseClass2index[rel]]
        sen_embeds = []
        dists = []
        for i, ins in enumerate(base_json_data[rel]):
            batch_set = {'word': []}

            word= __getraw__(ins, encoder)
            word = torch.tensor(word).long()
            __additem_simQ__(batch_set, word)
            for k in batch_set:
                batch_set[k] = torch.stack(batch_set[k], 0)

            word_embed, sentence_embed = basemodel.sentence_encoder(batch_set)
            sentence_embed = sentence_embed.cpu().data.numpy()
            dis = __dis__(sentence_embed, prototype)

            sen_embeds.extend(sentence_embed)
            dists.append(dis)

        #
        dists = np.array(dists)
        top_k_idx = dists.argsort()[::-1][-top_k:]
        strong_embeds = np.array(sen_embeds)[top_k_idx]  # shape=[10, hidden_size]
        topk_sen_embed.extend(strong_embeds)
        # print(topk_sen_embed)

    return topk_sen_embed


def __dis__(vec1, vec2):
    """

    :param embed1:
    :param embed2:
    :return:
    """
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


def __additem__(d, word, pos1, pos2, mask):
    d['word'].append(word)
    d["pos1"].append(pos1)
    d["pos2"].append(pos2)
    d["mask"].append(mask)

def __additem_simQ__(d, word):
    d['word'].append(word)


# def __getraw__(item, encoder):
#     word = encoder.tokenize(item['tokens'], item["pos1"], item["pos2"])
#     return word

def __getraw__(item, encoder):
    word, pos1, pos2, mask = encoder.tokenize(item['tokens'],
        item['h'][2][0],
        item['t'][2][0])
    return word, pos1, pos2, mask
