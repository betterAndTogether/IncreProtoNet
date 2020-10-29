# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '../../')))
from utils import json_util
from utils import path_util
import numpy as np
import os


def __relsIndex__(rels):
    rel2index = {}
    for i, rel in enumerate(rels):
        rel2index[rel] = i
    return rel2index


def train_test_split(data_file, train_rel_num, train_file, val_file, baserel2index_file):
    train_json_data = {}
    val_json_data = {}

    json_data = json_util.load(data_file)
    train_rels = np.random.choice(list(json_data.keys()), train_rel_num, False)
    novel_rels = [rel for rel in json_data.keys() if rel not in train_rels]

    for rel in json_data.keys():
        if rel in train_rels:
            train_json_data[rel] = json_data[rel]
        else:
            val_json_data[rel] = json_data[rel]

    # save
    json_util.dump(train_json_data, train_file)
    json_util.dump(val_json_data, val_file)
    # 固定classes_name的顺序
    json_util.dump(__relsIndex__(train_rels), baserel2index_file)
    json_util.dump(__relsIndex__(novel_rels), path_util.from_project_root("data/fewrel/novelrel2index.json"))


    print("train_fewrel rels_nums: {}".format(len(train_json_data.keys())))
    print("val_novel_fewrel rels_nums: {}".format(len(val_json_data.keys())))

    return None


def base_train_test_split(data_file, base_train_file, base_val_file, base_test_file, base_train_rel_num):
    base_train_json_data = {}
    base_val_json_data = {}
    base_test_json_data = {}

    json_data = json_util.load(data_file)
    for rel in json_data.keys():
        instances = json_data[rel]
        base_train_json_data[rel] = instances[:base_train_rel_num]
        base_val_json_data[rel] = instances[base_train_rel_num:base_train_rel_num+base_val_rel_num]
        base_test_json_data[rel] = instances[base_train_rel_num+base_val_rel_num:]

    # save
    json_util.dump(base_train_json_data, base_train_file)
    json_util.dump(base_val_json_data, base_val_file)
    json_util.dump(base_test_json_data, base_test_file)

    print("base_train_rel_num:{}".format(base_train_rel_num))
    print("base_val_rel_num:{}".format(base_val_rel_num))
    print("base_test_rel_num:{}".format(700 - base_train_rel_num - base_val_rel_num))

    return None


if __name__ == '__main__':
    """
        train_wiki.json 划分 54：10， 训练集 和 测试集
    """
    init_train_file = path_util.from_project_root("data/init_data/fewrel/train_wiki.json")
    train_file = path_util.from_project_root("data/fewrel/train_fewrel.json")
    val_file = path_util.from_project_root("data/fewrel/val_novel_fewrel.json")
    baserel2index_file = path_util.from_project_root("data/fewrel/baserel2index.json")
    train_rels_num = 54
    test_rels_num = 10
    train_test_split(init_train_file, train_rels_num, train_file, val_file, baserel2index_file)


    """
        从 train_fewrel.json中，每个关系划分， 550: 50：100， 作为base model的训练集、验证集、测试集
    """
    base_train_file = path_util.from_project_root("data/fewrel/base_train_fewrel.json")
    base_val_file = path_util.from_project_root("data/fewrel/base_val_fewrel.json")
    base_test_file = path_util.from_project_root("data/fewrel/base_test_fewrel.json")

    base_train_rel_num = 550
    base_val_rel_num = 50
    base_test_rel_num = 100
    base_train_test_split(train_file, base_train_file, base_val_file, base_test_file, base_train_rel_num)

    # 删除临时文件 train_fewrel.json"
    os.remove(train_file)
