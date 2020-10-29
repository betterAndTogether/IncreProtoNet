# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
import os
import argparse
import numpy as np
import json
import torch
from utils import path_util
from BaseModel.layers.sentence_encoder import CNNSentenceEncoder
from BaseModel.layers.sentence_encoder import BERTSentenceEncoder
from BaseModel.layers.data_loader import get_loader
from BaseModel.layers.DeepProtoNet import DeepProto
from BaseModel.layers.framework import BaseREFramework

def main():

    parser = argparse.ArgumentParser()
    """
        experimental settting
    """
    parser.add_argument("--num_rels", default=54, help="number of relations")
    parser.add_argument("--batch_size", default=100, help="batch_size")
    parser.add_argument("--max_length", default=40, help="max_length")
    parser.add_argument("--word_embedding_dim", default=768, help="word embedding size")
    parser.add_argument("--cnn_hidden_size", default=230, help="cnn_hidden_size")
    parser.add_argument("--pos_embedding_dim", default=5, help="position embedding size")
    parser.add_argument("--pl_weight", default=1e-1, type=float, help="the weight for the prototype loss")
    parser.add_argument("--lr", default=1e-2, type=float, help="initial learning rate")
    parser.add_argument("--train_iter", default=25000, help="iterate steps in training")
    parser.add_argument("--val_step", default=1000, type=int, help="evaluation steps")
    parser.add_argument("--val_iter", default=500, type=int, help="iterate steps in validation")

    parser.add_argument("--embedding_type", default="glove", help="bert or glove")
    parser.add_argument("--pretrain_ckpt",default="", help="bert ckpt")
    # fewrel
    parser.add_argument("--dataset", default="fewrel", help="base data train path")
    parser.add_argument("--base_train_file", default="data/fewrel/base_train_fewrel.json", help="base data train path")
    parser.add_argument("--base_test_file", default="data/fewrel/base_test_fewrel.json", help="base data test path")
    parser.add_argument("--baserel2index_file", default="data/fewrel/baserel2index.json", help="base2index file")

    parser.add_argument('--load_ckpt', default=None, help='load ckpt')
    parser.add_argument('--save_ckpt', default=None, help='save ckpt')

    opt = parser.parse_args()
    batch_size = opt.batch_size
    max_length = opt.max_length
    embedding_type = opt.embedding_type
    train_data_file = path_util.from_project_root(opt.base_train_file)
    test_data_file = path_util.from_project_root(opt.base_test_file)
    baserel2index_file = path_util.from_project_root(opt.baserel2index_file)

    print("embedding_type: {}".format(embedding_type))
    print("max_length: {}".format(max_length))

    sentence_encoder = None
    bert_optim = False

    """
        句子编码 
    """
    if embedding_type == "glove":
        try:
            glove_mat = np.load(path_util.from_project_root("data/pretrain/glove/glove_mat.npy"))
            glove_word2id = json.load(open(path_util.from_project_root("data/pretrain/glove/glove_word2id.json")))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        if opt.dataset == "simQ":
            sentence_encoder = CNNSentenceEncoder_simQ(
                glove_mat,
                glove_word2id,
                max_length, hidden_size=opt.cnn_hidden_size)
        else:
            sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length, hidden_size=opt.cnn_hidden_size)

    elif embedding_type == "bert":
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'
        if opt.dataset == "simQ":
            sentence_encoder = BERTSentenceEncoder_simQ(pretrain_ckpt, max_length, opt.word_embedding_dim,
                                                   opt.cnn_hidden_size,
                                                   opt.pos_embedding_dim)
        else:
            sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, opt.word_embedding_dim, opt.cnn_hidden_size,
                                                   opt.pos_embedding_dim)
        bert_optim = True


    """
        data loader生成
    """
    if opt.dataset == "simQ":
        train_data_loader = get_loader_simQ(train_data_file, baserel2index_file, sentence_encoder, batch_size)
        test_data_loader = get_loader_simQ(test_data_file, baserel2index_file, sentence_encoder, batch_size)
    else:
        train_data_loader = get_loader(train_data_file, baserel2index_file, sentence_encoder, batch_size)
        test_data_loader = get_loader(test_data_file, baserel2index_file, sentence_encoder, batch_size)

    """
        搭建framework
    """
    framework = BaseREFramework(train_data_loader, test_data_loader)

    model = DeepProto(sentence_encoder, opt.num_rels, opt.cnn_hidden_size, opt.pl_weight)

    # save model
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format("deepProto-{}-lr-{}-pl-{}-{}-v3".format(embedding_type, opt.lr, opt.pl_weight, opt.dataset))

    if torch.cuda.is_available():
        model.cuda()

    framework.train(model, bert_optim, opt.lr, opt.train_iter, opt.val_step, opt.val_iter, ckpt)


if __name__ == "__main__":
    main()



