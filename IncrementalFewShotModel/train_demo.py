# coding=utf-8
import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
import argparse
import numpy as np
import json
import os
import torch
from utils.path_util import from_project_root
from BaseModel.layers.sentence_encoder import CNNSentenceEncoder
from BaseModel.layers.sentence_encoder import BERTSentenceEncoder
from BaseModel.layers.DeepProtoNet import DeepProto
from IncrementalFewShotModel.layers.train_data_loader import get_loader as train_get_loader
from IncrementalFewShotModel.layers.test_data_loader import get_loader as test_get_loader


from IncrementalFewShotModel.layers.IncreFewProtoNet import IncreProto
from IncrementalFewShotModel.layers.framework import IncreFewShotREFramework
from IncrementalFewShotModel.layers.PrototypeGenerator import base_topk_selector
from IncrementalFewShotModel.layers.MetaCNNEncoder import MetaCNN

def main():

    parser = argparse.ArgumentParser()
    # shared parameter
    parser.add_argument("--baseNumRels", default=54, type=int, help="number of relations")
    parser.add_argument("--embedding_type", default="bert", type=str, help="bert or glove")
    parser.add_argument("--max_length", default=40, type=int, help="max_length")
    parser.add_argument("--cnn_hidden_size", default=230, help="cnn_hidden_size")
    parser.add_argument("--word_embedding_dim", default=768, help="word embedding size")
    parser.add_argument("--pos_embedding_dim", default=5, help="position embedding size")
    parser.add_argument("--pl_weight", default=1e-1, help="the weight for the prototype loss")
    parser.add_argument("--base_lr", default=1e-1, type=float, help="base model learning rate")
    parser.add_argument("--base_train_iter", default=8000, help="base model train iter")
    parser.add_argument("--pretrain_ckpt", default="", help="bert ckpt")

    # DeepProtoNet paramenter
    parser.add_argument("--base_batch_size", default=100, type=int, help="batch_size")

    # FewProtoNet parameter
    parser.add_argument("--train_baseN", default=10, type=int, help="trainN ways in training")
    parser.add_argument("--train_novelN", default=5, type=int, help="trainBaseN ways in training")
    parser.add_argument("--train_Q", default=5, type=int, help="Q number for each relation in training")

    parser.add_argument("--test_baseN", default=54, type=int, help="trainN ways in training")
    parser.add_argument("--test_novelN", default=5, type=int, help="trainBaseN ways in training")
    parser.add_argument("--test_Q", default=5, type=int, help="Q number for each relation in training")

    parser.add_argument("--K", default=5, type=int, help="K shots in testing")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size training for few shot module")
    parser.add_argument("--learn_rate", default=1e-3, type=float, help="learning rate in incremental few-shot learning")
    parser.add_argument("--triplet_num", default=10, type=int, help="triplet number")
    parser.add_argument("--margin", default=20.0, type=float, help="margin distance")
    parser.add_argument("--triplet_w", default=1.0, type=float, help="")

    parser.add_argument("--bi_neg_num", default=20, type=int, help="number of bi_neg_classifier learning")
    parser.add_argument("--bi_pos_num", default=1, type=int, help="number of bi_pos_classifier learning")

    parser.add_argument("--train_iter", default=20000, type=int, help="number of training iterate")
    parser.add_argument("--val_step", default=1000, type=int, help="valuation per val_steps")
    parser.add_argument("--val_iter", default=500, type=int, help="valuation iterates")
    parser.add_argument("--base_top_k", default=100, type=int, help="top k sentence for each base relations")
    parser.add_argument("--re_selected", default=False, type=bool, help="True or False")

    # File Path
    parser.add_argument("--dataset", default="fewrel", help="base pretrained file")
    parser.add_argument("--basemodel_ckpt", default="BaseModel/checkpoint/", help="base pretrained file")
    parser.add_argument("--baserel2index_path", default="data/fewrel/baserel2index.json", help="baseRel2index json path")
    parser.add_argument("--train_data_path", default="data/fewrel/base_train_fewrel.json", help="base training data path")
    parser.add_argument("--test_baseData_path", default="data/fewrel/base_test_fewrel.json", help="base testing data path")
    parser.add_argument("--test_novelData_path", default="data/fewrel/novel_test_fewrel.json", help="novel testing data path")

    opt = parser.parse_args()
    baseRel2index_path = from_project_root(opt.baserel2index_path)
    train_data_path = from_project_root(opt.train_data_path)
    test_baseData_path = from_project_root(opt.test_baseData_path)
    test_novelData_path = from_project_root(opt.test_novelData_path)
    is_simQ = False
    if opt.dataset == "simQ":
        is_simQ = True

    if opt.embedding_type == "glove":
        is_bert = False
        # fewrel
        basemodel_ckpt = opt.basemodel_ckpt+"deepProto-{}-lr-{}-pl-{}.pth.tar".format("glove", opt.base_lr, opt.pl_weight)
    else:
        is_bert = True
        # basemodel_ckpt = opt.basemodel_ckpt + "deepProto-bert-80.pth.tar"
        # fewrel
        basemodel_ckpt = opt.basemodel_ckpt + "deepProto-{}-lr-{}-pl-{}.pth.tar".format("bert", opt.base_lr, opt.pl_weight)

    baseModel_ckpt = from_project_root(basemodel_ckpt)


    """
        输出模型基本信息 
    """
    print("{}-baseN-{}-novelN-{}-shot IncreFew-Shot Relation Classification".format(opt.test_baseN, opt.test_novelN, opt.K))
    print("embedding_type: {}".format(opt.embedding_type))
    print("max_length: {}".format(opt.max_length))

    """
        加载baseModel模型 
    """
    basemodel, sentence_encoder = load_pretrain_model(baseModel_ckpt, opt.embedding_type, opt.max_length, opt.cnn_hidden_size,
                                   opt.word_embedding_dim, opt.pos_embedding_dim, opt.baseNumRels, opt.pl_weight, is_simQ)
    init_prototypes = basemodel.prototypes.cpu().data.numpy()
    print("selected top-{} sentence".format(opt.base_top_k))
    topK_sen_embeds = base_topk_selector(basemodel, sentence_encoder, init_prototypes, opt.base_top_k, train_data_path, baseRel2index_path, opt.re_selected, is_bert, is_simQ)
    print("load top-k base support sentences finished!")

    """
        创建novel_cnn_encoder  
    """
    if opt.embedding_type == "glove":
        if is_simQ:
            meta_cnn_encoder = MetaCNN(opt.max_length, word_embedding_dim=50, pos_embedding_dim=0,
                                       hidden_size=opt.cnn_hidden_size)
        else:
            meta_cnn_encoder = MetaCNN(opt.max_length, word_embedding_dim=50, pos_embedding_dim=5,
                                      hidden_size=opt.cnn_hidden_size)
    else:
        meta_cnn_encoder = MetaCNN(opt.max_length, word_embedding_dim=opt.word_embedding_dim, pos_embedding_dim=0,
                                      hidden_size=opt.cnn_hidden_size)

    """
        生成data_loader
    """

    train_data_loader = train_get_loader(train_data_path, baseRel2index_path, sentence_encoder, opt.batch_size, opt.train_baseN,
                                   opt.train_novelN, opt.K, opt.train_Q, opt.triplet_num, opt.bi_pos_num, opt.bi_neg_num)
    test_data_loader = test_get_loader(test_baseData_path, test_novelData_path, baseRel2index_path, sentence_encoder,
                                   opt.batch_size, opt.test_baseN, opt.test_novelN, opt.K, opt.test_Q, opt.bi_pos_num, opt.bi_neg_num)

    # 建立模型
    model = IncreProto(basemodel, meta_cnn_encoder, topK_sen_embeds, opt.base_top_k)

    framework = IncreFewShotREFramework(train_data_loader, test_data_loader)

    # save model
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format("IncreFewShotProto-{}-K-{}-m-{}-triplet_w-{}-att-{}".format(opt.embedding_type, opt.K, opt.margin, opt.triplet_w, False))

    if torch.cuda.is_available():
        model.cuda()

    """
        模型训练 
    """
    biQ = opt.bi_pos_num + opt.bi_neg_num
    framework.train(model, is_bert, opt.learn_rate, opt.train_iter, opt.val_step, opt.val_iter, ckpt, opt.K,
                    opt.cnn_hidden_size, opt.train_baseN, opt.train_novelN, opt.test_baseN, opt.test_novelN, opt.train_Q,
                    opt.test_Q, opt.triplet_num, opt.margin, biQ, opt.triplet_w)


def load_pretrain_model(base_ckpt, embedding_type, max_length, cnn_hidden_size, word_embedding_dim,
                        pos_embedding_dim, num_rels, pl_weight, is_simQ):
    """
        句子编码
    """
    sentence_encoder = None
    if embedding_type == "glove":
        try:
            glove_mat = np.load(from_project_root("data/pretrain/glove/glove_mat.npy"))
            glove_word2id = json.load(open(from_project_root("data/pretrain/glove/glove_word2id.json")))
        except:
            raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
        if is_simQ:
            sentence_encoder = CNNSentenceEncoder_simQ(
                glove_mat,
                glove_word2id,
                max_length, hidden_size=cnn_hidden_size)
        else:
            sentence_encoder = CNNSentenceEncoder(
                glove_mat,
                glove_word2id,
                max_length, hidden_size=cnn_hidden_size)
    elif embedding_type == "bert":
        pretrain_ckpt = 'bert-base-uncased'
        if is_simQ:
            sentence_encoder = BERTSentenceEncoder_simQ(pretrain_ckpt, max_length, word_embedding_dim, cnn_hidden_size,
                                                   pos_embedding_dim)
        else:
            sentence_encoder = BERTSentenceEncoder(pretrain_ckpt, max_length, word_embedding_dim, cnn_hidden_size, pos_embedding_dim)

    model = DeepProto(sentence_encoder, num_rels, cnn_hidden_size, pl_weight)

    state_dict = __load_model__(base_ckpt)['state_dict']  # 加载预训练好的模型
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)

    return model, sentence_encoder


def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


if __name__ == '__main__':
    main()