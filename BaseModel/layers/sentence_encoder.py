import sys
from os.path import normpath,join,dirname
# 先引入根目录路径，以后的导入将直接使用当前项目的绝对路径
sys.path.append(normpath(join(dirname(__file__), '..')))
import torch.nn as nn
import numpy as np
from BaseModel.layers.network import embedding
from BaseModel.layers.network import encoder
from BaseModel.layers.CNNEncoder import cnnEncoder
from transformers import BertTokenizer, BertModel

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, word2id, max_length, word_embedding_dim=50,
                 pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = embedding.Embedding(word_vec_mat, max_length,
                                             word_embedding_dim, pos_embedding_dim)
        self.encoder = encoder.Encoder(max_length, word_embedding_dim,
                                               pos_embedding_dim, hidden_size)
        self.word2id = word2id

    def forward(self, inputs):
        word_embed = self.embedding(inputs, True)
        x = self.encoder(word_embed)
        return word_embed, x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        pos1_in_index = min(self.max_length, pos_head[0])
        pos2_in_index = min(self.max_length, pos_tail[0])
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

    # add
    def tokenize_simQ(self, raw_tokens):
        # token -> index
        indexed_tokens = []
        for token in raw_tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id['[UNK]'])

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])
        indexed_tokens = indexed_tokens[:self.max_length]

        return indexed_tokens


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, word_embedding_dim, cnn_hidden_size, pos_embedding_dim):
        nn.Module.__init__(self)

        # Load pretrained model/tokenizer
        self.bert = BertModel.from_pretrained(pretrain_path)  # download bert 预训练模型
        self.cnn_encoder = cnnEncoder(max_length, word_embedding_dim, pos_embedding_dim, cnn_hidden_size)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.word_embedding_dim = word_embedding_dim
        self.cnn_hidden_size = cnn_hidden_size
        self.pos_embedding_dim = pos_embedding_dim

        self.drop = nn.Dropout(0.2)

    def forward(self, inputs):
        word_embed, _ = self.bert(inputs["word"], attention_mask=inputs["mask"])
        x = self.drop(word_embed)
        x = self.cnn_encoder(x)
        return word_embed, x

    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, pos1, pos2, mask

