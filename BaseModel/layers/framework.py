# coding=utf-8
import sys
import os
import torch
from torch import optim
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup


class BaseREModel(nn.Module):

    def __init__(self, sentence_encoder):
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        # self.sentence_encoder = sentence_encoder

    def forward(self, x, y):

        raise NotImplementedError


class BaseREFramework:

    def __init__(self, train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        """
        :param ckpt: path of the checkpoint
        :return: checkpoint dict
        """
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self, model, bert_optim, learning_rate, train_iter, val_step, val_iter, save_ckpt, warmup_step=300, weight_decay=1e-5,
              lr_step_size=20000, pytorch_optim=optim.SGD):

        print("Start training...")

        # Init
        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [
                {'params': [p for n, p in parameters_to_optimize
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                        num_training_steps=train_iter)
        else:
            optimizer = pytorch_optim(model.parameters(),
                                      learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        """
            training process  
        """
        model.train()

        start_iter = 0
        best_acc = 0.0
        for it in range(start_iter, start_iter+train_iter):
            data_set, data_label = next(self.train_data_loader)

            if torch.cuda.is_available():  # 实现gpu训练
                for k in data_set:
                    data_set[k] = data_set[k].cuda()
                data_label = data_label.cuda()

            preds, acc, loss = model(data_set, data_label)

            loss.backward()
            # 优化
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            sys.stdout.write(
                'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, loss,
                                                                           100 * acc) + '\r')
            sys.stdout.flush()

            # 验证模型
            if (it + 1) % val_step == 0:
                acc = self.eval(model, val_iter)
                model.train()
                if acc > best_acc:
                    print("Best checkpoint")
                    torch.save({'state_dict':model.state_dict()}, save_ckpt)
                    best_acc = acc

    def eval(self, model, val_iter, ckpt=None):
        print("")

        model.eval()  # 模型设置成验证模式

        if ckpt is None:
            test_dataset = self.test_data_loader
        else: # 加载训练好的模型
            state_dict = self.__load_model__(ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                own_state[name].copy_(param)
            test_dataset = self.test_data_loader

        iter_sample = 0.0
        iter_overall_acc = 0.0
        with torch.no_grad():
            for it in range(val_iter):
                data_set, data_label = next(test_dataset)

                if torch.cuda.is_available():
                    for k in data_set:
                        data_set[k] = data_set[k].cuda()
                    data_label = data_label.cuda()

                preds, iter_acc, loss = model(data_set, data_label)

                iter_sample += 1
                iter_overall_acc += self.item(iter_acc.data)
                sys.stdout.write(
                    '[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_overall_acc / iter_sample) + '\r')

                sys.stdout.flush()
            print("")

        return iter_overall_acc / iter_sample











