# coding=utf-8
import sys
from torch import nn
import os
import torch
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup

class IncreFewShotREModel(nn.Module):

    def __init__(self, MetaCNN):
        nn.Module.__init__(self)
        self.MetaCNN_Encoder = nn.DataParallel(MetaCNN)

    def forward(self, novelSupport_set, query_set, query_label, base_label, K, hidden_size, baseN,
                novelN, Q, triplet_base, triplet_novel, triplet_num, margin, biQ, triplet_loss_w, is_train):

        raise NotImplementedError


class IncreFewShotREFramework:

    def __init__(self,train_data_loader, test_data_loader):
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self, model, bert_optim, learning_rate, train_iter, val_step, val_iter, save_ckpt, K, hidden_size,
              train_baseN, train_novelN, test_baseN, test_novelN, train_Q, test_Q, triplet_num, margin, biQ, triplet_w,
              warmup_step=300,
              weight_decay=1e-5,
              lr_step_size=20000, pytorch_optim=optim.SGD):

        print("Start training...")

        #  过滤之前冻结的参数，不参与第二阶段的训练
        # if bert_optim:
        #     print('Use bert optim!')
        #     parameters_to_optimize = list(model.named_parameters())
        #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        #     parameters_to_optimize = [
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        #         {'params': [p for n, p in parameters_to_optimize
        #                     if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        #     ]
        #     optimizer = AdamW(parameters_to_optimize, lr=2e-5, correct_bias=False)
        #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
        #                                                 num_training_steps=train_iter)
        # else:
        optimizer = pytorch_optim(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, weight_decay=weight_decay)
        # optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)

        """
            training process  
        """
        model.train()  # 设置模型训练状态

        start_iter = 0
        best_acc = 0.0
        best_novel_acc = 0.0
        best_base_acc = 0.0
        iter_both_acc = 0.0
        iter_base_acc = 0.0
        iter_novel_acc = 0.0
        iter_sample = 0.0
        iter_loss = 0.0
        for it in range(start_iter, start_iter + train_iter):
            novelSupport_set, query_set, query_label, base_label, triplet_base, triplet_novel = next(self.train_data_loader)

            if torch.cuda.is_available():  # 实现gpu训练
                for k in novelSupport_set:
                    novelSupport_set[k] = novelSupport_set[k].cuda()
                for k in query_set:
                    query_set[k] = query_set[k].cuda()
                for k in triplet_base:
                    triplet_base[k] = triplet_base[k].cuda()
                for k in triplet_novel:
                    triplet_novel[k] = triplet_novel[k].cuda()
                query_label = query_label.cuda()
                base_label = base_label.cuda()

            base_acc, novel_acc, both_acc, preds, loss, logits= model(novelSupport_set, query_set, query_label, base_label, K, hidden_size, train_baseN,
                                     train_novelN, train_Q, triplet_base, triplet_novel, triplet_num,
                                                                      margin, biQ, triplet_w, True)

            loss.backward()
            # 优化
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            iter_both_acc += self.item(both_acc.data)
            iter_base_acc += self.item(base_acc.data)
            iter_novel_acc += self.item(novel_acc.data)
            iter_sample += 1
            iter_loss += self.item(loss.data)
            sys.stdout.write(
                'step: {0:4} | loss: {1:2.6f}, base_acc: {2:3.2f}, novel_acc:{3:3.2f}, both_acc: {4:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                           100 * iter_base_acc / iter_sample, 100 * iter_novel_acc / iter_sample, 100 * iter_both_acc / iter_sample,) + '\r')
            sys.stdout.flush()

            # 验证模型
            if (it + 1) % val_step == 0:
                base_acc, novel_acc, both_acc = self.eval(model, val_iter, K, hidden_size, test_baseN, test_novelN, test_Q, biQ)
                model.train()
                # impro_novel_acc = novel_acc - best_novel_acc
                # impro_base_acc = base_acc - best_base_acc
                # impro_both_acc = both_acc - best_acc
                if both_acc > best_acc:
                    print("Best checkpoint")
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = both_acc

                    # if (it+1) <= 400:  # 400为阈值，通过验证集判断模型稳定状态
                    #     print("Best checkpoint")
                    #     torch.save({'state_dict':model.state_dict()}, save_ckpt)
                    #     best_acc = both_acc
                    #     best_novel_acc = novel_acc
                    #     best_base_acc = base_acc
                    # else:
                    #     if impro_novel_acc >= 0:  # 如果novel relation提升
                    #         print("Best checkpoint")
                    #         torch.save({'state_dict':model.state_dict()}, save_ckpt)
                    #         best_acc = both_acc
                    #         best_novel_acc = novel_acc
                    #         best_base_acc = base_acc
                    #     else:
                    #         if impro_both_acc+impro_novel_acc > 0: # both的增益不能以损失novel的增益为代价
                    #             print("Best checkpoint")
                    #             torch.save({'state_dict':model.state_dict()}, save_ckpt)
                    #             best_acc = both_acc
                    #             best_novel_acc = novel_acc
                    #             best_base_acc = base_acc

                iter_sample = 0.0
                iter_both_acc = 0.0
                iter_base_acc = 0.0
                iter_novel_acc = 0.0
                iter_loss = 0.0

    def eval(self, model, val_iter, K, hidden_size, test_baseN, test_novelN, test_Q, biQ):
        print("")

        model.eval()  # 模型设置成验证模式
        model.baseModel.eval()

        iter_sample = 0.0
        iter_both_acc = 0.0
        iter_base_acc = 0.0
        iter_novel_acc = 0.0
        with torch.no_grad():
            for it in range(val_iter):

                novelSupport_set, query_set, query_label, base_label = next(self.test_data_loader)

                if torch.cuda.is_available():  # 实现gpu训练
                    for k in novelSupport_set:
                        novelSupport_set[k] = novelSupport_set[k].cuda()
                    for k in query_set:
                        query_set[k] = query_set[k].cuda()
                    query_label = query_label.cuda()
                    base_label = base_label.cuda()

                base_acc, novel_acc, both_acc, preds, loss, logits = model(novelSupport_set, query_set, query_label, base_label, K, hidden_size, test_baseN,
                                     test_novelN, test_Q, None, None, 0,  0.0, biQ, 0.0, False)

                iter_sample += 1
                iter_both_acc += self.item(both_acc.data)
                iter_base_acc += self.item(base_acc.data)
                iter_novel_acc += self.item(novel_acc.data)
                sys.stdout.write(
                    '[EVAL] step: {0:4} | base_acc:{1:3.2f}, novel_acc: {2:3.2f}, both_acc: {3:3.2f}%'.format(it + 1,
                                                                      100 * iter_base_acc / iter_sample, 100 * iter_novel_acc / iter_sample, 100 * iter_both_acc / iter_sample) + '\r')

                sys.stdout.flush()
            print("")

        return iter_base_acc / iter_sample, iter_novel_acc/iter_sample, iter_both_acc/iter_sample






