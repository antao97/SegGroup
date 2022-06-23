'''

Copyed from util.py of DGCNN: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py

'''

import numpy as np
import torch
import torch.nn.functional as F


def cross_entropy_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum()
    else:
        loss = F.cross_entropy(pred, gold, reduction='sum')

    return loss


def square_loss(euclidean_distance):
    """
    Square loss function.
    """
    loss = torch.sum(torch.pow(euclidean_distance, 2))

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
