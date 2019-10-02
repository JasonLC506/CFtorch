import numpy as np
import torch
from torch.autograd import Variable
import math

def init_weights(m):
    print m
    if hasattr(m, "weight"):
        print "model getting customized init", m
        stdv = 0.1 / math.sqrt(m.weight.size(1))
        print "stdv", stdv
        m.weight.data.uniform_(-stdv, stdv)
        if hasattr(m, "bias"):
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)


class CP(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(CP, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Embedding(self.n_users, self.k)
        self.v = torch.nn.Embedding(self.n_items, self.k)
        self.wbi = torch.nn.Parameter(torch.randn(self.k, self.n_labels).type(torch.FloatTensor), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(self.n_labels).type(torch.FloatTensor), requires_grad=True)
        self.apply(init_weights)
        
    def forward(self, users, items):
        us = self.u(users)
        vs = self.v(items)
        uv = us * vs
        bi = uv.mm(self.wbi)
        preds = bi + self.bias
        return preds


class TD(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(TD, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Embedding(self.n_users, self.k)
        self.v = torch.nn.Embedding(self.n_items, self.k)
        self.wbi = torch.nn.Bilinear(self.k, self.k, self.n_labels)

    def forward(self, users, items):
        preds = self.wbi(self.u(users), self.v(items))
        return preds


class MultiMF(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(MultiMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Parameter(torch.randn(self.n_users, self.n_labels, self.k).type(torch.FloatTensor), requires_grad=True)
        self.v = torch.nn.Parameter(torch.randn(self.n_items, self.n_labels, self.k).type(torch.FloatTensor), requires_grad=True)
    
    def forward(self, users, items):
        us = self.u[users]
        vs = self.v[items]
        hadamard = us*vs
        preds = hadamard.sum(-1)
        return preds


class MultiMA(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(MultiMA, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Embedding(self.n_users, self.n_labels)
        self.v = torch.nn.Embedding(self.n_items, self.n_labels)
    
    def forward(self, users, items):
        us = self.u(users)
        vs = self.v(items)
        preds = us + vs
        return preds


class NTN(torch.nn.Module):
    """ NTN model """
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(NTN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 2:
            raise ValueError("number of hyperparameters wrong")
        self.k1, self.k2 = hyperparameters
        self.u = torch.nn.Embedding(self.n_users, self.k1)
        self.v = torch.nn.Embedding(self.n_items, self.k1)
        self.wbi = torch.nn.Parameter(torch.randn(self.k1 * self.k1, self.n_labels, self.k2).type(torch.FloatTensor), requires_grad=True)
        self.wc = torch.nn.Parameter(torch.randn(self.k1 + self.k1, self.n_labels, self.k2).type(torch.FloatTensor), requires_grad=True)
        self.b = torch.nn.Parameter(torch.randn(self.n_labels, self.k2).type(torch.FloatTensor), requires_grad=True)
        self.act = torch.nn.Tanh()
        self.w2 = torch.nn.Parameter(torch.randn(self.n_labels, self.k2).type(torch.FloatTensor), requires_grad=True)
     
    def forward(self, users, items):
        us = self.u(users)
        vs = self.v(items)
        uvc = torch.cat((us, vs), 1)
        uvo = (us.unsqueeze(2) * vs.unsqueeze(1)).view(us.size()[0], self.k1 * self.k1)
        hbi = (uvo.mm(self.wbi.view(self.wbi.size()[0], self.n_labels * self.k2))).view(us.size()[0], self.n_labels, self.k2)
        hc = (uvc.mm(self.wc.view(self.wc.size()[0], self.n_labels * self.k2))).view(us.size()[0], self.n_labels, self.k2)
        h = self.act(hbi + hc + self.b)
        preds = (h * self.w2).sum(-1)
        return preds


class BiNNSeparateMLP(torch.nn.Module):
    """ similar to NTN model, difference: with separate multiplicative and additive embeddings;
                                          same weights of first layer for different classes
    """
    ## currently two layers ##
    def __init__(self, n_users, n_items, n_labels, hyperparameters, activation=torch.nn.Tanh):
        super(BiNNSeparateMLP, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 2:
            raise ValueError("number of hyperparameters wrong")
        self.k1, self.k2 = hyperparameters
        self.u = torch.nn.Embedding(self.n_users, self.k1)
        self.v = torch.nn.Embedding(self.n_items, self.k1)
        self.wbi = torch.nn.Bilinear(self.k1, self.k1, self.k2)
        self.ubias = torch.nn.Embedding(self.n_users, self.k2)
        self.vbias = torch.nn.Embedding(self.n_items, self.k2)
        self.act1 = activation()
        self.linear = torch.nn.Linear(self.k2, self.n_labels)
        ### test ###
        # self.apply(init_weights)
        
    def forward(self, users, items):
        preds = self.linear(self.act1(self.wbi(self.u(users), self.v(items)) + self.ubias(users) + self.vbias(items)))
        return preds


class MLP(torch.nn.Module):
    ## currently two layers ##
    def __init__(self, n_users, n_items, n_labels, hyperparameters, activation=torch.nn.Tanh):
        super(MLP, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 2:
            raise ValueError("number of hyperparameters wrong")
        self.k1, self.k2 = hyperparameters
        self.u = torch.nn.Embedding(self.n_users, self.k1)
        self.v = torch.nn.Embedding(self.n_items, self.k1)
        self.linear1u = torch.nn.Linear(self.k1, self.k2)
        self.linear1v = torch.nn.Linear(self.k1, self.k2)
        self.act1 = activation()
        self.linear2 = torch.nn.Linear(self.k2, self.n_labels)
        ### test ###
        self.apply(init_weights)
        
    def forward(self, users, items):
        us = self.u(users)
        vs = self.v(items)
        layer1 = self.linear1u(us) + self.linear1v(vs)
        preds = self.linear2(self.act1(layer1))
        return preds


class BiNNOnlyLinear(torch.nn.Module):
    """ also called Tucker Decompostion with same dimension for user, item mode """
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(BiNNOnlyLinear, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 2:
            raise ValueError("number of hyperparameters wrong")
        self.k1, self.k2 = hyperparameters
        self.u = torch.nn.Embedding(self.n_users, self.k1)
        self.v = torch.nn.Embedding(self.n_items, self.k1)
        self.wbi = torch.nn.Bilinear(self.k1, self.k1, self.k2)
        self.linear = torch.nn.Linear(self.k2, self.n_labels)
        # avoid explosion #
        self.apply(init_weights)
    
    def forward(self, users, items):
        preds = self.linear(self.wbi(self.u(users), self.v(items)))
        return preds


class BiNNSeparateSingle(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(BiNNSeparateSingle, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Embedding(self.n_users, self.k)
        self.v = torch.nn.Embedding(self.n_items, self.k)
        self.wbi = torch.nn.Bilinear(self.k, self.k, self.n_labels)
        self.ubias = torch.nn.Embedding(self.n_users, self.n_labels)
        self.vbias = torch.nn.Embedding(self.n_items, self.n_labels)
    
    def forward(self, users, items):
        preds = self.wbi(self.u(users), self.v(items))
        preds += self.ubias(users)
        preds += self.vbias(items)
        return preds


class BiNNSingle(torch.nn.Module):
    def __init__(self, n_users, n_items, n_labels, hyperparameters):
        super(BiNNSingle, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_labels = n_labels
        if len(hyperparameters) != 1:
            raise ValueError("number of hyperparameters wrong")
        self.k = hyperparameters[0]
        self.u = torch.nn.Embedding(self.n_users, self.k)
        self.v = torch.nn.Embedding(self.n_items, self.k)
        self.wbi = torch.nn.Bilinear(self.k, self.k, self.n_labels)
        self.wlu = torch.nn.Linear(self.k, self.n_labels)
        self.wlv = torch.nn.Linear(self.k, self.n_labels)
        
        self.apply(init_weights)
    
    def forward(self, users, items):
        us = self.u(users)
        vs = self.v(items)
        preds = self.wbi(us, vs) + self.wlu(us) + self.wlv(vs)
        return preds
