import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from advertorch import oplib
from advertorch.utils import clamp
from advertorch.oplib import getMNISTop
import random

class ZeroOneLoss(_Loss):
    """Zero-One Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean'):
        super(ZeroOneLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return logit_margin_loss(input, target, reduction=self.reduction)

class LogitMarginLoss(_Loss):
    """Logit Margin Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean', offset=0.):
        super(LogitMarginLoss, self).__init__(size_average, reduce, reduction)
        self.offset = offset

    def forward(self, input, target):
        return logit_margin_loss(
            input, target, reduction=self.reduction, offset=self.offset)

class CWLoss(_Loss):
    """CW Loss"""
    # TODO: combine with the CWLoss in advertorch.utils

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean'):
        super(CWLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return cw_loss(input, target, reduction=self.reduction)

class SoftLogitMarginLoss(_Loss):
    """Soft Logit Margin Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean', offset=0.):
        super(SoftLogitMarginLoss, self).__init__(
            size_average, reduce, reduction)
        self.offset = offset

    def forward(self, logits, targets):
        return soft_logit_margin_loss(
            logits, targets, reduction=self.reduction, offset=self.offset)

'''
A class which create a loss graph.
Parameter:
 - Z: input logits number 
 - M: Number of T2T/v2v after Logits or vector 
 - N: Number of v2v after 2v2v(m) 
Abbreviation: T2T:T ; T2v:t ; 2v2v:m ; v2v:v
Example :
 - ------------------------------------------------------------------------
 - | if K = 4 M = 2 N = 1, then the structure of Loss may looks like:      |
 - | Logits->T2T(1)->T2T(5)->T2v(09)->v2v->v2v->\                          |
 - | Logits->T2T(2)->T2T(6)->T2v(10)->v2v->v2v->2v2v->v2v->\               |
 - | Logits->T2T(3)->T2T(7)->T2v(11)->v2v->v2v->\          |               |
 - | Logits->T2T(4)->T2T(8)->T2v(12)->v2v->v2v->2v2v->v2v->2v2v->v2v->LOSS |
 - -------------------------------------------------------------------------
'''

class OpNode():
    middle_op = ['T','v','t']
    merge_op = 'm'
    special_op = ['leaf','loss']
    def __init__(self, type=None, op=None, lchild=None, rchild=None):
        self.type = type
        self.op,self.lchild,self.rchild = op,lchild,rchild
        self.name = type if type in self.special_op else op.__name__

    def __str__(self) -> str:
        intro = "type:{};name:{};".format(self.type,self.name)
        if self.result:intro+="result:{};".format(self.result)
        if self.exp:intro+="exp:{}".format(self.exp)
        return intro

    # NOTE : If op type is T\v\t, than it only has lchild
    def forward(self, logits, label, reduction='elementwise_mean'):
        if self.type in self.middle_op:
            input = self.lchild.forward(logits,label)
            self.result = self.op(input,label) if self.type=='t' else self.op(input)
        if self.type==self.merge_op:
            x,y = self.lchild.forward(logits,label),self.rchild.forward(logits,label)
            self.result = self.op(x,y)
        if self.type=='leaf':
            self.result = logits#.clone().detach()
        if self.type=='loss':
            self.result = _reduce_loss(self.lchild.forward(logits,label), reduction)
        return self.result
        
    def getExp(self):# TestOK
        if self.type in self.middle_op:
            self.exp = self.name + "(" + self.lchild.getExp() +")"
        elif self.type == self.merge_op:
            self.exp = self.name + "[" + self.lchild.getExp() + " , " + self.rchild.getExp() + "]"
        elif self.type=='leaf':
            self.exp = 'Z'
        elif self.type=='loss':
            self.exp = "loss = " + self.lchild.getExp()
        else:
            raise ValueError("type is not right")
        return self.exp

# Assume that all op need not keep requires_grad same as input tensor/vector
# 反转了 不用Assume,所有op都需要可导
# 反转反转了 貌似不是op的问题 是我的loss输出没有requires grad
class CompositeLoss(_Loss):
    def __init__(self, T2Tlist:list, T2vlist:list, v2vlist:list, tv2vlist:list ,K:int =2, M:int =1, N:int=1,
                size_average=None, reduce=None,reduction='elementwise_mean'):
        super(CompositeLoss, self).__init__(size_average, reduce, reduction)
        assert K>=2 and M>=0 and N>=0
        self.K, self.M, self.N, self.m_num= K, M, N, max(K-1,0)
        self.op = {'T':T2Tlist,'t':T2vlist,'v':v2vlist,'m':tv2vlist}
        self.OPORDER = 'T'*self.K*self.M + self.K*'t' + 'v'*self.K*self.M + self.m_num*('m'+self.N*'v')
        self.loss = None

    # TODO  Seems not working ,plz check out
    def _checkLegal(self,loss:str): # Use a str to represent loss.
        self.op_num = len(loss)//3
        for i in range(self.op_num):
            if int(loss[i*3+1:i*3+3]) >= len(self.op[loss[i*3]]):
                print("{} is wrong expression".format(loss[i*3:(i+1)*3]))
                return False
        return True

    def getLoss(self,loss):
        assert self._checkLegal(loss)
        self.branch = [[OpNode('leaf')] for _ in range(self.K)]
        # by this way ,you can actually Transfer to vector immediately after leaf,as the code here is unlimited.
        for i in range(self.K *(2*self.M+1)):
            type = loss[i*3]
            op = self.op[type][int(loss[i*3+1:i*3+3])]
            node = OpNode(type,op,self.branch[i%self.K][-1],None)
            self.branch[i%self.K].append(node)
        nodes2bmerge = [self.branch[k][-1] for k in range(self.K)]
        layer_node_num, leftnode = self.K, False
        while(layer_node_num!=1 or leftnode!=False):
            if not leftnode: # no node left to merge
                layer_node_num,leftnode = layer_node_num//2,False if layer_node_num%2==0 else True
            elif leftnode:
                layer_node_num,leftnode = layer_node_num//2+1 if layer_node_num%2 else layer_node_num//2,layer_node_num%2==0
            next_nodes2bmerge = []
            for j in range(layer_node_num):# product one layer with merge operations
                merge_node, i, type = [], i+1,'m'
                op = self.op[type][int(loss[i*3+1:i*3+3])]
                merge_node.append(OpNode(type,op,nodes2bmerge[j*2],nodes2bmerge[j*2+1]))
                for _ in range(self.N): # handle N v2vnode after tv2vnode
                    i,type= i+1,'v'
                    merge_node.append(OpNode(type,self.op[type][int(loss[i*3+1:i*3+3])],merge_node[-1],None))
                next_nodes2bmerge.append(merge_node[-1])
            # does it affect if it's position changed? Yes, after I change it here, result turn right
            if leftnode: next_nodes2bmerge.append(nodes2bmerge[-1])
            nodes2bmerge = next_nodes2bmerge
        assert len(nodes2bmerge)==1
        self.loss = OpNode('loss',op=None,lchild=nodes2bmerge[0],rchild=None)
        return self.loss

    def visualization(self):
        if self.loss:
            print(self.loss.getExp())
        else:
            print("plz specify loss by method:getLoss(lossExpr)")
    # TODO random constract loss
    def randomLossstr(self):
        str,loss = self.OPORDER,""
        for i in str:loss+=i+format(random.randint(0,len(self.op[i])-1),"02d")
        return loss
    def forward(self, logits, targets):
        return self.loss.forward(logits,targets,reduction=self.reduction)

# v2v,t2t,tv2v,t2v,_ = getMNISTop()
# cl = CompositeLoss(t2t,t2v,v2v,tv2v,K=5,M=1,N=1)
# loss = 'T01T02T01T02T01t03t04t03t04t03v05v06v05v06v05m07v08m07v08m07v08m07v08'
# loss = cl.getLoss(loss)
# # for visualization
# cl.visualization()
# print(cl.randomLossstr())
# print(cl.randomLossstr())

def zero_one_loss(input, target, reduction='elementwise_mean'):
    loss = (input != target)
    return _reduce_loss(loss, reduction)


def elementwise_margin(logits, label):
    batch_size = logits.size(0)
    topval, topidx = logits.topk(2, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse - logits[torch.arange(batch_size), label]


def logit_margin_loss(input, target, reduction='elementwise_mean', offset=0.):
    loss = elementwise_margin(input, target)
    return _reduce_loss(loss, reduction) + offset


def cw_loss(input, target, reduction='elementwise_mean'):
    loss = clamp(elementwise_margin(input, target) + 50, 0.)
    return _reduce_loss(loss, reduction)


def _reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(reduction + " is not valid")


def soft_logit_margin_loss(
        logits, targets, reduction='elementwise_mean', offset=0.):
    batch_size = logits.size(0)
    num_class = logits.size(1)
    mask = torch.ones_like(logits).byte()
    # TODO: need to cover different versions of torch
    # mask = torch.ones_like(logits).bool()
    mask[torch.arange(batch_size), targets] = 0
    logits_true_label = logits[torch.arange(batch_size), targets]
    logits_other_label = logits[mask].reshape(batch_size, num_class - 1)
    loss = torch.logsumexp(logits_other_label, dim=1) - logits_true_label
    return _reduce_loss(loss, reduction) + offset


# def myceloss(Z,label,reduction='elementwise_mean'):
#     opdict = getOP()
#     U,B,M,S = opdict['Unary'],opdict['Binary'],opdict['Multinary'],opdict['SpecialMulti']
#     zy = S['getLabellogit'](Z,label)
#     negzy = U['reverse'](zy)
#     logsumexpZ = U['logarithm'](M['Sum'](M['Exponential'](Z)))
#     # which is already in torch "torch.logsumexp"
#     loss = B['addtensor'](negzy,logsumexpZ)
#     return _reduce_loss(loss, reduction)