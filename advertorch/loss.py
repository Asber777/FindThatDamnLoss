import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from advertorch import oplib
from advertorch.utils import clamp
from advertorch.oplib import getMNISTop

# class MyCEloss(_Loss):
#     """CEloss which constracted by my oplib"""
#     def __init__(self, size_average=None, reduce=None,reduction='elementwise_mean'):
#         super(MyCEloss, self).__init__(size_average, reduce, reduction)

#     def forward(self, logits, targets):
#         return myceloss(logits, targets, reduction=self.reduction)

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
Create a LOSS graph .
input logits number ->Z
Number of T2T/v2v after Logits or vector -> M
Number of v2v after 2v2v(m) ->N
abbreviation - T2T:T ; T2v:t ; 2v2v:m ; v2v:v
====================Example=====================
if K = 4 M = 2 N = 1, then the structure of Loss may looks like:
Logits->T2T(1)->T2T(5)->T2v(09)->v2v->v2v->\
Logits->T2T(2)->T2T(6)->T2v(10)->v2v->v2v->2v2v->v2v->\
Logits->T2T(3)->T2T(7)->T2v(11)->v2v->v2v->\          |
Logits->T2T(4)->T2T(8)->T2v(12)->v2v->v2v->2v2v->v2v->2v2v->v2v->LOSS
'''
class OpNode():
    #FOR VISILAZTION
    LINK = "<-"
    MERGE = "/<-"
    def __init__(self, type=None, op=None, lchild=None, rchild=None):
        assert type in ['T','v','t','m','leaf','loss']
        self.type = type
        self.op,self.lchild,self.rchild = op,lchild,rchild
        self.name = type if type=='leaf' or type=='loss' else op.__name__[:3]
        self.result = 0
    # We assume that if op type is T v t, than it only has input from lchild
    def forward(self, logits, label=None):
        if self.type=='T':#T2T input:Tensor
            Z = self.lchild.forward(logits)
            assert isinstance(Z,torch.Tensor) and len(Z.shape)==2
            self.result = self.op(Z)
        if self.type=='v':#v2v input:vector
            x = self.lchild.forward(logits,label)
            assert isinstance(x,torch.Tensor) and len(x.shape)==1
            self.result = self.op(x)
        if self.type=='t':#T2v input:Tensor
            Z = self.lchild.forward(logits)
            assert isinstance(Z,torch.Tensor) and len(Z.shape)==1
            self.result = self.op(Z,self.label)
        if self.type=='m':#tv2v input:vector vector
            x = self.lchild.forward(logits,label)
            y = self.rchild.forward(logits,label)
            assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor)
            self.result = self.op(x,y)
        if self.type=='leaf':
            self.result = logits.clone().detach()
        if self.type=='loss':
            self.result = self.lchild.forward(logits,label)
            assert isinstance(Z,torch.Tensor) and len(Z.shape)==1
        return self.result

# Assume that all op need not keep requires_grad same as input tensor/vector
class CompositeLoss(_Loss):
    def __init__(self, T2Tlist:list, T2vlist:list, v2vlist:list, tv2vlist:list ,K:int =2, M:int =1, N:int=1,
                size_average=None, reduce=None,reduction='elementwise_mean'):
        super(CompositeLoss, self).__init__(size_average, reduce, reduction)
        assert K>=2 and M>=0 and N>=0
        self.K, self.M, self.N, self.m_num= K, M, N, max(K-1,0)
        self.opl = {'T':len(T2Tlist),'t':len(T2vlist),'v':len(v2vlist),'m':len(tv2vlist)}
        self.op = {'T':T2Tlist,'t':T2vlist,'v':v2vlist,'m':tv2vlist}
        self.example = 'T**'*self.K*self.M + self.K *'t**' + 'v**'*self.K*self.M + self.m_num*('m**'+self.N*'v**')
        self.loss = None

    def _checkLegal(self,loss:str): # Use a str to represent loss.
        self.op_num = len(loss)//3
        for i in range(self.op_num):
            if int(loss[i*3+1:i*3+3]) >= self.opl[loss[i*3]]:
                print("{} is wrong expression".format(loss[i*3:(i+1)*3]))
                return False
        print("len of loss str is {}".format(self.op_num))
        return True

    def getLoss(self,loss):
        assert self._checkLegal(loss)
        branch = [[OpNode('leaf')] for _ in range(self.K)]
        ahead = self.K *(2*self.M+1) # branch op num
        for i in range(ahead):# by this way ,you can actually Transfer to vector immediately after leaf,as the code here is unlimited.
            type = loss[i*3]
            op = self.op[type][int(loss[i*3+1:i*3+3])]
            node = OpNode(type,op,branch[i%self.K][-1],None)
            branch[i%self.K].append(node)
        nodes2bmerge = [branch[k][-1] for k in range(self.K)]
        layer_node_num, leftnode = self.K, False
        while(layer_node_num!=1 or leftnode!=False):
            if not leftnode:#no node left to merge
                layer_node_num,leftnode = layer_node_num//2,False if layer_node_num%2==0 else True
            elif leftnode and layer_node_num%2==0:#a node left to merge and the num of nodes product by last layer is even
                layer_node_num,leftnode = layer_node_num//2,True
            elif leftnode and layer_node_num%2==1:
                layer_node_num,leftnode = layer_node_num//2 + 1,False
            # According to layer_node_num:how much pair node to be merged, create next layer's nodes and add to self.merge
            next_nodes2bmerge = []
            for j in range(layer_node_num):#product one layer with merge operations
                merge_node = []
                i, type = i+1,'m'
                op = self.op[type][int(loss[i*3+1:i*3+3])]
                node = OpNode(type,op,nodes2bmerge[j*2],nodes2bmerge[j*2+1])#???
                merge_node.append(node)
                for _ in range(self.N): #handle N v2vnode after tv2vnode
                    i,type= i+1,'v'
                    op = self.op[type][int(loss[i*3+1:i*3+3])]
                    lchild = merge_node[-1]
                    node = OpNode(type,op,lchild,None)
                    merge_node.append(node)
                    # print("add {}{}node {}".format(node.type,node.name,node))
                next_nodes2bmerge.append(merge_node[-1])
            #does it affect if it's position changed? Yes, after I change it here, result turn right
            if leftnode: next_nodes2bmerge.append(nodes2bmerge[-1])
            nodes2bmerge = next_nodes2bmerge
        assert len(nodes2bmerge)==1
        self.loss = OpNode('loss',op=None,lchild=nodes2bmerge[0],rchild=None)
        return self.loss
    def visualization(self):
        pass
    # TODO random constract loss
    def randomLossstr(self):
        pass

    def forward(self, logits, targets):
        return self.loss.forward(logits,targets)

v2v,t2t,tv2v,t2v,_ = getMNISTop()
cl = CompositeLoss(t2t,t2v,v2v,tv2v,K=5,M=1,N=1)
loss = 'T01T02T01T02T01t03t04t03t04t03v05v06v05v06v05m07v08m07v08m07v08m07v08'
loss = cl.getLoss(loss)
r = loss
while(r!=None):
    print(r)
    r = r.lchild


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