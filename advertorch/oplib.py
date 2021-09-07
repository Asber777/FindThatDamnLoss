import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from advertorch.utils import clamp
'''Unary Operation'''
'''for simple,we assume x must be vector which size is batchsize'''
zero = lambda x,b:t.zeros_like(x)
one = lambda x,b:t.ones_like(x)
constant = lambda x,b:t.ones_like(x)*b
addconstant = lambda x,b:x+b
remain = lambda x,b:x
reverse = lambda x,b:-x
absolute = lambda x,b:abs(x)
square = lambda x,b:x**2
reciprocal = lambda x,b:1/x
multiply = lambda x,b:b*x
exponential = lambda x,b:t.exp(x)
logarithm = lambda x,b:t.log(x+b) #make sure b>0 and b>abs(min(x))
cosh = lambda x,b:t.cosh(x)
sinh = lambda x,b:t.sinh(x)
# TODO : sinc = lambda
maxima = lambda x,b:t.clamp(x,max=b)
minima = lambda x,b:t.clamp(x,min=b)
sigmoid = lambda x,b:t.sigmoid(x)
erf = lambda x,b:t.erf(x)
ceil = lambda x,b:t.ceil(x)
sign = lambda x,b:t.sign(x)
trunc = lambda x,b:t.trunc(x)

# TODO : test Unary Opration
'''Binary Operation'''
addtensor = lambda x,y,b:x+y
multitensor = lambda x,y,b:x*y
subtensor = lambda x,y,b:x-y
divtensor = lambda x,y,b:t.divide(x,y)
maxtensor = lambda x,y,b:t.max(t.stack([x,y]),0)[0]
mintensor = lambda x,y,b:t.min(t.stack([x,y]),0)[0]
sigmoidtensor = lambda x,y,b:y*t.sigmoid(x)
subsqexp = lambda x,y,b:t.exp(-b*(x-y)**2)
subabsexp = lambda x,y,b:t.exp(-b*abs(x-y))
firstord = lambda x,y,b:b*x+(1-b)*y
# addcdiv = lambda x,y:

'''Multinary opration'''
# TODO
dividenormal = lambda Z,b:Z/b
multipnormal = lambda Z,b:Z*b # which is same as "weight = lambda Z,w:Z*w"


def maxelse(logits,label):
    topval, topidx = logits.topk(2, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse
