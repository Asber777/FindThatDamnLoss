from numpy.lib.arraysetops import isin
import torch as t
import torch.nn as nn
import torch.nn.functional as F
TestOP = True

'''Unary Operation'''
'''for simple,we assume x must be vector which size is batchsize'''
'''make sure that b must be a number (float tensor with shape [1])'''
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
def logarithm(x,b):
    assert b>0 and b>abs(min(x))#make sure b>0 and b>abs(min(x))
    return t.log(x+b) 
cosh = lambda x,b:t.cosh(x)
sinh = lambda x,b:t.sinh(x)
# TODO : sinc = lambda
maxima = lambda x,b:t.clamp(x,min=b)
minima = lambda x,b:t.clamp(x,max=b)
sigmoid = lambda x,b:t.sigmoid(x)
erf = lambda x,b:t.erf(x)
ceil = lambda x,b:t.ceil(x)
sign = lambda x,b:t.sign(x)
trunc = lambda x,b:t.trunc(x)
softsign = lambda x,b:F.softsign(x)
softplus = lambda x,b:t.log(b*t.exp(x))
swish = lambda x,b:x*sigmoid(b*x,0)
# TODO : test Unary Opration
UnaryList = ["zero","one","constant","addconstant","remain","reverse","absolute","square","reciprocal","multiply","exponential",\
    "logarithm","cosh","sinh","maxima","minima","sigmoid","erf","ceil","sign","trunc","softsign","softplus","swish"]
def _testUnary(labelnum,unarylist):
    a = t.randn(labelnum)
    print(a)
    print("============a===========")
    for op in unarylist:
        print(op)
        print(eval(op)(a,10))
        print("--------------")
if TestOP: _testUnary(10,UnaryList)






'''Binary Operation'''
'''make sure that b must be a number (float tensor with shape [1])'''
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
# TODO test Binary OP
BinaryList = ["addtensor","multitensor","subtensor","divtensor","maxtensor","mintensor","sigmoidtensor","subsqexp","subabsexp","firstord"]
def _testBinary(labelnum,binarylist):
    a = t.randn(labelnum)
    b = t.randn(labelnum)
    print(a)
    print("===========a============")
    print(b)
    print("===========b============")
    for op in binarylist:
        print(op)
        print(eval(op)(a,b,10))
        print("--------------")
if TestOP: _testBinary(10,BinaryList)



'''Multinary opration on all logits'''
def WeightZ(Z,weight):
    assert isinstance(weight,t.Tensor)
    assert weight.shape[0]==Z.shape[1]
    return Z*weight

def MaxNlogit(Z,label,n=2):
    topval, topidx = Z.topk(n, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse



# The Multinary OP below is overall version of Unary OP
'''make sure that b must be a number (float tensor with shape [1])'''
Dividenormal = lambda Z,b:Z/b
Multipnormal = lambda Z,b:Z*b 
def Addconstant(Z,b):return Z+b
def Square(Z,b):return Z**2
def Reciprocal(Z,b):return 1/Z
def Exponential(Z,b):return t.exp(Z)
def Logarithm(Z,b):
    assert b>abs(min(t.min(Z,1)[0]))
    return t.log(Z+b) #make sure b>0 and b>abs(min(x))
def Cosh(Z,b):return t.cosh(Z)
def Sinh(Z,b):return t.sinh(Z)
def Erf(Z,b):return t.erf(Z)
def Ceil(Z,b):return t.ceil(Z)
def Sign(Z,b):return t.sign(Z)
def Trunc(Z,b):return t.trunc(Z)
def Softsign(Z,b):return F.softsign(Z)
def Swish(Z,b):return Z*sigmoid(b*Z,0)


def ShiftZ(Z,s,m):return s*Z+m
def Softplus(Z,a,b):return t.log(a+b*t.exp(Z))
# TODO test Multinary OP


TripMutinaryList = ["ShiftZ","Softplus"]
SpecialMutinaryList = ["WeightZ","MaxNlogit"]
MutinaryList = ["Dividenormal","Multipnormal","Addconstant","Square","Reciprocal","Exponential","Logarithm","Cosh","Sinh","Erf",\
    "Ceil","Sign","Trunc","Softsign","Swish"]
def _tesTripMutinary(batchsize,labelnum,List):
    logits = t.randn(batchsize,labelnum)
    print(logits)
    print("===========logits============")
    for op in List:
        print(op)
        print(eval(op)(logits,10,1))
        print("--------------")
def _testMutinary(batchsize,labelnum,List):
    logits = t.randn(batchsize,labelnum)
    print(logits)
    print("===========logits============")
    for op in List:
        print(op)
        print(eval(op)(logits,10))
        print("--------------")
if TestOP:
    _testMutinary(10,10,MutinaryList)
    _tesTripMutinary(10,10,TripMutinaryList)