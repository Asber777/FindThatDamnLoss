'''
operation lib and create a loss_space
make sure that b must be a number (float tensor with shape [1])
'''
# TODO 测试代码加上检测是否会使得tensor/vector输出无梯度 ->貌似并不需要,loss只需要计算数值
import torch as t
import torch.nn.functional as F
TestOP = True#Test OK -9/11

'''v2vOP: input a vector; return a vector'''
def zero(x,b=0):return t.zeros_like(x)
def one(x,b=0):return t.ones_like(x)
def constant(x,b=0.5):return t.ones_like(x)*b
def addconstant(x,b=1):return x+b
def remain(x,b=0):return x
def reverse(x,b=0):return -x+b
def absolute(x,b=0):return abs(x)+b
def square(x,b=2):return x**b
def reciprocal(x,b=0):return 1/(x+b)
def multiply(x,b=2):return b*x
def exponential(x,b=0):return t.exp(x+b)
def logarithm(x,b=0):return t.log(abs(x+b))# Use abs to makesure no bug
def cosh(x,b=1):return t.cosh(x)*b
def sinh(x,b=1):return t.sinh(x)*b
def maxima(x,b=0):return t.clamp(x,min=b)
def minima(x,b=0):return t.clamp(x,max=b)
def sigmoid(x,b=0):return t.sigmoid(x+b)
def erf(x,b=0):return t.erf(x+b)
def ceil(x,b=0):return t.ceil(x)+b
def sign(x,b=0):return t.sign(x+b)
def trunc(x,b=0):return t.trunc(x+b)
def softsign(x,b=0):return F.softsign(x)
def softplus(x,b=1):return t.log(b+t.exp(x))
def swish(x,b=1.702):return x*sigmoid(b*x)

'''T2T:input a logits Tensor and return a Tensor with the same shape'''
def Zero(Z,b=0):return t.zeros_like(Z)
def One(Z,b=0):return t.ones_like(Z)
def Constant(Z,b=0.5):return t.ones_like(Z)*b
def Addconstant(Z,b=1):return Z+b
def Remain(Z,b=0):return Z
def Reverse(Z,b=0):return -Z+b
def Absolute(Z,b=0):return abs(Z)+b
def Square(Z,b=2):return Z**b
def Reciprocal(Z,b=0):return 1/(Z+b)
def Multiply(Z,b=2):return b*Z
def Exponential(Z,b=0):return t.exp(Z+b)
def Logarithm(Z,b=0):return t.log(abs(Z+b)) #please make sure b>0 and b>abs(min(x))
def Cosh(Z,b=1):return t.cosh(Z)*b
def Sinh(Z,b=1):return t.sinh(Z)*b
def Maxima(Z,b=0):return t.clamp(Z,min=b)
def Minima(Z,b=0):return t.clamp(Z,max=b)
def Sigmoid(Z,b=0):return t.sigmoid(Z+b)
def Erf(Z,b=0):return t.erf(Z+b)
def Ceil(Z,b=0):return t.ceil(Z+b)
def Sign(Z,b=0):return t.sign(Z+b)
def Trunc(Z,b=0):return t.trunc(Z+b)
def Softsign(Z,b=0):return F.softsign(Z+b)
def Softplus(Z,b=1):return t.log(b+t.exp(Z))
def Swish(Z,b=1.702):return Z*sigmoid(b*Z,0)

v2vlist = [zero,one,constant,addconstant,remain,reverse,absolute,square,reciprocal,multiply,exponential,\
    logarithm,cosh,sinh,maxima,minima,sigmoid,erf,ceil,sign,trunc,softsign,softplus,swish]
T2Tlist = [Zero,One,Constant,Addconstant,Remain,Reverse,Absolute,Square,Reciprocal,Multiply,Exponential,\
    Logarithm,Cosh,Sinh,Maxima,Minima,Sigmoid,Erf,Ceil,Sign,Trunc,Softsign,Softplus,Swish]


def WeightZ(Z,weight=None):
    if weight==None:return Z
    assert isinstance(weight,t.Tensor)
    assert weight.shape[0]==Z.shape[1]
    return Z*weight
# This op is not include in our RL demo v0.1

#2v2v op:input 2 vector output 1 vector
def addtensor(x,y,b=0):return x+y
def multitensor(x,y,b=0):return x*y
def subtensor(x,y,b=0):return x-y
def divtensor(x,y,b=0):return t.divide(x,y)
def maxtensor(x,y,b=0):return t.max(t.stack([x,y]),0)[0]
def mintensor(x,y,b=0):return t.min(t.stack([x,y]),0)[0]
def sigmoidtensor(x,y,b=0):return y*t.sigmoid(x)
def subsqexp(x,y,b=1):return t.exp(-b*(x-y)**2)
def subabsexp(x,y,b=1):return t.exp(-b*abs(x-y))
def firstord(x,y,b=0.5):return b*x+(1-b)*y

#T2v op : input a logits tensor and a label ; return a logit vector which target label
def Top1else(Z,label,n=0):
    topval, topidx = Z.topk(2, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse
def Minelse(Z,label,n=0):
    label_num = Z.shape[1]
    minval, minidx = Z.topk(label_num, dim=1)
    minelse = ((label != minidx[:,label_num-1]).float() * minval[:, label_num-1]
               + (label == minidx[:,label_num-1]).float() * minval[:, label_num-2])
    return minelse
def TopNelse(Z,label,n):
    batchsize,label_num = Z.shape
    assert n<label_num and n>0
    newZ = t.stack([t.cat((Z[i][:label[i]],Z[i][label[i]+1:])) for i in range(batchsize)])
    topval, _ = newZ.topk(label_num-1, dim=1)
    return topval[:,n-1]
def getLabellogit(Z,label,n=0):return Z[t.arange(len(label)), label]
def Sum(Z,label=None,n=1):return t.sum(Z,dim=1)*n


tv2vlist = [addtensor,multitensor,subtensor,divtensor,maxtensor,mintensor,sigmoidtensor,subsqexp,subabsexp,firstord]
T2vlist = [Top1else,Minelse,TopNelse,getLabellogit,Sum]


def getMNISTop():
    def Top2else(Z,label):return TopNelse(Z,label,2)
    def Top3else(Z,label):return TopNelse(Z,label,3)
    def Top4else(Z,label):return TopNelse(Z,label,4)
    def Top5else(Z,label):return TopNelse(Z,label,5)
    def Top6else(Z,label):return TopNelse(Z,label,6)
    def Top7else(Z,label):return TopNelse(Z,label,7)
    def Top8else(Z,label):return TopNelse(Z,label,8)
    v2vlist = [zero,one,constant,addconstant,remain,reverse,absolute,square,reciprocal,multiply,exponential,\
    logarithm,cosh,sinh,maxima,minima,sigmoid,erf,ceil,sign,trunc,softsign,softplus,swish]
    T2Tlist = [Zero,One,Constant,Addconstant,Remain,Reverse,Absolute,Square,Reciprocal,Multiply,Exponential,\
    Logarithm,Cosh,Sinh,Maxima,Minima,Sigmoid,Erf,Ceil,Sign,Trunc,Softsign,Softplus,Swish]
    tv2vlist = [addtensor,multitensor,subtensor,divtensor,maxtensor,mintensor,sigmoidtensor,subsqexp,subabsexp,firstord]
    T2vlist = [Top1else,Top2else,Top3else,Top4else,Top5else,Top6else,Top7else,Top8else,Minelse,getLabellogit,Sum]
    opdict = {'v2v':v2vlist,'2v2v':tv2vlist,'T2T':T2Tlist,'T2v':T2vlist}
    return v2vlist,T2Tlist,tv2vlist,T2vlist,opdict

# - - For test - - - - -- - - - - - - - - - -- 
# TODO : modify ,cause I'm not done with reconstructing this function
def _test(batchsize,labelnum,T2Tlist,T2vlist,v2vlist,tv2vlist):
    logits = abs(t.randn(batchsize,labelnum))
    a = t.randn(labelnum)
    b = t.randn(labelnum)
    print(logits)
    print("===========logits============")
    print(a)
    print("==============a==============")
    print(b)
    print("==============b==============")
    for op in T2Tlist:
        print(op)
        r = op(logits)
        print(r)
        assert r.shape.numel()==batchsize*labelnum
        assert r.shape[0] == labelnum
        print("--------------")
    