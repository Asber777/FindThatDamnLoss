'''make sure that b must be a number (float tensor with shape [1])'''
from math import exp
from numpy.lib.arraysetops import isin
import torch as t
import torch.nn.functional as F
TestOP = False

'''Unary Operation: input a vector; return a vector'''
zero = lambda x,b=0:t.zeros_like(x)
one = lambda x,b=0:t.ones_like(x)
constant = lambda x,b=1:t.ones_like(x)*b
addconstant = lambda x,b=0:x+b
remain = lambda x,b=0:x
reverse = lambda x,b=0:-x+b
absolute = lambda x,b=0:abs(x)+b
square = lambda x,b=2:x**b
reciprocal = lambda x,b=0:1/(x+b)
multiply = lambda x,b=1:b*x
exponential = lambda x,b=0:t.exp(x+b)
logarithm = lambda x,b=0:t.log(x+b)
cosh = lambda x,b=1:t.cosh(x)*b
sinh = lambda x,b=1:t.sinh(x)*b
# TODO : sinc = lambda
maxima = lambda x,b=0:t.clamp(x,min=b)
minima = lambda x,b=0:t.clamp(x,max=b)
sigmoid = lambda x,b=0:t.sigmoid(x+b)
erf = lambda x,b=0:t.erf(x+b)
ceil = lambda x,b=0:t.ceil(x)+b
sign = lambda x,b=0:t.sign(x+b)
trunc = lambda x,b=0:t.trunc(x+b)
softsign = lambda x,b=0:F.softsign(x)
softplus = lambda x,b=1:t.log(b+t.exp(x))
swish = lambda x,b=1.702:x*sigmoid(b*x)

UnaryList = ["zero","one","constant","addconstant","remain","reverse","absolute","square","reciprocal","multiply","exponential",\
    "logarithm","cosh","sinh","maxima","minima","sigmoid","erf","ceil","sign","trunc","softsign","softplus","swish"]
UnaryDict = {"zero":zero,"one":one,"constant":constant,"addconstant":addconstant,"remain":remain,"reverse":reverse,"absolute":absolute,\
    "square":square,"reciprocal":reciprocal,"multiply":multiply,"exponential":exponential,"logarithm":logarithm,"cosh":cosh,"sinh":sinh,\
        "maxima":maxima,"minima":minima,"sigmoid":sigmoid,"erf":erf,"ceil":ceil,"sign":sign,"trunc":trunc,"softsign":softsign,\
            "softplus":softplus,"swish":swish}

def _testUnary(labelnum,unarylist):
    a = t.randn(labelnum)
    print(a)
    print("============a===========")
    for op in unarylist:
        print(op)
        print(eval(op)(a))
        print("--------------")
if TestOP: _testUnary(10,UnaryList)






'''Binary Operation:input two vectors; return a vector'''
addtensor = lambda x,y,b=0:x+y
multitensor = lambda x,y,b=0:x*y
subtensor = lambda x,y,b=0:x-y
divtensor = lambda x,y,b=0:t.divide(x,y)
maxtensor = lambda x,y,b=0:t.max(t.stack([x,y]),0)[0]
mintensor = lambda x,y,b=0:t.min(t.stack([x,y]),0)[0]
sigmoidtensor = lambda x,y,b=0:y*t.sigmoid(x)
subsqexp = lambda x,y,b=1:t.exp(-b*(x-y)**2)
subabsexp = lambda x,y,b=1:t.exp(-b*abs(x-y))
firstord = lambda x,y,b=0.5:b*x+(1-b)*y

BinaryList = ["addtensor","multitensor","subtensor","divtensor","maxtensor","mintensor","sigmoidtensor","subsqexp","subabsexp","firstord"]
BinaryDict = {"addtensor":addtensor,"multitensor":multitensor,"subtensor":subtensor,"divtensor":divtensor,"maxtensor":maxtensor,\
    "mintensor":mintensor,"sigmoidtensor":sigmoidtensor,"subsqexp":subsqexp,"subabsexp":subabsexp,"firstord":firstord}
def _testBinary(labelnum,binarylist):
    a = t.randn(labelnum)
    b = t.randn(labelnum)
    print(a)
    print("===========a============")
    print(b)
    print("===========b============")
    for op in binarylist:
        print(op)
        print(eval(op)(a,b))
        print("--------------")
if TestOP: _testBinary(10,BinaryList)







'''Multinary'''
''''SpecialMultiary: input logits tensor; return a Return specail logit vector'''
def Maxelse(Z,label):
    topval, topidx = Z.topk(2, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse
def Minelse(Z,label):
    label_num = Z.shape[1]
    minval, minidx = Z.topk(label_num, dim=1)
    minelse = ((label != minidx[:,label_num-1]).float() * minval[:, label_num-1]
               + (label == minidx[:,label_num-1]).float() * minval[:, label_num-2])
    return minelse
def Labelelse(Z,label,n):
    batchsize,label_num = Z.shape
    assert n<label_num and n>0
    newZ = t.stack([t.cat((Z[i][:label[i]],Z[i][label[i]+1:])) for i in range(batchsize)])
    topval, _ = newZ.topk(label_num-1, dim=1)
    return topval[:,n-1]
getLabellogit = lambda Z,label:t.tensor([z[label[i]] for i,z in enumerate(Z)],device=Z.device)


'''Multinary opration:input a logits tensor and return a tensor with the same shape'''
def WeightZ(Z,weight):
    assert isinstance(weight,t.Tensor)
    assert weight.shape[0]==Z.shape[1]
    return Z*weight
# The Multinary OP below is overall version of Unary OP
Dividenormal = lambda Z,b=1:Z/b
Multipnormal = lambda Z,b=1:Z*b 
def Addconstant(Z,b=0):return Z+b
def Square(Z,b=2):return Z**b
def Reciprocal(Z,b=0):return 1/(Z+b)
def Exponential(Z,b=0):return t.exp(Z+b)
def Logarithm(Z,b=0):
    assert b+min(t.min(Z,dim = 1)[0])>0
    return t.log(Z+b) #make sure b>0 and b>abs(min(x))
def Cosh(Z,b=1):return t.cosh(Z)*b
def Sinh(Z,b=1):return t.sinh(Z)*b
def Erf(Z,b=0):return t.erf(Z+b)
def Ceil(Z,b=0):return t.ceil(Z+b)
def Sign(Z,b=0):return t.sign(Z+b)
def Trunc(Z,b=0):return t.trunc(Z+b)
def Softsign(Z,b=0):return F.softsign(Z+b)
def Swish(Z,b=1.702):return Z*sigmoid(b*Z,0)
def Sum(Z,b=1):return t.sum(Z,dim=1)*b
'''TripMultinary'''
def ShiftZ(Z,s=1,m=1):return s*Z+m
def Softplus(Z,a=1,b=1):return t.log(a+b*t.exp(Z))

TripMultinaryList = ["ShiftZ","Softplus"]
SpecialMultinaryList = ["WeightZ","Maxelse","Minelse","Labelelse","getLabellogit"]
MultinaryList = ["Dividenormal","Multipnormal","Addconstant","Square","Reciprocal","Exponential","Logarithm","Cosh","Sinh","Erf",\
    "Ceil","Sign","Trunc","Softsign","Swish","Sum"]
TripMultinaryDict = {"ShiftZ":ShiftZ,"Softplus":Softplus}
SpecialMultinaryDict = {"WeightZ":WeightZ,"Maxelse":Maxelse,"Minelse":Minelse,"Labelelse":Labelelse,"getLabellogit":getLabellogit}
MultinaryDict = {"Dividenormal":Dividenormal,"Multipnormal":Multipnormal,"Addconstant":Addconstant,"Square":Square,\
    "Reciprocal":Reciprocal,"Exponential":Exponential,"Logarithm":Logarithm,"Cosh":Cosh,"Sinh":Sinh,"Erf":Erf,\
    "Ceil":Ceil,"Sign":Sign,"Trunc":Trunc,"Softsign":Softsign,"Swish":Swish,"Sum":Sum}
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
    _testMutinary(10,10,MultinaryList)
    _tesTripMutinary(10,10,TripMultinaryList)

def getOP(intro = False):
    opdict = {'Unary':UnaryDict,'Binary':BinaryDict,'Multinary':MultinaryDict,'TripMulti':TripMultinaryDict,\
        'SpecialMulti':SpecialMultinaryDict}
    oplist = [UnaryList,BinaryList,MultinaryList,TripMultinaryList,SpecialMultinaryList]
    if intro:
        opassemble_name = ["Unary","Bianry","Multinary","TripMulti","SpecialMulti"]
        for i,list in enumerate(oplist):
            print("{}:".format(opassemble_name[i]))
            print(list)
    return opdict,oplist