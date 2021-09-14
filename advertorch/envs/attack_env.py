'''
Create a env for RL to learn how to attack a picture with Unknow Model
state space : (x+delta_t, y), delta_t, logits(x+delta_t)_t, loss_cur_{t-1,t}
delta_0 = 0, loss_cur_{-1,0}= 0
action space : Loss(Z) Z=logits(x+delta_t) #TODO LOSS(Z,x,delta,loss_cur,logits,feature...)
reward : δ(Loss)/δ(delta)
'''
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
from advertorch.loss import CompositeLoss
from advertorch.oplib import getMNISTop
from advertorch.attacks import LinfPGDAttack
from advertorch_examples.benchmark_utils import benchmark_attack_success_rate










import gym

class AttackEnv(gym.Env):
    def __init__(self):
        '''
        - Z ->     Remain(Z):Z    -> GetLabellogit(Z):z_y -> reverse(z_y):-z_y \
        - Z -> exponential(Z):e^Z -> Sum(e^Z):sumeZ       -> logarithm(sumeZ)  ->addtensor(-z_y,logSumEZ) ->CELoss
        '''
        # CW(x, y) = −z_y + \max_{i\neq y} z_i
        # DLR(x,y) = - \frac{z_y - \max_{i\neq y} z_i} {z_{\pi_1}-z_{\pi_3}}
        self.batch_size = 100
        self.device = "cuda"
        v2vlist,T2Tlist,tv2vlist,T2vlist,op = getMNISTop()
        self.loss = CompositeLoss(T2Tlist,T2vlist,v2vlist,tv2vlist,K=2,M=1,N=0,reduction="sum")

    def action2str(self,action):
        return None

    def step(self,action)->list,int,bool,str:#observation reward done info
        lossstr = self.action2str(action)
        # CEloss = op['Remain']+op['exponential']+op['GetLabellogit']+op['Sum']+op['reverse']+op['logarithm']+op['addtensor']
        # self.loss.getLoss(CEloss)
        self.loss.getLoss(lossstr)
        lst_attack = [
            (LinfPGDAttack, dict(
                loss_fn=self.loss, eps=0.3,
                nb_iter=40, eps_iter=0.01, rand_init=False,
                clip_min=0.0, clip_max=1.0, targeted=False)),
        ]  # each element in the list is the tuple (attack_class, attack_kwargs)
        mnist_clntrained_model = get_mnist_lenet5_clntrained().to(self.device)
        mnist_advtrained_model = get_mnist_lenet5_advtrained().to(self.device)
        mnist_test_loader = get_mnist_test_loader(batch_size=self.batch_size)
        lst_setting = [
            (mnist_clntrained_model, mnist_test_loader),
            (mnist_advtrained_model, mnist_test_loader),
        ]

        self.lst_benchmark = []
        for model, loader in lst_setting:
            for attack_class, attack_kwargs in lst_attack:
                self.lst_benchmark.append(benchmark_attack_success_rate(
                    model, loader, attack_class, attack_kwargs, device=self.device))

    def reset(self):
        print('CustomEnv Environment reset')

    def render(self):
        info = get_benchmark_sys_info()
        print(info)
        for item in self.lst_benchmark:
            print(item)
        return 