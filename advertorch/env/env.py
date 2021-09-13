'''
Create a env for RL to learn how to attack a picture with Unknow Model
state space : (x+delta_t, y), delta_t, logits(x+delta_t)_t, loss_cur_{t-1,t}
delta_0 = 0, loss_cur_{-1,0}= 0
action space : Loss(Z) Z=logits(x+delta_t) #TODO LOSS(Z,x,delta,loss_cur,logits,feature...)
reward : δ(Loss)/δ(delta)
'''