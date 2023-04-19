from Reinforcement import PPO 
from env import Env
from policy import Policy
import torch 

env = Env(headless = True)
totel_steps = 10000000
batch_size = 100
epochs = 10 
obs_size = 3
n_steps = 1000
learning_rate = 0.001

net = Policy()
opt = torch.optim.Adam(net.parameters(),lr=learning_rate,eps=1e-5)

PPO(env,net,opt,totel_steps,epochs,batch_size,n_steps)
