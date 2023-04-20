from ppo import PPO 
from env import Env
from policy_ppo import Policy
import torch 

env = Env(headless = False)
totel_steps = 1_000_000
batch_size = 100
epochs = 10 
obs_size = 4 
n_steps = 3000
learning_rate = 0.001

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device {}'.format(device))
net = Policy().to(device)
opt = torch.optim.Adam(net.parameters(),lr=learning_rate,eps=1e-5)

PPO(env,net,opt,totel_steps,epochs,batch_size,n_steps,vec_obs_size=obs_size)
