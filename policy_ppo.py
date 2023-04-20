
import torch

class Policy(torch.nn.Module):
    def __init__(self,layers=[64,32,16],in_size=4,out_size=2,activation=torch.nn.ELU):
        super(Policy, self).__init__()
        layer_sizes = [in_size,*layers]
        linear_layers = [self.layer(in_f,out_f,activation) for in_f,out_f in zip(layer_sizes,layer_sizes[1:])]
        self.model = torch.nn.Sequential(
            *linear_layers,
            self.layer(layer_sizes[-1],out_size,torch.nn.Tanh) #bound [-1,1]
            )
        self.model_values = torch.nn.Sequential(
            *linear_layers,
            torch.nn.Linear(layer_sizes[-1],1)
            )
        self.sigma = torch.nn.Parameter(0.01*torch.ones(out_size,requires_grad=False,dtype = torch.float32),requires_grad =False)
    def layer(self,in_f,out_f,activation):
        return torch.nn.Sequential(torch.nn.Linear(in_f,out_f),activation())

    def forward(self,x):
        #channel first: x [N,4]
        return self.model(x) #[N,2]
    def get_action_and_value(self,obs, action=None):
        #print(obs.size())
        mu = self.model(obs)
        
        covar = self.sigma #torch.diag
        #print(covar)
        dist = torch.distributions.normal.Normal(mu,covar)
        if action is None:
            action =dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.model_values(obs)
        return action, action_logprob, entropy, value
            
if __name__ == "__main__":
    x = torch.rand(10,4)
    p = Policy()
    u = p(x)
    print(u.size())
