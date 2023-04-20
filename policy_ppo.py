
import torch

class Policy(torch.nn.Module):
    def __init__(self,layers=[64,32,16],in_size=4,out_size=2,activation=torch.nn.ELU):
        super(Policy, self).__init__()
        layer_sizes = [in_size,*layers]
        linear_layers = [self.layer(in_f,out_f,activation) for in_f,out_f in zip(layer_sizes,layer_sizes[1:])]
        self.backbone = torch.nn.Sequential(*linear_layers)
        self.action_head = torch.nn.Sequential(
            self.layer(layer_sizes[-1],out_size,torch.nn.Tanh)) #bound [-1,1]
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(layer_sizes[-1],1)
            )
        self.logstd= torch.nn.Parameter(torch.zeros(out_size,requires_grad=True,dtype = torch.float32),requires_grad=True)
    

    def layer(self,in_f,out_f,activation):
        return torch.nn.Sequential(torch.nn.Linear(in_f,out_f),activation())

    def forward(self,x):
        #channel first: x [N,4]
        return self.action_head(self.backbone(x)) #[N,2]

    def get_action_and_value(self,obs, action=None):
        #print(obs.size())
        mu = self.action_head(self.backbone(obs))
        
        covar = torch.exp(self.logstd.expand_as(mu))
        #print(covar)
        dist = torch.distributions.normal.Normal(mu,covar)
        if action is None:
            action =dist.sample()
        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(self.backbone(obs))
        return action, action_logprob, entropy, value
            
if __name__ == "__main__":
    x = torch.rand(10,4)
    p = Policy()
    u = p(x)
    print(u.size())
