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

    def layer(self,in_f,out_f,activation):
        return torch.nn.Sequential(torch.nn.Linear(in_f,out_f),activation())

    def forward(self,x):
        #channel first: x [N,4]
        #print(x.size())
        return self.model(x) #[N,2]

if __name__ == "__main__":
    x = torch.rand(10,4)
    p = Policy()
    u = p(x)
    print(u.size())
