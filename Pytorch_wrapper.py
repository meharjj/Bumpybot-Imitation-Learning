import torch

class Expert(torch.nn.Module):
    def __init__(self,model_name="rl_best_model.pt"):
        super(Expert, self).__init__()
        self.model = torch.jit.load(model_name)
        self.model.eval()
    def forward(self,x):
       
        u = self.model(x)
       
        return u.detach() #[N,2]

if __name__ == "__main__":
    from env import Env
    env = Env(1)
    expert = Expert()
    x = env.reset()
    for i in range(5000):
        u = expert(x)
        x,done = env.step(u)
        if sum(done) != 0:
            print("done.")
            break
