from copy import deepcopy
import matplotlib.pyplot as plt
import torch

from plot import Plot
from onnx_export import Export

def indicator(i,I=1):
    if i==I:
        return 1
    return 0

def exp_decay(i,p=0.5):
    return p**(i-1)

class beta(torch.nn.Module):
    def __init__(self,fn=exp_decay):
        super(beta, self).__init__()
        self.fn = fn
    def forward(self,i):
        return self.fn(i)

def get_validation_data(env,controller,steps=1_000, device="cpu"):
    x_data = torch.zeros(steps,env.x_dim)

    u_data = torch.zeros(steps,env.u_dim)
    
    x = env.reset()
    for i in range(0,steps,env.n):
        u = controller(x)
        x_data[i:i+env.n] = x.view(-1,env.x_dim)
        u_data[i:i+env.n] = u.view(-1,env.u_dim)
        x,_,_,_,_ = env.step(u)
    return torch.utils.data.TensorDataset(x_data[:i+env.n],u_data[:i+env.n]) #cut off trailing zeros, make dataset

def train(dataset,net,loss_cls=torch.nn.MSELoss,opt_fn=torch.optim.Adam,device="cpu"):
    epoch_loss = 0
    loss_fn = loss_cls()
    optimizer = opt_fn(net.parameters())
    for i, (x,u_star) in enumerate(dataset):
        optimizer.zero_grad()
        u = net(x)
        loss = loss_fn(u,u_star)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataset)

def evaluate(dataset,net,loss_cls=torch.nn.MSELoss,device="cpu"):
    net.train(False)
    loss_fn = loss_cls()
    loss = 0
    for i, (x,u_star) in enumerate(dataset):
        u = net(x.to(device))
        loss += loss_fn(u,u_star.to(device)).item()
    net.train(True)
    return loss / len(dataset)

def DAgger(Env,Expert,Policy,Beta=beta,Beta_fn=exp_decay,N=100,T=500,num_envs=10,plot=True,export=True,device="cpu"):
    env = Env(True)
    expert = Expert().to(device)
    net = Policy().to(device)
    beta = Beta(Beta_fn).to(device)
    v_dataset = get_validation_data(env,expert,device=device)
    best_loss = float("inf")

    train_losses = []
    val_losses = []
    avg_val_losses = []

    for i in range(1,N+1):
        x = env.reset()
        b = beta(i)
        x_data = torch.zeros(T,env.x_dim,device=device)
        u_data = torch.zeros(T,env.u_dim,device=device)
        for t in range(0,T,env.n):
            with torch.no_grad():
                u_star = expert(x)
                if b == 1:
                    u = u_star
                elif b == 0:
                    u = net(x)
                else:
                    u = b*u_star + (1-b)*net(x)
            x_data[t:t+env.n] = x.view(-1,env.x_dim)
            u_data[t:t+env.n] = u_star.view(-1,env.u_dim)
            x,_,done,_,_ = env.step(u)
            if done:
                x_data = x_data[:t+env.n] 
                u_data = u_data[:t+env.n] 
                break
        data = torch.utils.data.TensorDataset(x_data[:t+env.n],u_data[:t+env.n])
        if i == 1:
            dataset = torch.utils.data.ConcatDataset([data])
        else:
            dataset = torch.utils.data.ConcatDataset([dataset,data])
        
        train_loss = train(dataset,net,device=device)
        train_losses.append(train_loss)
        val_loss = evaluate(v_dataset,net,device=device)
        val_losses.append(val_loss)
        if len(val_losses) >= 5:
            avg_val_losses.append(sum(val_losses[-5:])/5)
        else:
            avg_val_losses.append(sum(val_losses)/len(val_losses))

        print("Epoch: {}".format(i))
        print("Training Loss: {}".format(train_loss))
        print("Validation Loss: {}".format(val_loss))
        print("Running Loss: {}".format(avg_val_losses[-1]))
        print()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(net)

        if len(val_losses) > 10 and avg_val_losses[-1] < 0.05:
            # train at least 10 epochs
            # stop if average of last 5 epochs are less than 5% error
            print("Stopping Early.")
            break

    print("Training Complete.")
    print()

    if plot:
        fig_loss,ax_loss = plt.subplots()
        fig_loss.suptitle("Loss")
        ax_loss.plot(train_losses,label="training")
        ax_loss.scatter([t for t in range(i)],val_losses,color="C1",s=5)
        ax_loss.plot([t for t in range(i)],avg_val_losses,label="validation",color="C1")
        ax_loss.legend()
        fig_loss.set_tight_layout(True)

        Plot(Env,expert,best_model,device="cpu")    

    best_model.train(False) # set model to evaluation mode
    model_scripted = torch.jit.script(best_model) # export to TorchScript
    model_scripted.save("il_model.pt") #save TorchScript model

    if export:
        x = env.reset()
        Export(model_scripted,x)

    return best_model

if __name__ == "__main__":
    from env import Env
    from bangbang import Control
    from policy import Policy

    DAgger(Env,Control,Policy)






