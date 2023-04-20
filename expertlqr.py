import torch
import scipy
def lqr( A, B, Q, R, H, dt):
    
    N = int(H/ dt)
    P = [None]*(N+1)
    P[N] =torch.from_numpy( scipy.linalg.solve_discrete_are(A, B, Q,R)).to(torch.float)
    
    # For i = N, ..., 1
    for i in range(N, 0, -1):
        
        # Discrete-time Algebraic Riccati equation to calculate the optimal 
        # state cost matrix
        P[i-1] = Q + torch.t(A) @ P[i] @ A - (torch.t(A) @ P[i] @ B) @ torch.linalg.pinv(
            R + torch.t(B) @ P[i] @ B) @ (torch.t(B) @ P[i] @ A)      
    
    # Create a list of N elements
    K = [None] * N
    #u = [None] * N
 
    # For i = 0, ..., N - 1
    for i in range(N):
 
        # Calculate the optimal feedback gain K
        K[i] = -torch.linalg.pinv(R + torch.t(B) @ P[i+1] @ B) @ torch.t(B) @ P[i+1] @ A
 
        #u[i] = K[i] @ x_error
 
    # Optimal control input is u_star
    #u_star = u[N-1]
 
    return K
A = torch.tensor([[1,0,0.01,0],[0,1,0,0.01],[0,0,1,0],[0,0,0,1]])
B = torch.tensor([[0,0],[0,0],[0.01,0],[0,0.01]])
Q = torch.diag(torch.tensor([35,35,1,1])).to(torch.float)
R = torch.eye(2)*100

class ExpertLQR(torch.nn.Module):
    def __init__(self,A=A, B=B, Q=Q, R=R, H=0.8, dt=0.01, dvel=1 ,device='cpu'):
        super(ExpertLQR, self).__init__()
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.H = H
        self.dt = dt
        self.dvel = dvel
        self.K = lqr( self.A, self.B, self.Q, self.R, self.H, self.dt)
        self.device = device
        
    def forward(self,x):
        
    
        #x[2:] = self.dvel - x[2:]
        #u = -K[0] @ x
        
        #x_ = [None] * (int(self.H/ self.dt) + 1)
        #x_[0]= x
        #u = [None] * int(self.H/ self.dt)
   
        #for i in range(len(self.K)):
        #    u[i] =  -self.K[i] @ x_ [i]
        #    x_[i+1] = A @ x_[i] + B @ u[i]
        u_star = -self.K[0].to(self.device) @ x 
        return u_star #,x_, self.K 
    
    def get_K(self):
        return self.K
    
if __name__ == "__main__":
    from env import Env
    import time 
    A = torch.tensor([[1,0,0.01,0],[0,1,0,0.01],[0,0,1,0],[0,0,0,1]])
    B = torch.tensor([[0,0],[0,0],[0.01,0],[0,0.01]])
    Q = torch.diag(torch.tensor([35,35,1,1])).to(torch.float)
    R = torch.eye(2)*100
    env = Env(0)
    expert = ExpertLQR(A,B,Q,R,H=0.8)
    K = expert.get_K()
    
    #obs= [diff[0],diff[1],theta,avel[2],vel[0],vel[1]]
    obs = env.reset()
    x = torch.zeros(4,1)
    obs_tensor= obs.view(-1,1)
    #x_star_tensor = torch.tensor([obs[0],obs[1],obs[4],obs[5]]).view(-1,1)
    u_tensor = torch.zeros(2,1)
  
    #goal for waypoi
    # nts
    goals = env.path
    idx = env.idx 
    
    for i in range(goals.shape[0]):
        print(i)
        for k in K:
            obs = torch.tensor(obs).view(-1,1)
            obs_tensor = torch.cat((obs_tensor,obs.view(-1,1)),dim=1)
            
            #x[2:] = obs[4:]
            #x[:2] = obs[:2]
            u =  -k @ obs
            u_tensor = torch.cat((u_tensor,u.view(-1,1)), dim=1)
            obs,reward,done,_,_ = env.step(u)
            #if idx != env.idx: 
            #    idx = env.idx
            #    break
            time.sleep(0.01)
            if done:
                print("done.")
                break
        if done:
            break
                
        
        
        #x[2:] = obs[4:]
                #x[:2] = obs[:2]
        #u,x_star = expert(x)
        #x_star_tensor = torch.cat((x_star_tensor,*x_star), dim=1)
        #u_tensor = torch.cat((u_tensor,*u), dim=1)
        #for uu in u: 
        #    obs_tensor = torch.cat((obs_tensor,obs.view(-1,1)),dim=1)
        #    obs,reward,done,_,_ = env.step(uu)
           #torch.tensor(self.obs).to(torch.float), self.reward, torch.tensor(done).to(torch.float), torch.tensor([0]).to(torch.float), {}     
        #    if done:
        #       print("done.")
        #        break
        #if done:
        #    break
    import matplotlib.pyplot as plt
    t = [100*i for i in range(51)]
    fig,ax = plt.subplots(5,1)
    ax[0].plot(obs_tensor[0,:],label="robot x err" ) 
    #ax[0].plot(x_star_tensor[0,:],label="lqr x err")
    ax[0].legend()
    ax[1].plot(obs_tensor[1,:],label="robot y err")
    #ax[1].plot(x_star_tensor[1,:],label="lqr y err")
    ax[1].legend()
    ax[2].plot(obs_tensor[2,:],label="robot vx err")
    #ax[2].plot(x_star_tensor[2,:],label="lqr vx err")
    ax[2].legend()
    ax[3].plot(obs_tensor[3,:],label="robot vy err")
    #ax[3].plot(x_star_tensor[3,:],label="lqr vy err")
    ax[3].legend()
    ax[4].plot(u_tensor[0,:],label="Fx")
    ax[4].plot(u_tensor[1,:],label="Fy")
    ax[4].legend()
  
   
    plt.show()
