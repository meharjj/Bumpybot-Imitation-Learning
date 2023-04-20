import matplotlib.pyplot as plt
import torch
import time
# policy dict like ~ policy_dict={'LQR':expert,'IL':policy}
def Plot_all(Env,policy_dict,max_iter=3_000,device="cpu"):
    
    envs = {}
    obs = {}
    xs = {}
    us = {}
    dones = {}
    for k,v in policy_dict.copy().items():
        policy_dict[k] = v.to(device)
        policy_dict[k].device = device
        envs[k] = Env(1,device=device)
        obs[k] = [envs[k].reset()]
        xs[k] = []
        us[k] = []
        dones[k] = False

        for t in range(max_iter):
            if not dones[k]:
                u = policy_dict[k](obs[k][t])
                x,_,done,_,_ = envs[k].step(u)
                #time.sleep(0.01)
                obs[k].append(x)
                xs[k].append(torch.tensor([*envs[k].pose[:2],*envs[k].vel[:2]]))
                us[k].append(u)
                dones[k] = done
            else:
                envs[k].close()
                break

    if  all(dones.values()):
        print("Test Success.")
    else:
        print("Test Failure.")
    print()

    fig,ax = plt.subplots()
    fig.suptitle("Trajectory")
    for k,v in policy_dict.items():
        ax.plot([x[0] for x in xs[k]],[x[1] for x in xs[k]],label=k)
    ax.set(xlim=(-3,3), ylim=(-3,3))
    ax.legend()
    fig.set_tight_layout(True)

    fig_v,ax_v = plt.subplots(2)
    fig_v.suptitle("Velocity")
    ax_v[0].get_xaxis().set_visible(False)
    for k,v in policy_dict.items():
        ax_v[0].plot([x[2] for x in xs[k]],label='{} vx'.format(k))
        ax_v[1].plot([x[3] for x in xs[k]],label='{} vy'.format(k))
    ax_v[0].set(ylim=(-1,1))
    ax_v[0].legend()
    ax_v[1].set(ylim=(-1,1))
    ax_v[1].legend()
    fig_v.set_tight_layout(True)

    fig_u,ax_u = plt.subplots(2)
    fig_u.suptitle("Control")
    ax_u[0].get_xaxis().set_visible(False)
    for k,v in policy_dict.items():
        ax_u[0].plot([u[0] for u in us[k]],label='{} ux'.format(k))
        ax_u[1].plot([u[1] for u in us[k]],label='{} uy'.format(k))
    ax_u[0].legend()
    ax_u[1].legend()
    fig_u.set_tight_layout(True)

    plt.show()

if __name__ == '__main__':
    from env import Env
    from expertlqr import ExpertLQR
    from Pytorch_wrapper import Expert
    expert = ExpertLQR()
    best_il = Expert('best_il_model.pt')
    il = Expert('il_model.pt')
    policy_dict = {'LQR':expert,'il':il,'best_il':best_il}
    Plot_all(Env,policy_dict)

