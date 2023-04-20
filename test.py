from env import Env 
from Pytorch_wrapper import Expert
from expertlqr import ExpertLQR

if __name__=="__main__":
    env=Env(False)
    ob=env.reset()
    expert= Expert("il_model.pt").to("cpu")
    #expert = ExpertLQR()
    import time

    for i in range(10000):
        u = expert(ob)
        ob,rew,done,trunc,info= env.step(u)
        print(rew)
        if done:
            print("done")
            env.close()
            break 
        time.sleep(.01)

