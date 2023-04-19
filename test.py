from env import Env 
from expert import Expert


if __name__=="__main__":
    env=Env(False)
    x=env.reset()
    expert= Expert("il_model.pt")
    import time

    for i in range(10000):
        u = expert(x)
        ob,rew,done,trunc,info= env.step(u)
        print(rew)
        if done:
            env.close()
            break 
        time.sleep(.01)

