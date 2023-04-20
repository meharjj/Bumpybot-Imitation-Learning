#from multiprocessing.context import BaseContext
import pybullet as p
from pybullet import getEulerFromQuaternion as Q2E
from pybullet import getQuaternionFromEuler as E2Q
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np
import EnvCreator
from requests import head
import torch 

class Env:
    def __init__(self, headless=False,device='cpu'):
        if headless:
            self.client = bc.BulletClient(connection_mode=p.DIRECT)
           
        else:
            
            self.client = bc.BulletClient(connection_mode=p.GUI)
           
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
       
        self.mass= 20
        self.max_acc = 0.2 
        self.mass_matrix = 0.5* self.mass * 0.3048**2
        self.max_aacc = 0.2
        self.ref_vel = 0.5
        self.max_vel = 1
        self.max_avel = 1
        self.min_err = 0.1
        self.max_err = 0.5
        self.n = 1
        self.x_dim = 4
        self.u_dim = 2
        self.dt = 0.01
    def reset(self):
        self.setup()
        self.obs = self.get_obs()
        return torch.tensor(self.obs).to(torch.float)

    def setup(self):
        ## Initiate simulation
        self.client.resetSimulation()

        ## Set up simulation
        self.client.setTimeStep(self.dt)
        self.client.setPhysicsEngineParameter(numSolverIterations=int(30))
        self.client.setPhysicsEngineParameter(enableConeFriction=0)
        self.client.setGravity(0,0,-9.8)
       

        # create a ground plane 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane.urdf")
        p.changeDynamics(ground, -1, lateralFriction = 0.0, spinningFriction=0.0)


        #create walls
        #wall_path_file= "./maps/L-shape.png"
        #env_c =EnvCreator.envCreator(wall_path_file,
        #                             flip= False
        #                             )
        #env_urdf = env_c.get_urdf(output_dir=".")
        #self.walls = p.loadURDF(env_urdf,useFixedBase=True)
        #p.changeDynamics(self.walls, -1, lateralFriction = 0.0, spinningFriction=0.0)

        
        #get A* path
        #start = (0,0)
        #target = (6,14)
        #filter_dist = 0.45
        #path = env_c.get_path(start,target,filter_dist) 
        #self.path = np.array([*path,path[-1]])##see code for options
        #print(self.path)
        #self.path = np.array([[0,0],[0.5,0.5],[0.5,0.5],[0.5,0.5]])
       
        #path_urdf = env_c.path2urdf(output_dir=".")
        #path_asset = p.loadURDF(path_urdf,useFixedBase=True)
        
        self.path = np.array([[0,0],[0,0.5],[0,1],[0,1.5],[0,2],[0.5,2],[1,2],[1.5,2],[2,2],[2,1.5],[2,1],[2,0.5],[2,0],[1.5,0],[1,0],[0.5,0],[0,0],[0,0]])

        #create robot
        self.robot = p.loadURDF(
            "./robot.urdf",
            basePosition = [0,0,0.6319/2],
            baseOrientation= E2Q([0,0,0]),
            flags= p.URDF_USE_IMPLICIT_CYLINDER
        )
        
        self.idx = 1


    def get_obs(self):
       self.pose,ori_q=p.getBasePositionAndOrientation(self.robot)
       self.vel,avel= p.getBaseVelocity(self.robot)
       ori=Q2E(ori_q)[2]
       ori_to_next = np.arctan2(self.path[self.idx+1,0] - self.pose[0],self.path[self.idx+1,1]-self.pose[1])
       #print(ori_to_next)
       if self.check_waypoints(self.pose[:2],self.path,self.idx):
           self.idx += 1
       #print(ori)
       #print(self.idx)
       target = self.path[self.idx]
       #print(target)

       diff= target-self.pose[:2]
       diff_vel = np.array([np.sin(ori_to_next),np.cos(ori_to_next)])*self.ref_vel - np.array(self.vel[:2]) 
       self.theta = np.arctan2(diff[0],diff[1])
       #theta = theta*180/np.pi
       #print(theta)
       #vel,avel= p.getBaseVelocity(self.robot)

       #obs= [diff[0],diff[1],theta,avel[2],diff_vel[0],diff_vel[1]]
       obs= [diff[0],diff[1],diff_vel[0],diff_vel[1]]
     
       return obs

    def check_waypoints(self,robot,path,idx):
        """
        Robot: [2]
        Waypts: [nx2]
        idx: [1]
        path[idx]: [2]
        Returns True where target needs to be updated to next waypoint, False otherwise
        """
        m = (path[idx+1,1] - path[idx-1,1] + 1e-7)/(path[idx+1,0] - path[idx-1,0] + 1e-7)
        b = - m * path[idx+1,0] + path[idx+1,1]

        m2 = - 1 / m
        b2 = - m2 * path[idx,0] + path[idx,1]

        R = m2 * robot[0] + b2 - robot[1]
        W = m2 * path[idx-1,0] + b2 - path[idx-1,1]

        return (np.linalg.norm(robot-path[idx]) < self.min_err) or (np.sign(R) != np.sign(W) and np.linalg.norm(robot-path[idx]) < self.max_err) #increment if the robot is past the waypoint


    def get_rewards(self):
        #contact_reward = 0 if self.in_contact else 1
        #heading_rew = 1/(1+self.theta))
        #print(self.obs)
        target_rew = 1/(1+abs(self.obs[0]))+  1/(1+abs(self.obs[1]))
        #print(target_rew)
        vel_rew = 1/(1+abs(self.obs[2])) + 1/(1+abs(self.obs[3]))
        #print(vel_rew)
        control_rew = 0#1/(1+abs(self.force[0])/self.mass) + 1/(1+abs(self.force[1])/self.mass)
        #print(self.force)
        #print(control_rew)
        reward = (target_rew + vel_rew + control_rew)/4
        return reward 

    def step(self, action): # action[zero] is the desired velocity and action[one] is the desired heading 
       #self.in_contact = 0
       """
       print(action)
       fd = torch.linalg.norm(action).item()
       #print(fd)
       th_d = torch.atan2(action[1],action[0]).item()
      
       force = self.mass * fd * np.array([np.sin(th_d),np.cos(th_d),0])
       print(force)
       torque = np.array([0,0,th_d-self.obs[3]])
       
       f_next =np.linalg.norm( np.array([self.obs[4]+force[0]*self.dt,self.obs[5]+force[1]*self.dt,0]))
       avel_next = np.linalg.norm( np.array([0,0,self.obs[3]+torque[2]*self.dt]))

       if f_next > self.max_vel: 
           force = np.array([0,0,0])
       elif -f_next < -self.max_vel: 
           force = np.array([0,0,0])
       
       #print(force)
       if avel_next> self.max_avel: 
           torque = np.array([0,0,0])
       elif avel_next< -self.max_avel: 
           torque = np.array([0,0,0])

       p.applyExternalForce(self.robot,-1,force,[0,0,0],p.WORLD_FRAME)
       p.applyExternalTorque(self.robot,-1,torque,p.WORLD_FRAME)
       #print(force)
       
       """
       self.force = [self.mass*action[0],self.mass*action[1],0]
       p.applyExternalForce(self.robot,-1,self.force,[0,0,0],p.LINK_FRAME)
       
       self.client.stepSimulation()
     
       #ctx = p.getContactPoints(self.robot,self.walls)
       #if len(ctx) > 0:
       #    self.in_contact= 1
        
       self.obs= self.get_obs()
       
       self.reward = self.get_rewards()
       #print(self.idx)
       #print(len(self.path)-2)
       #print()
       if self.idx == len(self.path)-2: 
           done =   True #[1] 
       else: 
           done = False #[0]
       #print(self.reward)
       #print(done)
       return torch.tensor(self.obs).to(torch.float), self.reward, done, torch.tensor([0]).to(torch.float), {}
    
    def close(self):
        p.disconnect()

if __name__=="__main__":
    env=Env(False)
    env.reset()
    import time 
    for i in range(10000):

        ob,rew,done,trunc,info= env.step([1,0])
        if sum(done):
            env.close()
            break 
        time.sleep(.02)

