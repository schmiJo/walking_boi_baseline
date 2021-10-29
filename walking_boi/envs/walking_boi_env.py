import gym
import numpy as np
import pybullet as p

import time

import math
 

from walking_boi.resources.humanoid import Humanoid 
from walking_boi.resources.plane import Plane 

class WalkingBoiEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
  
    def __init__(self):
        
        self.action_space = gym.spaces.box.Box( 
            low = np.array([-1]*15),
            high = np.array([1] * 15)
        )
        
        
        self.observation_space = gym.spaces.box.Box(
            low = np.array([-1]*36),
            high = np.array([1] * 36)
            
        )
        
        
        
        self.np_random, _ = gym.utils.seeding.np_random()
        
        
        #self.client = p.connect(p.GUI)
        self.client = p.connect(p.DIRECT)
        # Reduce length of episodes for RL algorithms
        p.setTimeStep(1/30, self.client)
        self.prev_distance_to_origin = 0
        
        self.reward_factor = 20
        
        self.humanoid = None
        self.plane = None 
        p.resetSimulation(self.client)
        p.setGravity(0,0,-1)
        
        #load the Humanoid
        self.plane = Plane(self.client)
        self.humanoid = Humanoid(self.client)
        
    
    def step(self, action):
        #Feed the action to the humanoid ang get observation of the state
        self.humanoid.apply_action(action)
        p.stepSimulation()
        humanoid_ob = np.hstack(self.humanoid.get_observation())
        
        # Compute rewards as in distance to origin
        
        x_humanoid = humanoid_ob[0];
        y_humanoid = humanoid_ob[1];
    
         
        dist_to_origin = -y_humanoid
        reward = max(dist_to_origin - self.prev_distance_to_origin, 0) * self.reward_factor;
        
        self.prev_distance_to_origin = dist_to_origin
        
        #Done by running off boundaries
        if (x_humanoid >= 100 or x_humanoid <= -100 or
                y_humanoid >= 100 or y_humanoid <= -100):
            self.done = True
            reward += 50
         
        ob = np.array(humanoid_ob, dtype=np.float32)
        
        
        reward += 0.1
        
        contactPoints =   p.getContactPoints(self.humanoid.get_humanoid_id(), self.plane.get_plane_id() );
         
        for contact in contactPoints:
            if contact[1] == 1 and contact[2] == 0: 
                
                # The Id of the left and right foot are 2 and 7
                if contact[3] == 2 or contact[3] == 7:
                    continue;
                
                # If anything other than the feet touch the ground penalize and be done with the episode
                
                reward -= 10
                
                
                self.done = True                

        
        return ob, reward, self.done, dict()
    
    
    def reset(self):
        
        self.humanoid.reset()
        
        self.done = False
        
        # Set the goal to a random target
        #x = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #     self.np_random.uniform(-5, -9))
        #y = (self.np_random.uniform(5, 9) if self.np_random.randint(2) else
        #     self.np_random.uniform(-5, -9))
        #self.goal = (x, y)
        
        
        # Get observation to return
        humanoid_ob = self.humanoid.get_observation()
         
        return np.hstack(humanoid_ob)
        
        
    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
    
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]