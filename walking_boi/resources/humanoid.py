from numpy.random.mtrand import f
import pybullet as p
import os
import math
import numpy as np
import time

class Humanoid:
    def __init__(self, client) -> None:
        f_name = os.path.join(os.path.dirname(__file__), 'humanoid/robot.urdf') 
        self.humanoid = p.loadURDF(fileName=f_name, basePosition=[0,0, 0], physicsClientId=client)
        self.client = client
        # rotor Joint indices
        self.joint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        
        #The minimum value that joint can achive (angles given in radians)
        self.joint_min = []
        
        #The maximum value that joint can achive (angles given in radians)
        self.joint_max = []
        
        self.num_joints = p.getNumJoints(self.humanoid)
        
        for i in self.joint_indices:
            joint_info = p.getJointInfo(self.humanoid , i)
            self.joint_min.append(joint_info[8])
            self.joint_max.append(joint_info[9])
            
        
        self.base_pos_and_orient = p.getBasePositionAndOrientation(self.humanoid)
          
        
        # joint speed
        self.joint_speed = 0
        
        
        # Drag constants (Todo: Domain Randomize here)
        self.c_rolling = 0.2
        self.c_drag = 0.01
        
    def get_ids(self):
        return self.humanoid, self.client
    
    def get_humanoid_id(self):
        return self.humanoid
    
    
    def apply_action(self, action : np.array):
        # expects action to be of shape (1,8) and the values between -1 and 1
        
        # might delete this for performance reasons
        action[action < -1] = -1
        action[action > 1] = 1
         
        p.setJointMotorControlArray(self.humanoid, self.joint_indices, controlMode=p.VELOCITY_CONTROL, forces=action*50)
    
    
    def get_observation(self):
        # Get the position and orientation of the car in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.humanoid, self.client)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        # Get the velocity of the car
        vel = p.getBaseVelocity(self.humanoid, self.client)[0][0:2]

        # Concatenate position, orientation, velocity
        observation = (pos + ori + vel)

        for i in self.joint_indices:
            joint_state = p.getJointState(self.humanoid, i)
            observation = observation +  (Humanoid.normalize_angle(joint_state[0], self.joint_min[i], self.joint_max[i]),)
            observation = observation + (joint_state[1],)

        return observation
    
    
    def reset(self): 
        p.resetBasePositionAndOrientation(self.humanoid,  self.base_pos_and_orient[0], self.base_pos_and_orient[1])
    
        for i in range(self.num_joints):
            p.resetJointState(self.humanoid, i, 0,0)
        
        
        # Change the friction for the left and right foot
        p.changeDynamics(self.humanoid, 2, lateralFriction = 0.5)
        p.changeDynamics(self.humanoid, 7, lateralFriction = 0.5)
        
        
        pass
    
    
    def normalize_angle(value: float, min: float, max: float) -> float:
        if max == min:
            return 0 
        return (value- min)/(max-min)