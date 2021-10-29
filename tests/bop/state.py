class State:
    
    
    def __init__(self, obs, hx ) -> None:
        """
        obs: Observations made by the robot, (can include joint position, ray perception etc)
        hx: Hidden states of the action network
        """
        self.obs = obs
        self.hx = hx
    
    
    