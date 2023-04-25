import gym
from gym import spaces
from gym import wrappers
import numpy as np
import rospy

class GoalConditionedEnv(gym.Env):

    def __init__(self, env, params):
        self.env = env
        self.params = params
        
        self.goal_distribution = True if self.params['alg']['goal_dist']=='normal' else False
        self.new_low_ac = -1
        self.new_high_ac = 1
        if isinstance(self.env, gym.Env):
            if self.params['env']['env_name'] == 'reacher':
                self.new_low_obs = -0.3
                self.new_high_obs = 0.3
                self.shape_ac = (7,)
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(20,))
                #self.action_space = spaces.Box(low=self.new_low_ac, high=self.new_high_ac, shape=self.shape_ac)
                self.action_space = self.env.action_space
                #goal_space = gym.spaces.Box(low=self.goal_distribution.min(), high=self.goal_distribution.max(), shape=self.goal_distribution.shape, dtype=np.float32)
                #self.observation_space = gym.spaces.Tuple((self.env.observation_space, goal_space))
                
                
            elif self.params['env']['env_name']== 'antmaze':
                self.new_low_obs = -4.0
                self.new_high_obs = 20.0
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(2,))
                self.action_space = self.env.action_space
                self.shape_ac = (8,)
            
            else:
                self.action_space = self.env.action_space
                self.observation_space = self.env.observation_space
                
        self.goal = None
        self.state = None
        self.stepK = 0
        self.reset()
    
    def reset(self):
        # Add code to generate a goal from a distribution
        if self.params['env']['env_name'] == 'reacher':
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            self.state[-3:] = self.goal - self.params['alg']['relative_goal']*self.state[-6:-3]
            return self.state
        else:
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            rospy.logwarn('Given target pose: ' + str(self.goal))
            self.env.publish_marker(self.goal)
            self.env.setGoal(self.goal)
            self.state[-3:] = self.goal 
            #- self.params['alg']['relative_goal']*self.state[-6:-3]
            return self.state
    
    def render(self, mode):
        return self.env.render(mode)

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        if self.params['env']['env_name'] == 'reacher' :
            self.state, new_reward, done, info = self.env.step(action)
            return self.state, new_reward, done, info
        else :
            self.state, new_reward, done, info = self.env.step(action)
            rospy.logwarn('current target pose:' + str( self.goal))
            rospy.logwarn('current reward : ' + str(new_reward))
            return self.state, new_reward, done, info
        
    def createState(self):
        ## Add the goal to the state
        return np.append(self.state, self.goal - self.params['alg']['relative_goal']*self.state)
    
    def sample_goal(self, normal=False):
        if normal :
            goal = np.random.normal(loc=[20, -0.7, 0.0], scale=[0.3, 0.4, 0.0])
        else:
            goal = np.random.uniform(low=[-2, -1, 0.0], high=[2, 1, 0.0])
        return goal 
        
class GoalConditionedEnvV2(GoalConditionedEnv):

    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.goal_distribution = True if self.params['alg']['goal_dist']=='normal' else False
        self.new_low_ac = -1
        self.new_high_ac = 1
        if isinstance(self.env, gym.Env):
            if self.params['env']['env_name'] == 'reacher':
                self.new_low_obs = -0.3
                self.new_high_obs = 0.3
                self.shape_ac = (7,)
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(20,))
                #self.action_space = spaces.Box(low=self.new_low_ac, high=self.new_high_ac, shape=self.shape_ac)
                self.action_space = self.env.action_space
                #goal_space = gym.spaces.Box(low=self.goal_distribution.min(), high=self.goal_distribution.max(), shape=self.goal_distribution.shape, dtype=np.float32)
                #self.observation_space = gym.spaces.Tuple((self.env.observation_space, goal_space))
                
                
            elif self.params['env']['env_name']== 'antmaze':
                self.new_low_obs = -4.0
                self.new_high_obs = 20.0
                self.observation_space = spaces.Box(low=self.new_low_obs, high=self.new_high_obs, shape=(2,))
                self.action_space = self.env.action_space
                self.shape_ac = (8,)
                  
        self.goal = None
        self.state = None
        self.stepK = 0
        self.reset()
    
    def sample_goal(self, normal=False):
        if self.params['env']['env_name'] == 'reacher':
            if normal :
                goal = np.random.normal(loc=[0.2, -0.7, 0.0], scale=[0.3, 0.4, 0.05])
            else:
                goal = np.random.uniform(low=[-0.6, -1.4, -0.4], high=[0.8, 0.2, 0.5])
        elif self.params['env']['env_name']== 'antmaze':
            if normal :
                goal = np.random.normal(loc=[0, 8], scale=[4, 4])
            else:
                goal = np.random.uniform(low=[-4, 4], high=[20, 4])
        return goal
    
    def reset(self):
        # Add code to generate a goal from a distribution
        self.stepK = 0
        if self.params['env']['env_name'] == 'reacher':
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            self.state[-3:] = self.goal 
            return self.state
        elif self.params['env']['env_name']== 'antmaze':
            self.state = self.env.reset()
            self.goal = self.sample_goal(self.goal_distribution)
            return self.createState() 

    def step(self, action):
        ## Add code to compute a new goal-conditioned reward
        self.stepK +=1
        if self.params['env']['env_name'] == 'reacher' :
            self.state, new_reward, done, info = self.env.step(action)
            
            if self.stepK % self.params['alg']['goal_frequency'] == 0:
                self.goal =self.sample_goal(self.goal_distribution)
                self.state[-3:] = self.goal - self.params['alg']['relative_goal']*self.state[-6:-3]
                
            return self.state, new_reward, done, info 
        
        elif self.params['env']['env_name']== 'antmaze':
            self.state, reward, done, info = self.env.step(action)
            
            if self.stepK % self.params['alg']['goal_frequency'] == 0:
                self.goal =self.sample_goal(self.goal_distribution)
                
            new_reward = -np.linalg.norm(self.state[:2] - self.goal)
            return self.createState(), new_reward, done, info 
        
    def createState(self):
        ## Add the goal to the state
        return np.append(self.state, self.goal)
        