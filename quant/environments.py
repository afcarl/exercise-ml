import gym
from gym import Env
from gym import spaces
import numpy as np

# environment wrapper
class Environment:
    def __init__(self):
        # self.env = gym.make("CartPole-v0")
        self.env = StockMarketEnv()

class StockMarketEnv(Env):
    def __init__(self):
        self.state = 0
        self.low = -1
        self.high = 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, shape=(25,1))

    # returns the first state
    def reset(self):
        return self.observation_space

    def step(self, action):
        # check for invalid action
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        reward = 0
        done = True
        return self.state, reward, done, {}

    def render(self, close):
        # dont do anything
        return 0
