import gym
from gym import Env
from gym import spaces
import numpy as np

# environment wrapper
class Environment:
    def __init__(self):
        # self.env = gym.make("CartPole-v0")
        self.env = StockMarketEnv()

    # patience threshold is the number of episodes the rewards did not improve
    def train(self, agent, patience=1000):
        print("Training agent")
        # episodes e
        e = 0
        e_without_improvement = 0
        reward_max_obtained = 0

        while e_without_improvement < patience:
            s = self.env.reset()
            reward_total = 0
            done = False
            e += 1
            while not done:
                # get action a from state s
                a = agent.act(s)
                print(a, end="")
                # print(" %r" % agent.is_exploring(), end="")
                # print(" %0.2f  " % agent.epsilon, end="")
                s_, r, done, info = self.env.step(a)

                if done: # yep we are done.
                    s_ = None
                agent.observe( (s, a, r, s_) )
                s = s_
                reward_total += r

            print(" %02.0f" % reward_total)
            # save everytime reward is better
            if reward_total > reward_max_obtained:
                reward_max_obtained = reward_total
                e_without_improvement = 0
                agent.save_model()
            else:
                # dont penalize exploration
                if not agent.is_exploring():
                    e_without_improvement = e_without_improvement + 1

    # run the simulations with number of episodes
    def run(self, agent, episodes=10):
        agent.training = False
        for e in range(episodes):
            s = self.env.reset()
            done = False
            while not done:
                self.env.render()
                # get action a from state s
                a = agent.act(s)
                print(a, end="")
                s_, r, done, info = self.env.step(a)
                agent.observe( (s, a, r, s_) )
                s = s_
            print("")

class StockMarketEnv(Env):
    def __init__(self):
        self.state = 0
        self.low = -1
        self.high = 1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, shape=(3,1))
        self.data = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
        self.count = 0

    # returns the first state
    def reset(self):
        self.count = 0
        self.state = np.array([0,0,self.count])
        # numpy so it can be reshaped
        return self.state

    def step(self, action):
        if action == 0:
            reward = 100
        else:
            reward = 0

        self.count += 1
        if self.count < len(self.data) - 1:
            done = False
        else:
            done = True
        self.state = np.array([0,0,self.count])

        return self.state, reward, done, {}

    def render(self, close=False):
        # dont do anything
        return 0
