import gym
import time
import sys

# environment wrapper
class Environment:
    def __init__(self, name):
        self.env = gym.make(name)

    # patience threshold is the number of episodes the rewards did not improve
    def train(self, agent, patience=100):
        # episodes e
        e_without_improvement = 0
        reward_max_obtained = 0

        while e_without_improvement < patience:
            s = self.env.reset()
            reward_total = 0
            done = False
            while not done:
                # get action a from state s
                a = agent.act(s)
                s_, r, done, info = self.env.step(a)
                # check if the epsilon greedy has went to the lowest
                # possible value, and then stop. its already in full exploit
                if agent.epsilon == agent.epsilon_min:
                    done = True
                if done: # yep we are done.
                    s_ = None
                agent.observe( (s, a, r, s_) )
                agent.replay()
                s = s_
                reward_total += r

            # save everytime reward is better
            if(reward_total > reward_max_obtained):
                print('%02.0f' % e_without_improvement, reward_total)
                sys.stdout.flush()
                reward_max_obtained = reward_total
                e_without_improvement = 0
                agent.save_model()
            else:
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
                s_, r, done, info = self.env.step(a)
                agent.observe( (s, a, r, s_) )
                s = s_

# class StockMarket:
#     def __init__(self):
#         self.state = 0
#         self.observation_space = []
#         self.action_space.n = 3
#
#     def reset():
#         return False
#
#     def render():
#         return 0
#
#     def step():
#         return 0
