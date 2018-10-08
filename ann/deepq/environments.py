import gym
import time
from simulations import StockMarket

# environment wrapper
class CartPoleEnvironment:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.global_time_start = time.time()
        self.save_frequency = 10

    # run the simulations with number of episodes
    def run(self, agent, episodes=1000):
        for e in range(episodes):
            s = self.env.reset()
            self.episode_time_start = time.time()
            reward_total = 0
            done = False

            while not done:
                self.env.render()

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

            end = time.time()
            global_elapsed = end - self.global_time_start
            episode_elapsed = end - self.episode_time_start
            # print("%04.0f" % e, "%04.2f" % episode_elapsed, reward_total, end="\r")

            # save every 10th episode
            if(e % self.save_frequency == 0):
                agent.save_model()
