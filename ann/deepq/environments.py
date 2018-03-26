import gym
import time

# environment wrapper
class CartPoleEnvironment:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.time_start = time.time()

    # run the whole simulation
    def run(self, agent):
        s = self.env.reset()
        reward_total = 0

        while True:
            self.env.render()

            # get action a from state s
            a = agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done: # yep we are done.
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()

            s = s_
            reward_total += r

            if done:
                break

        end = time.time()
        elapsed = end - self.time_start
        print(time.strftime("%H:%M:%S", time.gmtime(elapsed)), reward_total, end="\r")
