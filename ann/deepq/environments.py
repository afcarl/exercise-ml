import gym
import time

# environment wrapper
class CartPoleEnvironment:
    def __init__(self):
        self.env = gym.make('MountainCar-v0')
        self.global_time_start = time.time()
        self.save_frequency = 10

    # run the whole simulation
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

                if done: # yep we are done.
                    s_ = None

                agent.observe( (s, a, r, s_) )
                agent.replay()

                s = s_
                reward_total += r

            end = time.time()
            global_elapsed = end - self.global_time_start
            episode_elapsed = end - self.episode_time_start
            print("Episode: %04.0f" % e, time.strftime("%H:%M:%S", time.gmtime(global_elapsed)), "%02.2fs" % episode_elapsed, reward_total, end="\r")
            # save every 10th episode
            if(e % self.save_frequency == 0):
                agent.save_model()
