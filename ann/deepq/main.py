from environments import Environment
from agents import DeepQAgent
import os, warnings, sys
# hide warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

cartpole = 'CartPole-v0'
mountaincar = 'MountainCar-v0'
# stockmarket = 'StockMarket'

current_env = mountaincar
environment = Environment(current_env)

# get the shape of the observation and action space
state_num  = environment.env.observation_space.shape[0]
action_num = environment.env.action_space.n

agent = DeepQAgent(state_num, action_num, current_env)

if len(sys.argv) > 1 and sys.argv[1] == 'train':
    environment.train(agent)
else:
    agent.is_training = False
    environment.run(agent)
