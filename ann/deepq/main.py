from environments import CartPoleEnvironment
from agents import DeepQAgent

# doing the cart pole with deep q
cartpole = CartPoleEnvironment()

# get the shape of the observation and action space
state_num  = cartpole.env.observation_space.shape[0]
action_num = cartpole.env.action_space.n

agent = DeepQAgent(state_num, action_num)

try:
    while True:
        cartpole.run(agent)
finally:
    agent.save_model()
