from environment import Linear
from agents import DeepSARSA
env = Linear()
env.reset()

agent = DeepSARSA(10,3)
