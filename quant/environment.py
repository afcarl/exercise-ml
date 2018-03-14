from enum import Enum, unique
import numpy as np
from abc import ABC

@unique
class Action(Enum):
    DO_NOTHING = 0
    BUY = 1
    SELL = 2

ACTION_SPACE = [Action.DO_NOTHING, Action.BUY, Action.SELL]

class Env(ABC):
    def __init__(self):
        super(Env, self).__init__()
        self.actionSpace = ACTION_SPACE
        self.reset()

    # resets the environment
    def reset(self):
        self.done = False
        self.data = self.__generate_data()

    # private function to generate data
    def __generate_data(self):
        pass

    # move one time step
    def step(self):
        pass

    # render the simulation
    def render(self):
        pass


class Linear(Env):
    def __init__(self):
        super(Env, self).__init__()
        self.reset()

    def __generate_data(self):
        # Generate linear data
        data = np.arange(200)
        return data
